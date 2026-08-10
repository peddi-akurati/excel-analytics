# Databricks notebook source
# MAGIC %md
# MAGIC # Single-Table Databricks Schema Validation with Row Hash
# MAGIC
# MAGIC Configure the variables in Section 1 and run the notebook.
# MAGIC
# MAGIC Validates one table between two Databricks schemas using:
# MAGIC - total row count
# MAGIC - schema/data-type comparison
# MAGIC - categorical count and distinct-count comparison
# MAGIC - numerical min/max/sum/median comparison
# MAGIC - date/timestamp min/max comparison with timezone reconciliation
# MAGIC - row-level SHA-256 hash comparison
# MAGIC - SOURCE_ONLY / TARGET_ONLY / CHANGED record identification
# MAGIC - mismatch-column drill-down with primary keys and source/target values
# MAGIC
# MAGIC Serverless compatible: no CACHE, PERSIST, UNPERSIST, DBFS mkdir, or report export.

# COMMAND ----------

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from functools import reduce
from typing import Dict, List, Sequence, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Configuration variables

# COMMAND ----------

SOURCE_CATALOG = "main"
SOURCE_SCHEMA = "source_schema"

TARGET_CATALOG = "main"
TARGET_SCHEMA = "target_schema"

TABLE_NAME = "claims"

# Composite primary keys are supported.
PRIMARY_KEY_COLUMNS = ["claim_id"]

# Optional filters. Keep empty to validate the complete table.
SOURCE_WHERE_CLAUSE = ""
TARGET_WHERE_CLAUSE = ""

ROW_HASH_ENABLED = True

# 0 = full table.
# Example: 1000000 = first 1M rows ordered by ROW_ORDER_COLUMNS.
ROW_LIMIT = 0
ROW_ORDER_COLUMNS = PRIMARY_KEY_COLUMNS

# Columns to ignore in row hash and mismatch drill-down.
EXCLUDED_HASH_COLUMNS = [
    # "etl_loaded_ts",
]

# 0 = unlimited mismatch detail rows.
MAX_MISMATCH_DETAILS = 100000

# Timestamp interpretation.
SOURCE_TIMESTAMP_TIMEZONE = "Asia/Kolkata"
TARGET_TIMESTAMP_TIMEZONE = "Asia/Kolkata"
COMPARISON_TIMEZONE = "UTC"
TIMESTAMP_TOLERANCE_SECONDS = 0

# Numeric comparison tolerance.
NUMERIC_ABSOLUTE_TOLERANCE = 0.0
NUMERIC_RELATIVE_TOLERANCE = 0.0

MEDIAN_ACCURACY = 10000

# COMMAND ----------

if not SOURCE_SCHEMA or not TARGET_SCHEMA or not TABLE_NAME:
    raise ValueError(
        "SOURCE_SCHEMA, TARGET_SCHEMA and TABLE_NAME are required."
    )

if not PRIMARY_KEY_COLUMNS:
    raise ValueError(
        "PRIMARY_KEY_COLUMNS must contain at least one key."
    )

spark.conf.set("spark.sql.session.timeZone", "UTC")

RUN_ID = str(uuid.uuid4())
RUN_TS = datetime.now(timezone.utc)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Helpers

# COMMAND ----------

def q(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def fqtn(catalog: str, schema: str, table: str) -> str:
    return f"{q(catalog)}.{q(schema)}.{q(table)}"


def empty_df(schema: str) -> DataFrame:
    return spark.createDataFrame([], schema=schema)


def union_all(
    frames: Sequence[DataFrame],
    schema: str,
) -> DataFrame:
    frames = [x for x in frames if x is not None]
    if not frames:
        return empty_df(schema)

    return reduce(
        lambda a, b: a.unionByName(
            b,
            allowMissingColumns=True,
        ),
        frames,
    )


def read_table(
    catalog: str,
    schema: str,
    table: str,
    where_clause: str,
) -> DataFrame:
    df = spark.table(
        fqtn(catalog, schema, table)
    )
    return df.where(where_clause) if where_clause else df


def classify(field: T.StructField) -> str:
    dt = field.dataType

    if isinstance(
        dt,
        (
            T.ByteType,
            T.ShortType,
            T.IntegerType,
            T.LongType,
            T.FloatType,
            T.DoubleType,
            T.DecimalType,
        ),
    ):
        return "NUMERICAL"

    if isinstance(dt, T.DateType):
        return "DATE"

    if isinstance(dt, T.TimestampNTZType):
        return "TIMESTAMP_NTZ"

    if isinstance(dt, T.TimestampType):
        return "TIMESTAMP"

    return "CATEGORICAL"


def apply_row_limit(
    df: DataFrame,
    row_limit: int,
    order_columns: Sequence[str],
) -> DataFrame:

    if row_limit <= 0:
        return df

    if order_columns:
        return (
            df.orderBy(
                *[
                    F.col(q(c))
                    for c in order_columns
                ]
            )
            .limit(row_limit)
        )

    return df.limit(row_limit)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Load source and target

# COMMAND ----------

source_df = read_table(
    SOURCE_CATALOG,
    SOURCE_SCHEMA,
    TABLE_NAME,
    SOURCE_WHERE_CLAUSE,
)

target_df = read_table(
    TARGET_CATALOG,
    TARGET_SCHEMA,
    TABLE_NAME,
    TARGET_WHERE_CLAUSE,
)

source_columns = {
    c.lower(): c
    for c in source_df.columns
}

target_columns = {
    c.lower(): c
    for c in target_df.columns
}

missing_source_keys = [
    k
    for k in PRIMARY_KEY_COLUMNS
    if k.lower() not in source_columns
]

missing_target_keys = [
    k
    for k in PRIMARY_KEY_COLUMNS
    if k.lower() not in target_columns
]

if missing_source_keys or missing_target_keys:
    raise ValueError(
        "Primary key validation failed. "
        f"source missing={missing_source_keys}, "
        f"target missing={missing_target_keys}"
    )

PRIMARY_KEYS = [
    source_columns[k.lower()]
    for k in PRIMARY_KEY_COLUMNS
]

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Schema comparison

# COMMAND ----------

SCHEMA_REPORT_SCHEMA = """
run_id string,
table_name string,
column_name string,
source_data_type string,
target_data_type string,
source_classification string,
target_classification string,
status string,
run_ts timestamp
"""

source_fields = {
    f.name.lower(): f
    for f in source_df.schema.fields
}

target_fields = {
    f.name.lower(): f
    for f in target_df.schema.fields
}

schema_rows = []

for name in sorted(
    set(source_fields)
    | set(target_fields)
):
    sf = source_fields.get(name)
    tf = target_fields.get(name)

    if sf is None:
        status = "TARGET_ONLY_COLUMN"

    elif tf is None:
        status = "SOURCE_ONLY_COLUMN"

    elif (
        sf.dataType.simpleString()
        != tf.dataType.simpleString()
    ):
        status = "TYPE_MISMATCH"

    elif classify(sf) != classify(tf):
        status = "CLASSIFICATION_MISMATCH"

    else:
        status = "PASS"

    schema_rows.append(
        {
            "run_id": RUN_ID,
            "table_name": TABLE_NAME,
            "column_name": name,
            "source_data_type": (
                sf.dataType.simpleString()
                if sf else None
            ),
            "target_data_type": (
                tf.dataType.simpleString()
                if tf else None
            ),
            "source_classification": (
                classify(sf)
                if sf else None
            ),
            "target_classification": (
                classify(tf)
                if tf else None
            ),
            "status": status,
            "run_ts": RUN_TS,
        }
    )

schema_report = spark.createDataFrame(
    schema_rows,
    schema=SCHEMA_REPORT_SCHEMA,
)

common_fields: Dict[
    str,
    Tuple[T.StructField, T.StructField],
] = {
    name: (
        source_fields[name],
        target_fields[name],
    )
    for name in sorted(
        set(source_fields)
        & set(target_fields)
    )
    if (
        source_fields[name]
        .dataType
        .simpleString()
        ==
        target_fields[name]
        .dataType
        .simpleString()
    )
}

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Total count comparison

# COMMAND ----------

COUNT_REPORT_SCHEMA = """
run_id string,
table_name string,
source_row_count long,
target_row_count long,
difference long,
status string,
run_ts timestamp
"""

source_total_count = source_df.count()
target_total_count = target_df.count()

count_report = spark.createDataFrame(
    [{
        "run_id": RUN_ID,
        "table_name": TABLE_NAME,
        "source_row_count": source_total_count,
        "target_row_count": target_total_count,
        "difference": (
            target_total_count
            - source_total_count
        ),
        "status": (
            "PASS"
            if source_total_count
            == target_total_count
            else "FAIL"
        ),
        "run_ts": RUN_TS,
    }],
    schema=COUNT_REPORT_SCHEMA,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Categorical comparison

# COMMAND ----------

CATEGORY_REPORT_SCHEMA = """
run_id string,
table_name string,
column_name string,
category_value string,
source_count long,
target_count long,
difference long,
source_distinct_count long,
target_distinct_count long,
status string,
run_ts timestamp
"""

category_frames: List[DataFrame] = []

for _, (sf, tf) in common_fields.items():

    if classify(sf) != "CATEGORICAL":
        continue

    source_distinct = (
        source_df
        .select(
            F.col(q(sf.name))
        )
        .distinct()
        .count()
    )

    target_distinct = (
        target_df
        .select(
            F.col(q(tf.name))
        )
        .distinct()
        .count()
    )

    source_counts = (
        source_df
        .groupBy(
            F.coalesce(
                F.col(
                    q(sf.name)
                ).cast("string"),
                F.lit("<NULL>"),
            ).alias(
                "category_value"
            )
        )
        .count()
        .withColumnRenamed(
            "count",
            "source_count",
        )
    )

    target_counts = (
        target_df
        .groupBy(
            F.coalesce(
                F.col(
                    q(tf.name)
                ).cast("string"),
                F.lit("<NULL>"),
            ).alias(
                "category_value"
            )
        )
        .count()
        .withColumnRenamed(
            "count",
            "target_count",
        )
    )

    category_frames.append(
        source_counts
        .join(
            target_counts,
            "category_value",
            "full",
        )
        .select(
            F.lit(RUN_ID)
            .alias("run_id"),

            F.lit(TABLE_NAME)
            .alias("table_name"),

            F.lit(sf.name)
            .alias("column_name"),

            F.col(
                "category_value"
            ),

            F.coalesce(
                F.col(
                    "source_count"
                ),
                F.lit(0),
            )
            .cast("long")
            .alias(
                "source_count"
            ),

            F.coalesce(
                F.col(
                    "target_count"
                ),
                F.lit(0),
            )
            .cast("long")
            .alias(
                "target_count"
            ),

            (
                F.coalesce(
                    F.col(
                        "target_count"
                    ),
                    F.lit(0),
                )
                -
                F.coalesce(
                    F.col(
                        "source_count"
                    ),
                    F.lit(0),
                )
            )
            .cast("long")
            .alias(
                "difference"
            ),

            F.lit(
                source_distinct
            )
            .cast("long")
            .alias(
                "source_distinct_count"
            ),

            F.lit(
                target_distinct
            )
            .cast("long")
            .alias(
                "target_distinct_count"
            ),

            F.when(
                F.coalesce(
                    F.col(
                        "source_count"
                    ),
                    F.lit(0),
                )
                ==
                F.coalesce(
                    F.col(
                        "target_count"
                    ),
                    F.lit(0),
                ),
                "PASS",
            )
            .otherwise(
                "FAIL"
            )
            .alias(
                "status"
            ),

            F.lit(
                RUN_TS
            )
            .cast("timestamp")
            .alias(
                "run_ts"
            ),
        )
    )

category_report = union_all(
    category_frames,
    CATEGORY_REPORT_SCHEMA,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Numerical comparison

# COMMAND ----------

NUMERIC_REPORT_SCHEMA = """
run_id string,
table_name string,
column_name string,
statistic_name string,
source_value double,
target_value double,
absolute_difference double,
relative_difference double,
status string,
run_ts timestamp
"""

numeric_rows = []


def numeric_stats(
    df: DataFrame,
    column_name: str,
) -> dict:

    c = F.col(
        q(column_name)
    ).cast(
        "double"
    )

    return (
        df
        .agg(
            F.min(c)
            .alias("MIN"),

            F.max(c)
            .alias("MAX"),

            F.sum(c)
            .alias("SUM"),

            F.expr(
                f"percentile_approx("
                f"cast({q(column_name)} as double), "
                f"0.5, {MEDIAN_ACCURACY})"
            )
            .alias("MEDIAN"),
        )
        .first()
        .asDict()
    )


for _, (sf, tf) in common_fields.items():

    if classify(sf) != "NUMERICAL":
        continue

    ss = numeric_stats(
        source_df,
        sf.name,
    )

    ts = numeric_stats(
        target_df,
        tf.name,
    )

    for metric in [
        "MIN",
        "MAX",
        "SUM",
        "MEDIAN",
    ]:
        sv = ss[metric]
        tv = ts[metric]

        if (
            sv is None
            and tv is None
        ):
            abs_diff = None
            rel_diff = None
            status = "PASS"

        elif (
            sv is None
            or tv is None
        ):
            abs_diff = None
            rel_diff = None
            status = "FAIL"

        else:
            sv = float(sv)
            tv = float(tv)

            abs_diff = abs(
                tv - sv
            )

            denominator = max(
                abs(sv),
                abs(tv),
                1.0,
            )

            rel_diff = (
                abs_diff
                / denominator
            )

            allowed = max(
                NUMERIC_ABSOLUTE_TOLERANCE,
                NUMERIC_RELATIVE_TOLERANCE
                * denominator,
            )

            status = (
                "PASS"
                if abs_diff
                <= allowed
                else "FAIL"
            )

        numeric_rows.append(
            {
                "run_id": RUN_ID,
                "table_name": TABLE_NAME,
                "column_name": sf.name,
                "statistic_name": metric,
                "source_value": (
                    None
                    if sv is None
                    else float(sv)
                ),
                "target_value": (
                    None
                    if tv is None
                    else float(tv)
                ),
                "absolute_difference": (
                    abs_diff
                ),
                "relative_difference": (
                    rel_diff
                ),
                "status": status,
                "run_ts": RUN_TS,
            }
        )

numeric_report = (
    spark.createDataFrame(
        numeric_rows,
        schema=NUMERIC_REPORT_SCHEMA,
    )
    if numeric_rows
    else empty_df(
        NUMERIC_REPORT_SCHEMA
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Date / timestamp comparison

# COMMAND ----------

DATETIME_REPORT_SCHEMA = """
run_id string,
table_name string,
column_name string,
classification string,
statistic_name string,
source_value string,
target_value string,
difference_seconds double,
status string,
run_ts timestamp
"""

datetime_rows = []


def datetime_expression(
    column_name: str,
    classification: str,
    assumed_timezone: str,
) -> F.Column:

    c = F.col(
        q(column_name)
    )

    if classification == "DATE":
        return c.cast(
            "date"
        )

    if classification == "TIMESTAMP_NTZ":
        return F.to_utc_timestamp(
            c.cast(
                "timestamp"
            ),
            assumed_timezone,
        )

    return c.cast(
        "timestamp"
    )


for _, (sf, tf) in common_fields.items():

    cls = classify(sf)

    if cls not in {
        "DATE",
        "TIMESTAMP_NTZ",
        "TIMESTAMP",
    }:
        continue

    se = datetime_expression(
        sf.name,
        cls,
        SOURCE_TIMESTAMP_TIMEZONE,
    )

    te = datetime_expression(
        tf.name,
        cls,
        TARGET_TIMESTAMP_TIMEZONE,
    )

    ss = (
        source_df
        .agg(
            F.min(se)
            .alias("MIN"),

            F.max(se)
            .alias("MAX"),
        )
        .first()
        .asDict()
    )

    ts = (
        target_df
        .agg(
            F.min(te)
            .alias("MIN"),

            F.max(te)
            .alias("MAX"),
        )
        .first()
        .asDict()
    )

    for metric in [
        "MIN",
        "MAX",
    ]:
        sv = ss[metric]
        tv = ts[metric]

        if (
            sv is None
            and tv is None
        ):
            difference_seconds = None
            status = "PASS"

        elif (
            sv is None
            or tv is None
        ):
            difference_seconds = None
            status = "FAIL"

        elif cls == "DATE":
            difference_seconds = (
                tv.toordinal()
                - sv.toordinal()
            ) * 86400.0

            status = (
                "PASS"
                if sv == tv
                else "FAIL"
            )

        else:
            difference_seconds = (
                tv.timestamp()
                - sv.timestamp()
            )

            status = (
                "PASS"
                if abs(
                    difference_seconds
                )
                <= TIMESTAMP_TOLERANCE_SECONDS
                else "FAIL"
            )

        datetime_rows.append(
            {
                "run_id": RUN_ID,
                "table_name": TABLE_NAME,
                "column_name": sf.name,
                "classification": cls,
                "statistic_name": metric,
                "source_value": (
                    None
                    if sv is None
                    else str(sv)
                ),
                "target_value": (
                    None
                    if tv is None
                    else str(tv)
                ),
                "difference_seconds": (
                    difference_seconds
                ),
                "status": status,
                "run_ts": RUN_TS,
            }
        )

datetime_report = (
    spark.createDataFrame(
        datetime_rows,
        schema=DATETIME_REPORT_SCHEMA,
    )
    if datetime_rows
    else empty_df(
        DATETIME_REPORT_SCHEMA
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Row-hash canonicalization

# COMMAND ----------

NUMERIC_TYPES = (
    T.ByteType,
    T.ShortType,
    T.IntegerType,
    T.LongType,
    T.FloatType,
    T.DoubleType,
    T.DecimalType,
)


def canonical_value(
    col_expr: F.Column,
    data_type: T.DataType,
    timezone_name: str,
) -> F.Column:

    if isinstance(
        data_type,
        T.StringType,
    ):
        return col_expr.cast(
            "string"
        )

    if isinstance(
        data_type,
        T.BooleanType,
    ):
        return (
            col_expr
            .cast("boolean")
            .cast("string")
        )

    if isinstance(
        data_type,
        NUMERIC_TYPES,
    ):
        return (
            col_expr
            .cast(
                "decimal(38,18)"
            )
            .cast("string")
        )

    if isinstance(
        data_type,
        T.DateType,
    ):
        return F.date_format(
            col_expr.cast(
                "date"
            ),
            "yyyy-MM-dd",
        )

    if isinstance(
        data_type,
        T.TimestampNTZType,
    ):
        utc_ts = (
            F.to_utc_timestamp(
                col_expr.cast(
                    "timestamp"
                ),
                timezone_name,
            )
        )

        return F.date_format(
            utc_ts,
            "yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'",
        )

    if isinstance(
        data_type,
        T.TimestampType,
    ):
        return F.date_format(
            col_expr.cast(
                "timestamp"
            ),
            "yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'",
        )

    if isinstance(
        data_type,
        T.BinaryType,
    ):
        return F.base64(
            col_expr
        )

    if isinstance(
        data_type,
        (
            T.ArrayType,
            T.MapType,
            T.StructType,
        ),
    ):
        return F.to_json(
            col_expr,
            options={
                "ignoreNullFields": "false"
            },
        )

    return col_expr.cast(
        "string"
    )


def add_hash(
    df: DataFrame,
    columns: Sequence[str],
    timezone_name: str,
    hash_column_name: str,
) -> DataFrame:

    field_map = {
        f.name.lower(): f
        for f in df.schema.fields
    }

    canonical_columns = []

    for col_name in columns:
        field = field_map[
            col_name.lower()
        ]

        canonical_columns.append(
            F.coalesce(
                canonical_value(
                    F.col(
                        q(field.name)
                    ),
                    field.dataType,
                    timezone_name,
                ),
                F.lit("∅"),
            ).alias(
                field.name
            )
        )

    payload = F.to_json(
        F.struct(
            *canonical_columns
        ),
        options={
            "ignoreNullFields": "false"
        },
    )

    return (
        df
        .withColumn(
            "__row_payload",
            payload,
        )
        .withColumn(
            hash_column_name,
            F.sha2(
                F.col(
                    "__row_payload"
                ),
                256,
            ),
        )
        .drop(
            "__row_payload"
        )
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Row-hash comparison and mismatch drill-down

# COMMAND ----------

ROW_HASH_SUMMARY_SCHEMA = """
run_id string,
table_name string,
requested_row_limit long,
source_rows_hashed long,
target_rows_hashed long,
source_only_rows long,
target_only_rows long,
matched_rows long,
changed_rows long,
hash_match_pct double,
status string,
run_ts timestamp
"""

ROW_STATUS_SCHEMA = """
run_id string,
table_name string,
key_json string,
record_status string,
source_row_hash string,
target_row_hash string,
run_ts timestamp
"""

MISMATCH_DETAIL_SCHEMA = """
run_id string,
table_name string,
key_json string,
mismatch_column string,
mismatch_type string,
source_value string,
target_value string,
source_row_hash string,
target_row_hash string,
run_ts timestamp
"""


def equal_value(
    source_col: F.Column,
    target_col: F.Column,
    field: T.StructField,
) -> F.Column:

    cls = classify(
        field
    )

    both_null = (
        source_col.isNull()
        & target_col.isNull()
    )

    one_null = (
        (
            source_col.isNull()
            & target_col.isNotNull()
        )
        |
        (
            source_col.isNotNull()
            & target_col.isNull()
        )
    )

    if isinstance(
        field.dataType,
        NUMERIC_TYPES,
    ):
        s = source_col.cast(
            "double"
        )

        t = target_col.cast(
            "double"
        )

        abs_diff = F.abs(
            t - s
        )

        denominator = F.greatest(
            F.abs(s),
            F.abs(t),
            F.lit(1.0),
        )

        allowed = F.greatest(
            F.lit(
                NUMERIC_ABSOLUTE_TOLERANCE
            ),
            F.lit(
                NUMERIC_RELATIVE_TOLERANCE
            )
            * denominator,
        )

        non_null_equal = (
            abs_diff
            <= allowed
        )

    elif cls == "TIMESTAMP_NTZ":

        s = (
            F.to_utc_timestamp(
                source_col.cast(
                    "timestamp"
                ),
                SOURCE_TIMESTAMP_TIMEZONE,
            )
            .cast("double")
        )

        t = (
            F.to_utc_timestamp(
                target_col.cast(
                    "timestamp"
                ),
                TARGET_TIMESTAMP_TIMEZONE,
            )
            .cast("double")
        )

        non_null_equal = (
            F.abs(
                t - s
            )
            <= F.lit(
                TIMESTAMP_TOLERANCE_SECONDS
            )
        )

    elif cls == "TIMESTAMP":

        non_null_equal = (
            F.abs(
                target_col
                .cast("timestamp")
                .cast("double")
                -
                source_col
                .cast("timestamp")
                .cast("double")
            )
            <= F.lit(
                TIMESTAMP_TOLERANCE_SECONDS
            )
        )

    else:
        non_null_equal = (
            source_col.eqNullSafe(
                target_col
            )
        )

    return (
        F.when(
            both_null,
            F.lit(True),
        )
        .when(
            one_null,
            F.lit(False),
        )
        .otherwise(
            non_null_equal
        )
    )


if ROW_HASH_ENABLED:

    source_hash_df = apply_row_limit(
        source_df,
        ROW_LIMIT,
        ROW_ORDER_COLUMNS,
    )

    target_hash_df = apply_row_limit(
        target_df,
        ROW_LIMIT,
        ROW_ORDER_COLUMNS,
    )

    source_hash_count = (
        source_hash_df.count()
    )

    target_hash_count = (
        target_hash_df.count()
    )

    excluded = {
        c.lower()
        for c in EXCLUDED_HASH_COLUMNS
    }

    key_names_lower = {
        k.lower()
        for k in PRIMARY_KEYS
    }

    hash_columns = [
        sf.name
        for logical_name, (
            sf,
            tf,
        ) in common_fields.items()
        if (
            logical_name
            not in excluded
        )
        and (
            sf.name.lower()
            not in key_names_lower
        )
    ]

    if not hash_columns:
        raise ValueError(
            "No common non-key columns remain for row hashing."
        )

    target_hash_columns = [
        common_fields[
            c.lower()
        ][1].name
        for c in hash_columns
    ]

    source_hashed = add_hash(
        source_hash_df,
        hash_columns,
        SOURCE_TIMESTAMP_TIMEZONE,
        "__source_hash",
    ).alias("s")

    target_hashed = add_hash(
        target_hash_df,
        target_hash_columns,
        TARGET_TIMESTAMP_TIMEZONE,
        "__target_hash",
    ).alias("t")

    join_condition = reduce(
        lambda a, b: a & b,
        [
            F.col(
                f"s.{q(source_columns[k.lower()])}"
            ).eqNullSafe(
                F.col(
                    f"t.{q(target_columns[k.lower()])}"
                )
            )
            for k in PRIMARY_KEYS
        ],
    )

    joined = source_hashed.join(
        target_hashed,
        join_condition,
        "fullouter",
    )

    source_exists = (
        F.col(
            "s.__source_hash"
        )
        .isNotNull()
    )

    target_exists = (
        F.col(
            "t.__target_hash"
        )
        .isNotNull()
    )

    record_status_expr = (
        F.when(
            source_exists
            & ~target_exists,
            "SOURCE_ONLY",
        )
        .when(
            ~source_exists
            & target_exists,
            "TARGET_ONLY",
        )
        .when(
            F.col(
                "s.__source_hash"
            )
            ==
            F.col(
                "t.__target_hash"
            ),
            "MATCH",
        )
        .otherwise(
            "CHANGED"
        )
    )

    key_fields = []

    for key in PRIMARY_KEYS:

        source_key = (
            source_columns[
                key.lower()
            ]
        )

        target_key = (
            target_columns[
                key.lower()
            ]
        )

        key_fields.append(
            F.coalesce(
                F.col(
                    f"s.{q(source_key)}"
                ),
                F.col(
                    f"t.{q(target_key)}"
                ),
            )
            .alias(
                key
            )
        )

    key_json_expr = F.to_json(
        F.struct(
            *key_fields
        ),
        options={
            "ignoreNullFields": "false"
        },
    )

    row_status_report = (
        joined
        .select(
            F.lit(
                RUN_ID
            )
            .alias(
                "run_id"
            ),

            F.lit(
                TABLE_NAME
            )
            .alias(
                "table_name"
            ),

            key_json_expr
            .alias(
                "key_json"
            ),

            record_status_expr
            .alias(
                "record_status"
            ),

            F.col(
                "s.__source_hash"
            )
            .alias(
                "source_row_hash"
            ),

            F.col(
                "t.__target_hash"
            )
            .alias(
                "target_row_hash"
            ),

            F.lit(
                RUN_TS
            )
            .cast(
                "timestamp"
            )
            .alias(
                "run_ts"
            ),
        )
    )

    rc = (
        row_status_report
        .agg(
            F.sum(
                F.when(
                    F.col(
                        "record_status"
                    )
                    ==
                    "SOURCE_ONLY",
                    1,
                )
                .otherwise(0)
            )
            .alias(
                "source_only"
            ),

            F.sum(
                F.when(
                    F.col(
                        "record_status"
                    )
                    ==
                    "TARGET_ONLY",
                    1,
                )
                .otherwise(0)
            )
            .alias(
                "target_only"
            ),

            F.sum(
                F.when(
                    F.col(
                        "record_status"
                    )
                    ==
                    "MATCH",
                    1,
                )
                .otherwise(0)
            )
            .alias(
                "matched"
            ),

            F.sum(
                F.when(
                    F.col(
                        "record_status"
                    )
                    ==
                    "CHANGED",
                    1,
                )
                .otherwise(0)
            )
            .alias(
                "changed"
            ),
        )
        .first()
        .asDict()
    )

    source_only = int(
        rc["source_only"]
        or 0
    )

    target_only = int(
        rc["target_only"]
        or 0
    )

    matched = int(
        rc["matched"]
        or 0
    )

    changed = int(
        rc["changed"]
        or 0
    )

    denominator = max(
        source_hash_count,
        target_hash_count,
        1,
    )

    row_hash_summary = (
        spark.createDataFrame(
            [{
                "run_id": RUN_ID,
                "table_name": TABLE_NAME,
                "requested_row_limit": ROW_LIMIT,
                "source_rows_hashed": source_hash_count,
                "target_rows_hashed": target_hash_count,
                "source_only_rows": source_only,
                "target_only_rows": target_only,
                "matched_rows": matched,
                "changed_rows": changed,
                "hash_match_pct": (
                    matched
                    / denominator
                    * 100.0
                ),
                "status": (
                    "PASS"
                    if (
                        source_only == 0
                        and target_only == 0
                        and changed == 0
                    )
                    else "FAIL"
                ),
                "run_ts": RUN_TS,
            }],
            schema=ROW_HASH_SUMMARY_SCHEMA,
        )
    )

    changed_rows_df = joined.where(
        source_exists
        & target_exists
        & (
            F.col(
                "s.__source_hash"
            )
            !=
            F.col(
                "t.__target_hash"
            )
        )
    )

    mismatch_frames: List[
        DataFrame
    ] = []

    for logical_name, (
        sf,
        tf,
    ) in common_fields.items():

        if logical_name in excluded:
            continue

        if (
            sf.name.lower()
            in key_names_lower
        ):
            continue

        source_value_col = F.col(
            f"s.{q(sf.name)}"
        )

        target_value_col = F.col(
            f"t.{q(tf.name)}"
        )

        equal_expr = equal_value(
            source_value_col,
            target_value_col,
            sf,
        )

        mismatch_frames.append(
            changed_rows_df
            .where(
                ~equal_expr
            )
            .select(
                F.lit(
                    RUN_ID
                )
                .alias(
                    "run_id"
                ),

                F.lit(
                    TABLE_NAME
                )
                .alias(
                    "table_name"
                ),

                key_json_expr
                .alias(
                    "key_json"
                ),

                F.lit(
                    sf.name
                )
                .alias(
                    "mismatch_column"
                ),

                F.when(
                    source_value_col.isNull()
                    & target_value_col.isNotNull(),
                    "SOURCE_NULL",
                )
                .when(
                    source_value_col.isNotNull()
                    & target_value_col.isNull(),
                    "TARGET_NULL",
                )
                .otherwise(
                    "VALUE_MISMATCH"
                )
                .alias(
                    "mismatch_type"
                ),

                canonical_value(
                    source_value_col,
                    sf.dataType,
                    SOURCE_TIMESTAMP_TIMEZONE,
                )
                .alias(
                    "source_value"
                ),

                canonical_value(
                    target_value_col,
                    tf.dataType,
                    TARGET_TIMESTAMP_TIMEZONE,
                )
                .alias(
                    "target_value"
                ),

                F.col(
                    "s.__source_hash"
                )
                .alias(
                    "source_row_hash"
                ),

                F.col(
                    "t.__target_hash"
                )
                .alias(
                    "target_row_hash"
                ),

                F.lit(
                    RUN_TS
                )
                .cast(
                    "timestamp"
                )
                .alias(
                    "run_ts"
                ),
            )
        )

    mismatch_detail_report = union_all(
        mismatch_frames,
        MISMATCH_DETAIL_SCHEMA,
    )

    if MAX_MISMATCH_DETAILS > 0:
        mismatch_detail_report = (
            mismatch_detail_report
            .limit(
                MAX_MISMATCH_DETAILS
            )
        )

else:

    row_hash_summary = empty_df(
        ROW_HASH_SUMMARY_SCHEMA
    )

    row_status_report = empty_df(
        ROW_STATUS_SCHEMA
    )

    mismatch_detail_report = empty_df(
        MISMATCH_DETAIL_SCHEMA
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. Display reports

# COMMAND ----------

print("TOTAL COUNT")
display(
    count_report
)

# COMMAND ----------

print("SCHEMA COMPARISON")
display(
    schema_report
    .orderBy(
        "column_name"
    )
)

# COMMAND ----------

print("CATEGORICAL COMPARISON")
display(
    category_report
    .orderBy(
        "column_name",
        F.desc(
            "source_count"
        ),
    )
)

# COMMAND ----------

print("NUMERICAL COMPARISON")
display(
    numeric_report
    .orderBy(
        "column_name",
        "statistic_name",
    )
)

# COMMAND ----------

print("DATETIME COMPARISON")
display(
    datetime_report
    .orderBy(
        "column_name",
        "statistic_name",
    )
)

# COMMAND ----------

print("ROW HASH SUMMARY")
display(
    row_hash_summary
)

# COMMAND ----------

print(
    "SOURCE_ONLY / TARGET_ONLY / CHANGED RECORDS"
)
display(
    row_status_report
    .where(
        F.col(
            "record_status"
        )
        !=
        "MATCH"
    )
    .orderBy(
        "key_json"
    )
)

# COMMAND ----------

print(
    "COLUMN-LEVEL MISMATCH DETAILS"
)
display(
    mismatch_detail_report
    .orderBy(
        "key_json",
        "mismatch_column",
    )
)
