# Databricks notebook source
# MAGIC %md
# MAGIC # Single-Table Databricks Schema Statistical Validation
# MAGIC
# MAGIC Configure the variables in Section 1 and run the notebook.
# MAGIC
# MAGIC Validates one table between two Databricks schemas using:
# MAGIC - total row count
# MAGIC - schema/data-type comparison
# MAGIC - categorical count and distinct-count comparison
# MAGIC - numerical min/max/sum/median comparison
# MAGIC - date/timestamp min/max comparison with timezone reconciliation
# MAGIC
# MAGIC No row hash, no PK drill-down, no export/download logic.

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

# Optional filters.
SOURCE_WHERE_CLAUSE = "intimation_date > DATE '2026-07-01'"
TARGET_WHERE_CLAUSE = "intimation_date > DATE '2026-07-01'"

# Timestamp handling.
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


def build_select_statement(
    catalog: str,
    schema: str,
    table: str,
    where_clause: str,
) -> str:

    statement = (
        f"SELECT * FROM {fqtn(catalog, schema, table)}"
    )

    if where_clause:
        statement += f" WHERE {where_clause}"

    return statement


def read_table(
    catalog: str,
    schema: str,
    table: str,
    where_clause: str,
) -> DataFrame:

    df = spark.table(
        fqtn(
            catalog,
            schema,
            table,
        )
    )

    return (
        df.where(where_clause)
        if where_clause
        else df
    )


def classify(
    field: T.StructField,
) -> str:

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

    if isinstance(
        dt,
        T.DateType,
    ):
        return "DATE"

    if isinstance(
        dt,
        T.TimestampNTZType,
    ):
        return "TIMESTAMP_NTZ"

    if isinstance(
        dt,
        T.TimestampType,
    ):
        return "TIMESTAMP"

    return "CATEGORICAL"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Final source and target SELECT statements

# COMMAND ----------

SOURCE_SELECT_STATEMENT = build_select_statement(
    SOURCE_CATALOG,
    SOURCE_SCHEMA,
    TABLE_NAME,
    SOURCE_WHERE_CLAUSE,
)

TARGET_SELECT_STATEMENT = build_select_statement(
    TARGET_CATALOG,
    TARGET_SCHEMA,
    TABLE_NAME,
    TARGET_WHERE_CLAUSE,
)

print("FINAL SOURCE SELECT STATEMENT")
print(SOURCE_SELECT_STATEMENT)

print("\nFINAL TARGET SELECT STATEMENT")
print(TARGET_SELECT_STATEMENT)

select_statement_report = spark.createDataFrame(
    [
        {
            "side": "SOURCE",
            "select_statement": SOURCE_SELECT_STATEMENT,
        },
        {
            "side": "TARGET",
            "select_statement": TARGET_SELECT_STATEMENT,
        },
    ]
)

display(select_statement_report)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Load source and target

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

source_fields = {
    f.name.lower(): f
    for f in source_df.schema.fields
}

target_fields = {
    f.name.lower(): f
    for f in target_df.schema.fields
}

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Schema comparison

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
    Tuple[
        T.StructField,
        T.StructField,
    ],
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
# MAGIC ## 6. Total count comparison

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

source_total_count = (
    source_df.count()
)

target_total_count = (
    target_df.count()
)

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
# MAGIC ## 7. Categorical comparison

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

category_frames: List[
    DataFrame
] = []

for _, (
    sf,
    tf,
) in common_fields.items():

    if classify(sf) != "CATEGORICAL":
        continue

    source_distinct = (
        source_df
        .select(
            F.col(
                q(sf.name)
            )
        )
        .distinct()
        .count()
    )

    target_distinct = (
        target_df
        .select(
            F.col(
                q(tf.name)
            )
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
                ).cast(
                    "string"
                ),
                F.lit(
                    "<NULL>"
                ),
            )
            .alias(
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
                ).cast(
                    "string"
                ),
                F.lit(
                    "<NULL>"
                ),
            )
            .alias(
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

            F.lit(
                sf.name
            )
            .alias(
                "column_name"
            ),

            F.col(
                "category_value"
            ),

            F.coalesce(
                F.col(
                    "source_count"
                ),
                F.lit(0),
            )
            .cast(
                "long"
            )
            .alias(
                "source_count"
            ),

            F.coalesce(
                F.col(
                    "target_count"
                ),
                F.lit(0),
            )
            .cast(
                "long"
            )
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
            .cast(
                "long"
            )
            .alias(
                "difference"
            ),

            F.lit(
                source_distinct
            )
            .cast(
                "long"
            )
            .alias(
                "source_distinct_count"
            ),

            F.lit(
                target_distinct
            )
            .cast(
                "long"
            )
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
            .cast(
                "timestamp"
            )
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
# MAGIC ## 8. Numerical comparison

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
            .alias(
                "MIN"
            ),

            F.max(c)
            .alias(
                "MAX"
            ),

            F.sum(c)
            .alias(
                "SUM"
            ),

            F.expr(
                f"percentile_approx("
                f"cast({q(column_name)} as double), "
                f"0.5, {MEDIAN_ACCURACY})"
            )
            .alias(
                "MEDIAN"
            ),
        )
        .first()
        .asDict()
    )


for _, (
    sf,
    tf,
) in common_fields.items():

    if classify(sf) != "NUMERICAL":
        continue

    source_stats = numeric_stats(
        source_df,
        sf.name,
    )

    target_stats = numeric_stats(
        target_df,
        tf.name,
    )

    for metric in [
        "MIN",
        "MAX",
        "SUM",
        "MEDIAN",
    ]:

        source_value = (
            source_stats[
                metric
            ]
        )

        target_value = (
            target_stats[
                metric
            ]
        )

        if (
            source_value is None
            and target_value is None
        ):

            absolute_difference = None
            relative_difference = None
            status = "PASS"

        elif (
            source_value is None
            or target_value is None
        ):

            absolute_difference = None
            relative_difference = None
            status = "FAIL"

        else:

            source_value = float(
                source_value
            )

            target_value = float(
                target_value
            )

            absolute_difference = abs(
                target_value
                - source_value
            )

            denominator = max(
                abs(
                    source_value
                ),
                abs(
                    target_value
                ),
                1.0,
            )

            relative_difference = (
                absolute_difference
                / denominator
            )

            allowed_difference = max(
                NUMERIC_ABSOLUTE_TOLERANCE,
                NUMERIC_RELATIVE_TOLERANCE
                * denominator,
            )

            status = (
                "PASS"
                if absolute_difference
                <= allowed_difference
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
                    if source_value is None
                    else float(
                        source_value
                    )
                ),
                "target_value": (
                    None
                    if target_value is None
                    else float(
                        target_value
                    )
                ),
                "absolute_difference": (
                    absolute_difference
                ),
                "relative_difference": (
                    relative_difference
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
# MAGIC ## 9. Date / timestamp comparison

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


for _, (
    sf,
    tf,
) in common_fields.items():

    classification = classify(
        sf
    )

    if classification not in {
        "DATE",
        "TIMESTAMP_NTZ",
        "TIMESTAMP",
    }:
        continue

    source_expr = datetime_expression(
        sf.name,
        classification,
        SOURCE_TIMESTAMP_TIMEZONE,
    )

    target_expr = datetime_expression(
        tf.name,
        classification,
        TARGET_TIMESTAMP_TIMEZONE,
    )

    source_stats = (
        source_df
        .agg(
            F.min(
                source_expr
            )
            .alias(
                "MIN"
            ),

            F.max(
                source_expr
            )
            .alias(
                "MAX"
            ),
        )
        .first()
        .asDict()
    )

    target_stats = (
        target_df
        .agg(
            F.min(
                target_expr
            )
            .alias(
                "MIN"
            ),

            F.max(
                target_expr
            )
            .alias(
                "MAX"
            ),
        )
        .first()
        .asDict()
    )

    for metric in [
        "MIN",
        "MAX",
    ]:

        source_value = (
            source_stats[
                metric
            ]
        )

        target_value = (
            target_stats[
                metric
            ]
        )

        if (
            source_value is None
            and target_value is None
        ):

            difference_seconds = None
            status = "PASS"

        elif (
            source_value is None
            or target_value is None
        ):

            difference_seconds = None
            status = "FAIL"

        elif classification == "DATE":

            difference_seconds = (
                target_value.toordinal()
                - source_value.toordinal()
            ) * 86400.0

            status = (
                "PASS"
                if source_value
                == target_value
                else "FAIL"
            )

        else:

            difference_seconds = (
                target_value.timestamp()
                - source_value.timestamp()
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
                "classification": classification,
                "statistic_name": metric,
                "source_value": (
                    None
                    if source_value is None
                    else str(
                        source_value
                    )
                ),
                "target_value": (
                    None
                    if target_value is None
                    else str(
                        target_value
                    )
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
# MAGIC ## 10. Display reports

# COMMAND ----------

print(
    "TOTAL COUNT"
)
display(
    count_report
)

# COMMAND ----------

print(
    "SCHEMA COMPARISON"
)
display(
    schema_report
    .orderBy(
        "column_name"
    )
)

# COMMAND ----------

print(
    "CATEGORICAL COMPARISON"
)
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

print(
    "NUMERICAL COMPARISON"
)
display(
    numeric_report
    .orderBy(
        "column_name",
        "statistic_name",
    )
)

# COMMAND ----------

print(
    "DATETIME COMPARISON"
)
display(
    datetime_report
    .orderBy(
        "column_name",
        "statistic_name",
    )
)
