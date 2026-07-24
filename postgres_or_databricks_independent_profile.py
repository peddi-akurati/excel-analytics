# Databricks notebook source
# MAGIC %md
# MAGIC # Independent Statistical Profiling: PostgreSQL or Databricks
# MAGIC
# MAGIC Run this notebook in one of two independent modes:
# MAGIC
# MAGIC - **POSTGRES**: profile configured PostgreSQL schema tables.
# MAGIC - **DATABRICKS**: profile configured Databricks catalog/schema tables.
# MAGIC
# MAGIC The notebook does not compare the two systems during the same execution.
# MAGIC It produces normalized CSV reports that can be manually compared later.
# MAGIC
# MAGIC For each configured table it produces:
# MAGIC
# MAGIC 1. Total row count
# MAGIC 2. Metadata-driven column classification
# MAGIC 3. Categorical category counts and column distinct count
# MAGIC 4. Numerical minimum, maximum, sum, and median
# MAGIC 5. Date/time minimum and maximum
# MAGIC 6. Timestamp normalization using configured timezone assumptions
# MAGIC 7. Execution status and error report
# MAGIC 8. Downloadable CSV files

# COMMAND ----------

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from functools import reduce
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Interactive parameters

# COMMAND ----------

dbutils.widgets.dropdown("execution_mode", "POSTGRES", ["POSTGRES", "DATABRICKS"])

# Comma-separated list, for example:
# claims,claim_line,policy,customer
dbutils.widgets.text("table_list", "")

# Optional JSON configuration by table.
# Supported values:
# {
#   "claims": {
#       "where_clause": "created_date >= DATE '2026-01-01'",
#       "partition_column": "claim_id",
#       "lower_bound": "1",
#       "upper_bound": "500000000",
#       "num_partitions": 32
#   },
#   "policy": {
#       "where_clause": ""
#   }
# }
dbutils.widgets.text("table_config_json", "{}")

# PostgreSQL mode
dbutils.widgets.text("pg_host", "")
dbutils.widgets.text("pg_port", "5432")
dbutils.widgets.text("pg_database", "")
dbutils.widgets.text("pg_schema", "public")
dbutils.widgets.text("pg_user", "")
dbutils.widgets.text("pg_password", "")
dbutils.widgets.text("pg_fetch_size", "10000")
dbutils.widgets.text("pg_sslmode", "require")

# Databricks mode
dbutils.widgets.text("dbx_catalog", "main")
dbutils.widgets.text("dbx_schema", "default")

# Datetime handling
# TIMESTAMP WITHOUT TIME ZONE values are interpreted in this timezone.
dbutils.widgets.text("source_timestamp_timezone", "Asia/Kolkata")

# Timestamp values are normalized and exported in this timezone.
dbutils.widgets.text("output_timestamp_timezone", "UTC")

# Profiling controls
dbutils.widgets.text("median_accuracy", "10000")
dbutils.widgets.text("category_value_max_length", "1000")

# Export
# FileStore is used so the result can be downloaded from the notebook.
dbutils.widgets.text(
    "export_base_path",
    "dbfs:/FileStore/data_validation_profiles"
)

# COMMAND ----------

EXECUTION_MODE = dbutils.widgets.get("execution_mode").strip().upper()
TABLE_LIST_RAW = dbutils.widgets.get("table_list").strip()
TABLE_CONFIG_JSON = dbutils.widgets.get("table_config_json").strip() or "{}"

PG_HOST = dbutils.widgets.get("pg_host").strip()
PG_PORT = dbutils.widgets.get("pg_port").strip()
PG_DATABASE = dbutils.widgets.get("pg_database").strip()
PG_SCHEMA = dbutils.widgets.get("pg_schema").strip()
PG_USER = dbutils.widgets.get("pg_user").strip()
PG_PASSWORD = dbutils.widgets.get("pg_password")
PG_FETCH_SIZE = dbutils.widgets.get("pg_fetch_size").strip()
PG_SSLMODE = dbutils.widgets.get("pg_sslmode").strip()

DBX_CATALOG = dbutils.widgets.get("dbx_catalog").strip()
DBX_SCHEMA = dbutils.widgets.get("dbx_schema").strip()

SOURCE_TIMESTAMP_TIMEZONE = dbutils.widgets.get(
    "source_timestamp_timezone"
).strip()
OUTPUT_TIMESTAMP_TIMEZONE = dbutils.widgets.get(
    "output_timestamp_timezone"
).strip()

MEDIAN_ACCURACY = int(dbutils.widgets.get("median_accuracy").strip())
CATEGORY_VALUE_MAX_LENGTH = int(
    dbutils.widgets.get("category_value_max_length").strip()
)
EXPORT_BASE_PATH = dbutils.widgets.get("export_base_path").strip().rstrip("/")

if EXECUTION_MODE not in {"POSTGRES", "DATABRICKS"}:
    raise ValueError("execution_mode must be POSTGRES or DATABRICKS.")

TABLES = list(dict.fromkeys(
    table.strip()
    for table in TABLE_LIST_RAW.split(",")
    if table.strip()
))
if not TABLES:
    raise ValueError(
        "Enter at least one table in table_list, separated by commas."
    )

try:
    TABLE_CONFIG: Dict[str, dict] = json.loads(TABLE_CONFIG_JSON)
except json.JSONDecodeError as exc:
    raise ValueError(f"table_config_json is not valid JSON: {exc}") from exc

if not isinstance(TABLE_CONFIG, dict):
    raise ValueError("table_config_json must be a JSON object.")

if not SOURCE_TIMESTAMP_TIMEZONE:
    raise ValueError("source_timestamp_timezone cannot be empty.")
if not OUTPUT_TIMESTAMP_TIMEZONE:
    raise ValueError("output_timestamp_timezone cannot be empty.")

if EXECUTION_MODE == "POSTGRES":
    required = {
        "pg_host": PG_HOST,
        "pg_database": PG_DATABASE,
        "pg_schema": PG_SCHEMA,
        "pg_user": PG_USER,
    }
else:
    required = {
        "dbx_catalog": DBX_CATALOG,
        "dbx_schema": DBX_SCHEMA,
    }

missing = [name for name, value in required.items() if not value]
if missing:
    raise ValueError(
        f"Missing required {EXECUTION_MODE} parameters: {', '.join(missing)}"
    )

# Use UTC internally. Explicit conversion is applied when exporting timestamps.
spark.conf.set("spark.sql.session.timeZone", "UTC")

RUN_ID = str(uuid.uuid4())
RUN_TS = datetime.now(timezone.utc)
RUN_FOLDER = (
    f"{EXPORT_BASE_PATH}/"
    f"{EXECUTION_MODE.lower()}_"
    f"{RUN_TS.strftime('%Y%m%d_%H%M%S')}_"
    f"{RUN_ID[:8]}"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Shared helpers

# COMMAND ----------

def quote_pg_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_spark_identifier(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def table_configuration(table: str) -> dict:
    # Case-insensitive lookup while retaining exact configured table name.
    by_lower = {str(k).lower(): v for k, v in TABLE_CONFIG.items()}
    value = by_lower.get(table.lower(), {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"Configuration for table {table} must be a JSON object."
        )
    return value


def safe_file_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def empty_df(schema: str) -> DataFrame:
    return spark.createDataFrame([], schema=schema)


def union_frames(frames: Sequence[DataFrame], schema: str) -> DataFrame:
    if not frames:
        return empty_df(schema)
    return reduce(
        lambda left, right: left.unionByName(
            right, allowMissingColumns=True
        ),
        frames,
    )


def write_single_csv(df: DataFrame, output_folder: str) -> str:
    """
    Writes a Spark DataFrame as one CSV part file and returns the DBFS file path.
    This is intended for statistical result sets, not source data extraction.
    """
    temp_folder = output_folder + "_tmp"
    dbutils.fs.rm(temp_folder, True)
    dbutils.fs.rm(output_folder, True)

    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .option("quoteAll", "true")
        .csv(temp_folder)
    )

    part_files = [
        item.path
        for item in dbutils.fs.ls(temp_folder)
        if item.name.startswith("part-") and item.name.endswith(".csv")
    ]
    if not part_files:
        raise RuntimeError(f"No CSV part file generated in {temp_folder}")

    final_file = output_folder + ".csv"
    dbutils.fs.rm(final_file, True)
    dbutils.fs.mv(part_files[0], final_file)
    dbutils.fs.rm(temp_folder, True)
    return final_file


def file_store_download_url(dbfs_path: str) -> Optional[str]:
    prefix = "dbfs:/FileStore/"
    if not dbfs_path.startswith(prefix):
        return None
    relative_path = dbfs_path[len("dbfs:/FileStore"):]
    return f"/files{relative_path}"


def truncate_category(column: F.Column) -> F.Column:
    return F.substring(
        F.coalesce(column.cast("string"), F.lit("<NULL>")),
        1,
        CATEGORY_VALUE_MAX_LENGTH,
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Source-specific metadata and table readers

# COMMAND ----------

JDBC_URL = None
JDBC_BASE_OPTIONS: Dict[str, str] = {}

if EXECUTION_MODE == "POSTGRES":
    JDBC_URL = (
        f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
        f"?sslmode={PG_SSLMODE}"
        "&ApplicationName=databricks-independent-profiler"
    )
    JDBC_BASE_OPTIONS = {
        "url": JDBC_URL,
        "user": PG_USER,
        "password": PG_PASSWORD,
        "driver": "org.postgresql.Driver",
        "fetchsize": PG_FETCH_SIZE,
    }


def read_postgres_query(query: str) -> DataFrame:
    options = dict(JDBC_BASE_OPTIONS)
    options["dbtable"] = f"({query}) profile_query"
    return spark.read.format("jdbc").options(**options).load()


def postgres_metadata(table: str) -> DataFrame:
    query = f"""
        SELECT
            ordinal_position,
            column_name,
            data_type,
            udt_name,
            is_nullable,
            numeric_precision,
            numeric_scale,
            datetime_precision
        FROM information_schema.columns
        WHERE table_schema = {sql_literal(PG_SCHEMA)}
          AND table_name = {sql_literal(table)}
        ORDER BY ordinal_position
    """
    return read_postgres_query(query)


def read_postgres_table(table: str, config: dict) -> DataFrame:
    relation = (
        f"{quote_pg_identifier(PG_SCHEMA)}."
        f"{quote_pg_identifier(table)}"
    )

    where_clause = str(config.get("where_clause", "")).strip()
    dbtable = (
        f"(SELECT * FROM {relation} WHERE {where_clause}) source_table"
        if where_clause
        else relation
    )

    options = dict(JDBC_BASE_OPTIONS)
    options["dbtable"] = dbtable

    partition_column = str(config.get("partition_column", "")).strip()
    lower_bound = str(config.get("lower_bound", "")).strip()
    upper_bound = str(config.get("upper_bound", "")).strip()
    num_partitions = int(config.get("num_partitions", 1))

    partition_values = [
        partition_column,
        lower_bound,
        upper_bound,
    ]
    if any(partition_values) and not all(partition_values):
        raise ValueError(
            f"Table {table}: partition_column, lower_bound, and upper_bound "
            "must all be supplied together."
        )

    if partition_column and num_partitions > 1:
        options.update(
            {
                "partitionColumn": partition_column,
                "lowerBound": lower_bound,
                "upperBound": upper_bound,
                "numPartitions": str(num_partitions),
            }
        )

    return spark.read.format("jdbc").options(**options).load()


def databricks_metadata(table: str) -> DataFrame:
    full_name = (
        f"{quote_spark_identifier(DBX_CATALOG)}."
        f"{quote_spark_identifier(DBX_SCHEMA)}."
        f"{quote_spark_identifier(table)}"
    )
    schema = spark.table(full_name).schema

    rows = []
    for position, field in enumerate(schema.fields, start=1):
        rows.append(
            {
                "ordinal_position": position,
                "column_name": field.name,
                "data_type": field.dataType.simpleString(),
                "udt_name": field.dataType.typeName(),
                "is_nullable": "YES" if field.nullable else "NO",
                "numeric_precision": None,
                "numeric_scale": None,
                "datetime_precision": None,
            }
        )

    return spark.createDataFrame(rows)


def read_databricks_table(table: str, config: dict) -> DataFrame:
    full_name = (
        f"{quote_spark_identifier(DBX_CATALOG)}."
        f"{quote_spark_identifier(DBX_SCHEMA)}."
        f"{quote_spark_identifier(table)}"
    )
    df = spark.table(full_name)
    where_clause = str(config.get("where_clause", "")).strip()
    return df.where(where_clause) if where_clause else df


def load_metadata(table: str) -> DataFrame:
    if EXECUTION_MODE == "POSTGRES":
        return postgres_metadata(table)
    return databricks_metadata(table)


def load_table(table: str, config: dict) -> DataFrame:
    if EXECUTION_MODE == "POSTGRES":
        return read_postgres_table(table, config)
    return read_databricks_table(table, config)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Metadata classification

# COMMAND ----------

POSTGRES_NUMERIC_TYPES = {
    "smallint", "integer", "bigint", "decimal", "numeric",
    "real", "double precision", "smallserial", "serial",
    "bigserial", "money",
}
POSTGRES_DATE_TYPES = {"date"}
POSTGRES_TIMESTAMP_WITHOUT_TZ_TYPES = {"timestamp without time zone"}
POSTGRES_TIMESTAMP_WITH_TZ_TYPES = {"timestamp with time zone"}
POSTGRES_TIME_TYPES = {
    "time without time zone",
    "time with time zone",
}

SPARK_NUMERIC_TYPE_NAMES = {
    "byte", "short", "integer", "long", "float", "double", "decimal"
}
SPARK_DATE_TYPE_NAMES = {"date"}
SPARK_TIMESTAMP_TYPE_NAMES = {"timestamp", "timestamp_ntz"}


def classify_postgres_type(data_type: str, udt_name: str) -> str:
    dt = (data_type or "").lower()

    if dt in POSTGRES_NUMERIC_TYPES:
        return "NUMERICAL"
    if dt in POSTGRES_DATE_TYPES:
        return "DATE"
    if dt in POSTGRES_TIMESTAMP_WITHOUT_TZ_TYPES:
        return "TIMESTAMP_WITHOUT_TIME_ZONE"
    if dt in POSTGRES_TIMESTAMP_WITH_TZ_TYPES:
        return "TIMESTAMP_WITH_TIME_ZONE"
    if dt in POSTGRES_TIME_TYPES:
        return "TIME"

    # Strings, booleans, UUID, JSON, arrays, enums, and other non-numeric
    # values are treated as categorical for profiling.
    return "CATEGORICAL"


def classify_spark_field(field: T.StructField) -> str:
    data_type = field.dataType

    if isinstance(
        data_type,
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
    if isinstance(data_type, T.DateType):
        return "DATE"
    if isinstance(data_type, T.TimestampNTZType):
        return "TIMESTAMP_WITHOUT_TIME_ZONE"
    if isinstance(data_type, T.TimestampType):
        return "TIMESTAMP_WITH_TIME_ZONE"

    return "CATEGORICAL"


def classified_metadata(
    table: str,
    metadata_df: DataFrame,
    data_df: DataFrame,
) -> Tuple[DataFrame, Dict[str, str]]:
    metadata_rows = metadata_df.collect()
    actual_fields = {f.name.lower(): f for f in data_df.schema.fields}
    classifications: Dict[str, str] = {}
    output_rows = []

    for row in metadata_rows:
        values = row.asDict()
        column_name = values["column_name"]

        if EXECUTION_MODE == "POSTGRES":
            classification = classify_postgres_type(
                values.get("data_type"),
                values.get("udt_name"),
            )
        else:
            field = actual_fields.get(column_name.lower())
            if field is None:
                classification = "UNSUPPORTED"
            else:
                classification = classify_spark_field(field)

        classifications[column_name] = classification

        output_rows.append(
            {
                "run_id": RUN_ID,
                "execution_mode": EXECUTION_MODE,
                "database_or_catalog": (
                    PG_DATABASE
                    if EXECUTION_MODE == "POSTGRES"
                    else DBX_CATALOG
                ),
                "schema_name": (
                    PG_SCHEMA
                    if EXECUTION_MODE == "POSTGRES"
                    else DBX_SCHEMA
                ),
                "table_name": table,
                "ordinal_position": int(values["ordinal_position"]),
                "column_name": column_name,
                "source_data_type": values.get("data_type"),
                "udt_name": values.get("udt_name"),
                "is_nullable": values.get("is_nullable"),
                "classification": classification,
                "source_timestamp_timezone": (
                    SOURCE_TIMESTAMP_TIMEZONE
                    if classification == "TIMESTAMP_WITHOUT_TIME_ZONE"
                    else None
                ),
                "output_timestamp_timezone": (
                    OUTPUT_TIMESTAMP_TIMEZONE
                    if classification.startswith("TIMESTAMP")
                    else None
                ),
                "run_ts": RUN_TS,
            }
        )

    schema = """
        run_id string,
        execution_mode string,
        database_or_catalog string,
        schema_name string,
        table_name string,
        ordinal_position integer,
        column_name string,
        source_data_type string,
        udt_name string,
        is_nullable string,
        classification string,
        source_timestamp_timezone string,
        output_timestamp_timezone string,
        run_ts timestamp
    """
    return spark.createDataFrame(output_rows, schema=schema), classifications

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Profiling functions

# COMMAND ----------

COUNT_SCHEMA = """
    run_id string,
    execution_mode string,
    database_or_catalog string,
    schema_name string,
    table_name string,
    metric_name string,
    metric_value string,
    status string,
    run_ts timestamp
"""

CATEGORY_SCHEMA = """
    run_id string,
    execution_mode string,
    database_or_catalog string,
    schema_name string,
    table_name string,
    column_name string,
    category_value string,
    category_count long,
    column_distinct_count long,
    column_null_count long,
    run_ts timestamp
"""

NUMERIC_SCHEMA = """
    run_id string,
    execution_mode string,
    database_or_catalog string,
    schema_name string,
    table_name string,
    column_name string,
    statistic_name string,
    statistic_value string,
    non_null_count long,
    null_count long,
    run_ts timestamp
"""

DATETIME_SCHEMA = """
    run_id string,
    execution_mode string,
    database_or_catalog string,
    schema_name string,
    table_name string,
    column_name string,
    classification string,
    statistic_name string,
    statistic_value string,
    source_timestamp_timezone string,
    output_timestamp_timezone string,
    null_count long,
    run_ts timestamp
"""

STATUS_SCHEMA = """
    run_id string,
    execution_mode string,
    database_or_catalog string,
    schema_name string,
    table_name string,
    status string,
    message string,
    started_ts timestamp,
    completed_ts timestamp,
    duration_seconds double
"""


def common_dimensions(table: str) -> dict:
    return {
        "run_id": RUN_ID,
        "execution_mode": EXECUTION_MODE,
        "database_or_catalog": (
            PG_DATABASE if EXECUTION_MODE == "POSTGRES" else DBX_CATALOG
        ),
        "schema_name": (
            PG_SCHEMA if EXECUTION_MODE == "POSTGRES" else DBX_SCHEMA
        ),
        "table_name": table,
    }


def profile_count(table: str, df: DataFrame) -> DataFrame:
    row_count = df.count()
    payload = {
        **common_dimensions(table),
        "metric_name": "TOTAL_ROW_COUNT",
        "metric_value": str(row_count),
        "status": "COMPLETED",
        "run_ts": RUN_TS,
    }
    return spark.createDataFrame([payload], schema=COUNT_SCHEMA)


def profile_categorical_column(
    table: str,
    df: DataFrame,
    column_name: str,
) -> DataFrame:
    column = F.col(quote_spark_identifier(column_name))

    summary = (
        df.agg(
            F.countDistinct(column).alias("distinct_count"),
            F.sum(F.when(column.isNull(), 1).otherwise(0)).alias("null_count"),
        )
        .first()
        .asDict()
    )

    distinct_count = int(summary["distinct_count"] or 0)
    null_count = int(summary["null_count"] or 0)

    return (
        df.groupBy(truncate_category(column).alias("category_value"))
        .agg(F.count(F.lit(1)).alias("category_count"))
        .select(
            F.lit(RUN_ID).alias("run_id"),
            F.lit(EXECUTION_MODE).alias("execution_mode"),
            F.lit(
                PG_DATABASE
                if EXECUTION_MODE == "POSTGRES"
                else DBX_CATALOG
            ).alias("database_or_catalog"),
            F.lit(
                PG_SCHEMA
                if EXECUTION_MODE == "POSTGRES"
                else DBX_SCHEMA
            ).alias("schema_name"),
            F.lit(table).alias("table_name"),
            F.lit(column_name).alias("column_name"),
            F.col("category_value"),
            F.col("category_count").cast("long"),
            F.lit(distinct_count).cast("long").alias(
                "column_distinct_count"
            ),
            F.lit(null_count).cast("long").alias("column_null_count"),
            F.lit(RUN_TS).cast("timestamp").alias("run_ts"),
        )
    )


def profile_numeric_columns(
    table: str,
    df: DataFrame,
    columns: Sequence[str],
) -> DataFrame:
    frames: List[DataFrame] = []

    for column_name in columns:
        column = F.col(quote_spark_identifier(column_name)).cast(
            "decimal(38,18)"
        )

        stats = (
            df.agg(
                F.min(column).alias("minimum"),
                F.max(column).alias("maximum"),
                F.sum(column).alias("sum"),
                F.expr(
                    f"percentile_approx("
                    f"cast({quote_spark_identifier(column_name)} "
                    f"as decimal(38,18)), 0.5, {MEDIAN_ACCURACY})"
                ).alias("median"),
                F.count(column).alias("non_null_count"),
                F.sum(
                    F.when(
                        F.col(quote_spark_identifier(column_name)).isNull(),
                        1,
                    ).otherwise(0)
                ).alias("null_count"),
            )
            .first()
            .asDict()
        )

        for statistic_name in ["minimum", "maximum", "sum", "median"]:
            payload = {
                **common_dimensions(table),
                "column_name": column_name,
                "statistic_name": statistic_name.upper(),
                "statistic_value": (
                    None
                    if stats[statistic_name] is None
                    else str(stats[statistic_name])
                ),
                "non_null_count": int(stats["non_null_count"] or 0),
                "null_count": int(stats["null_count"] or 0),
                "run_ts": RUN_TS,
            }
            frames.append(
                spark.createDataFrame([payload], schema=NUMERIC_SCHEMA)
            )

    return union_frames(frames, NUMERIC_SCHEMA)


def normalized_timestamp_expr(
    column_name: str,
    classification: str,
) -> F.Column:
    column = F.col(quote_spark_identifier(column_name))

    if classification == "TIMESTAMP_WITHOUT_TIME_ZONE":
        # Interpret the naive value using the configured source timezone,
        # convert it to a UTC instant, then render it in the output timezone.
        utc_timestamp = F.to_utc_timestamp(
            column.cast("timestamp"),
            SOURCE_TIMESTAMP_TIMEZONE,
        )
    else:
        # TIMESTAMP WITH TIME ZONE / Spark TimestampType represents an instant.
        utc_timestamp = column.cast("timestamp")

    return F.from_utc_timestamp(
        utc_timestamp,
        OUTPUT_TIMESTAMP_TIMEZONE,
    )


def profile_datetime_column(
    table: str,
    df: DataFrame,
    column_name: str,
    classification: str,
) -> DataFrame:
    original = F.col(quote_spark_identifier(column_name))

    if classification == "DATE":
        normalized = original.cast("date")
        display_format = "yyyy-MM-dd"
        source_timezone = None
        output_timezone = None
    elif classification == "TIME":
        # Time-of-day values do not identify an instant. They are profiled
        # lexically and are not timezone-shifted.
        normalized = original.cast("string")
        display_format = None
        source_timezone = None
        output_timezone = None
    else:
        normalized = normalized_timestamp_expr(
            column_name,
            classification,
        )
        display_format = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
        source_timezone = (
            SOURCE_TIMESTAMP_TIMEZONE
            if classification == "TIMESTAMP_WITHOUT_TIME_ZONE"
            else None
        )
        output_timezone = OUTPUT_TIMESTAMP_TIMEZONE

    aggregate = (
        df.agg(
            F.min(normalized).alias("minimum"),
            F.max(normalized).alias("maximum"),
            F.sum(F.when(original.isNull(), 1).otherwise(0)).alias(
                "null_count"
            ),
        )
        .first()
        .asDict()
    )

    def format_value(value) -> Optional[str]:
        if value is None:
            return None
        if classification == "DATE":
            return value.isoformat()
        if classification.startswith("TIMESTAMP"):
            # Spark has already shifted the displayed wall-clock value into
            # OUTPUT_TIMESTAMP_TIMEZONE. Add the timezone name explicitly.
            return f"{value.isoformat(sep=' ')} [{OUTPUT_TIMESTAMP_TIMEZONE}]"
        return str(value)

    frames = []
    for statistic_name in ["minimum", "maximum"]:
        payload = {
            **common_dimensions(table),
            "column_name": column_name,
            "classification": classification,
            "statistic_name": statistic_name.upper(),
            "statistic_value": format_value(aggregate[statistic_name]),
            "source_timestamp_timezone": source_timezone,
            "output_timestamp_timezone": output_timezone,
            "null_count": int(aggregate["null_count"] or 0),
            "run_ts": RUN_TS,
        }
        frames.append(
            spark.createDataFrame([payload], schema=DATETIME_SCHEMA)
        )

    return union_frames(frames, DATETIME_SCHEMA)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Execute configured tables

# COMMAND ----------

metadata_frames: List[DataFrame] = []
count_frames: List[DataFrame] = []
category_frames: List[DataFrame] = []
numeric_frames: List[DataFrame] = []
datetime_frames: List[DataFrame] = []
status_frames: List[DataFrame] = []

for table in TABLES:
    started = datetime.now(timezone.utc)
    print(f"[{EXECUTION_MODE}] Profiling {table}")

    try:
        config = table_configuration(table)
        data_df = load_table(table, config)
        metadata_df = load_metadata(table)

        if not metadata_df.take(1):
            raise ValueError(f"No metadata found for table {table}")

        classified_df, classifications = classified_metadata(
            table,
            metadata_df,
            data_df,
        )
        metadata_frames.append(classified_df)

        count_frames.append(profile_count(table, data_df))

        actual_columns = {c.lower(): c for c in data_df.columns}

        for configured_name, classification in classifications.items():
            actual_name = actual_columns.get(configured_name.lower())
            if actual_name is None:
                continue

            if classification == "CATEGORICAL":
                category_frames.append(
                    profile_categorical_column(
                        table,
                        data_df,
                        actual_name,
                    )
                )
            elif classification == "NUMERICAL":
                # Numeric columns are grouped after this loop.
                pass
            elif classification in {
                "DATE",
                "TIME",
                "TIMESTAMP_WITHOUT_TIME_ZONE",
                "TIMESTAMP_WITH_TIME_ZONE",
            }:
                datetime_frames.append(
                    profile_datetime_column(
                        table,
                        data_df,
                        actual_name,
                        classification,
                    )
                )

        numeric_columns = [
            actual_columns[name.lower()]
            for name, classification in classifications.items()
            if classification == "NUMERICAL"
            and name.lower() in actual_columns
        ]
        numeric_frames.append(
            profile_numeric_columns(
                table,
                data_df,
                numeric_columns,
            )
        )

        completed = datetime.now(timezone.utc)
        status_payload = {
            **common_dimensions(table),
            "status": "COMPLETED",
            "message": None,
            "started_ts": started,
            "completed_ts": completed,
            "duration_seconds": (
                completed - started
            ).total_seconds(),
        }
        status_frames.append(
            spark.createDataFrame([status_payload], schema=STATUS_SCHEMA)
        )

    except Exception as exc:
        completed = datetime.now(timezone.utc)
        status_payload = {
            **common_dimensions(table),
            "status": "FAILED",
            "message": f"{type(exc).__name__}: {str(exc)}"[:10000],
            "started_ts": started,
            "completed_ts": completed,
            "duration_seconds": (
                completed - started
            ).total_seconds(),
        }
        status_frames.append(
            spark.createDataFrame([status_payload], schema=STATUS_SCHEMA)
        )
        print(
            f"[{EXECUTION_MODE}] {table} failed: "
            f"{type(exc).__name__}: {exc}"
        )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Consolidated result DataFrames

# COMMAND ----------

METADATA_SCHEMA = """
    run_id string,
    execution_mode string,
    database_or_catalog string,
    schema_name string,
    table_name string,
    ordinal_position integer,
    column_name string,
    source_data_type string,
    udt_name string,
    is_nullable string,
    classification string,
    source_timestamp_timezone string,
    output_timestamp_timezone string,
    run_ts timestamp
"""

metadata_result = union_frames(metadata_frames, METADATA_SCHEMA)
count_result = union_frames(count_frames, COUNT_SCHEMA)
category_result = union_frames(category_frames, CATEGORY_SCHEMA)
numeric_result = union_frames(numeric_frames, NUMERIC_SCHEMA)
datetime_result = union_frames(datetime_frames, DATETIME_SCHEMA)
status_result = union_frames(status_frames, STATUS_SCHEMA)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Display execution reports

# COMMAND ----------

print("TABLE EXECUTION STATUS")
display(status_result.orderBy("table_name"))

# COMMAND ----------

print("TOTAL ROW COUNTS")
display(count_result.orderBy("table_name"))

# COMMAND ----------

print("METADATA AND CLASSIFICATION")
display(
    metadata_result.orderBy(
        "table_name",
        "ordinal_position",
    )
)

# COMMAND ----------

print("CATEGORICAL COUNTS")
display(
    category_result.orderBy(
        "table_name",
        "column_name",
        F.desc("category_count"),
    )
)

# COMMAND ----------

print("NUMERICAL STATISTICS")
display(
    numeric_result.orderBy(
        "table_name",
        "column_name",
        "statistic_name",
    )
)

# COMMAND ----------

print("DATE AND TIME STATISTICS")
display(
    datetime_result.orderBy(
        "table_name",
        "column_name",
        "statistic_name",
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Export CSV reports

# COMMAND ----------

dbutils.fs.mkdirs(RUN_FOLDER)

export_datasets = {
    "01_table_status": status_result,
    "02_total_counts": count_result,
    "03_metadata_classification": metadata_result,
    "04_categorical_counts": category_result,
    "05_numerical_statistics": numeric_result,
    "06_datetime_statistics": datetime_result,
}

download_rows = []

for report_name, report_df in export_datasets.items():
    output_path = write_single_csv(
        report_df,
        f"{RUN_FOLDER}/{report_name}",
    )
    download_rows.append(
        {
            "report_name": report_name,
            "dbfs_path": output_path,
            "download_url": file_store_download_url(output_path),
        }
    )

download_df = spark.createDataFrame(download_rows)

print(f"Execution mode: {EXECUTION_MODE}")
print(f"Run ID: {RUN_ID}")
print(f"Result folder: {RUN_FOLDER}")
display(download_df)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Clickable download links

# COMMAND ----------

links = []
for item in download_rows:
    if item["download_url"]:
        links.append(
            f'<li><a href="{item["download_url"]}" target="_blank">'
            f'Download {item["report_name"]}.csv</a></li>'
        )
    else:
        links.append(
            f'<li>{item["report_name"]}: {item["dbfs_path"]}</li>'
        )

displayHTML(
    "<h3>CSV downloads</h3>"
    "<ul>"
    + "".join(links)
    + "</ul>"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Example configurations
# MAGIC
# MAGIC ### PostgreSQL execution
# MAGIC
# MAGIC ```text
# MAGIC execution_mode = POSTGRES
# MAGIC table_list = claims,claim_line,policy
# MAGIC pg_host = your-host
# MAGIC pg_database = insurance
# MAGIC pg_schema = public
# MAGIC pg_user = validation_user
# MAGIC ```
# MAGIC
# MAGIC Example `table_config_json`:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "claims": {
# MAGIC     "where_clause": "created_date >= DATE '2026-01-01'",
# MAGIC     "partition_column": "claim_id",
# MAGIC     "lower_bound": "1",
# MAGIC     "upper_bound": "500000000",
# MAGIC     "num_partitions": 32
# MAGIC   },
# MAGIC   "claim_line": {
# MAGIC     "partition_column": "claim_id",
# MAGIC     "lower_bound": "1",
# MAGIC     "upper_bound": "500000000",
# MAGIC     "num_partitions": 32
# MAGIC   },
# MAGIC   "policy": {}
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC ### Databricks execution
# MAGIC
# MAGIC ```text
# MAGIC execution_mode = DATABRICKS
# MAGIC table_list = claims,claim_line,policy
# MAGIC dbx_catalog = main
# MAGIC dbx_schema = migrated
# MAGIC ```
# MAGIC
# MAGIC Example `table_config_json`:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "claims": {
# MAGIC     "where_clause": "created_date >= DATE '2026-01-01'"
# MAGIC   },
# MAGIC   "claim_line": {},
# MAGIC   "policy": {}
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC Run the notebook once in POSTGRES mode and download its CSVs. Run it
# MAGIC separately in DATABRICKS mode and download the second set. The report
# MAGIC schemas are identical, allowing manual or spreadsheet-based comparison.
