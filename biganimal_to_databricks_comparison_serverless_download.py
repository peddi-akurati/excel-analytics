# Databricks notebook source
# MAGIC %md
# MAGIC # PostgreSQL vs Databricks Statistical Comparison
# MAGIC
# MAGIC This notebook provides only:
# MAGIC 1. Interactive notebook parameters
# MAGIC 2. PostgreSQL metadata-driven datatype classification
# MAGIC 3. Total record-count comparison
# MAGIC 4. Categorical value-count and distinct-count comparison
# MAGIC 5. Numerical min, max, sum, and median comparison
# MAGIC 6. Date/time min and max comparison with timezone reconciliation
# MAGIC 7. Final tabular PASS/FAIL reports

# COMMAND ----------

from __future__ import annotations

from functools import reduce
from typing import Dict, List, Optional
from datetime import datetime
import os
import re
import shutil

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Interactive parameters

# COMMAND ----------

# PostgreSQL source connection
# For production, prefer a Databricks secret instead of entering the password directly.
dbutils.widgets.text("pg_host", "")
dbutils.widgets.text("pg_port", "5432")
dbutils.widgets.text("pg_database", "")
dbutils.widgets.text("pg_schema", "public")
dbutils.widgets.text("pg_table", "")
dbutils.widgets.text("pg_user", "")
dbutils.widgets.text("pg_password", "")

# Databricks target
dbutils.widgets.text("target_catalog", "main")
dbutils.widgets.text("target_schema", "default")
dbutils.widgets.text("target_table", "")

# Optional equivalent filters on source and target
dbutils.widgets.text("source_where_clause", "")
dbutils.widgets.text("target_where_clause", "")

# JDBC tuning
dbutils.widgets.text("fetch_size", "10000")
dbutils.widgets.text("partition_column", "")
dbutils.widgets.text("lower_bound", "")
dbutils.widgets.text("upper_bound", "")
dbutils.widgets.text("num_partitions", "1")

# Comparison controls
dbutils.widgets.text("absolute_tolerance", "0.000001")
dbutils.widgets.text("timestamp_tolerance_seconds", "0")

# Timezone reconciliation
# PostgreSQL TIMESTAMP WITHOUT TIME ZONE has no embedded timezone. This widget
# defines the timezone in which those source values should be interpreted.
dbutils.widgets.text("source_timestamp_timezone", "Asia/Kolkata")

# Databricks timestamp interpretation timezone. For ordinary Spark TimestampType,
# values are instants and this setting controls normalization/display assumptions.
dbutils.widgets.text("target_timestamp_timezone", "UTC")

# All timestamps are normalized to this timezone before comparison.
dbutils.widgets.text("comparison_timezone", "UTC")

# Result export and download
# Keep this under dbfs:/FileStore to generate browser-downloadable links.
dbutils.widgets.text("export_base_path", "dbfs:/FileStore/data_validation_exports")
dbutils.widgets.text("excel_max_detail_rows", "200000")

# COMMAND ----------

PG_HOST = dbutils.widgets.get("pg_host").strip()
PG_PORT = dbutils.widgets.get("pg_port").strip()
PG_DATABASE = dbutils.widgets.get("pg_database").strip()
PG_SCHEMA = dbutils.widgets.get("pg_schema").strip()
PG_TABLE = dbutils.widgets.get("pg_table").strip()
PG_USER = dbutils.widgets.get("pg_user").strip()
PG_PASSWORD = dbutils.widgets.get("pg_password")

TARGET_CATALOG = dbutils.widgets.get("target_catalog").strip()
TARGET_SCHEMA = dbutils.widgets.get("target_schema").strip()
TARGET_TABLE = dbutils.widgets.get("target_table").strip() or PG_TABLE

SOURCE_WHERE = dbutils.widgets.get("source_where_clause").strip()
TARGET_WHERE = dbutils.widgets.get("target_where_clause").strip()

FETCH_SIZE = dbutils.widgets.get("fetch_size").strip()
PARTITION_COLUMN = dbutils.widgets.get("partition_column").strip()
LOWER_BOUND = dbutils.widgets.get("lower_bound").strip()
UPPER_BOUND = dbutils.widgets.get("upper_bound").strip()
NUM_PARTITIONS = int(dbutils.widgets.get("num_partitions").strip())

ABSOLUTE_TOLERANCE = float(dbutils.widgets.get("absolute_tolerance").strip())
TIMESTAMP_TOLERANCE_SECONDS = float(
    dbutils.widgets.get("timestamp_tolerance_seconds").strip()
)
SOURCE_TIMESTAMP_TIMEZONE = dbutils.widgets.get("source_timestamp_timezone").strip()
TARGET_TIMESTAMP_TIMEZONE = dbutils.widgets.get("target_timestamp_timezone").strip()
COMPARISON_TIMEZONE = dbutils.widgets.get("comparison_timezone").strip()
EXPORT_BASE_PATH = dbutils.widgets.get("export_base_path").strip().rstrip("/")
EXCEL_MAX_DETAIL_ROWS = int(dbutils.widgets.get("excel_max_detail_rows").strip())

required_parameters = {
    "pg_host": PG_HOST,
    "pg_database": PG_DATABASE,
    "pg_table": PG_TABLE,
    "pg_user": PG_USER,
    "target_catalog": TARGET_CATALOG,
    "target_schema": TARGET_SCHEMA,
    "target_table": TARGET_TABLE,
    "source_timestamp_timezone": SOURCE_TIMESTAMP_TIMEZONE,
    "target_timestamp_timezone": TARGET_TIMESTAMP_TIMEZONE,
    "comparison_timezone": COMPARISON_TIMEZONE,
}
missing_parameters = [name for name, value in required_parameters.items() if not value]
if missing_parameters:
    raise ValueError(
        "Enter values for these notebook parameters: " + ", ".join(missing_parameters)
    )

# Keep the Spark session deterministic for timestamp parsing and display.
spark.conf.set("spark.sql.session.timeZone", COMPARISON_TIMEZONE)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Connection and metadata helpers

# COMMAND ----------

def quote_pg_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_spark_identifier(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


JDBC_URL = (
    f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
    "?sslmode=require&ApplicationName=databricks-data-comparison"
)

JDBC_BASE_OPTIONS: Dict[str, str] = {
    "url": JDBC_URL,
    "user": PG_USER,
    "password": PG_PASSWORD,
    "driver": "org.postgresql.Driver",
    "fetchsize": FETCH_SIZE,
}


def read_postgres_query(query: str) -> DataFrame:
    options = dict(JDBC_BASE_OPTIONS)
    options["dbtable"] = f"({query}) postgres_query"
    return spark.read.format("jdbc").options(**options).load()


metadata_query = f"""
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
  AND table_name = {sql_literal(PG_TABLE)}
ORDER BY ordinal_position
"""

postgres_metadata_df = read_postgres_query(metadata_query)
postgres_metadata_rows = postgres_metadata_df.collect()

if not postgres_metadata_rows:
    raise ValueError(
        f"PostgreSQL table {PG_SCHEMA}.{PG_TABLE} was not found or has no columns."
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Metadata-driven datatype classification

# COMMAND ----------

# Classification is intentionally driven by PostgreSQL metadata rather than by
# Spark's inferred JDBC schema.
POSTGRES_NUMERIC_TYPES = {
    "smallint", "integer", "bigint", "decimal", "numeric", "real",
    "double precision", "smallserial", "serial", "bigserial", "money",
}

POSTGRES_CATEGORICAL_TYPES = {
    "character varying", "character", "text", "boolean", "uuid", "json",
    "jsonb", "xml", "bytea", "name", "inet", "cidr", "macaddr", "macaddr8",
    "bit", "bit varying", "USER-DEFINED",
}

POSTGRES_DATE_TYPES = {"date"}
POSTGRES_TIMESTAMP_WITHOUT_TZ_TYPES = {"timestamp without time zone"}
POSTGRES_TIMESTAMP_WITH_TZ_TYPES = {"timestamp with time zone"}
POSTGRES_TIME_TYPES = {"time without time zone", "time with time zone"}

metadata_by_lower_name = {row["column_name"].lower(): row.asDict() for row in postgres_metadata_rows}

categorical_columns: List[str] = []
numerical_columns: List[str] = []
date_columns: List[str] = []
timestamp_without_tz_columns: List[str] = []
timestamp_with_tz_columns: List[str] = []
time_columns: List[str] = []
unsupported_columns: List[str] = []

for row in postgres_metadata_rows:
    column_name = row["column_name"]
    data_type = row["data_type"]

    if data_type in POSTGRES_NUMERIC_TYPES:
        numerical_columns.append(column_name)
    elif data_type in POSTGRES_DATE_TYPES:
        date_columns.append(column_name)
    elif data_type in POSTGRES_TIMESTAMP_WITHOUT_TZ_TYPES:
        timestamp_without_tz_columns.append(column_name)
    elif data_type in POSTGRES_TIMESTAMP_WITH_TZ_TYPES:
        timestamp_with_tz_columns.append(column_name)
    elif data_type in POSTGRES_TIME_TYPES:
        time_columns.append(column_name)
    elif data_type in POSTGRES_CATEGORICAL_TYPES or data_type.startswith("ARRAY"):
        categorical_columns.append(column_name)
    else:
        unsupported_columns.append(column_name)

classification_rows = []
for row in postgres_metadata_rows:
    name = row["column_name"]
    if name in numerical_columns:
        classification = "NUMERICAL"
    elif name in categorical_columns:
        classification = "CATEGORICAL"
    elif name in date_columns:
        classification = "DATE"
    elif name in timestamp_without_tz_columns:
        classification = "TIMESTAMP_WITHOUT_TIME_ZONE"
    elif name in timestamp_with_tz_columns:
        classification = "TIMESTAMP_WITH_TIME_ZONE"
    elif name in time_columns:
        classification = "TIME"
    else:
        classification = "UNSUPPORTED"

    classification_rows.append(
        (
            int(row["ordinal_position"]),
            name,
            row["data_type"],
            row["udt_name"],
            classification,
            row["is_nullable"],
        )
    )

metadata_classification_report = spark.createDataFrame(
    classification_rows,
    "ordinal_position int, column_name string, postgres_data_type string, "
    "postgres_udt_name string, comparison_classification string, nullable string",
)

print("POSTGRESQL METADATA CLASSIFICATION")
display(metadata_classification_report.orderBy("ordinal_position"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Read source and target tables

# COMMAND ----------

source_relation = f"{quote_pg_identifier(PG_SCHEMA)}.{quote_pg_identifier(PG_TABLE)}"
source_dbtable = (
    f"(SELECT * FROM {source_relation} WHERE {SOURCE_WHERE}) source_data"
    if SOURCE_WHERE
    else source_relation
)

jdbc_options = dict(JDBC_BASE_OPTIONS)
jdbc_options["dbtable"] = source_dbtable

if PARTITION_COLUMN and LOWER_BOUND and UPPER_BOUND and NUM_PARTITIONS > 1:
    jdbc_options.update(
        {
            "partitionColumn": PARTITION_COLUMN,
            "lowerBound": LOWER_BOUND,
            "upperBound": UPPER_BOUND,
            "numPartitions": str(NUM_PARTITIONS),
        }
    )

source_df = spark.read.format("jdbc").options(**jdbc_options).load()

target_full_name = (
    f"{quote_spark_identifier(TARGET_CATALOG)}."
    f"{quote_spark_identifier(TARGET_SCHEMA)}."
    f"{quote_spark_identifier(TARGET_TABLE)}"
)
target_df = spark.table(target_full_name)
if TARGET_WHERE:
    target_df = target_df.where(TARGET_WHERE)

# Serverless compute does not support DataFrame cache/persist operations.
# The comparison therefore evaluates each aggregation directly.

source_column_map = {column.lower(): column for column in source_df.columns}
target_column_map = {column.lower(): column for column in target_df.columns}
common_lower_names = set(source_column_map) & set(target_column_map)

if not common_lower_names:
    raise ValueError("No common columns exist between source and target tables.")

# Only compare columns that exist on both sides.
def common_columns(columns: List[str]) -> List[str]:
    return [column for column in columns if column.lower() in common_lower_names]

categorical_columns = common_columns(categorical_columns)
numerical_columns = common_columns(numerical_columns)
date_columns = common_columns(date_columns)
timestamp_without_tz_columns = common_columns(timestamp_without_tz_columns)
timestamp_with_tz_columns = common_columns(timestamp_with_tz_columns)
time_columns = common_columns(time_columns)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Total-count comparison

# COMMAND ----------

source_total_count = source_df.count()
target_total_count = target_df.count()

count_report = spark.createDataFrame(
    [
        (
            "TOTAL_COUNT",
            PG_TABLE,
            int(source_total_count),
            int(target_total_count),
            int(target_total_count - source_total_count),
            "PASS" if source_total_count == target_total_count else "FAIL",
        )
    ],
    "comparison_type string, table_name string, source_value long, "
    "target_value long, difference long, status string",
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Categorical comparison

# COMMAND ----------

categorical_detail_frames: List[DataFrame] = []
categorical_summary_rows = []

for source_column in categorical_columns:
    target_column = target_column_map[source_column.lower()]

    source_categories = (
        source_df
        .select(
            F.coalesce(
                F.col(quote_spark_identifier(source_column)).cast("string"),
                F.lit("<NULL>"),
            ).alias("category_value")
        )
        .groupBy("category_value")
        .count()
        .withColumnRenamed("count", "source_category_count")
    )

    target_categories = (
        target_df
        .select(
            F.coalesce(
                F.col(quote_spark_identifier(target_column)).cast("string"),
                F.lit("<NULL>"),
            ).alias("category_value")
        )
        .groupBy("category_value")
        .count()
        .withColumnRenamed("count", "target_category_count")
    )

    detail = (
        source_categories
        .join(target_categories, "category_value", "full")
        .fillna(0, subset=["source_category_count", "target_category_count"])
        .withColumn("column_name", F.lit(source_column))
        .withColumn(
            "count_difference",
            F.col("target_category_count") - F.col("source_category_count"),
        )
        .withColumn(
            "status",
            F.when(
                F.col("source_category_count") == F.col("target_category_count"),
                F.lit("PASS"),
            ).otherwise(F.lit("FAIL")),
        )
        .select(
            "column_name",
            "category_value",
            "source_category_count",
            "target_category_count",
            "count_difference",
            "status",
        )
    )

    categorical_detail_frames.append(detail)

    # countDistinct excludes null, while the detailed comparison represents null
    # as a separate category. This report explicitly includes null as a category.
    source_distinct = source_categories.count()
    target_distinct = target_categories.count()
    mismatched_category_count = detail.where(F.col("status") == "FAIL").count()

    categorical_summary_rows.append(
        (
            source_column,
            int(source_distinct),
            int(target_distinct),
            int(target_distinct - source_distinct),
            int(mismatched_category_count),
            "PASS"
            if source_distinct == target_distinct and mismatched_category_count == 0
            else "FAIL",
        )
    )

if categorical_detail_frames:
    categorical_detail_report = reduce(
        lambda left, right: left.unionByName(right), categorical_detail_frames
    )
else:
    categorical_detail_report = spark.createDataFrame(
        [],
        "column_name string, category_value string, source_category_count long, "
        "target_category_count long, count_difference long, status string",
    )

categorical_summary_report = spark.createDataFrame(
    categorical_summary_rows,
    "column_name string, source_distinct_count long, target_distinct_count long, "
    "distinct_count_difference long, mismatched_category_count long, status string",
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Numerical comparison

# COMMAND ----------

numerical_rows = []

for source_column in numerical_columns:
    target_column = target_column_map[source_column.lower()]
    source_expression = quote_spark_identifier(source_column)
    target_expression = quote_spark_identifier(target_column)

    source_stats = source_df.agg(
        F.min(F.col(source_expression).cast("decimal(38,18)")).alias("min_value"),
        F.max(F.col(source_expression).cast("decimal(38,18)")).alias("max_value"),
        F.sum(F.col(source_expression).cast("decimal(38,18)")).alias("sum_value"),
        F.expr(
            f"percentile_approx(cast({source_expression} as decimal(38,18)), 0.5, 10000)"
        ).alias("median_value"),
    ).first()

    target_stats = target_df.agg(
        F.min(F.col(target_expression).cast("decimal(38,18)")).alias("min_value"),
        F.max(F.col(target_expression).cast("decimal(38,18)")).alias("max_value"),
        F.sum(F.col(target_expression).cast("decimal(38,18)")).alias("sum_value"),
        F.expr(
            f"percentile_approx(cast({target_expression} as decimal(38,18)), 0.5, 10000)"
        ).alias("median_value"),
    ).first()

    for statistic_name in ["min_value", "max_value", "sum_value", "median_value"]:
        source_value = source_stats[statistic_name]
        target_value = target_stats[statistic_name]

        if source_value is None and target_value is None:
            difference = 0.0
            status = "PASS"
        elif source_value is None or target_value is None:
            difference = None
            status = "FAIL"
        else:
            difference = float(target_value) - float(source_value)
            status = "PASS" if abs(difference) <= ABSOLUTE_TOLERANCE else "FAIL"

        numerical_rows.append(
            (
                source_column,
                statistic_name.replace("_value", "").upper(),
                str(source_value) if source_value is not None else None,
                str(target_value) if target_value is not None else None,
                difference,
                status,
            )
        )

numerical_report = spark.createDataFrame(
    numerical_rows,
    "column_name string, statistic string, source_value string, target_value string, "
    "difference double, status string",
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Date and timestamp comparison with timezone reconciliation

# COMMAND ----------

def source_normalized_timestamp(column_name: str, postgres_data_type: str):
    column = F.col(quote_spark_identifier(column_name)).cast("timestamp")

    if postgres_data_type == "timestamp without time zone":
        # Interpret the timezone-free source wall-clock value in the configured
        # source timezone and convert it to a UTC instant.
        utc_value = F.to_utc_timestamp(column, SOURCE_TIMESTAMP_TIMEZONE)
    else:
        # PostgreSQL timestamptz represents an instant. Spark JDBC reads it as a
        # timestamp instant; normalize explicitly through the session timezone.
        utc_value = F.to_utc_timestamp(column, COMPARISON_TIMEZONE)

    return F.from_utc_timestamp(utc_value, COMPARISON_TIMEZONE)


def target_normalized_timestamp(column_name: str):
    column = F.col(quote_spark_identifier(column_name)).cast("timestamp")
    utc_value = F.to_utc_timestamp(column, TARGET_TIMESTAMP_TIMEZONE)
    return F.from_utc_timestamp(utc_value, COMPARISON_TIMEZONE)


def timestamp_difference_seconds(source_value, target_value) -> Optional[float]:
    if source_value is None and target_value is None:
        return 0.0
    if source_value is None or target_value is None:
        return None
    return float((target_value - source_value).total_seconds())


datetime_rows = []

# Date columns do not require timezone conversion.
for source_column in date_columns:
    target_column = target_column_map[source_column.lower()]

    source_stats = source_df.agg(
        F.min(F.col(quote_spark_identifier(source_column)).cast("date")).alias("min_value"),
        F.max(F.col(quote_spark_identifier(source_column)).cast("date")).alias("max_value"),
    ).first()
    target_stats = target_df.agg(
        F.min(F.col(quote_spark_identifier(target_column)).cast("date")).alias("min_value"),
        F.max(F.col(quote_spark_identifier(target_column)).cast("date")).alias("max_value"),
    ).first()

    for statistic_name in ["min_value", "max_value"]:
        source_value = source_stats[statistic_name]
        target_value = target_stats[statistic_name]
        if source_value is None and target_value is None:
            difference_seconds = 0.0
            status = "PASS"
        elif source_value is None or target_value is None:
            difference_seconds = None
            status = "FAIL"
        else:
            difference_seconds = float((target_value - source_value).days * 86400)
            status = "PASS" if difference_seconds == 0 else "FAIL"

        datetime_rows.append(
            (
                source_column,
                "DATE",
                statistic_name.replace("_value", "").upper(),
                str(source_value) if source_value is not None else None,
                str(target_value) if target_value is not None else None,
                difference_seconds,
                None,
                None,
                COMPARISON_TIMEZONE,
                status,
            )
        )

for source_column in timestamp_without_tz_columns + timestamp_with_tz_columns:
    target_column = target_column_map[source_column.lower()]
    postgres_data_type = metadata_by_lower_name[source_column.lower()]["data_type"]

    source_normalized = source_normalized_timestamp(source_column, postgres_data_type)
    target_normalized = target_normalized_timestamp(target_column)

    source_stats = source_df.agg(
        F.min(source_normalized).alias("min_value"),
        F.max(source_normalized).alias("max_value"),
    ).first()
    target_stats = target_df.agg(
        F.min(target_normalized).alias("min_value"),
        F.max(target_normalized).alias("max_value"),
    ).first()

    for statistic_name in ["min_value", "max_value"]:
        source_value = source_stats[statistic_name]
        target_value = target_stats[statistic_name]
        difference_seconds = timestamp_difference_seconds(source_value, target_value)
        status = (
            "PASS"
            if difference_seconds is not None
            and abs(difference_seconds) <= TIMESTAMP_TOLERANCE_SECONDS
            else "FAIL"
        )

        datetime_rows.append(
            (
                source_column,
                postgres_data_type.upper().replace(" ", "_"),
                statistic_name.replace("_value", "").upper(),
                str(source_value) if source_value is not None else None,
                str(target_value) if target_value is not None else None,
                difference_seconds,
                SOURCE_TIMESTAMP_TIMEZONE,
                TARGET_TIMESTAMP_TIMEZONE,
                COMPARISON_TIMEZONE,
                status,
            )
        )

datetime_report = spark.createDataFrame(
    datetime_rows,
    "column_name string, date_time_type string, statistic string, source_value string, "
    "target_value string, difference_seconds double, source_timezone string, "
    "target_timezone string, comparison_timezone string, status string",
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Final tabular reports

# COMMAND ----------

print("TOTAL COUNT COMPARISON")
display(count_report)

print("CATEGORICAL COLUMN SUMMARY")
display(categorical_summary_report.orderBy("column_name"))

print("CATEGORICAL CATEGORY-LEVEL DETAILS")
display(categorical_detail_report.orderBy("column_name", "category_value"))

print("NUMERICAL COLUMN STATISTICS")
display(numerical_report.orderBy("column_name", "statistic"))

print("DATE/TIMESTAMP STATISTICS AFTER TIMEZONE RECONCILIATION")
display(datetime_report.orderBy("column_name", "statistic"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Overall final status

# COMMAND ----------

count_failed = count_report.where(F.col("status") == "FAIL").count()
categorical_failed = categorical_summary_report.where(F.col("status") == "FAIL").count()
numerical_failed = numerical_report.where(F.col("status") == "FAIL").count()
datetime_failed = datetime_report.where(F.col("status") == "FAIL").count()

overall_status = (
    "PASS"
    if count_failed == 0
    and categorical_failed == 0
    and numerical_failed == 0
    and datetime_failed == 0
    else "FAIL"
)

overall_report = spark.createDataFrame(
    [
        (
            PG_TABLE,
            TARGET_TABLE,
            int(source_total_count),
            int(target_total_count),
            len(categorical_columns),
            int(categorical_failed),
            len(numerical_columns),
            int(numerical_failed),
            len(date_columns),
            len(timestamp_without_tz_columns) + len(timestamp_with_tz_columns),
            int(datetime_failed),
            overall_status,
        )
    ],
    "source_table string, target_table string, source_total_count long, "
    "target_total_count long, categorical_columns_checked int, "
    "categorical_columns_failed int, numerical_columns_checked int, "
    "numerical_statistics_failed int, date_columns_checked int, "
    "timestamp_columns_checked int, datetime_statistics_failed int, "
    "overall_status string",
)

print("OVERALL COMPARISON STATUS")
display(overall_report)

# No unpersist calls are required because no DataFrames are cached or persisted.


# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Export and download execution results
# MAGIC
# MAGIC The notebook exports:
# MAGIC - One CSV file for every report
# MAGIC - One consolidated Excel workbook
# MAGIC - Clickable download links when `export_base_path` is under `dbfs:/FileStore`

# COMMAND ----------

def safe_file_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "comparison"


def export_single_csv(df: DataFrame, output_directory: str, final_file_name: str) -> str:
    """Write a Spark DataFrame as one CSV and return the final DBFS path."""
    temporary_directory = f"{output_directory}/_{final_file_name}_parts"
    final_path = f"{output_directory}/{final_file_name}"

    dbutils.fs.rm(temporary_directory, True)
    dbutils.fs.rm(final_path, True)

    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .option("quoteAll", "true")
        .csv(temporary_directory)
    )

    part_files = [
        item.path
        for item in dbutils.fs.ls(temporary_directory)
        if item.name.startswith("part-") and item.name.endswith(".csv")
    ]
    if not part_files:
        raise RuntimeError(f"No CSV part file was generated at {temporary_directory}")

    dbutils.fs.cp(part_files[0], final_path)
    dbutils.fs.rm(temporary_directory, True)
    return final_path


def filestore_download_url(dbfs_path: str) -> Optional[str]:
    prefix = "dbfs:/FileStore/"
    if dbfs_path.startswith(prefix):
        return "/files/" + dbfs_path[len(prefix):]
    return None


execution_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
export_folder_name = (
    f"{safe_file_component(PG_SCHEMA)}_{safe_file_component(PG_TABLE)}_"
    f"to_{safe_file_component(TARGET_SCHEMA)}_{safe_file_component(TARGET_TABLE)}_"
    f"{execution_timestamp}"
)
execution_export_path = f"{EXPORT_BASE_PATH}/{export_folder_name}"
dbutils.fs.mkdirs(execution_export_path)

reports = {
    "01_metadata_classification": metadata_classification_report.orderBy("ordinal_position"),
    "02_total_count": count_report,
    "03_categorical_summary": categorical_summary_report.orderBy("column_name"),
    "04_categorical_detail": categorical_detail_report.orderBy("column_name", "category_value"),
    "05_numerical_statistics": numerical_report.orderBy("column_name", "statistic"),
    "06_datetime_statistics": datetime_report.orderBy("column_name", "statistic"),
    "07_overall_status": overall_report,
}

exported_files = []
for report_name, report_df in reports.items():
    csv_name = f"{report_name}.csv"
    csv_path = export_single_csv(report_df, execution_export_path, csv_name)
    exported_files.append((report_name, "CSV", csv_path, filestore_download_url(csv_path)))

# Create a consolidated Excel workbook on the driver. The detailed categorical
# sheet is capped to protect the driver from high-cardinality columns.
excel_file_name = "comparison_results.xlsx"
local_excel_path = f"/tmp/{export_folder_name}_{excel_file_name}"
excel_dbfs_path = f"{execution_export_path}/{excel_file_name}"

try:
    import pandas as pd

    with pd.ExcelWriter(local_excel_path, engine="openpyxl") as writer:
        metadata_classification_report.orderBy("ordinal_position").toPandas().to_excel(
            writer, sheet_name="Metadata", index=False
        )
        count_report.toPandas().to_excel(writer, sheet_name="Total_Count", index=False)
        categorical_summary_report.orderBy("column_name").toPandas().to_excel(
            writer, sheet_name="Categorical_Summary", index=False
        )
        categorical_detail_report.orderBy("column_name", "category_value").limit(
            EXCEL_MAX_DETAIL_ROWS
        ).toPandas().to_excel(writer, sheet_name="Categorical_Detail", index=False)
        numerical_report.orderBy("column_name", "statistic").toPandas().to_excel(
            writer, sheet_name="Numerical", index=False
        )
        datetime_report.orderBy("column_name", "statistic").toPandas().to_excel(
            writer, sheet_name="Date_Time", index=False
        )
        overall_report.toPandas().to_excel(writer, sheet_name="Overall_Status", index=False)

    dbutils.fs.cp(f"file:{local_excel_path}", excel_dbfs_path, True)
    exported_files.append(
        ("consolidated_comparison_results", "EXCEL", excel_dbfs_path, filestore_download_url(excel_dbfs_path))
    )
finally:
    if os.path.exists(local_excel_path):
        os.remove(local_excel_path)

export_manifest = spark.createDataFrame(
    [
        (name, file_type, path, url or "NOT_AVAILABLE_OUTSIDE_FILESTORE")
        for name, file_type, path, url in exported_files
    ],
    "report_name string, file_type string, storage_path string, download_url string",
)

print("EXPORTED RESULT FILES")
display(export_manifest.orderBy("report_name"))

link_rows = []
for name, file_type, path, url in exported_files:
    if url:
        link_rows.append(
            f'<tr><td>{name}</td><td>{file_type}</td>'
            f'<td><a href="{url}" target="_blank" download>Download</a></td></tr>'
        )
    else:
        link_rows.append(
            f'<tr><td>{name}</td><td>{file_type}</td>'
            f'<td>Saved at <code>{path}</code></td></tr>'
        )

html = f"""
<h3>Download comparison results</h3>
<p><b>Overall status:</b> {overall_status}</p>
<p>Excel categorical-detail rows are limited to {EXCEL_MAX_DETAIL_ROWS:,}. The CSV contains the complete report.</p>
<table style="border-collapse:collapse" border="1" cellpadding="6">
  <thead><tr><th>Report</th><th>Format</th><th>Download</th></tr></thead>
  <tbody>{''.join(link_rows)}</tbody>
</table>
"""
displayHTML(html)

print(f"All result files were saved under: {execution_export_path}")
