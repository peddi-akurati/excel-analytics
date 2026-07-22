# Databricks notebook source
# MAGIC %md
# MAGIC # BigAnimal PostgreSQL → Databricks Record-Level Reconciliation
# MAGIC
# MAGIC Scalable validation framework that produces:
# MAGIC - table/schema comparison
# MAGIC - source/target row counts
# MAGIC - duplicate and null primary-key statistics
# MAGIC - source-only, target-only, matched, and changed row counts
# MAGIC - canonical row hashes
# MAGIC - column-level mismatch counts and percentages
# MAGIC - numeric difference statistics
# MAGIC - mismatch samples with source/target values
# MAGIC - partition/bucket-level reconciliation
# MAGIC - run and error audit tables
# MAGIC
# MAGIC **Important:** Configure primary/business keys and JDBC partition columns before production execution.

# COMMAND ----------

from __future__ import annotations

import json
import re
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from functools import reduce
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pyspark import StorageLevel
from pyspark.sql import DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql import types as T
from delta.tables import DeltaTable

# COMMAND ----------

# -----------------------------
# Databricks widgets
# -----------------------------
dbutils.widgets.text("source_schema", "public")
dbutils.widgets.text("target_catalog", "main")
dbutils.widgets.text("target_schema", "migrated")
dbutils.widgets.text("audit_catalog", "main")
dbutils.widgets.text("audit_schema", "migration_reconciliation")
dbutils.widgets.text("table_include_regex", ".*")
dbutils.widgets.text("table_exclude_regex", "^$")
dbutils.widgets.text("max_tables", "0")
dbutils.widgets.text("sample_mismatches_per_table", "1000")
dbutils.widgets.dropdown("run_mode", "FULL", ["FULL", "METADATA_ONLY", "COUNTS_ONLY"])
dbutils.widgets.dropdown("fail_on_schema_mismatch", "false", ["true", "false"])

SOURCE_SCHEMA = dbutils.widgets.get("source_schema")
TARGET_CATALOG = dbutils.widgets.get("target_catalog")
TARGET_SCHEMA = dbutils.widgets.get("target_schema")
AUDIT_CATALOG = dbutils.widgets.get("audit_catalog")
AUDIT_SCHEMA = dbutils.widgets.get("audit_schema")
TABLE_INCLUDE_REGEX = dbutils.widgets.get("table_include_regex")
TABLE_EXCLUDE_REGEX = dbutils.widgets.get("table_exclude_regex")
MAX_TABLES = int(dbutils.widgets.get("max_tables"))
SAMPLE_MISMATCHES_PER_TABLE = int(dbutils.widgets.get("sample_mismatches_per_table"))
RUN_MODE = dbutils.widgets.get("run_mode")
FAIL_ON_SCHEMA_MISMATCH = dbutils.widgets.get("fail_on_schema_mismatch").lower() == "true"

RUN_ID = str(uuid.uuid4())
RUN_TS = datetime.now(timezone.utc)

# COMMAND ----------

# -----------------------------
# Secrets and JDBC configuration
# -----------------------------
# Replace the scope/key names with your Databricks secret scope.
PG_HOST = dbutils.secrets.get("biganimal", "host")
PG_PORT = dbutils.secrets.get("biganimal", "port")
PG_DATABASE = dbutils.secrets.get("biganimal", "database")
PG_USER = dbutils.secrets.get("biganimal", "username")
PG_PASSWORD = dbutils.secrets.get("biganimal", "password")

JDBC_URL = (
    f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
    "?sslmode=require"
    "&reWriteBatchedInserts=true"
    "&ApplicationName=databricks-migration-reconciliation"
)

JDBC_BASE_OPTIONS = {
    "url": JDBC_URL,
    "user": PG_USER,
    "password": PG_PASSWORD,
    "driver": "org.postgresql.Driver",
    "fetchsize": "10000",
    "queryTimeout": "0",
    "sessionInitStatement": "SET statement_timeout = 0",
}

# COMMAND ----------

@dataclass
class TableRule:
    """
    Per-table reconciliation configuration.

    key_columns:
        Required for deterministic record-to-record comparison.
    partition_column:
        Numeric/date/timestamp source column used for parallel JDBC reads.
    lower_bound / upper_bound:
        JDBC partition bounds. They control stride, not filtering.
    num_partitions:
        Maximum simultaneous JDBC connections/tasks for the table.
    where_clause:
        Optional source and target filter, e.g. migration batch or date range.
    excluded_columns:
        Columns ignored during value comparison.
    timestamp_tolerance_seconds:
        Permitted timestamp difference.
    numeric_tolerance_abs / numeric_tolerance_rel:
        Permitted numeric differences.
    trim_strings / case_insensitive_strings:
        String normalization.
    null_equivalents:
        String values interpreted as null during comparison.
    """
    key_columns: List[str]
    partition_column: Optional[str] = None
    lower_bound: Optional[str] = None
    upper_bound: Optional[str] = None
    num_partitions: int = 16
    where_clause: Optional[str] = None
    excluded_columns: List[str] = field(default_factory=list)
    timestamp_tolerance_seconds: int = 0
    numeric_tolerance_abs: float = 0.0
    numeric_tolerance_rel: float = 0.0
    trim_strings: bool = True
    case_insensitive_strings: bool = False
    null_equivalents: List[str] = field(
        default_factory=lambda: ["", "NULL", "null", "N/A", "NA"]
    )
    hash_buckets: int = 1024


# Populate this dictionary for all large tables.
# Keys must use lower-case PostgreSQL table names.
TABLE_RULES: Dict[str, TableRule] = {
    # "claims": TableRule(
    #     key_columns=["claim_id"],
    #     partition_column="claim_id",
    #     lower_bound="1",
    #     upper_bound="500000000",
    #     num_partitions=64,
    #     excluded_columns=["etl_created_ts", "etl_updated_ts"],
    #     numeric_tolerance_abs=0.01,
    #     timestamp_tolerance_seconds=1,
    # ),
    # "policy": TableRule(
    #     key_columns=["policy_id", "policy_version"],
    #     partition_column="policy_id",
    #     lower_bound="1",
    #     upper_bound="100000000",
    #     num_partitions=32,
    # ),
}

# Optional default only for genuinely small tables.
DEFAULT_SMALL_TABLE_RULE = TableRule(
    key_columns=[],
    num_partitions=1,
    hash_buckets=128,
)

# COMMAND ----------

# -----------------------------
# Audit table names
# -----------------------------
AUDIT_DB = f"`{AUDIT_CATALOG}`.`{AUDIT_SCHEMA}`"

T_RUN = f"{AUDIT_DB}.recon_run"
T_TABLE = f"{AUDIT_DB}.recon_table_summary"
T_SCHEMA = f"{AUDIT_DB}.recon_schema_difference"
T_COLUMN = f"{AUDIT_DB}.recon_column_summary"
T_SAMPLE = f"{AUDIT_DB}.recon_mismatch_sample"
T_BUCKET = f"{AUDIT_DB}.recon_bucket_summary"
T_ERROR = f"{AUDIT_DB}.recon_error"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {AUDIT_DB}")

# COMMAND ----------

# -----------------------------
# Generic helpers
# -----------------------------
def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def spark_qident(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def now_col():
    return F.current_timestamp()


def append_delta(df: DataFrame, table_name: str) -> None:
    (
        df.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(table_name)
    )


def one_row_df(payload: dict) -> DataFrame:
    return spark.createDataFrame([Row(**payload)])


def jdbc_query_df(query: str, extra_options: Optional[dict] = None) -> DataFrame:
    options = dict(JDBC_BASE_OPTIONS)
    options["dbtable"] = f"({query}) src"
    if extra_options:
        options.update({k: str(v) for k, v in extra_options.items() if v is not None})
    return spark.read.format("jdbc").options(**options).load()


def read_postgres_table(schema: str, table: str, rule: TableRule) -> DataFrame:
    relation = f"{qident(schema)}.{qident(table)}"
    if rule.where_clause:
        dbtable = f"(SELECT * FROM {relation} WHERE {rule.where_clause}) src"
    else:
        dbtable = relation

    options = dict(JDBC_BASE_OPTIONS)
    options["dbtable"] = dbtable

    if (
        rule.partition_column
        and rule.lower_bound is not None
        and rule.upper_bound is not None
        and rule.num_partitions > 1
    ):
        options.update(
            {
                "partitionColumn": rule.partition_column,
                "lowerBound": str(rule.lower_bound),
                "upperBound": str(rule.upper_bound),
                "numPartitions": str(rule.num_partitions),
            }
        )

    return spark.read.format("jdbc").options(**options).load()


def read_target_table(table: str, rule: TableRule) -> DataFrame:
    full_name = (
        f"{spark_qident(TARGET_CATALOG)}."
        f"{spark_qident(TARGET_SCHEMA)}."
        f"{spark_qident(table)}"
    )
    df = spark.table(full_name)
    return df.where(rule.where_clause) if rule.where_clause else df


def pg_tables(schema: str) -> List[str]:
    query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{schema.replace("'", "''")}'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """
    return [r["table_name"] for r in jdbc_query_df(query).collect()]


def dbx_tables() -> List[str]:
    rows = spark.sql(
        f"SHOW TABLES IN {spark_qident(TARGET_CATALOG)}.{spark_qident(TARGET_SCHEMA)}"
    ).collect()
    return sorted([r["tableName"] for r in rows if not r["isTemporary"]])


def pg_primary_key_columns(schema: str, table: str) -> List[str]:
    query = f"""
        SELECT a.attname AS column_name
        FROM pg_index i
        JOIN pg_class t ON t.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
        WHERE i.indisprimary
          AND n.nspname = '{schema.replace("'", "''")}'
          AND t.relname = '{table.replace("'", "''")}'
        ORDER BY k.ord
    """
    return [r["column_name"] for r in jdbc_query_df(query).collect()]


def pg_estimated_rows(schema: str, table: str) -> Optional[int]:
    query = f"""
        SELECT GREATEST(c.reltuples::bigint, 0) AS estimated_rows
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = '{schema.replace("'", "''")}'
          AND c.relname = '{table.replace("'", "''")}'
    """
    rows = jdbc_query_df(query).collect()
    return int(rows[0]["estimated_rows"]) if rows else None


def source_exact_count(schema: str, table: str, rule: TableRule) -> int:
    where = f" WHERE {rule.where_clause}" if rule.where_clause else ""
    query = f"SELECT COUNT(*)::bigint AS row_count FROM {qident(schema)}.{qident(table)}{where}"
    return int(jdbc_query_df(query).first()["row_count"])


# COMMAND ----------

# -----------------------------
# Canonicalization
# -----------------------------
NUMERIC_TYPES = (
    T.ByteType,
    T.ShortType,
    T.IntegerType,
    T.LongType,
    T.FloatType,
    T.DoubleType,
    T.DecimalType,
)
DATE_TYPES = (T.DateType,)
TIMESTAMP_TYPES = (T.TimestampType, T.TimestampNTZType)
COMPLEX_TYPES = (T.ArrayType, T.MapType, T.StructType)


def canonical_expr(field: T.StructField, rule: TableRule) -> F.Column:
    c = F.col(spark_qident(field.name))
    dt = field.dataType

    if isinstance(dt, T.StringType):
        out = c
        if rule.trim_strings:
            out = F.trim(out)
        if rule.case_insensitive_strings:
            out = F.lower(out)
        if rule.null_equivalents:
            nulls = [x.lower() if rule.case_insensitive_strings else x for x in rule.null_equivalents]
            out = F.when(out.isin(nulls), F.lit(None)).otherwise(out)
        return out.cast("string")

    if isinstance(dt, T.BooleanType):
        return c.cast("boolean").cast("string")

    if isinstance(dt, NUMERIC_TYPES):
        # Decimal string avoids scientific-notation inconsistencies.
        return c.cast("decimal(38,18)").cast("string")

    if isinstance(dt, DATE_TYPES):
        return F.date_format(c.cast("date"), "yyyy-MM-dd")

    if isinstance(dt, TIMESTAMP_TYPES):
        return F.date_format(
            F.to_utc_timestamp(c.cast("timestamp"), "UTC"),
            "yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'",
        )

    if isinstance(dt, T.BinaryType):
        return F.base64(c)

    if isinstance(dt, COMPLEX_TYPES):
        return F.to_json(c, options={"ignoreNullFields": "false"})

    return c.cast("string")


def canonicalize(
    df: DataFrame,
    common_columns: Sequence[str],
    key_columns: Sequence[str],
    rule: TableRule,
    side: str,
) -> DataFrame:
    field_map = {f.name.lower(): f for f in df.schema.fields}
    actual_map = {c.lower(): c for c in df.columns}

    selected = []
    for logical_name in common_columns:
        actual = actual_map[logical_name.lower()]
        field = field_map[logical_name.lower()]
        selected.append(
            canonical_expr(
                T.StructField(logical_name, field.dataType, field.nullable), rule
            ).alias(logical_name)
        )

    out = df.select(*selected)

    non_key_cols = [c for c in common_columns if c.lower() not in {k.lower() for k in key_columns}]
    hash_inputs = [
        F.coalesce(F.col(spark_qident(c)), F.lit("∅")).alias(c)
        for c in non_key_cols
    ]

    # SHA-256 is used for robust equality checking. xxhash64 is also retained
    # as a fast bucketing/checksum statistic.
    canonical_payload = F.to_json(
        F.struct(*hash_inputs), options={"ignoreNullFields": "false"}
    )

    return (
        out.withColumn(f"__{side}_payload", canonical_payload)
        .withColumn(f"__{side}_sha256", F.sha2(F.col(f"__{side}_payload"), 256))
        .withColumn(f"__{side}_xxhash64", F.xxhash64(*hash_inputs))
    )


# COMMAND ----------

# -----------------------------
# Schema comparison
# -----------------------------
def schema_rows(
    source_df: DataFrame,
    target_df: DataFrame,
    table: str,
) -> List[dict]:
    src = {f.name.lower(): f for f in source_df.schema.fields}
    tgt = {f.name.lower(): f for f in target_df.schema.fields}
    rows = []

    for col in sorted(set(src) | set(tgt)):
        sf = src.get(col)
        tf = tgt.get(col)

        if sf is None:
            status = "TARGET_ONLY_COLUMN"
        elif tf is None:
            status = "SOURCE_ONLY_COLUMN"
        elif sf.dataType.simpleString() != tf.dataType.simpleString():
            status = "TYPE_MISMATCH"
        elif sf.nullable != tf.nullable:
            status = "NULLABILITY_MISMATCH"
        else:
            status = "MATCH"

        rows.append(
            {
                "run_id": RUN_ID,
                "table_name": table,
                "column_name": col,
                "source_data_type": sf.dataType.simpleString() if sf else None,
                "target_data_type": tf.dataType.simpleString() if tf else None,
                "source_nullable": sf.nullable if sf else None,
                "target_nullable": tf.nullable if tf else None,
                "status": status,
                "run_ts": RUN_TS,
            }
        )
    return rows


# COMMAND ----------

# -----------------------------
# Key quality statistics
# -----------------------------
def key_quality(df: DataFrame, keys: Sequence[str], prefix: str) -> dict:
    if not keys:
        return {
            f"{prefix}_null_key_rows": None,
            f"{prefix}_duplicate_key_groups": None,
            f"{prefix}_duplicate_key_rows": None,
        }

    null_condition = reduce(
        lambda a, b: a | b,
        [F.col(spark_qident(k)).isNull() for k in keys],
    )
    null_key_rows = df.where(null_condition).count()

    duplicate_groups_df = (
        df.groupBy(*[F.col(spark_qident(k)) for k in keys])
        .count()
        .where(F.col("count") > 1)
    )
    duplicate_key_groups = duplicate_groups_df.count()
    duplicate_key_rows = (
        duplicate_groups_df.agg(
            F.coalesce(F.sum(F.col("count") - 1), F.lit(0)).alias("n")
        ).first()["n"]
    )

    return {
        f"{prefix}_null_key_rows": int(null_key_rows),
        f"{prefix}_duplicate_key_groups": int(duplicate_key_groups),
        f"{prefix}_duplicate_key_rows": int(duplicate_key_rows),
    }


# COMMAND ----------

# -----------------------------
# Data-type-aware value equality
# -----------------------------
def equality_expr(
    src_col: F.Column,
    tgt_col: F.Column,
    data_type: T.DataType,
    rule: TableRule,
) -> F.Column:
    both_null = src_col.isNull() & tgt_col.isNull()
    one_null = src_col.isNull() ^ tgt_col.isNull()

    if isinstance(data_type, NUMERIC_TYPES):
        s = src_col.cast("decimal(38,18)")
        t = tgt_col.cast("decimal(38,18)")
        abs_diff = F.abs(s - t)
        scale = F.greatest(F.abs(s), F.abs(t), F.lit(1.0))
        allowed = F.greatest(
            F.lit(float(rule.numeric_tolerance_abs)),
            F.lit(float(rule.numeric_tolerance_rel)) * scale,
        )
        equal_non_null = abs_diff <= allowed

    elif isinstance(data_type, TIMESTAMP_TYPES):
        diff_seconds = F.abs(
            src_col.cast("timestamp").cast("double")
            - tgt_col.cast("timestamp").cast("double")
        )
        equal_non_null = diff_seconds <= F.lit(rule.timestamp_tolerance_seconds)

    else:
        equal_non_null = src_col.eqNullSafe(tgt_col)

    return F.when(both_null, F.lit(True)).when(one_null, F.lit(False)).otherwise(equal_non_null)


# COMMAND ----------

# -----------------------------
# Bucket-level reconciliation
# -----------------------------
def build_bucket_summary(
    src: DataFrame,
    tgt: DataFrame,
    keys: Sequence[str],
    table: str,
    buckets: int,
) -> DataFrame:
    key_exprs = [F.coalesce(F.col(spark_qident(k)).cast("string"), F.lit("∅")) for k in keys]

    def summarize(df: DataFrame, side: str) -> DataFrame:
        return (
            df.withColumn("__bucket_id", F.pmod(F.xxhash64(*key_exprs), F.lit(buckets)))
            .groupBy("__bucket_id")
            .agg(
                F.count(F.lit(1)).alias(f"{side}_row_count"),
                F.sum(F.col(f"__{side}_xxhash64")).alias(f"{side}_hash_sum"),
                F.min(F.col(f"__{side}_xxhash64")).alias(f"{side}_hash_min"),
                F.max(F.col(f"__{side}_xxhash64")).alias(f"{side}_hash_max"),
                F.approx_count_distinct(
                    F.struct(*[F.col(spark_qident(k)) for k in keys])
                ).alias(f"{side}_approx_distinct_keys"),
            )
        )

    s = summarize(src, "source")
    t = summarize(tgt, "target")

    return (
        s.join(t, "__bucket_id", "full")
        .select(
            F.lit(RUN_ID).alias("run_id"),
            F.lit(table).alias("table_name"),
            F.col("__bucket_id").cast("int").alias("bucket_id"),
            F.col("source_row_count"),
            F.col("target_row_count"),
            F.col("source_hash_sum"),
            F.col("target_hash_sum"),
            F.col("source_hash_min"),
            F.col("target_hash_min"),
            F.col("source_hash_max"),
            F.col("target_hash_max"),
            F.col("source_approx_distinct_keys"),
            F.col("target_approx_distinct_keys"),
            (
                F.coalesce(F.col("source_row_count"), F.lit(0))
                == F.coalesce(F.col("target_row_count"), F.lit(0))
            ).alias("row_count_match"),
            (
                F.coalesce(F.col("source_hash_sum"), F.lit(0))
                == F.coalesce(F.col("target_hash_sum"), F.lit(0))
            ).alias("hash_sum_match"),
            now_col().alias("run_ts"),
        )
    )


# COMMAND ----------

# -----------------------------
# Column statistics
# -----------------------------
def column_statistics(
    joined_changed_rows: DataFrame,
    source_schema: T.StructType,
    common_non_key_columns: Sequence[str],
    table: str,
    total_joined_keys: int,
) -> DataFrame:
    field_map = {f.name.lower(): f for f in source_schema.fields}
    stat_frames = []

    for column_name in common_non_key_columns:
        field = field_map[column_name.lower()]
        s = F.col(f"s.{spark_qident(column_name)}")
        t = F.col(f"t.{spark_qident(column_name)}")
        is_equal = equality_expr(s, t, field.dataType, TABLE_RULES.get(table, DEFAULT_SMALL_TABLE_RULE))
        is_mismatch = ~is_equal

        base_aggs = [
            F.count(F.lit(1)).alias("changed_row_candidates"),
            F.sum(F.when(is_mismatch, 1).otherwise(0)).alias("mismatch_count"),
            F.sum(F.when(s.isNull(), 1).otherwise(0)).alias("source_null_count"),
            F.sum(F.when(t.isNull(), 1).otherwise(0)).alias("target_null_count"),
            F.sum(F.when(s.isNull() & t.isNotNull(), 1).otherwise(0)).alias("source_null_target_not_null"),
            F.sum(F.when(s.isNotNull() & t.isNull(), 1).otherwise(0)).alias("source_not_null_target_null"),
            F.approx_count_distinct(s).alias("source_approx_distinct"),
            F.approx_count_distinct(t).alias("target_approx_distinct"),
        ]

        if isinstance(field.dataType, NUMERIC_TYPES):
            diff = t.cast("double") - s.cast("double")
            abs_diff = F.abs(diff)
            numeric_aggs = [
                F.avg(F.when(is_mismatch, diff)).alias("mean_signed_difference"),
                F.avg(F.when(is_mismatch, abs_diff)).alias("mean_absolute_difference"),
                F.max(F.when(is_mismatch, abs_diff)).alias("max_absolute_difference"),
                F.expr(
                    f"percentile_approx(CASE WHEN NOT ({is_equal._jc.toString()}) "
                    f"THEN abs(cast(t.{spark_qident(column_name)} as double) - "
                    f"cast(s.{spark_qident(column_name)} as double)) END, 0.5, 10000)"
                ).alias("median_absolute_difference"),
                F.expr(
                    f"percentile_approx(CASE WHEN NOT ({is_equal._jc.toString()}) "
                    f"THEN abs(cast(t.{spark_qident(column_name)} as double) - "
                    f"cast(s.{spark_qident(column_name)} as double)) END, 0.95, 10000)"
                ).alias("p95_absolute_difference"),
            ]
        else:
            numeric_aggs = [
                F.lit(None).cast("double").alias("mean_signed_difference"),
                F.lit(None).cast("double").alias("mean_absolute_difference"),
                F.lit(None).cast("double").alias("max_absolute_difference"),
                F.lit(None).cast("double").alias("median_absolute_difference"),
                F.lit(None).cast("double").alias("p95_absolute_difference"),
            ]

        stat = (
            joined_changed_rows.agg(*(base_aggs + numeric_aggs))
            .select(
                F.lit(RUN_ID).alias("run_id"),
                F.lit(table).alias("table_name"),
                F.lit(column_name).alias("column_name"),
                F.lit(field.dataType.simpleString()).alias("data_type"),
                "*",
                F.when(
                    F.lit(total_joined_keys) > 0,
                    F.col("mismatch_count") / F.lit(total_joined_keys) * 100.0,
                ).otherwise(F.lit(0.0)).alias("mismatch_pct_of_joined_keys"),
                now_col().alias("run_ts"),
            )
        )
        stat_frames.append(stat)

    if not stat_frames:
        return spark.createDataFrame([], schema="""
            run_id string, table_name string, column_name string, data_type string,
            changed_row_candidates long, mismatch_count long,
            source_null_count long, target_null_count long,
            source_null_target_not_null long, source_not_null_target_null long,
            source_approx_distinct long, target_approx_distinct long,
            mean_signed_difference double, mean_absolute_difference double,
            max_absolute_difference double, median_absolute_difference double,
            p95_absolute_difference double, mismatch_pct_of_joined_keys double,
            run_ts timestamp
        """)

    return reduce(DataFrame.unionByName, stat_frames)


# COMMAND ----------

# -----------------------------
# Mismatch samples, one row per key + column
# -----------------------------
def mismatch_samples(
    joined_changed_rows: DataFrame,
    source_schema: T.StructType,
    keys: Sequence[str],
    common_non_key_columns: Sequence[str],
    table: str,
    rule: TableRule,
    limit_per_table: int,
) -> DataFrame:
    field_map = {f.name.lower(): f for f in source_schema.fields}
    sample_frames = []

    key_json = F.to_json(
        F.struct(
            *[
                F.coalesce(
                    F.col(f"s.{spark_qident(k)}"),
                    F.col(f"t.{spark_qident(k)}"),
                ).alias(k)
                for k in keys
            ]
        ),
        options={"ignoreNullFields": "false"},
    )

    for column_name in common_non_key_columns:
        field = field_map[column_name.lower()]
        s = F.col(f"s.{spark_qident(column_name)}")
        t = F.col(f"t.{spark_qident(column_name)}")
        is_equal = equality_expr(s, t, field.dataType, rule)

        if isinstance(field.dataType, NUMERIC_TYPES):
            absolute_difference = F.abs(t.cast("double") - s.cast("double"))
            relative_difference = absolute_difference / F.greatest(
                F.abs(s.cast("double")), F.lit(1.0)
            )
        else:
            absolute_difference = F.lit(None).cast("double")
            relative_difference = F.lit(None).cast("double")

        frame = (
            joined_changed_rows.where(~is_equal)
            .select(
                F.lit(RUN_ID).alias("run_id"),
                F.lit(table).alias("table_name"),
                key_json.alias("key_json"),
                F.lit(column_name).alias("column_name"),
                F.lit(field.dataType.simpleString()).alias("data_type"),
                s.cast("string").alias("source_value"),
                t.cast("string").alias("target_value"),
                F.when(s.isNull() & t.isNotNull(), "SOURCE_NULL")
                .when(s.isNotNull() & t.isNull(), "TARGET_NULL")
                .otherwise("VALUE_MISMATCH")
                .alias("mismatch_type"),
                absolute_difference.alias("absolute_difference"),
                relative_difference.alias("relative_difference"),
                now_col().alias("run_ts"),
            )
            .limit(limit_per_table)
        )
        sample_frames.append(frame)

    if not sample_frames:
        return spark.createDataFrame([], schema="""
            run_id string, table_name string, key_json string, column_name string,
            data_type string, source_value string, target_value string,
            mismatch_type string, absolute_difference double,
            relative_difference double, run_ts timestamp
        """)

    # Final table-wide limit. Change to per-column limits if required.
    return reduce(DataFrame.unionByName, sample_frames).limit(limit_per_table)


# COMMAND ----------

# -----------------------------
# Main table comparison
# -----------------------------
def compare_table(table: str, rule: TableRule) -> None:
    started = datetime.now(timezone.utc)
    print(f"[{table}] Starting")

    source_df = read_postgres_table(SOURCE_SCHEMA, table, rule)
    target_df = read_target_table(table, rule)

    schema_result = schema_rows(source_df, target_df, table)
    append_delta(spark.createDataFrame(schema_result), T_SCHEMA)

    schema_mismatches = [r for r in schema_result if r["status"] != "MATCH"]
    if FAIL_ON_SCHEMA_MISMATCH and schema_mismatches:
        raise ValueError(f"Schema mismatch for {table}: {schema_mismatches}")

    if RUN_MODE == "METADATA_ONLY":
        print(f"[{table}] Metadata complete")
        return

    source_count = source_df.count()
    target_count = target_df.count()

    key_columns = rule.key_columns or pg_primary_key_columns(SOURCE_SCHEMA, table)
    if not key_columns:
        raise ValueError(
            f"No key configured or detected for {table}. "
            "Record-to-record comparison requires a primary/business key."
        )

    src_actual = {c.lower(): c for c in source_df.columns}
    tgt_actual = {c.lower(): c for c in target_df.columns}

    missing_source_keys = [k for k in key_columns if k.lower() not in src_actual]
    missing_target_keys = [k for k in key_columns if k.lower() not in tgt_actual]
    if missing_source_keys or missing_target_keys:
        raise ValueError(
            f"Missing key columns. source={missing_source_keys}, "
            f"target={missing_target_keys}"
        )

    # Normalize key spelling to source logical names.
    key_columns = [src_actual[k.lower()] for k in key_columns]

    source_key_stats = key_quality(source_df, key_columns, "source")
    target_key_stats = key_quality(target_df, key_columns, "target")

    if RUN_MODE == "COUNTS_ONLY":
        result = {
            "run_id": RUN_ID,
            "table_name": table,
            "status": "COUNTS_ONLY",
            "source_row_count": source_count,
            "target_row_count": target_count,
            "row_count_difference": target_count - source_count,
            "source_only_rows": None,
            "target_only_rows": None,
            "matched_equal_rows": None,
            "matched_changed_rows": None,
            "matched_key_rows": None,
            "match_pct": None,
            "common_column_count": None,
            "source_only_column_count": len(
                [x for x in schema_result if x["status"] == "SOURCE_ONLY_COLUMN"]
            ),
            "target_only_column_count": len(
                [x for x in schema_result if x["status"] == "TARGET_ONLY_COLUMN"]
            ),
            **source_key_stats,
            **target_key_stats,
            "started_ts": started,
            "completed_ts": datetime.now(timezone.utc),
            "duration_seconds": (
                datetime.now(timezone.utc) - started
            ).total_seconds(),
            "error_message": None,
        }
        append_delta(one_row_df(result), T_TABLE)
        return

    excluded = {c.lower() for c in rule.excluded_columns}
    common_columns = [
        src_actual[c]
        for c in sorted(set(src_actual) & set(tgt_actual))
        if c not in excluded
    ]

    # Ensure keys occur first and exactly once.
    non_key_columns = [
        c for c in common_columns
        if c.lower() not in {k.lower() for k in key_columns}
    ]
    ordered_columns = key_columns + non_key_columns

    src_c = canonicalize(
        source_df, ordered_columns, key_columns, rule, "source"
    ).alias("s")
    tgt_c = canonicalize(
        target_df, ordered_columns, key_columns, rule, "target"
    ).alias("t")

    # Persist canonical representations so count, bucket, join, and diagnostics
    # do not repeatedly pull BigAnimal data.
    src_c.persist(StorageLevel.DISK_ONLY)
    tgt_c.persist(StorageLevel.DISK_ONLY)
    src_c.count()
    tgt_c.count()

    bucket_df = build_bucket_summary(
        src_c, tgt_c, key_columns, table, rule.hash_buckets
    )
    append_delta(bucket_df, T_BUCKET)

    join_condition = reduce(
        lambda a, b: a & b,
        [
            F.col(f"s.{spark_qident(k)}").eqNullSafe(
                F.col(f"t.{spark_qident(k)}")
            )
            for k in key_columns
        ],
    )

    joined = src_c.join(tgt_c, join_condition, "fullouter").persist(StorageLevel.DISK_ONLY)

    source_present = F.col("s.__source_sha256").isNotNull()
    target_present = F.col("t.__target_sha256").isNotNull()
    same_hash = F.col("s.__source_sha256").eqNullSafe(F.col("t.__target_sha256"))

    counts = (
        joined.agg(
            F.sum(F.when(source_present & ~target_present, 1).otherwise(0)).alias("source_only_rows"),
            F.sum(F.when(~source_present & target_present, 1).otherwise(0)).alias("target_only_rows"),
            F.sum(F.when(source_present & target_present & same_hash, 1).otherwise(0)).alias("matched_equal_rows"),
            F.sum(F.when(source_present & target_present & ~same_hash, 1).otherwise(0)).alias("matched_changed_rows"),
            F.sum(F.when(source_present & target_present, 1).otherwise(0)).alias("matched_key_rows"),
        )
        .first()
        .asDict()
    )

    changed = joined.where(
        source_present & target_present & ~same_hash
    ).persist(StorageLevel.DISK_ONLY)

    total_joined_keys = int(counts["matched_key_rows"] or 0)

    column_df = column_statistics(
        changed,
        source_df.schema,
        non_key_columns,
        table,
        total_joined_keys,
    )
    append_delta(column_df, T_COLUMN)

    sample_df = mismatch_samples(
        changed,
        source_df.schema,
        key_columns,
        non_key_columns,
        table,
        rule,
        SAMPLE_MISMATCHES_PER_TABLE,
    )
    append_delta(sample_df, T_SAMPLE)

    matched_equal = int(counts["matched_equal_rows"] or 0)
    matched_changed = int(counts["matched_changed_rows"] or 0)
    source_only = int(counts["source_only_rows"] or 0)
    target_only = int(counts["target_only_rows"] or 0)
    denominator = max(source_count, target_count, 1)
    match_pct = matched_equal / denominator * 100.0

    result = {
        "run_id": RUN_ID,
        "table_name": table,
        "status": "PASS" if source_only == 0 and target_only == 0 and matched_changed == 0 else "FAIL",
        "source_row_count": int(source_count),
        "target_row_count": int(target_count),
        "row_count_difference": int(target_count - source_count),
        "source_only_rows": source_only,
        "target_only_rows": target_only,
        "matched_equal_rows": matched_equal,
        "matched_changed_rows": matched_changed,
        "matched_key_rows": total_joined_keys,
        "match_pct": float(match_pct),
        "common_column_count": len(ordered_columns),
        "source_only_column_count": len(
            [x for x in schema_result if x["status"] == "SOURCE_ONLY_COLUMN"]
        ),
        "target_only_column_count": len(
            [x for x in schema_result if x["status"] == "TARGET_ONLY_COLUMN"]
        ),
        **source_key_stats,
        **target_key_stats,
        "started_ts": started,
        "completed_ts": datetime.now(timezone.utc),
        "duration_seconds": (
            datetime.now(timezone.utc) - started
        ).total_seconds(),
        "error_message": None,
    }
    append_delta(one_row_df(result), T_TABLE)

    changed.unpersist()
    joined.unpersist()
    src_c.unpersist()
    tgt_c.unpersist()

    print(f"[{table}] Completed: {result['status']} | match={match_pct:.6f}%")


# COMMAND ----------

# -----------------------------
# Register run
# -----------------------------
append_delta(
    one_row_df(
        {
            "run_id": RUN_ID,
            "run_ts": RUN_TS,
            "source_system": "BigAnimal PostgreSQL",
            "source_database": PG_DATABASE,
            "source_schema": SOURCE_SCHEMA,
            "target_catalog": TARGET_CATALOG,
            "target_schema": TARGET_SCHEMA,
            "run_mode": RUN_MODE,
            "status": "RUNNING",
            "configuration_json": json.dumps(
                {
                    "table_include_regex": TABLE_INCLUDE_REGEX,
                    "table_exclude_regex": TABLE_EXCLUDE_REGEX,
                    "max_tables": MAX_TABLES,
                    "sample_mismatches_per_table": SAMPLE_MISMATCHES_PER_TABLE,
                    "table_rules": {k: asdict(v) for k, v in TABLE_RULES.items()},
                },
                default=str,
            ),
            "completed_ts": None,
        }
    ),
    T_RUN,
)

# COMMAND ----------

# -----------------------------
# Discover and filter tables
# -----------------------------
source_tables = set(pg_tables(SOURCE_SCHEMA))
target_tables = set(dbx_tables())

all_tables = sorted(source_tables | target_tables)
selected_tables = [
    t
    for t in all_tables
    if re.search(TABLE_INCLUDE_REGEX, t)
    and not re.search(TABLE_EXCLUDE_REGEX, t)
]
if MAX_TABLES > 0:
    selected_tables = selected_tables[:MAX_TABLES]

print(f"Run ID: {RUN_ID}")
print(f"Selected {len(selected_tables)} tables")

# Record tables existing on only one side.
table_presence_rows = []
for table in selected_tables:
    if table not in source_tables or table not in target_tables:
        table_presence_rows.append(
            {
                "run_id": RUN_ID,
                "table_name": table,
                "column_name": None,
                "source_data_type": None,
                "target_data_type": None,
                "source_nullable": None,
                "target_nullable": None,
                "status": (
                    "SOURCE_ONLY_TABLE"
                    if table in source_tables
                    else "TARGET_ONLY_TABLE"
                ),
                "run_ts": RUN_TS,
            }
        )
if table_presence_rows:
    append_delta(spark.createDataFrame(table_presence_rows), T_SCHEMA)

comparable_tables = [
    t for t in selected_tables if t in source_tables and t in target_tables
]

# COMMAND ----------

# -----------------------------
# Execute comparisons
# -----------------------------
failure_count = 0

for table in comparable_tables:
    try:
        configured_rule = TABLE_RULES.get(table.lower())

        if configured_rule is None:
            detected_keys = pg_primary_key_columns(SOURCE_SCHEMA, table)
            configured_rule = TableRule(
                key_columns=detected_keys,
                num_partitions=1,
                hash_buckets=DEFAULT_SMALL_TABLE_RULE.hash_buckets,
            )

        compare_table(table, configured_rule)

    except Exception as exc:
        failure_count += 1
        error_text = f"{type(exc).__name__}: {str(exc)}"
        print(f"[{table}] ERROR: {error_text}")

        append_delta(
            one_row_df(
                {
                    "run_id": RUN_ID,
                    "table_name": table,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:10000],
                    "stack_trace": traceback.format_exc()[:30000],
                    "run_ts": datetime.now(timezone.utc),
                }
            ),
            T_ERROR,
        )

        append_delta(
            one_row_df(
                {
                    "run_id": RUN_ID,
                    "table_name": table,
                    "status": "ERROR",
                    "source_row_count": None,
                    "target_row_count": None,
                    "row_count_difference": None,
                    "source_only_rows": None,
                    "target_only_rows": None,
                    "matched_equal_rows": None,
                    "matched_changed_rows": None,
                    "matched_key_rows": None,
                    "match_pct": None,
                    "common_column_count": None,
                    "source_only_column_count": None,
                    "target_only_column_count": None,
                    "source_null_key_rows": None,
                    "source_duplicate_key_groups": None,
                    "source_duplicate_key_rows": None,
                    "target_null_key_rows": None,
                    "target_duplicate_key_groups": None,
                    "target_duplicate_key_rows": None,
                    "started_ts": None,
                    "completed_ts": datetime.now(timezone.utc),
                    "duration_seconds": None,
                    "error_message": error_text[:10000],
                }
            ),
            T_TABLE,
        )

# COMMAND ----------

# -----------------------------
# Complete run
# -----------------------------
spark.sql(
    f"""
    UPDATE {T_RUN}
       SET status = '{"COMPLETED_WITH_ERRORS" if failure_count else "COMPLETED"}',
           completed_ts = current_timestamp()
     WHERE run_id = '{RUN_ID}'
    """
)

print(
    f"Reconciliation completed. run_id={RUN_ID}, "
    f"tables={len(comparable_tables)}, failures={failure_count}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary queries

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT *
        FROM {T_TABLE}
        WHERE run_id = '{RUN_ID}'
        ORDER BY
          CASE status WHEN 'ERROR' THEN 1 WHEN 'FAIL' THEN 2 ELSE 3 END,
          table_name
        """
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT
          table_name,
          column_name,
          data_type,
          mismatch_count,
          mismatch_pct_of_joined_keys,
          source_null_count,
          target_null_count,
          mean_absolute_difference,
          max_absolute_difference,
          p95_absolute_difference
        FROM {T_COLUMN}
        WHERE run_id = '{RUN_ID}'
          AND mismatch_count > 0
        ORDER BY mismatch_count DESC, table_name, column_name
        """
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT *
        FROM {T_SAMPLE}
        WHERE run_id = '{RUN_ID}'
        ORDER BY table_name, key_json, column_name
        """
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recommended production orchestration
# MAGIC
# MAGIC 1. Run one Databricks Workflow task per large table rather than one notebook task for the full 13 TB.
# MAGIC 2. Pass `table_include_regex=^table_name$` to each task.
# MAGIC 3. Use a read replica or controlled reconciliation window on BigAnimal.
# MAGIC 4. Configure `partition_column`, bounds, and `num_partitions` per large table.
# MAGIC 5. Validate immutable historical partitions once; rerun only open/current partitions.
# MAGIC 6. Use the bucket summary to isolate bad key ranges before materializing detailed mismatches.
# MAGIC 7. Freeze source writes or use a transactionally consistent source snapshot and a Delta table version/timestamp.
