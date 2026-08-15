"""
PostgreSQL <-> Databricks In-Memory Reconciliation Utility
==========================================================

Supported checks:
    Counts_Check
    Type_Check
    Sample_Check
    Duplicate_Check
    All_Check

All_Check runs Counts_Check + Type_Check + Sample_Check.
Duplicate_Check remains available only when explicitly selected.

All data remains in DataFrames in memory.
No Volumes, DBFS paths, or CSV files are used.

Designed to run inside Databricks where:
    - Spark is available.
    - PostgreSQL is reachable over the network.
    - psycopg2 and pandas are installed.

Requirements:
    pip install psycopg2-binary pandas

Notes:
    - PostgreSQL is treated as SOURCE.
    - Databricks is treated as TARGET.
    - Sample_Check selects the latest N PostgreSQL rows using DATE_COLUMN.
    - The sampled unique keys are used to retrieve the corresponding
      Databricks rows.
    - SHA-256 row hashes are calculated on columns common to both systems,
      excluding IGNORE_COLUMNS.
    - Reconciliation is performed using UNIQUE_KEY_COLUMNS.
"""

import hashlib
import json
from decimal import Decimal

import pandas as pd
import psycopg2
from psycopg2 import sql

from pyspark.sql import functions as F
from pyspark.sql import types as T


# ============================================================
# CONFIGURATION
# ============================================================

# PostgreSQL SOURCE
POSTGRES_CONFIG = {
    "host": "your-postgres-host",
    "port": 5432,
    "database": "your_database",
    "user": "your_username",
    "password": "your_password"
}

POSTGRES_SCHEMA = "public"
POSTGRES_TABLE = "your_table"

# Databricks TARGET
DATABRICKS_CATALOG = "your_catalog"
DATABRICKS_SCHEMA = "your_schema"
DATABRICKS_TABLE = "your_table"

# Allowed:
# Counts_Check
# Type_Check
# Sample_Check
# Duplicate_Check
# All_Check
CHECK_FLAG = "All_Check"

# Sample Check configuration
UNIQUE_KEY_COLUMNS = "policy_id,claim_id"
DATE_COLUMN = "intimation_date"
IGNORE_COLUMNS = "created_timestamp,updated_timestamp"
SAMPLE_SIZE = 1000


# ============================================================
# GENERAL HELPERS
# ============================================================

def parse_column_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


def is_datetime_like_value(value):
    if value is None:
        return False
    if isinstance(value, pd.Timestamp):
        return True
    module_name = type(value).__module__
    class_name = type(value).__name__.lower()
    return (
        "datetime" in class_name
        or class_name == "date"
        or module_name == "datetime"
    )


def safe_scalar_for_dataframe(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if is_datetime_like_value(value):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def make_pandas_dataframe_timestamp_safe(dataframe):
    if dataframe is None:
        return dataframe
    safe_df = dataframe.copy()
    for column in safe_df.columns:
        safe_df[column] = safe_df[column].map(safe_scalar_for_dataframe)
    return safe_df


def pandas_to_spark_safe(dataframe):
    if dataframe.empty:
        return spark.createDataFrame([], T.StructType([]))
    safe_df = make_pandas_dataframe_timestamp_safe(dataframe)
    clean_df = safe_df.astype(object).where(pd.notnull(safe_df), None)
    return spark.createDataFrame(clean_df)


def spark_dataframe_to_pandas_safe(dataframe, columns=None):
    working_df = dataframe
    if columns is not None:
        working_df = working_df.select(
            *[F.col(c) for c in unique_preserve_order(columns)]
        )
    converted = []
    for field in working_df.schema.fields:
        name = field.name
        dt = field.dataType
        if isinstance(dt, T.TimestampType):
            converted.append(
                F.date_format(F.col(name), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS").alias(name)
            )
        elif isinstance(dt, T.DateType):
            converted.append(
                F.date_format(F.col(name), "yyyy-MM-dd").alias(name)
            )
        else:
            converted.append(F.col(name))
    return working_df.select(*converted).toPandas()


def unique_preserve_order(values):
    """
    Remove duplicate items while preserving their original order.
    Useful when primary-key columns are also part of hash columns.
    """
    seen = set()
    result = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


# ============================================================
# POSTGRES HELPERS
# ============================================================

def get_postgres_connection():
    return psycopg2.connect(
        host=POSTGRES_CONFIG["host"],
        port=POSTGRES_CONFIG["port"],
        database=POSTGRES_CONFIG["database"],
        user=POSTGRES_CONFIG["user"],
        password=POSTGRES_CONFIG["password"]
    )


def validate_postgres_table(conn):
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
        );
    """

    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (POSTGRES_SCHEMA, POSTGRES_TABLE)
        )
        exists = cursor.fetchone()[0]

    if not exists:
        raise ValueError(
            f"PostgreSQL table "
            f"{POSTGRES_SCHEMA}.{POSTGRES_TABLE} does not exist."
        )


def get_postgres_columns(conn):
    query = """
        SELECT
            ordinal_position,
            column_name,
            data_type,
            udt_name,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            datetime_precision,
            is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
    """

    return pd.read_sql_query(
        query,
        conn,
        params=(POSTGRES_SCHEMA, POSTGRES_TABLE)
    )


# ============================================================
# DATABRICKS HELPERS
# ============================================================

def databricks_full_name():
    return (
        f"{DATABRICKS_CATALOG}."
        f"{DATABRICKS_SCHEMA}."
        f"{DATABRICKS_TABLE}"
    )


def validate_databricks_table():
    if not spark.catalog.tableExists(databricks_full_name()):
        raise ValueError(
            f"Databricks table {databricks_full_name()} does not exist."
        )


def get_databricks_dataframe():
    return spark.table(databricks_full_name())


# ============================================================
# HASH NORMALIZATION
# ============================================================

def normalize_python_value(value):
    if value is None:
        return "<NULL>"

    try:
        if pd.isna(value):
            return "<NULL>"
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, Decimal):
        return format(value, "f")

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            sort_keys=True,
            default=str
        )

    return str(value)


def generate_postgres_row_hash(row, hash_columns):
    parts = []

    for column in hash_columns:
        parts.append(
            f"{column}={normalize_python_value(row[column])}"
        )

    return hashlib.sha256(
        "||".join(parts).encode("utf-8")
    ).hexdigest()


def databricks_normalized_expression(column_name, data_type):
    column = F.col(
        f"`{column_name.replace('`', '``')}`"
    )

    if isinstance(data_type, T.DateType):
        value_expr = F.date_format(
            column,
            "yyyy-MM-dd"
        )

    elif isinstance(data_type, T.TimestampType):
        value_expr = F.date_format(
            column,
            "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
        )

    elif isinstance(data_type, T.BooleanType):
        value_expr = (
            F.when(
                column == F.lit(True),
                F.lit("true")
            )
            .otherwise(F.lit("false"))
        )

    elif isinstance(
        data_type,
        (T.ArrayType, T.MapType, T.StructType)
    ):
        value_expr = F.to_json(column)

    else:
        value_expr = column.cast("string")

    normalized_value = F.coalesce(
        value_expr,
        F.lit("<NULL>")
    )

    return F.concat(
        F.lit(f"{column_name}="),
        normalized_value
    )


# ============================================================
# COUNTS CHECK
# ============================================================

def postgres_counts_check(conn):
    query = sql.SQL("""
        SELECT COUNT(*) AS total_count
        FROM {}.{}
    """).format(
        sql.Identifier(POSTGRES_SCHEMA),
        sql.Identifier(POSTGRES_TABLE)
    )

    with conn.cursor() as cursor:
        cursor.execute(query)
        total_count = cursor.fetchone()[0]

    return pd.DataFrame([
        {
            "system": "PostgreSQL",
            "database_or_catalog": POSTGRES_CONFIG["database"],
            "schema": POSTGRES_SCHEMA,
            "table": POSTGRES_TABLE,
            "total_count": int(total_count)
        }
    ])


def databricks_counts_check():
    total_count = get_databricks_dataframe().count()

    return spark.createDataFrame(
        [
            (
                "Databricks",
                DATABRICKS_CATALOG,
                DATABRICKS_SCHEMA,
                DATABRICKS_TABLE,
                int(total_count)
            )
        ],
        [
            "system",
            "database_or_catalog",
            "schema",
            "table",
            "total_count"
        ]
    )


def reconcile_counts(postgres_df, databricks_df):
    postgres_count = int(
        postgres_df.iloc[0]["total_count"]
    )

    databricks_count = int(
        databricks_df.collect()[0]["total_count"]
    )

    difference = databricks_count - postgres_count

    status = "MATCH" if difference == 0 else "MISMATCH"

    return spark.createDataFrame(
        [
            (
                postgres_count,
                databricks_count,
                difference,
                status
            )
        ],
        [
            "postgres_count",
            "databricks_count",
            "difference",
            "status"
        ]
    )


# ============================================================
# TYPE CHECK
# ============================================================

def postgres_type_check(conn):
    df = get_postgres_columns(conn).copy()
    df["system"] = "PostgreSQL"

    return df[
        [
            "system",
            "ordinal_position",
            "column_name",
            "data_type",
            "udt_name",
            "character_maximum_length",
            "numeric_precision",
            "numeric_scale",
            "datetime_precision",
            "is_nullable"
        ]
    ]


def databricks_type_check():
    table_df = get_databricks_dataframe()

    rows = []

    for ordinal_position, field in enumerate(
        table_df.schema.fields,
        start=1
    ):
        rows.append(
            (
                "Databricks",
                ordinal_position,
                field.name,
                field.dataType.simpleString(),
                field.dataType.typeName(),
                field.nullable
            )
        )

    return spark.createDataFrame(
        rows,
        [
            "system",
            "ordinal_position",
            "column_name",
            "data_type",
            "type_name",
            "nullable"
        ]
    )


def normalize_type_name(value):
    if value is None:
        return None

    value = str(value).lower()

    mappings = {
        "int2": "smallint",
        "int4": "integer",
        "int8": "bigint",
        "float4": "real",
        "float8": "double precision",
        "varchar": "character varying",
        "bool": "boolean",
        "timestamp_ntz": "timestamp",
        "timestamp_ltz": "timestamp",
        "string": "character varying",
        "long": "bigint",
        "integer": "integer",
        "short": "smallint",
        "double": "double precision",
        "float": "real",
        "boolean": "boolean",
        "date": "date",
        "binary": "binary",
        "decimal": "numeric"
    }

    return mappings.get(value, value)


def reconcile_types(postgres_type_df, databricks_type_df):
    pg = postgres_type_df.copy()
    dbx = databricks_type_df.toPandas()

    pg["column_key"] = pg["column_name"].str.lower()
    dbx["column_key"] = dbx["column_name"].str.lower()

    merged = pd.merge(
        pg,
        dbx,
        how="outer",
        on="column_key",
        suffixes=("_postgres", "_databricks")
    )

    merged["postgres_normalized_type"] = merged[
        "udt_name"
    ].apply(normalize_type_name)

    merged["databricks_normalized_type"] = merged[
        "type_name"
    ].apply(normalize_type_name)

    def type_status(row):
        pg_col = row.get("column_name_postgres")
        dbx_col = row.get("column_name_databricks")

        if pd.isna(pg_col):
            return "MISSING_IN_POSTGRES"

        if pd.isna(dbx_col):
            return "MISSING_IN_DATABRICKS"

        if (
            row["postgres_normalized_type"]
            == row["databricks_normalized_type"]
        ):
            return "MATCH"

        return "TYPE_MISMATCH"

    merged["status"] = merged.apply(
        type_status,
        axis=1
    )

    columns = [
        "column_name_postgres",
        "column_name_databricks",
        "ordinal_position_postgres",
        "ordinal_position_databricks",
        "data_type_postgres",
        "udt_name",
        "data_type_databricks",
        "type_name",
        "postgres_normalized_type",
        "databricks_normalized_type",
        "status"
    ]

    clean = merged[columns].astype(object).where(
        pd.notnull(merged[columns]),
        None
    )

    return spark.createDataFrame(clean)


# ============================================================
# SAMPLE CHECK - POSTGRES
# ============================================================

def postgres_sample_check(conn):
    key_columns = parse_column_list(UNIQUE_KEY_COLUMNS)
    ignore_columns = parse_column_list(IGNORE_COLUMNS)

    column_meta = get_postgres_columns(conn)
    table_columns = column_meta["column_name"].tolist()

    missing_keys = [
        column
        for column in key_columns
        if column not in table_columns
    ]

    if missing_keys:
        raise ValueError(
            "PostgreSQL unique key columns not found: "
            + ", ".join(missing_keys)
        )

    if DATE_COLUMN not in table_columns:
        raise ValueError(
            f"PostgreSQL date column '{DATE_COLUMN}' not found."
        )

    valid_ignore_columns = [
        column
        for column in ignore_columns
        if column in table_columns
    ]

    hash_columns = [
        column
        for column in table_columns
        if column not in valid_ignore_columns
    ]

    order_parts = [
        sql.SQL(
            "{} DESC NULLS LAST"
        ).format(
            sql.Identifier(DATE_COLUMN)
        )
    ]

    for key_column in key_columns:
        if key_column != DATE_COLUMN:
            order_parts.append(
                sql.SQL(
                    "{} ASC NULLS LAST"
                ).format(
                    sql.Identifier(key_column)
                )
            )

    query = sql.SQL("""
        SELECT *
        FROM {}.{}
        ORDER BY {}
        LIMIT %s
    """).format(
        sql.Identifier(POSTGRES_SCHEMA),
        sql.Identifier(POSTGRES_TABLE),
        sql.SQL(", ").join(order_parts)
    )

    sample_df = pd.read_sql_query(
        query.as_string(conn),
        conn,
        params=(SAMPLE_SIZE,)
    )

    # Preserve PostgreSQL extreme timestamps as Python objects.
    # Do not call pd.to_datetime; sentinel values like 9999-12-31
    # exceed pandas timestamp[ns] range.
    sample_df = sample_df.astype(object)

    return sample_df, hash_columns


# ============================================================
# SAMPLE CHECK - DATABRICKS
# ============================================================

def databricks_sample_for_keys(
    postgres_sample_df,
    postgres_hash_columns
):
    key_columns = parse_column_list(UNIQUE_KEY_COLUMNS)

    target_df = get_databricks_dataframe()
    target_columns = target_df.columns

    missing_keys = [
        column
        for column in key_columns
        if column not in target_columns
    ]

    if missing_keys:
        raise ValueError(
            "Databricks unique key columns not found: "
            + ", ".join(missing_keys)
        )

    common_hash_columns = [
        column
        for column in postgres_hash_columns
        if column in target_columns
    ]

    source_only_hash_columns = [
        column
        for column in postgres_hash_columns
        if column not in target_columns
    ]

    key_pdf = (
        postgres_sample_df[key_columns]
        .drop_duplicates()
        .copy()
    )

    if key_pdf.empty:
        empty_target = target_df.limit(0)
        return (
            empty_target,
            common_hash_columns,
            source_only_hash_columns
        )

    key_sdf = spark.createDataFrame(
        key_pdf.astype(object).where(
            pd.notnull(key_pdf),
            None
        )
    )

    joined_df = target_df.join(
        F.broadcast(key_sdf),
        on=key_columns,
        how="inner"
    )

    schema_by_column = {
        field.name: field.dataType
        for field in target_df.schema.fields
    }

    hash_components = [
        databricks_normalized_expression(
            column,
            schema_by_column[column]
        )
        for column in common_hash_columns
    ]

    result_df = joined_df.withColumn(
        "ROW_HASH",
        F.sha2(
            F.concat_ws(
                "||",
                *hash_components
            ),
            256
        )
    )

    return (
        result_df,
        common_hash_columns,
        source_only_hash_columns
    )


def postgres_rehash_with_common_columns(
    postgres_sample_df,
    common_hash_columns
):
    result = postgres_sample_df.copy()

    if result.empty:
        result["ROW_HASH"] = pd.Series(dtype="object")
        return result

    result["ROW_HASH"] = result.apply(
        lambda row:
        generate_postgres_row_hash(
            row,
            common_hash_columns
        ),
        axis=1
    )

    return result


def reconcile_samples(
    postgres_sample_df,
    databricks_sample_df
):
    key_columns = parse_column_list(UNIQUE_KEY_COLUMNS)

    pg = postgres_sample_df[
        key_columns + ["ROW_HASH"]
    ].copy()

    pg = pg.rename(
        columns={
            "ROW_HASH": "POSTGRES_ROW_HASH"
        }
    )

    dbx = spark_dataframe_to_pandas_safe(
        databricks_sample_df.select(
            *key_columns,
            F.col("ROW_HASH").alias("DATABRICKS_ROW_HASH")
        )
    )

    merged = pd.merge(
        pg,
        dbx,
        on=key_columns,
        how="outer",
        indicator=True
    )

    def status(row):
        if row["_merge"] == "left_only":
            return "MISSING_IN_DATABRICKS"

        if row["_merge"] == "right_only":
            return "MISSING_IN_POSTGRES"

        if (
            row["POSTGRES_ROW_HASH"]
            == row["DATABRICKS_ROW_HASH"]
        ):
            return "MATCH"

        return "HASH_MISMATCH"

    merged["RECON_STATUS"] = merged.apply(
        status,
        axis=1
    )

    merged = merged.drop(columns=["_merge"])

    clean = merged.astype(object).where(
        pd.notnull(merged),
        None
    )

    return spark.createDataFrame(clean)




# ============================================================
# DUPLICATE CHECK
# ============================================================

def validate_primary_key_columns(
    postgres_columns,
    databricks_columns
):
    """
    Validate the supplied UNIQUE_KEY_COLUMNS against both systems.
    """
    key_columns = parse_column_list(
        UNIQUE_KEY_COLUMNS
    )

    if not key_columns:
        raise ValueError(
            "UNIQUE_KEY_COLUMNS must contain at least one "
            "primary-key column for Duplicate_Check."
        )

    missing_pg = [
        column
        for column in key_columns
        if column not in postgres_columns
    ]

    missing_dbx = [
        column
        for column in key_columns
        if column not in databricks_columns
    ]

    if missing_pg:
        raise ValueError(
            "Primary-key columns missing in PostgreSQL: "
            + ", ".join(missing_pg)
        )

    if missing_dbx:
        raise ValueError(
            "Primary-key columns missing in Databricks: "
            + ", ".join(missing_dbx)
        )

    return key_columns


def postgres_duplicate_check(
    conn
):
    """
    Return duplicate primary-key combinations from PostgreSQL.

    Output:
        key columns + duplicate_count
    """
    key_columns = parse_column_list(
        UNIQUE_KEY_COLUMNS
    )

    postgres_columns = (
        get_postgres_columns(
            conn
        )["column_name"]
        .tolist()
    )

    missing_keys = [
        column
        for column in key_columns
        if column not in postgres_columns
    ]

    if missing_keys:
        raise ValueError(
            "PostgreSQL primary-key columns not found: "
            + ", ".join(missing_keys)
        )

    key_sql = sql.SQL(", ").join(
        [
            sql.Identifier(column)
            for column in key_columns
        ]
    )

    query = sql.SQL("""
        SELECT
            {},
            COUNT(*) AS duplicate_count
        FROM {}.{}
        GROUP BY {}
        HAVING COUNT(*) > 1
        ORDER BY duplicate_count DESC
    """).format(
        key_sql,
        sql.Identifier(
            POSTGRES_SCHEMA
        ),
        sql.Identifier(
            POSTGRES_TABLE
        ),
        key_sql
    )

    return pd.read_sql_query(
        query.as_string(conn),
        conn
    )


def databricks_duplicate_check():
    """
    Return duplicate primary-key combinations from Databricks.

    Output:
        key columns + duplicate_count
    """
    key_columns = parse_column_list(
        UNIQUE_KEY_COLUMNS
    )

    target_df = get_databricks_dataframe()

    missing_keys = [
        column
        for column in key_columns
        if column not in target_df.columns
    ]

    if missing_keys:
        raise ValueError(
            "Databricks primary-key columns not found: "
            + ", ".join(missing_keys)
        )

    return (
        target_df
        .groupBy(
            *[
                F.col(column)
                for column in key_columns
            ]
        )
        .count()
        .filter(
            F.col("count") > 1
        )
        .withColumnRenamed(
            "count",
            "duplicate_count"
        )
        .orderBy(
            F.col(
                "duplicate_count"
            ).desc()
        )
    )


def reconcile_duplicates(
    postgres_duplicate_df,
    databricks_duplicate_df
):
    """
    Reconcile duplicate key combinations between PostgreSQL
    and Databricks.

    RECON_STATUS:
        MATCH
        COUNT_MISMATCH
        ONLY_IN_POSTGRES
        ONLY_IN_DATABRICKS
    """
    key_columns = parse_column_list(
        UNIQUE_KEY_COLUMNS
    )

    pg = postgres_duplicate_df.copy()

    if pg.empty:
        pg = pd.DataFrame(
            columns=(
                key_columns
                + [
                    "duplicate_count"
                ]
            )
        )

    pg = pg.rename(
        columns={
            "duplicate_count":
            "POSTGRES_DUPLICATE_COUNT"
        }
    )

    dbx = spark_dataframe_to_pandas_safe(
        databricks_duplicate_df
    )

    if dbx.empty:
        dbx = pd.DataFrame(
            columns=(
                key_columns
                + [
                    "duplicate_count"
                ]
            )
        )

    dbx = dbx.rename(
        columns={
            "duplicate_count":
            "DATABRICKS_DUPLICATE_COUNT"
        }
    )

    merged = pd.merge(
        pg,
        dbx,
        on=key_columns,
        how="outer",
        indicator=True
    )

    def duplicate_status(row):
        if row["_merge"] == "left_only":
            return "ONLY_IN_POSTGRES"

        if row["_merge"] == "right_only":
            return "ONLY_IN_DATABRICKS"

        pg_count = int(
            row[
                "POSTGRES_DUPLICATE_COUNT"
            ]
        )

        dbx_count = int(
            row[
                "DATABRICKS_DUPLICATE_COUNT"
            ]
        )

        if pg_count == dbx_count:
            return "MATCH"

        return "COUNT_MISMATCH"

    if merged.empty:
        merged = pd.DataFrame(
            columns=(
                key_columns
                + [
                    "POSTGRES_DUPLICATE_COUNT",
                    "DATABRICKS_DUPLICATE_COUNT",
                    "RECON_STATUS"
                ]
            )
        )

        return merged

    merged[
        "RECON_STATUS"
    ] = merged.apply(
        duplicate_status,
        axis=1
    )

    merged = merged.drop(
        columns=[
            "_merge"
        ]
    )

    return merged


def build_duplicate_summary(
    postgres_duplicate_df,
    databricks_duplicate_df,
    duplicate_reconciliation_df
):
    """
    Create a compact duplicate reconciliation summary.
    """

    pg_duplicate_keys = len(
        postgres_duplicate_df
    )

    dbx_duplicate_keys = (
        databricks_duplicate_df
        .count()
    )

    pg_duplicate_rows = (
        int(
            postgres_duplicate_df[
                "duplicate_count"
            ].sum()
        )
        if not postgres_duplicate_df.empty
        else 0
    )

    dbx_duplicate_rows = (
        databricks_duplicate_df
        .agg(
            F.coalesce(
                F.sum(
                    "duplicate_count"
                ),
                F.lit(0)
            ).alias(
                "duplicate_rows"
            )
        )
        .collect()[0][
            "duplicate_rows"
        ]
    )

    if duplicate_reconciliation_df.empty:
        mismatch_keys = 0
        matched_keys = 0
    else:
        mismatch_keys = int(
            (
                duplicate_reconciliation_df[
                    "RECON_STATUS"
                ]
                != "MATCH"
            ).sum()
        )

        matched_keys = int(
            (
                duplicate_reconciliation_df[
                    "RECON_STATUS"
                ]
                == "MATCH"
            ).sum()
        )

    overall_status = (
        "MATCH"
        if mismatch_keys == 0
        and pg_duplicate_keys
        == dbx_duplicate_keys
        and pg_duplicate_rows
        == int(dbx_duplicate_rows)
        else "MISMATCH"
    )

    return pd.DataFrame(
        [
            {
                "postgres_duplicate_key_count":
                    pg_duplicate_keys,
                "databricks_duplicate_key_count":
                    int(dbx_duplicate_keys),
                "postgres_duplicate_row_count":
                    pg_duplicate_rows,
                "databricks_duplicate_row_count":
                    int(dbx_duplicate_rows),
                "matched_duplicate_keys":
                    matched_keys,
                "mismatched_duplicate_keys":
                    mismatch_keys,
                "status":
                    overall_status
            }
        ]
    )



# ============================================================
# SEMANTIC / TOLERANT MISMATCH CLASSIFICATION
# ============================================================

def _safe_decimal(value):
    """
    Convert numeric-looking values to Decimal without losing textual
    decimal precision. Returns None if conversion is not possible.
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        if isinstance(value, Decimal):
            return value

        if isinstance(value, bool):
            return None

        text = str(value).strip()

        if text == "":
            return None

        return Decimal(text)

    except Exception:
        return None


def _parse_datetime_relaxed(value):
    """
    Parse a datetime-like value without forcing pandas timestamp[ns].

    Returns a tuple:
        (year, month, day, hour, minute, second, microsecond, tz_text)

    This uses Python datetime parsing where possible and falls back to
    string decomposition for extreme years such as 9999.
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    # Datetime/date objects
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        try:
            return (
                int(value.year),
                int(value.month),
                int(value.day),
                int(getattr(value, "hour", 0)),
                int(getattr(value, "minute", 0)),
                int(getattr(value, "second", 0)),
                int(getattr(value, "microsecond", 0)),
                str(getattr(value, "tzinfo", "") or "")
            )
        except Exception:
            pass

    text = str(value).strip()

    if not text:
        return None

    # Normalize common separators.
    text = text.replace("T", " ")

    # Remove timezone suffix for structural comparison.
    # Keep only the date/time core.
    core = text

    if core.endswith("Z"):
        core = core[:-1]

    # Split timezone offsets if present after time portion.
    if " " in core:
        date_part, time_part = core.split(" ", 1)
    else:
        date_part = core
        time_part = ""

    # Remove UTC offset suffixes from time part.
    # Examples: 10:20:30+00:00 / 10:20:30-05:30
    tz_pos = None
    for idx in range(1, len(time_part)):
        if time_part[idx] in ("+", "-"):
            tz_pos = idx
            break

    if tz_pos is not None:
        time_part = time_part[:tz_pos]

    try:
        y, m, d = [int(x) for x in date_part.split("-")[:3]]
    except Exception:
        return None

    hh = mm = ss = micros = 0

    if time_part:
        pieces = time_part.split(":")

        try:
            if len(pieces) >= 1:
                hh = int(pieces[0])
            if len(pieces) >= 2:
                mm = int(pieces[1])
            if len(pieces) >= 3:
                sec_text = pieces[2]
                if "." in sec_text:
                    sec_main, frac = sec_text.split(".", 1)
                    ss = int(sec_main)
                    frac_digits = "".join(
                        ch for ch in frac
                        if ch.isdigit()
                    )
                    micros = int(
                        (frac_digits + "000000")[:6]
                    ) if frac_digits else 0
                else:
                    ss = int(sec_text)
        except Exception:
            return None

    return (
        y, m, d, hh, mm, ss, micros, ""
    )


def classify_semantic_mismatch(
    postgres_value,
    databricks_value,
    postgres_type_name=None,
    databricks_type_name=None
):
    """
    Return:
        (classification, reason)

    Classification:
        EXACT_MATCH
        FALSE_MISMATCH
        TRUE_MISMATCH

    False mismatch rules:
      1. Datetime/timestamp values are considered equivalent when
         year/month/day/hour/minute match and the only difference
         is seconds/microseconds/timezone representation.
      2. Numeric/decimal values are considered equivalent when their
         numeric value is the same after removing representation/
         precision differences (e.g. 10.0 vs 10.0000).
      3. String-equivalent values after trimming are also treated as
         false mismatches only when their normalized values are equal.
    """

    pg_norm = normalize_python_value(
        postgres_value
    )

    dbx_norm = normalize_python_value(
        databricks_value
    )

    if pg_norm == dbx_norm:
        return (
            "EXACT_MATCH",
            "Values match exactly after standard normalization"
        )

    pg_type = (
        str(postgres_type_name or "")
        .lower()
    )

    dbx_type = (
        str(databricks_type_name or "")
        .lower()
    )

    type_text = (
        pg_type
        + " "
        + dbx_type
    )

    # --------------------------------------------------------
    # Datetime tolerance:
    # Ignore differences in seconds, microseconds and timezone
    # representation if date/hour/minute are otherwise equal.
    # --------------------------------------------------------

    looks_datetime = any(
        token in type_text
        for token in [
            "timestamp",
            "datetime",
            "date"
        ]
    )

    if looks_datetime:
        pg_dt = _parse_datetime_relaxed(
            postgres_value
        )

        dbx_dt = _parse_datetime_relaxed(
            databricks_value
        )

        if (
            pg_dt is not None
            and dbx_dt is not None
        ):
            pg_to_minute = pg_dt[:5]
            dbx_to_minute = dbx_dt[:5]

            if pg_to_minute == dbx_to_minute:
                return (
                    "FALSE_MISMATCH",
                    "Datetime differs only at seconds/microseconds/timezone representation"
                )

    # --------------------------------------------------------
    # Numeric / decimal tolerance:
    # Ignore representation/scale/precision differences when
    # actual numeric value is identical.
    # --------------------------------------------------------

    looks_numeric = any(
        token in type_text
        for token in [
            "decimal",
            "numeric",
            "number",
            "double",
            "float",
            "real",
            "int",
            "long",
            "short",
            "bigint",
            "smallint"
        ]
    )

    if looks_numeric:
        pg_num = _safe_decimal(
            postgres_value
        )

        dbx_num = _safe_decimal(
            databricks_value
        )

        if (
            pg_num is not None
            and dbx_num is not None
            and pg_num == dbx_num
        ):
            return (
                "FALSE_MISMATCH",
                "Numeric values are equal; difference is only precision/scale/representation"
            )

    return (
        "TRUE_MISMATCH",
        "Values remain different after semantic tolerance rules"
    )


def get_postgres_type_map(conn):
    """
    Map PostgreSQL column name -> PostgreSQL type.
    """
    meta = get_postgres_columns(
        conn
    )

    return {
        row["column_name"]:
            (
                row["udt_name"]
                if pd.notna(row["udt_name"])
                else row["data_type"]
            )
        for _, row in meta.iterrows()
    }


def get_databricks_type_map():
    """
    Map Databricks column name -> Spark simple type.
    """
    table_df = get_databricks_dataframe()

    return {
        field.name:
            field.dataType.simpleString()
        for field in table_df.schema.fields
    }


# ============================================================
# COLUMN-LEVEL HASH MISMATCH ANALYSIS
# ============================================================

def values_equal_for_reconciliation(left_value, right_value):
    """
    Compare PostgreSQL and Databricks values using the same
    normalization approach used for row-hash reconciliation.
    """
    return (
        normalize_python_value(left_value)
        == normalize_python_value(right_value)
    )


def build_column_level_mismatches(
    postgres_sample_df,
    databricks_sample_df,
    sample_reconciliation_df,
    common_hash_columns,
    postgres_type_map=None,
    databricks_type_map=None
):
    """
    For rows classified as HASH_MISMATCH:
      - identify exactly which hash-participating columns differ,
      - classify each difference as TRUE_MISMATCH or FALSE_MISMATCH,
      - include IGNORE_COLUMNS as contextual source/target values,
      - derive a row-level mismatch classification.

    Ignored columns are included in the final result for visibility,
    but do NOT influence hash reconciliation or true/false mismatch
    classification.
    """

    postgres_type_map = (
        postgres_type_map
        or {}
    )

    databricks_type_map = (
        databricks_type_map
        or {}
    )

    key_columns = parse_column_list(
        UNIQUE_KEY_COLUMNS
    )

    ignore_columns = parse_column_list(
        IGNORE_COLUMNS
    )

    # Keep only ignored columns that exist in both samples.
    pg_columns = list(
        postgres_sample_df.columns
    )

    dbx_columns = list(
        databricks_sample_df.columns
    )

    common_ignore_columns = [
        column
        for column in ignore_columns
        if (
            column in pg_columns
            and column in dbx_columns
        )
    ]

    # Do not select the same column twice.
    compare_columns = unique_preserve_order(
        key_columns
        + common_hash_columns
        + common_ignore_columns
    )

    base_output_columns = (
        key_columns
        + [
            "mismatch_column",
            "postgres_value",
            "databricks_value",
            "postgres_data_type",
            "databricks_data_type",
            "COLUMN_MISMATCH_CLASS",
            "MISMATCH_REASON",
            "ROW_MISMATCH_CLASS"
        ]
    )

    # Add ignored columns as context in wide form.
    ignored_context_columns = []

    for column in common_ignore_columns:
        ignored_context_columns.extend(
            [
                f"POSTGRES_IGNORE_{column}",
                f"DATABRICKS_IGNORE_{column}"
            ]
        )

    output_columns = (
        base_output_columns
        + ignored_context_columns
    )

    mismatch_keys_pdf = spark_dataframe_to_pandas_safe(
        sample_reconciliation_df
        .filter(
            F.col("RECON_STATUS")
            == F.lit("HASH_MISMATCH")
        ),
        key_columns
    )

    if mismatch_keys_pdf.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    pg_compare = postgres_sample_df[
        compare_columns
    ].copy()

    dbx_compare = spark_dataframe_to_pandas_safe(
        databricks_sample_df,
        compare_columns
    )

    # Restrict to rows already known to have row-hash mismatch.
    pg_compare = pd.merge(
        mismatch_keys_pdf,
        pg_compare,
        on=key_columns,
        how="inner"
    )

    dbx_compare = pd.merge(
        mismatch_keys_pdf,
        dbx_compare,
        on=key_columns,
        how="inner"
    )

    merged = pd.merge(
        pg_compare,
        dbx_compare,
        on=key_columns,
        how="inner",
        suffixes=(
            "_POSTGRES",
            "_DATABRICKS"
        )
    )

    mismatch_rows = []

    # Keys are used to match rows and should not appear as
    # content mismatch columns.
    non_key_hash_columns = [
        column
        for column in common_hash_columns
        if column not in key_columns
    ]

    # First pass: create column-level mismatch rows.
    for _, row in merged.iterrows():

        key_values = {
            key: row[key]
            for key in key_columns
        }

        row_details = []

        ignored_context = {}

        for ignore_column in common_ignore_columns:
            pg_ignore_name = (
                ignore_column
                + "_POSTGRES"
                if ignore_column not in key_columns
                else ignore_column
            )

            dbx_ignore_name = (
                ignore_column
                + "_DATABRICKS"
                if ignore_column not in key_columns
                else ignore_column
            )

            pg_ignore_value = (
                row[pg_ignore_name]
                if pg_ignore_name in merged.columns
                else None
            )

            dbx_ignore_value = (
                row[dbx_ignore_name]
                if dbx_ignore_name in merged.columns
                else None
            )

            ignored_context[
                f"POSTGRES_IGNORE_{ignore_column}"
            ] = normalize_python_value(
                pg_ignore_value
            )

            ignored_context[
                f"DATABRICKS_IGNORE_{ignore_column}"
            ] = normalize_python_value(
                dbx_ignore_value
            )

        for column in non_key_hash_columns:

            pg_column = (
                column
                + "_POSTGRES"
            )

            dbx_column = (
                column
                + "_DATABRICKS"
            )

            if (
                pg_column not in merged.columns
                or dbx_column not in merged.columns
            ):
                continue

            pg_value = row[
                pg_column
            ]

            dbx_value = row[
                dbx_column
            ]

            if values_equal_for_reconciliation(
                pg_value,
                dbx_value
            ):
                continue

            classification, reason = (
                classify_semantic_mismatch(
                    postgres_value=pg_value,
                    databricks_value=dbx_value,
                    postgres_type_name=(
                        postgres_type_map.get(
                            column
                        )
                    ),
                    databricks_type_name=(
                        databricks_type_map.get(
                            column
                        )
                    )
                )
            )

            # Skip exact matches defensively.
            if classification == "EXACT_MATCH":
                continue

            detail = dict(
                key_values
            )

            detail[
                "mismatch_column"
            ] = column

            detail[
                "postgres_value"
            ] = normalize_python_value(
                pg_value
            )

            detail[
                "databricks_value"
            ] = normalize_python_value(
                dbx_value
            )

            detail[
                "postgres_data_type"
            ] = postgres_type_map.get(
                column
            )

            detail[
                "databricks_data_type"
            ] = databricks_type_map.get(
                column
            )

            detail[
                "COLUMN_MISMATCH_CLASS"
            ] = classification

            detail[
                "MISMATCH_REASON"
            ] = reason

            detail.update(
                ignored_context
            )

            row_details.append(
                detail
            )

        # ----------------------------------------------------
        # Row-level classification
        #
        # FALSE_MISMATCH:
        #   every differing column in the row is semantically
        #   equivalent under tolerance rules.
        #
        # TRUE_MISMATCH:
        #   at least one differing column remains genuinely
        #   different after tolerance rules.
        # ----------------------------------------------------

        if row_details:
            row_class = (
                "TRUE_MISMATCH"
                if any(
                    detail[
                        "COLUMN_MISMATCH_CLASS"
                    ] == "TRUE_MISMATCH"
                    for detail in row_details
                )
                else "FALSE_MISMATCH"
            )

            for detail in row_details:
                detail[
                    "ROW_MISMATCH_CLASS"
                ] = row_class

                mismatch_rows.append(
                    detail
                )

    return pd.DataFrame(
        mismatch_rows,
        columns=output_columns
    )


def build_column_mismatch_summary(
    column_mismatch_df
):
    """
    Summarize true/false mismatches by column.
    """

    if column_mismatch_df.empty:
        return pd.DataFrame(
            columns=[
                "mismatch_column",
                "true_mismatch_count",
                "false_mismatch_count",
                "total_mismatch_count"
            ]
        )

    summary = (
        column_mismatch_df
        .groupby(
            [
                "mismatch_column",
                "COLUMN_MISMATCH_CLASS"
            ],
            dropna=False
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    pivot = (
        summary
        .pivot_table(
            index="mismatch_column",
            columns="COLUMN_MISMATCH_CLASS",
            values="count",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
    )

    if "TRUE_MISMATCH" not in pivot.columns:
        pivot["TRUE_MISMATCH"] = 0

    if "FALSE_MISMATCH" not in pivot.columns:
        pivot["FALSE_MISMATCH"] = 0

    pivot = pivot.rename(
        columns={
            "TRUE_MISMATCH":
                "true_mismatch_count",
            "FALSE_MISMATCH":
                "false_mismatch_count"
        }
    )

    pivot[
        "total_mismatch_count"
    ] = (
        pivot[
            "true_mismatch_count"
        ]
        + pivot[
            "false_mismatch_count"
        ]
    )

    return (
        pivot[
            [
                "mismatch_column",
                "true_mismatch_count",
                "false_mismatch_count",
                "total_mismatch_count"
            ]
        ]
        .sort_values(
            [
                "true_mismatch_count",
                "false_mismatch_count"
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


def build_row_mismatch_summary(
    column_mismatch_df
):
    """
    Return one row per mismatched business key with its final
    TRUE_MISMATCH / FALSE_MISMATCH classification.
    """

    key_columns = parse_column_list(
        UNIQUE_KEY_COLUMNS
    )

    if column_mismatch_df.empty:
        return pd.DataFrame(
            columns=(
                key_columns
                + [
                    "ROW_MISMATCH_CLASS"
                ]
            )
        )

    return (
        column_mismatch_df[
            key_columns
            + [
                "ROW_MISMATCH_CLASS"
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# MAIN
# ============================================================

def run_reconciliation():
    valid_flags = {
        "counts_check",
        "type_check",
        "sample_check",
        "duplicate_check",
        "all_check"
    }

    check_flag = CHECK_FLAG.strip().lower()

    if check_flag not in valid_flags:
        raise ValueError(
            f"Invalid CHECK_FLAG: {CHECK_FLAG}. "
            "Allowed values: Counts_Check, Type_Check, "
            "Sample_Check, Duplicate_Check, All_Check"
        )

    print(
        "\nPostgreSQL <-> Databricks "
        "In-Memory Reconciliation Utility"
    )

    print(
        f"PostgreSQL : "
        f"{POSTGRES_CONFIG['database']}."
        f"{POSTGRES_SCHEMA}."
        f"{POSTGRES_TABLE}"
    )

    print(
        f"Databricks : "
        f"{DATABRICKS_CATALOG}."
        f"{DATABRICKS_SCHEMA}."
        f"{DATABRICKS_TABLE}"
    )

    print(
        f"Check      : {CHECK_FLAG}"
    )

    if check_flag == "all_check":
        print(
            "All_Check  : Counts_Check + Type_Check + Sample_Check "
            "(Duplicate_Check is excluded unless explicitly selected)"
        )

    validate_databricks_table()

    conn = None
    results = {}

    try:
        conn = get_postgres_connection()
        validate_postgres_table(conn)

        # ====================================================
        # COUNTS CHECK
        # ====================================================

        if check_flag in {
            "counts_check",
            "all_check"
        }:
            print("\n" + "=" * 80)
            print("COUNTS CHECK")
            print("=" * 80)

            pg_counts = postgres_counts_check(conn)
            dbx_counts = databricks_counts_check()
            recon_counts = reconcile_counts(
                pg_counts,
                dbx_counts
            )

            print("\nPostgreSQL Count:")
            display(
                pandas_to_spark_safe(pg_counts)
            )

            print("\nDatabricks Count:")
            display(dbx_counts)

            print("\nCount Reconciliation:")
            display(recon_counts)

            results["Counts_Check"] = {
                "postgres": pg_counts,
                "databricks": dbx_counts,
                "reconciliation": recon_counts
            }

        # ====================================================
        # TYPE CHECK
        # ====================================================

        if check_flag in {
            "type_check",
            "all_check"
        }:
            print("\n" + "=" * 80)
            print("TYPE CHECK")
            print("=" * 80)

            pg_types = postgres_type_check(conn)
            dbx_types = databricks_type_check()
            recon_types = reconcile_types(
                pg_types,
                dbx_types
            )

            print("\nPostgreSQL Types:")
            display(
                pandas_to_spark_safe(pg_types)
            )

            print("\nDatabricks Types:")
            display(dbx_types)

            print("\nType Reconciliation:")
            display(recon_types)

            results["Type_Check"] = {
                "postgres": pg_types,
                "databricks": dbx_types,
                "reconciliation": recon_types
            }

        # ====================================================
        # SAMPLE CHECK
        # ====================================================

        if check_flag in {
            "sample_check",
            "all_check"
        }:
            print("\n" + "=" * 80)
            print("SAMPLE CHECK")
            print("=" * 80)

            pg_sample_raw, pg_hash_columns = (
                postgres_sample_check(conn)
            )

            (
                dbx_sample,
                common_hash_columns,
                source_only_hash_columns
            ) = databricks_sample_for_keys(
                pg_sample_raw,
                pg_hash_columns
            )

            pg_sample = (
                postgres_rehash_with_common_columns(
                    pg_sample_raw,
                    common_hash_columns
                )
            )

            sample_recon = reconcile_samples(
                pg_sample,
                dbx_sample
            )

            sample_summary = (
                sample_recon
                .groupBy("RECON_STATUS")
                .count()
                .orderBy("RECON_STATUS")
            )

            postgres_type_map = (
                get_postgres_type_map(
                    conn
                )
            )

            databricks_type_map = (
                get_databricks_type_map()
            )

            column_mismatch_df = (
                build_column_level_mismatches(
                    pg_sample,
                    dbx_sample,
                    sample_recon,
                    common_hash_columns,
                    postgres_type_map,
                    databricks_type_map
                )
            )

            column_mismatch_summary_df = (
                build_column_mismatch_summary(
                    column_mismatch_df
                )
            )

            row_mismatch_summary_df = (
                build_row_mismatch_summary(
                    column_mismatch_df
                )
            )

            print(
                "\nHash columns used across both systems:"
            )
            print(common_hash_columns)

            if source_only_hash_columns:
                print(
                    "\nPostgreSQL columns excluded from "
                    "cross-system hash because they do not "
                    "exist in Databricks:"
                )
                print(source_only_hash_columns)

            extreme_timestamp_examples = []
            for column in pg_sample.columns:
                for value in pg_sample[column].head(100).tolist():
                    if is_datetime_like_value(value):
                        try:
                            year_value = value.year
                        except Exception:
                            continue
                        if year_value < 1678 or year_value > 2262:
                            extreme_timestamp_examples.append({
                                "column": column,
                                "value": safe_scalar_for_dataframe(value)
                            })
                            if len(extreme_timestamp_examples) >= 10:
                                break
                if len(extreme_timestamp_examples) >= 10:
                    break

            if extreme_timestamp_examples:
                print(
                    "\nExtreme timestamp values detected. "
                    "They are treated as strings to avoid pandas timestamp overflow:"
                )
                display(
                    pandas_to_spark_safe(
                        pd.DataFrame(extreme_timestamp_examples)
                    )
                )

            print("\nPostgreSQL Sample:")
            display(
                pandas_to_spark_safe(pg_sample)
            )

            print("\nDatabricks Sample:")
            display(dbx_sample)

            print("\nSample Reconciliation:")
            display(sample_recon)

            print("\nSample Reconciliation Summary:")
            display(sample_summary)

            results["Sample_Check"] = {
                "postgres": pg_sample,
                "databricks": dbx_sample,
                "reconciliation": sample_recon,
                "summary": sample_summary,
                "column_mismatches": column_mismatch_df,
                "column_mismatch_summary": (
                    column_mismatch_summary_df
                ),
                "row_mismatch_summary": (
                    row_mismatch_summary_df
                ),
                "hash_columns": common_hash_columns,
                "excluded_source_columns": (
                    source_only_hash_columns
                )
            }

        # ====================================================
        # DUPLICATE CHECK
        # ====================================================

        if check_flag == "duplicate_check":
            print("\n" + "=" * 80)
            print("DUPLICATE CHECK")
            print("=" * 80)

            key_columns = parse_column_list(
                UNIQUE_KEY_COLUMNS
            )

            pg_columns = (
                get_postgres_columns(
                    conn
                )["column_name"]
                .tolist()
            )

            dbx_columns = (
                get_databricks_dataframe()
                .columns
            )

            validate_primary_key_columns(
                pg_columns,
                dbx_columns
            )

            pg_duplicates = (
                postgres_duplicate_check(
                    conn
                )
            )

            dbx_duplicates = (
                databricks_duplicate_check()
            )

            duplicate_recon = (
                reconcile_duplicates(
                    pg_duplicates,
                    dbx_duplicates
                )
            )

            duplicate_summary = (
                build_duplicate_summary(
                    pg_duplicates,
                    dbx_duplicates,
                    duplicate_recon
                )
            )

            print(
                "\nPrimary Key Columns:"
            )
            print(
                key_columns
            )

            print(
                "\nPostgreSQL Duplicate Keys:"
            )

            if pg_duplicates.empty:
                print(
                    "No duplicate primary-key "
                    "combinations found in PostgreSQL."
                )
            else:
                display(
                    pandas_to_spark_safe(
                        pg_duplicates
                    )
                )

            print(
                "\nDatabricks Duplicate Keys:"
            )

            if dbx_duplicates.limit(1).count() == 0:
                print(
                    "No duplicate primary-key "
                    "combinations found in Databricks."
                )
            else:
                display(
                    dbx_duplicates
                )

            print(
                "\nDuplicate Reconciliation:"
            )

            if duplicate_recon.empty:
                print(
                    "No duplicate primary-key "
                    "combinations found in either system."
                )
            else:
                display(
                    pandas_to_spark_safe(
                        duplicate_recon
                    )
                )

            print(
                "\nDuplicate Reconciliation Summary:"
            )

            display(
                pandas_to_spark_safe(
                    duplicate_summary
                )
            )

            results[
                "Duplicate_Check"
            ] = {
                "postgres":
                    pg_duplicates,
                "databricks":
                    dbx_duplicates,
                "reconciliation":
                    duplicate_recon,
                "summary":
                    duplicate_summary,
                "primary_key_columns":
                    key_columns
            }

        # ====================================================
        # FINAL HASH MISMATCH DETAIL SECTION
        # ====================================================

        if (
            "Sample_Check" in results
            and check_flag in {
                "sample_check",
                "all_check"
            }
        ):
            print("\n" + "=" * 80)
            print("HASH ROW MISMATCH - COLUMN LEVEL DETAILS")
            print("=" * 80)

            final_column_mismatches = (
                results[
                    "Sample_Check"
                ][
                    "column_mismatches"
                ]
            )

            final_column_summary = (
                results[
                    "Sample_Check"
                ][
                    "column_mismatch_summary"
                ]
            )

            if final_column_mismatches.empty:
                print(
                    "\nNo HASH_MISMATCH rows found. "
                    "No column-level mismatches to display."
                )

            else:
                print(
                    "\nMismatch Summary by Column:"
                )

                display(
                    pandas_to_spark_safe(
                        final_column_summary
                    )
                )

                print(
                    "\nDetailed Mismatched Columns "
                    "(with TRUE/FALSE mismatch classification):"
                )

                display(
                    pandas_to_spark_safe(
                        final_column_mismatches
                    )
                )

                print(
                    "\nRow-Level TRUE/FALSE Mismatch Summary:"
                )

                display(
                    pandas_to_spark_safe(
                        results[
                            "Sample_Check"
                        ][
                            "row_mismatch_summary"
                        ]
                    )
                )

                print(
                    "\nTotal mismatched column values:",
                    len(
                        final_column_mismatches
                    )
                )

                true_count = int(
                    (
                        final_column_mismatches[
                            "COLUMN_MISMATCH_CLASS"
                        ]
                        == "TRUE_MISMATCH"
                    ).sum()
                )

                false_count = int(
                    (
                        final_column_mismatches[
                            "COLUMN_MISMATCH_CLASS"
                        ]
                        == "FALSE_MISMATCH"
                    ).sum()
                )

                print(
                    "True mismatched column values:",
                    true_count
                )

                print(
                    "False mismatched column values:",
                    false_count
                )

        print("\n" + "=" * 80)
        print("RECONCILIATION COMPLETED")
        print("=" * 80)

        return results

    finally:
        if conn is not None:
            conn.close()
            print("\nPostgreSQL connection closed.")


# ============================================================
# EXECUTION
# ============================================================

reconciliation_results = run_reconciliation()


# ============================================================
# EXAMPLES OF ACCESSING RESULTS IN THE NOTEBOOK
# ============================================================
#
# Count reconciliation:
# display(
#     reconciliation_results[
#         "Counts_Check"
#     ]["reconciliation"]
# )
#
# Type mismatches only:
# display(
#     reconciliation_results[
#         "Type_Check"
#     ]["reconciliation"]
#     .filter(F.col("status") != "MATCH")
# )
#
# Sample mismatches only:
# display(
#     reconciliation_results[
#         "Sample_Check"
#     ]["reconciliation"]
#     .filter(F.col("RECON_STATUS") != "MATCH")
# )
#
# Sample summary:
# display(
#     reconciliation_results[
#         "Sample_Check"
#     ]["summary"]
# )
#
# Column-level hash mismatches:
# display(
#     pandas_to_spark_safe(
#         reconciliation_results[
#             "Sample_Check"
#         ]["column_mismatches"]
#     )
# )
#
# Mismatch count by column:
# display(
#     pandas_to_spark_safe(
#         reconciliation_results[
#             "Sample_Check"
#         ]["column_mismatch_summary"]
#     )
# )

#
# Duplicate reconciliation:
# display(
#     pandas_to_spark_safe(
#         reconciliation_results[
#             "Duplicate_Check"
#         ]["reconciliation"]
#     )
# )
#
# Duplicate reconciliation summary:
# display(
#     pandas_to_spark_safe(
#         reconciliation_results[
#             "Duplicate_Check"
#         ]["summary"]
#     )
# )

#
# Row-level true/false mismatch classification:
# display(
#     pandas_to_spark_safe(
#         reconciliation_results[
#             "Sample_Check"
#         ]["row_mismatch_summary"]
#     )
# )
