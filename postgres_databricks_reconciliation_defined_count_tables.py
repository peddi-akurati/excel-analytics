"""
PostgreSQL <-> Databricks In-Memory Reconciliation Utility
==========================================================

Supported checks:
    Counts_Check
    Type_Check
    Type_Check_All
    Sample_Check
    Duplicate_Check
    All_Check

All_Check runs Counts_Check + Type_Check + Sample_Check.
Duplicate_Check remains available only when explicitly selected.

All outputs remain available in DataFrames in memory.
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
    - Sample_Check supports separate custom WHERE clauses for PostgreSQL and Databricks.
    - PostgreSQL sampling then selects the latest N rows using DATE_COLUMN.
    - The sampled unique keys are used to retrieve the corresponding
      Databricks rows.
    - SHA-256 row hashes are calculated on columns common to both systems,
      excluding IGNORE_COLUMNS.
    - Reconciliation is performed using UNIQUE_KEY_COLUMNS.
"""

import hashlib
import json
from decimal import Decimal
from datetime import datetime, date, timezone

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

# Reconciliation audit storage location.
# All check outputs will be appended to Delta tables here.

# Optional table-name prefix for audit tables.

# Raw Sample_Check / Duplicate_Check audit outputs use table-specific
# Delta table names automatically. This prevents schema collisions when
# the same column name (for example row_wid) has different datatypes
# in different business tables.

# True  -> truncate all reconciliation audit tables matching the prefix
#          before the reconciliation starts.
# False -> preserve history and append the current run.

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

# Counts_Check table scope.
# Only these tables will be checked by Counts_Check / All_Check.
# Can be a comma-separated string or a Python list.
#
# Examples:
# COUNTS_TABLE_LIST = "claims,policies,members"
# COUNTS_TABLE_LIST = ["claims", "policies", "members"]
COUNTS_TABLE_LIST = "your_table"

# Optional custom WHERE clauses for Sample_Check.
# Provide only the filter condition, WITHOUT the WHERE keyword.
#
# Examples:
# POSTGRES_SAMPLE_WHERE = "intimation_date >= DATE '2026-07-01'"
# DATABRICKS_SAMPLE_WHERE = "intimation_date >= DATE('2026-07-01')"
#
# Leave blank for no additional filtering.
POSTGRES_SAMPLE_WHERE = ""
DATABRICKS_SAMPLE_WHERE = ""



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
    """
    Convert a pandas DataFrame to Spark safely.

    Handles mixed Python numeric types such as:
        float + Decimal
        int + Decimal
        numpy numeric + Decimal

    Spark schema inference cannot merge DoubleType and DecimalType
    inside the same pandas/object column. For display-oriented
    DataFrames, heterogeneous object columns are therefore normalized
    to strings before Spark DataFrame creation.

    This conversion does NOT alter the original reconciliation
    DataFrames or hash calculations.
    """
    if dataframe is None:
        return None

    if dataframe.empty:
        # Preserve column names even when there are no rows.
        empty_schema = T.StructType([
            T.StructField(
                str(column),
                T.StringType(),
                True
            )
            for column in dataframe.columns
        ])

        return spark.createDataFrame(
            [],
            empty_schema
        )

    safe_df = dataframe.copy()

    def is_null(value):
        if value is None:
            return True

        try:
            result = pd.isna(value)

            if isinstance(result, bool):
                return result
        except Exception:
            pass

        return False

    def normalize_display_value(value):
        if is_null(value):
            return None

        # Avoid timestamp overflow.
        if is_datetime_like_value(value):
            try:
                return value.isoformat()
            except Exception:
                return str(value)

        # Decimal is intentionally converted to text for display
        # when mixed with float/int in a pandas object column.
        if isinstance(value, Decimal):
            return format(
                value,
                "f"
            )

        if isinstance(value, (dict, list)):
            return json.dumps(
                value,
                sort_keys=True,
                default=str
            )

        return value

    for column in safe_df.columns:
        series = safe_df[column].map(
            normalize_display_value
        )

        non_null_values = [
            value
            for value in series.tolist()
            if value is not None
        ]

        if not non_null_values:
            safe_df[column] = series.astype(
                object
            )
            continue

        python_types = {
            type(value)
            for value in non_null_values
        }

        has_decimal = any(
            isinstance(value, Decimal)
            for value in non_null_values
        )

        has_float = any(
            isinstance(value, float)
            for value in non_null_values
        )

        has_int = any(
            isinstance(value, int)
            and not isinstance(value, bool)
            for value in non_null_values
        )

        has_string = any(
            isinstance(value, str)
            for value in non_null_values
        )

        # Most important fix:
        # a column mixing Decimal with float/int cannot be inferred
        # by Spark as one numeric type.
        heterogeneous_numeric = (
            has_decimal
            and (
                has_float
                or has_int
            )
        )

        # Also protect general mixed object columns because Spark may
        # infer incompatible schemas across rows.
        generally_mixed = (
            len(python_types) > 1
            and (
                has_string
                or heterogeneous_numeric
            )
        )

        if heterogeneous_numeric or generally_mixed:
            safe_df[column] = series.map(
                lambda value:
                    None
                    if value is None
                    else str(value)
            )
        else:
            safe_df[column] = series

    clean_df = safe_df.astype(
        object
    ).where(
        pd.notnull(safe_df),
        None
    )

    try:
        return spark.createDataFrame(
            clean_df
        )

    except Exception as error:
        error_text = str(error)

        if (
            "cannot merge type DoubleType and DecimalType"
            in error_text
            or "CANNOT_MERGE_TYPE"
            in error_text
            or "Can not merge type"
            in error_text
            or "DoubleType" in error_text
            and "DecimalType" in error_text
        ):
            print(
                "\nSpark schema inference detected incompatible "
                "mixed numeric types. Falling back to string-safe "
                "display conversion."
            )

            print(
                "Columns being converted to string for safe display:",
                list(clean_df.columns)
            )

            fallback_df = clean_df.copy()

            for column in fallback_df.columns:
                fallback_df[column] = fallback_df[
                    column
                ].map(
                    lambda value:
                        None
                        if value is None
                        else str(value)
                )

            return spark.createDataFrame(
                fallback_df
            )

        raise

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
# COUNTS CHECK - COMMON TABLES
# ============================================================

def get_postgres_table_names(conn):
    """
    Return all base-table names from the configured PostgreSQL schema.
    """
    query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(POSTGRES_SCHEMA,)
    )

    return df["table_name"].tolist()


def get_databricks_table_names():
    """
    Return all non-temporary tables from the configured Databricks schema.
    """
    tables_df = spark.sql(
        f"""
        SHOW TABLES IN
        `{DATABRICKS_CATALOG}`.`{DATABRICKS_SCHEMA}`
        """
    )

    return [
        row["tableName"]
        for row in tables_df.collect()
        if not row["isTemporary"]
    ]


def get_requested_count_tables(conn):
    """
    Resolve the user-defined Counts_Check table list.

    Only requested tables are considered. Matching against PostgreSQL
    and Databricks is case-insensitive.

    Returns:
        valid_tables:
            list of (postgres_table_name, databricks_table_name)

        missing_rows:
            list of tuples:
                requested_table,
                postgres_exists,
                databricks_exists,
                status
    """
    requested_tables = parse_column_list(
        COUNTS_TABLE_LIST
    )

    if not requested_tables:
        raise ValueError(
            "COUNTS_TABLE_LIST must contain at least one table "
            "for Counts_Check."
        )

    pg_tables = get_postgres_table_names(
        conn
    )

    dbx_tables = get_databricks_table_names()

    pg_map = {
        table.lower(): table
        for table in pg_tables
    }

    dbx_map = {
        table.lower(): table
        for table in dbx_tables
    }

    valid_tables = []
    missing_rows = []

    for requested_table in requested_tables:
        key = requested_table.lower()

        pg_exists = key in pg_map
        dbx_exists = key in dbx_map

        if pg_exists and dbx_exists:
            valid_tables.append(
                (
                    pg_map[key],
                    dbx_map[key]
                )
            )
            continue

        if not pg_exists and not dbx_exists:
            status = "MISSING_IN_BOTH"
        elif not pg_exists:
            status = "MISSING_IN_POSTGRES"
        else:
            status = "MISSING_IN_DATABRICKS"

        missing_rows.append(
            (
                requested_table,
                pg_exists,
                dbx_exists,
                status
            )
        )

    return (
        valid_tables,
        missing_rows
    )


def postgres_table_count(conn, table_name):
    """
    Count rows in one PostgreSQL table.
    """
    query = sql.SQL("""
        SELECT COUNT(*) AS total_count
        FROM {}.{}
    """).format(
        sql.Identifier(POSTGRES_SCHEMA),
        sql.Identifier(table_name)
    )

    with conn.cursor() as cursor:
        cursor.execute(query)
        return int(cursor.fetchone()[0])


def databricks_table_count(table_name):
    """
    Count rows in one Databricks table.
    """
    full_name = (
        f"{DATABRICKS_CATALOG}."
        f"{DATABRICKS_SCHEMA}."
        f"{table_name}"
    )

    return int(
        spark.table(full_name).count()
    )


def counts_check_common_tables(conn):
    """
    Compare row counts ONLY for tables listed in COUNTS_TABLE_LIST.

    Requested tables missing on either side are retained in the output
    with the appropriate status rather than silently ignored.
    """
    (
        valid_tables,
        missing_rows
    ) = get_requested_count_tables(
        conn
    )

    requested_tables = parse_column_list(
        COUNTS_TABLE_LIST
    )

    print(
        "\nCounts_Check requested tables:",
        requested_tables
    )

    output_schema = T.StructType([
        T.StructField(
            "requested_table",
            T.StringType(),
            True
        ),
        T.StructField(
            "postgres_table",
            T.StringType(),
            True
        ),
        T.StructField(
            "databricks_table",
            T.StringType(),
            True
        ),
        T.StructField(
            "postgres_count",
            T.LongType(),
            True
        ),
        T.StructField(
            "databricks_count",
            T.LongType(),
            True
        ),
        T.StructField(
            "difference",
            T.LongType(),
            True
        ),
        T.StructField(
            "status",
            T.StringType(),
            True
        )
    ])

    rows = []

    total_tables = len(
        valid_tables
    )

    for index, (
        pg_table,
        dbx_table
    ) in enumerate(
        valid_tables,
        start=1
    ):
        print(
            f"Counting requested table "
            f"{index}/{total_tables}: "
            f"{pg_table}"
        )

        pg_count = postgres_table_count(
            conn,
            pg_table
        )

        dbx_count = databricks_table_count(
            dbx_table
        )

        difference = (
            dbx_count
            - pg_count
        )

        rows.append(
            (
                pg_table,
                pg_table,
                dbx_table,
                pg_count,
                dbx_count,
                difference,
                (
                    "MATCH"
                    if difference == 0
                    else "MISMATCH"
                )
            )
        )

    for (
        requested_table,
        pg_exists,
        dbx_exists,
        status
    ) in missing_rows:
        rows.append(
            (
                requested_table,
                (
                    requested_table
                    if pg_exists
                    else None
                ),
                (
                    requested_table
                    if dbx_exists
                    else None
                ),
                None,
                None,
                None,
                status
            )
        )

    return spark.createDataFrame(
        rows,
        schema=output_schema
    )


def counts_check_summary(
    counts_reconciliation_df
):
    """
    Summarize requested-table count reconciliation.
    """
    return (
        counts_reconciliation_df
        .agg(
            F.count(
                F.lit(1)
            ).alias(
                "requested_table_count"
            ),
            F.sum(
                F.when(
                    F.col("status") == F.lit("MATCH"),
                    F.lit(1)
                ).otherwise(
                    F.lit(0)
                )
            ).alias(
                "matched_table_count"
            ),
            F.sum(
                F.when(
                    F.col("status") == F.lit("MISMATCH"),
                    F.lit(1)
                ).otherwise(
                    F.lit(0)
                )
            ).alias(
                "count_mismatch_table_count"
            ),
            F.sum(
                F.when(
                    F.col("status").isin(
                        "MISSING_IN_POSTGRES",
                        "MISSING_IN_DATABRICKS",
                        "MISSING_IN_BOTH"
                    ),
                    F.lit(1)
                ).otherwise(
                    F.lit(0)
                )
            ).alias(
                "missing_table_count"
            )
        )
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

    return pandas_to_spark_safe(clean)



# ============================================================
# TYPE CHECK ALL - ALL TABLES IN BOTH SCHEMAS
# ============================================================

def get_postgres_all_table_types(conn):
    """
    Return column metadata for all BASE TABLES in POSTGRES_SCHEMA.
    """
    query = """
        SELECT
            table_schema,
            table_name,
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
          AND table_name IN (
              SELECT table_name
              FROM information_schema.tables
              WHERE table_schema = %s
                AND table_type = 'BASE TABLE'
          )
        ORDER BY
            table_name,
            ordinal_position
    """

    return pd.read_sql_query(
        query,
        conn,
        params=(
            POSTGRES_SCHEMA,
            POSTGRES_SCHEMA
        )
    )


def get_databricks_all_table_types():
    """
    Return column metadata for all tables in the selected Databricks schema.
    """
    tables_df = spark.sql(
        f"""
        SHOW TABLES IN
        `{DATABRICKS_CATALOG}`.`{DATABRICKS_SCHEMA}`
        """
    )

    table_names = [
        row["tableName"]
        for row in tables_df.collect()
        if not row["isTemporary"]
    ]

    rows = []

    for table_name in table_names:
        full_name = (
            f"{DATABRICKS_CATALOG}."
            f"{DATABRICKS_SCHEMA}."
            f"{table_name}"
        )

        table_df = spark.table(
            full_name
        )

        for ordinal_position, field in enumerate(
            table_df.schema.fields,
            start=1
        ):
            rows.append(
                (
                    DATABRICKS_CATALOG,
                    DATABRICKS_SCHEMA,
                    table_name,
                    ordinal_position,
                    field.name,
                    field.dataType.simpleString(),
                    field.dataType.typeName(),
                    field.nullable
                )
            )

    schema = T.StructType([
        T.StructField("catalog", T.StringType(), False),
        T.StructField("schema", T.StringType(), False),
        T.StructField("table_name", T.StringType(), False),
        T.StructField("ordinal_position", T.IntegerType(), False),
        T.StructField("column_name", T.StringType(), False),
        T.StructField("data_type", T.StringType(), False),
        T.StructField("type_name", T.StringType(), False),
        T.StructField("nullable", T.BooleanType(), False),
    ])

    return spark.createDataFrame(
        rows,
        schema=schema
    )


def reconcile_all_table_types(
    postgres_all_types_df,
    databricks_all_types_df
):
    """
    Reconcile all tables and columns between PostgreSQL and Databricks.

    Comparison dimensions:
      - table existence
      - column existence
      - ordinal position
      - normalized datatype

    Returns:
      detailed mismatch Spark DataFrame
      summary-by-table Spark DataFrame
      overall summary Spark DataFrame
    """
    pg = postgres_all_types_df.copy()

    dbx = databricks_all_types_df.toPandas()

    pg["table_key"] = (
        pg["table_name"]
        .astype(str)
        .str.lower()
    )

    pg["column_key"] = (
        pg["column_name"]
        .astype(str)
        .str.lower()
    )

    dbx["table_key"] = (
        dbx["table_name"]
        .astype(str)
        .str.lower()
    )

    dbx["column_key"] = (
        dbx["column_name"]
        .astype(str)
        .str.lower()
    )

    merged = pd.merge(
        pg,
        dbx,
        how="outer",
        on=[
            "table_key",
            "column_key"
        ],
        suffixes=(
            "_postgres",
            "_databricks"
        )
    )

    merged[
        "postgres_normalized_type"
    ] = merged[
        "udt_name"
    ].apply(
        normalize_type_name
    )

    merged[
        "databricks_normalized_type"
    ] = merged[
        "type_name"
    ].apply(
        normalize_type_name
    )

    def classify(row):
        pg_table = row.get(
            "table_name_postgres"
        )

        dbx_table = row.get(
            "table_name_databricks"
        )

        pg_col = row.get(
            "column_name_postgres"
        )

        dbx_col = row.get(
            "column_name_databricks"
        )

        if pd.isna(pg_table):
            return "TABLE_MISSING_IN_POSTGRES"

        if pd.isna(dbx_table):
            return "TABLE_MISSING_IN_DATABRICKS"

        if pd.isna(pg_col):
            return "COLUMN_MISSING_IN_POSTGRES"

        if pd.isna(dbx_col):
            return "COLUMN_MISSING_IN_DATABRICKS"

        pg_pos = row.get(
            "ordinal_position_postgres"
        )

        dbx_pos = row.get(
            "ordinal_position_databricks"
        )

        pg_type = row.get(
            "postgres_normalized_type"
        )

        dbx_type = row.get(
            "databricks_normalized_type"
        )

        pos_match = (
            pd.notna(pg_pos)
            and pd.notna(dbx_pos)
            and int(pg_pos) == int(dbx_pos)
        )

        type_match = (
            pg_type == dbx_type
        )

        if pos_match and type_match:
            return "MATCH"

        if (not pos_match) and (not type_match):
            return "ORDINAL_AND_TYPE_MISMATCH"

        if not pos_match:
            return "ORDINAL_MISMATCH"

        return "TYPE_MISMATCH"

    merged[
        "RECON_STATUS"
    ] = merged.apply(
        classify,
        axis=1
    )

    detailed_columns = [
        "table_name_postgres",
        "table_name_databricks",
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
        "RECON_STATUS"
    ]

    detailed = merged[
        detailed_columns
    ].copy()

    # Table name for summary
    detailed[
        "table_name"
    ] = detailed[
        "table_name_postgres"
    ].where(
        detailed[
            "table_name_postgres"
        ].notna(),
        detailed[
            "table_name_databricks"
        ]
    )

    mismatch_only = detailed[
        detailed["RECON_STATUS"] != "MATCH"
    ].copy()

    if mismatch_only.empty:
        mismatch_only = pd.DataFrame(
            columns=[
                "table_name",
                *detailed_columns
            ]
        )
    else:
        mismatch_only = mismatch_only[
            [
                "table_name",
                *detailed_columns
            ]
        ]

    # Summary by table
    summary_rows = []

    all_tables = sorted(
        detailed["table_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    for table_name in all_tables:
        subset = detailed[
            detailed["table_name"] == table_name
        ]

        mismatches = subset[
            subset["RECON_STATUS"] != "MATCH"
        ]

        counts = (
            mismatches[
                "RECON_STATUS"
            ]
            .value_counts()
            .to_dict()
        )

        summary_rows.append(
            {
                "table_name":
                    table_name,
                "total_columns_compared":
                    int(len(subset)),
                "matched_columns":
                    int(
                        (
                            subset[
                                "RECON_STATUS"
                            ]
                            == "MATCH"
                        ).sum()
                    ),
                "mismatched_columns":
                    int(len(mismatches)),
                "table_missing_in_postgres":
                    int(
                        counts.get(
                            "TABLE_MISSING_IN_POSTGRES",
                            0
                        )
                    ),
                "table_missing_in_databricks":
                    int(
                        counts.get(
                            "TABLE_MISSING_IN_DATABRICKS",
                            0
                        )
                    ),
                "column_missing_in_postgres":
                    int(
                        counts.get(
                            "COLUMN_MISSING_IN_POSTGRES",
                            0
                        )
                    ),
                "column_missing_in_databricks":
                    int(
                        counts.get(
                            "COLUMN_MISSING_IN_DATABRICKS",
                            0
                        )
                    ),
                "ordinal_mismatch":
                    int(
                        counts.get(
                            "ORDINAL_MISMATCH",
                            0
                        )
                    ),
                "type_mismatch":
                    int(
                        counts.get(
                            "TYPE_MISMATCH",
                            0
                        )
                    ),
                "ordinal_and_type_mismatch":
                    int(
                        counts.get(
                            "ORDINAL_AND_TYPE_MISMATCH",
                            0
                        )
                    ),
                "status":
                    (
                        "MATCH"
                        if len(mismatches) == 0
                        else "MISMATCH"
                    )
            }
        )

    summary_pdf = pd.DataFrame(
        summary_rows
    )

    overall_pdf = pd.DataFrame([
        {
            "postgres_table_count":
                int(
                    postgres_all_types_df[
                        "table_name"
                    ].nunique()
                ),
            "databricks_table_count":
                int(
                    dbx[
                        "table_name"
                    ].nunique()
                ),
            "tables_compared":
                int(
                    summary_pdf[
                        "table_name"
                    ].nunique()
                )
                if not summary_pdf.empty
                else 0,
            "tables_with_mismatch":
                int(
                    (
                        summary_pdf[
                            "status"
                        ] == "MISMATCH"
                    ).sum()
                )
                if not summary_pdf.empty
                else 0,
            "total_mismatch_records":
                int(
                    len(
                        mismatch_only
                    )
                )
        }
    ])

    detailed_sdf = (
        pandas_to_spark_safe(
            mismatch_only
        )
        if not mismatch_only.empty
        else spark.createDataFrame(
            [],
            T.StructType([
                T.StructField(
                    "table_name",
                    T.StringType(),
                    True
                )
            ])
        )
    )

    summary_sdf = (
        pandas_to_spark_safe(
            summary_pdf
        )
        if not summary_pdf.empty
        else spark.createDataFrame(
            [],
            T.StructType([
                T.StructField(
                    "table_name",
                    T.StringType(),
                    True
                )
            ])
        )
    )

    overall_sdf = pandas_to_spark_safe(
        overall_pdf
    )

    return (
        detailed_sdf,
        summary_sdf,
        overall_sdf
    )


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

    where_sql = sql.SQL("")

    if POSTGRES_SAMPLE_WHERE and POSTGRES_SAMPLE_WHERE.strip():
        # Trusted configuration. Supply only the condition,
        # not the WHERE keyword itself.
        where_sql = sql.SQL(
            " WHERE " + POSTGRES_SAMPLE_WHERE.strip()
        )

    query = sql.SQL("""
        SELECT *
        FROM {}.{}
        {}
        ORDER BY {}
        LIMIT %s
    """).format(
        sql.Identifier(POSTGRES_SCHEMA),
        sql.Identifier(POSTGRES_TABLE),
        where_sql,
        sql.SQL(", ").join(order_parts)
    )

    print(
        "\nPostgreSQL Sample WHERE:",
        POSTGRES_SAMPLE_WHERE.strip()
        if POSTGRES_SAMPLE_WHERE.strip()
        else "<none>"
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

def coerce_value_for_spark_type(value, data_type):
    """
    Coerce a PostgreSQL/pandas key value to the Databricks target
    column's exact Spark datatype.

    This prevents Spark from inferring a mixed DoubleType/DecimalType
    schema for sampled primary-key columns.
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        if isinstance(data_type, T.StringType):
            return str(value)

        if isinstance(
            data_type,
            (
                T.ByteType,
                T.ShortType,
                T.IntegerType,
                T.LongType
            )
        ):
            return int(value)

        if isinstance(
            data_type,
            (
                T.FloatType,
                T.DoubleType
            )
        ):
            return float(value)

        if isinstance(data_type, T.DecimalType):
            if isinstance(value, Decimal):
                return value
            return Decimal(str(value))

        if isinstance(data_type, T.BooleanType):
            if isinstance(value, bool):
                return value

            text = str(value).strip().lower()

            if text in ("true", "1", "t", "yes", "y"):
                return True

            if text in ("false", "0", "f", "no", "n"):
                return False

            return bool(value)

        if isinstance(data_type, T.DateType):
            # Handle pandas Timestamp first.
            if isinstance(value, pd.Timestamp):
                py_dt = value.to_pydatetime()
                return py_dt.date()

            if isinstance(value, datetime):
                return value.date()

            if isinstance(value, date):
                return value

            text = str(value).strip().split("T")[0].split(" ")[0]
            year, month, day = [
                int(piece)
                for piece in text.split("-")[:3]
            ]
            return date(year, month, day)

        if isinstance(data_type, T.TimestampType):
            # Spark TimestampType requires native datetime.datetime.
            # Handle pandas Timestamp before generic datetime.
            if isinstance(value, pd.Timestamp):
                py_dt = value.to_pydatetime()

                if py_dt.tzinfo is not None:
                    py_dt = (
                        py_dt
                        .astimezone(timezone.utc)
                        .replace(tzinfo=None)
                    )

                return py_dt

            if isinstance(value, datetime):
                py_dt = value

                if py_dt.tzinfo is not None:
                    py_dt = (
                        py_dt
                        .astimezone(timezone.utc)
                        .replace(tzinfo=None)
                    )

                return py_dt

            text = str(value).strip().replace("Z", "+00:00")

            try:
                py_dt = datetime.fromisoformat(text)

                if py_dt.tzinfo is not None:
                    py_dt = (
                        py_dt
                        .astimezone(timezone.utc)
                        .replace(tzinfo=None)
                    )

                return py_dt

            except Exception:
                date_text = text.split("T")[0].split(" ")[0]
                year, month, day = [
                    int(piece)
                    for piece in date_text.split("-")[:3]
                ]
                return datetime(year, month, day)

        # For uncommon key datatypes, retain the original Python value.
        return value

    except Exception as coercion_error:
        raise ValueError(
            f"Unable to coerce sampled key value '{value}' "
            f"to Databricks type '{data_type.simpleString()}'."
        ) from coercion_error


def build_sample_key_spark_dataframe(
    key_pdf,
    target_df,
    key_columns
):
    """
    Build sampled-key Spark DataFrame using the Databricks target key
    schema explicitly instead of schema inference.
    """
    target_field_map = {
        field.name: field
        for field in target_df.schema.fields
    }

    key_schema = T.StructType([
        T.StructField(
            column,
            target_field_map[column].dataType,
            True
        )
        for column in key_columns
    ])

    rows = []

    for record in key_pdf.to_dict(
        orient="records"
    ):
        rows.append(
            tuple(
                coerce_value_for_spark_type(
                    record.get(column),
                    target_field_map[column].dataType
                )
                for column in key_columns
            )
        )

    # Defensive validation before Spark ingestion.
    for row_index, row_values in enumerate(rows):
        for field_index, field in enumerate(key_schema.fields):
            if isinstance(field.dataType, T.TimestampType):
                field_value = row_values[field_index]

                if (
                    field_value is not None
                    and type(field_value) is not datetime
                ):
                    raise TypeError(
                        f"Sample key column '{field.name}' at row "
                        f"{row_index} contains timestamp object "
                        f"{type(field_value)} instead of native "
                        f"datetime.datetime."
                    )

    return spark.createDataFrame(
        rows,
        schema=key_schema
    )


def databricks_sample_for_keys(
    postgres_sample_df,
    postgres_hash_columns
):
    key_columns = parse_column_list(UNIQUE_KEY_COLUMNS)

    target_df = get_databricks_dataframe()

    if DATABRICKS_SAMPLE_WHERE and DATABRICKS_SAMPLE_WHERE.strip():
        target_df = target_df.filter(
            DATABRICKS_SAMPLE_WHERE.strip()
        )

    print(
        "\nDatabricks Sample WHERE:",
        DATABRICKS_SAMPLE_WHERE.strip()
        if DATABRICKS_SAMPLE_WHERE.strip()
        else "<none>"
    )

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

    key_sdf = build_sample_key_spark_dataframe(
        key_pdf,
        target_df,
        key_columns
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

    return pandas_to_spark_safe(clean)




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
        "type_check_all",
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
            "(Duplicate_Check and Type_Check_All are excluded "
            "unless explicitly selected)"
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
            print("COUNTS CHECK - DEFINED TABLE LIST")
            print("=" * 80)

            counts_recon = (
                counts_check_common_tables(
                    conn
                )
            )

            counts_summary = (
                counts_check_summary(
                    counts_recon
                )
            )

            print(
                "\nCounts Reconciliation "
                "(defined table list only):"
            )

            display(
                counts_recon
            )

            print(
                "\nCounts Reconciliation Summary:"
            )

            display(
                counts_summary
            )

            results["Counts_Check"] = {
                "reconciliation":
                    counts_recon,
                "summary":
                    counts_summary
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
        # TYPE CHECK ALL
        # ====================================================

        if check_flag == "type_check_all":
            print("\n" + "=" * 80)
            print("TYPE CHECK ALL - ALL TABLES")
            print("=" * 80)

            pg_all_types = (
                get_postgres_all_table_types(
                    conn
                )
            )

            dbx_all_types = (
                get_databricks_all_table_types()
            )

            (
                type_all_mismatches,
                type_all_summary,
                type_all_overall
            ) = reconcile_all_table_types(
                pg_all_types,
                dbx_all_types
            )

            print(
                "\nOverall Summary:"
            )
            display(
                type_all_overall
            )

            print(
                "\nSummary by Table:"
            )
            display(
                type_all_summary
            )

            print(
                "\nDetailed Type / Ordinal Mismatches "
                "(showing up to 100,000 records):"
            )
            display(
                type_all_mismatches.limit(100000)
            )

            results[
                "Type_Check_All"
            ] = {
                "postgres":
                    pg_all_types,
                "databricks":
                    dbx_all_types,
                "reconciliation":
                    type_all_mismatches,
                "summary":
                    type_all_summary,
                "overall_summary":
                    type_all_overall
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
                ),
                "postgres_sample_where":
                    POSTGRES_SAMPLE_WHERE,
                "databricks_sample_where":
                    DATABRICKS_SAMPLE_WHERE
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

#
# Type_Check_All summary by table:
# display(
#     reconciliation_results[
#         "Type_Check_All"
#     ]["summary"]
# )
#
# Type_Check_All detailed mismatches:
# display(
#     reconciliation_results[
#         "Type_Check_All"
#     ]["reconciliation"]
# )
