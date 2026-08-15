"""
PostgreSQL <-> Databricks In-Memory Reconciliation Utility
==========================================================

Supported checks:
    Counts_Check
    Type_Check
    Sample_Check
    All_Check

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


def pandas_to_spark_safe(dataframe):
    """
    Convert a pandas DataFrame to Spark DataFrame.
    """
    if dataframe.empty:
        return spark.createDataFrame([], T.StructType([]))

    clean_df = dataframe.astype(object).where(
        pd.notnull(dataframe),
        None
    )

    return spark.createDataFrame(clean_df)


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

    dbx = (
        databricks_sample_df
        .select(
            *key_columns,
            F.col("ROW_HASH").alias(
                "DATABRICKS_ROW_HASH"
            )
        )
        .toPandas()
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
    common_hash_columns
):
    """
    For rows classified as HASH_MISMATCH, identify exactly
    which columns differ between PostgreSQL and Databricks.

    Returns a pandas DataFrame with:
        unique-key columns
        mismatch_column
        postgres_value
        databricks_value
    """

    key_columns = parse_column_list(
        UNIQUE_KEY_COLUMNS
    )

    mismatch_keys_pdf = (
        sample_reconciliation_df
        .filter(
            F.col("RECON_STATUS")
            == F.lit("HASH_MISMATCH")
        )
        .select(
            *key_columns
        )
        .toPandas()
    )

    output_columns = (
        key_columns
        + [
            "mismatch_column",
            "postgres_value",
            "databricks_value"
        ]
    )

    if mismatch_keys_pdf.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    pg_compare = postgres_sample_df[
        key_columns + common_hash_columns
    ].copy()

    dbx_compare = (
        databricks_sample_df
        .select(
            *[
                F.col(column)
                for column in (
                    key_columns
                    + common_hash_columns
                )
            ]
        )
        .toPandas()
    )

    # Restrict both sides to rows already known to have hash mismatches.
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

    for _, row in merged.iterrows():

        key_values = {
            key: row[key]
            for key in key_columns
        }

        for column in common_hash_columns:

            # A unique key may also be included in the hash columns.
            # Since it exists unsuffixed after the merge, it cannot
            # differ for an already matched key and can be skipped.
            if column in key_columns:
                continue

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

            if not values_equal_for_reconciliation(
                pg_value,
                dbx_value
            ):
                mismatch_row = dict(
                    key_values
                )

                mismatch_row[
                    "mismatch_column"
                ] = column

                mismatch_row[
                    "postgres_value"
                ] = normalize_python_value(
                    pg_value
                )

                mismatch_row[
                    "databricks_value"
                ] = normalize_python_value(
                    dbx_value
                )

                mismatch_rows.append(
                    mismatch_row
                )

    return pd.DataFrame(
        mismatch_rows,
        columns=output_columns
    )


def build_column_mismatch_summary(
    column_mismatch_df
):
    """
    Summarize the number of mismatched values by column.
    """

    if column_mismatch_df.empty:
        return pd.DataFrame(
            columns=[
                "mismatch_column",
                "mismatch_count"
            ]
        )

    return (
        column_mismatch_df
        .groupby(
            "mismatch_column",
            dropna=False
        )
        .size()
        .reset_index(
            name="mismatch_count"
        )
        .sort_values(
            "mismatch_count",
            ascending=False
        )
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
        "all_check"
    }

    check_flag = CHECK_FLAG.strip().lower()

    if check_flag not in valid_flags:
        raise ValueError(
            f"Invalid CHECK_FLAG: {CHECK_FLAG}. "
            "Allowed values: Counts_Check, Type_Check, "
            "Sample_Check, All_Check"
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

            column_mismatch_df = (
                build_column_level_mismatches(
                    pg_sample,
                    dbx_sample,
                    sample_recon,
                    common_hash_columns
                )
            )

            column_mismatch_summary_df = (
                build_column_mismatch_summary(
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
                "hash_columns": common_hash_columns,
                "excluded_source_columns": (
                    source_only_hash_columns
                )
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
                    "\nDetailed Mismatched Columns:"
                )

                display(
                    pandas_to_spark_safe(
                        final_column_mismatches
                    )
                )

                print(
                    "\nTotal mismatched column values:",
                    len(
                        final_column_mismatches
                    )
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
