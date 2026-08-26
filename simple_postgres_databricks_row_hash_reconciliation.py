"""
Simple PostgreSQL vs Databricks SQL Reconciliation using Row Hash
================================================================

Supports TWO modes:

1. KEYED MODE
   - Set KEY_COLUMNS, e.g. "claim_id,member_id"
   - Rows are aligned by key.
   - A deterministic SHA-256 row hash is generated from all common
     non-ignored columns.
   - Hashes are compared for each key.

2. NON-KEY MODE
   - Set KEY_COLUMNS = ""
   - No unique key is required.
   - Each output row is converted to a deterministic SHA-256 hash.
   - PostgreSQL and Databricks are compared as multisets of row hashes.
   - Duplicate rows are handled using hash counts.

Requirements:
    pip install psycopg2-binary pandas

Designed to run inside Databricks.
"""

import hashlib
from decimal import Decimal

import pandas as pd
import psycopg2
from pyspark.sql import functions as F


# ============================================================
# CONFIGURATION
# ============================================================

POSTGRES_CONFIG = {
    "host": "your-postgres-host",
    "port": 5432,
    "database": "your_database",
    "user": "your_username",
    "password": "your_password"
}


POSTGRES_SQL = """
SELECT
    claim_id,
    member_id,
    status,
    amount,
    intimation_date
FROM claims.claim_details
WHERE intimation_date >= DATE '2026-07-01'
"""


DATABRICKS_SQL = """
SELECT
    claim_id,
    member_id,
    status,
    amount,
    intimation_date
FROM catalog.schema.claim_details
WHERE intimation_date >= DATE('2026-07-01')
"""


# Optional.
#
# KEYED mode:
# KEY_COLUMNS = "claim_id,member_id"
#
# NON-KEY mode:
# KEY_COLUMNS = ""
KEY_COLUMNS = ""


# Optional columns excluded from row-hash calculation.
IGNORE_COLUMNS = ""


# ============================================================
# HELPERS
# ============================================================

def parse_list(value):
    if not value:
        return []

    if isinstance(value, list):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    return [
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    ]


def normalize_value(value):
    """
    Convert PostgreSQL/Databricks values into deterministic strings
    before row hashing.
    """
    if value is None:
        return "<NULL>"

    try:
        if pd.isna(value):
            return "<NULL>"
    except Exception:
        pass

    if isinstance(value, Decimal):
        return format(value, "f")

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    if isinstance(value, bool):
        return (
            "true"
            if value
            else "false"
        )

    return str(value).strip()


def calculate_row_hash(
    row,
    hash_columns
):
    """
    Deterministic SHA-256 row hash.

    Column names are included so column ordering does not become ambiguous.
    """
    parts = []

    for column in hash_columns:
        value = normalize_value(
            row[column]
        )

        parts.append(
            f"{column}={value}"
        )

    hash_input = "||".join(
        parts
    )

    return hashlib.sha256(
        hash_input.encode("utf-8")
    ).hexdigest()


def postgres_query(sql_text):
    conn = psycopg2.connect(
        host=POSTGRES_CONFIG["host"],
        port=POSTGRES_CONFIG["port"],
        database=POSTGRES_CONFIG["database"],
        user=POSTGRES_CONFIG["user"],
        password=POSTGRES_CONFIG["password"]
    )

    try:
        return pd.read_sql_query(
            sql_text,
            conn
        )

    finally:
        conn.close()


def databricks_query(sql_text):
    """
    Convert Databricks date/timestamp fields to strings before toPandas()
    to avoid pandas timestamp range problems.
    """
    df = spark.sql(
        sql_text
    )

    select_exprs = []

    for field in df.schema.fields:

        if field.dataType.typeName() in {
            "timestamp",
            "timestamp_ntz",
            "date"
        }:
            select_exprs.append(
                F.col(field.name)
                .cast("string")
                .alias(field.name)
            )

        else:
            select_exprs.append(
                F.col(field.name)
            )

    return (
        df
        .select(*select_exprs)
        .toPandas()
    )


def safe_display(pdf):
    if pdf is None or pdf.empty:
        print("No rows.")
        return

    safe_pdf = pdf.copy()

    for column in safe_pdf.columns:
        safe_pdf[column] = safe_pdf[
            column
        ].map(
            lambda value:
                None
                if value is None
                else str(value)
        )

    display(
        spark.createDataFrame(
            safe_pdf
        )
    )


def get_comparison_columns(
    postgres_df,
    databricks_df
):
    ignore_columns = parse_list(
        IGNORE_COLUMNS
    )

    common_columns = [
        column
        for column in postgres_df.columns
        if (
            column in databricks_df.columns
            and column not in ignore_columns
        )
    ]

    if not common_columns:
        raise ValueError(
            "No common columns are available for row-hash comparison."
        )

    return common_columns


def add_row_hash(
    dataframe,
    hash_columns
):
    result = dataframe.copy()

    result["ROW_HASH"] = result.apply(
        lambda row:
            calculate_row_hash(
                row,
                hash_columns
            ),
        axis=1
    )

    return result


# ============================================================
# KEYED RECONCILIATION
# ============================================================

def reconcile_with_keys(
    postgres_df,
    databricks_df,
    hash_columns,
    key_columns
):
    """
    Align rows by key columns and compare row hashes.
    """

    for key in key_columns:

        if key not in postgres_df.columns:
            raise ValueError(
                f"Key column '{key}' missing in PostgreSQL result."
            )

        if key not in databricks_df.columns:
            raise ValueError(
                f"Key column '{key}' missing in Databricks result."
            )

    pg = add_row_hash(
        postgres_df,
        hash_columns
    )

    dbx = add_row_hash(
        databricks_df,
        hash_columns
    )

    pg_hash = pg[
        key_columns
        + [
            "ROW_HASH"
        ]
    ].rename(
        columns={
            "ROW_HASH":
                "POSTGRES_ROW_HASH"
        }
    )

    dbx_hash = dbx[
        key_columns
        + [
            "ROW_HASH"
        ]
    ].rename(
        columns={
            "ROW_HASH":
                "DATABRICKS_ROW_HASH"
        }
    )

    merged = pd.merge(
        pg_hash,
        dbx_hash,
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

    merged[
        "RECON_STATUS"
    ] = merged.apply(
        status,
        axis=1
    )

    merged = merged.drop(
        columns=[
            "_merge"
        ]
    )

    summary = (
        merged
        .groupby(
            "RECON_STATUS",
            dropna=False
        )
        .size()
        .reset_index(
            name="row_count"
        )
    )

    mismatch_keys = merged[
        merged[
            "RECON_STATUS"
        ] == "HASH_MISMATCH"
    ][
        key_columns
    ]

    detailed_mismatches = []

    if not mismatch_keys.empty:

        pg_detail = pd.merge(
            mismatch_keys,
            pg,
            on=key_columns,
            how="left"
        )

        dbx_detail = pd.merge(
            mismatch_keys,
            dbx,
            on=key_columns,
            how="left"
        )

        detailed = pd.merge(
            pg_detail,
            dbx_detail,
            on=key_columns,
            how="inner",
            suffixes=(
                "__POSTGRES",
                "__DATABRICKS"
            )
        )

        for _, row in detailed.iterrows():

            key_values = {
                key: row[key]
                for key in key_columns
            }

            for column in hash_columns:

                if column in key_columns:
                    continue

                pg_column = (
                    f"{column}__POSTGRES"
                )

                dbx_column = (
                    f"{column}__DATABRICKS"
                )

                if (
                    pg_column not in detailed.columns
                    or dbx_column not in detailed.columns
                ):
                    continue

                pg_value = normalize_value(
                    row[pg_column]
                )

                dbx_value = normalize_value(
                    row[dbx_column]
                )

                if pg_value != dbx_value:

                    detailed_mismatches.append(
                        {
                            **key_values,
                            "column_name":
                                column,
                            "postgres_value":
                                pg_value,
                            "databricks_value":
                                dbx_value
                        }
                    )

    detailed_mismatch_df = pd.DataFrame(
        detailed_mismatches
    )

    return {
        "mode":
            "KEYED",

        "postgres_with_hash":
            pg,

        "databricks_with_hash":
            dbx,

        "reconciliation":
            merged,

        "summary":
            summary,

        "detailed_mismatches":
            detailed_mismatch_df
    }


# ============================================================
# NON-KEY / HASH-ONLY RECONCILIATION
# ============================================================

def reconcile_without_keys(
    postgres_df,
    databricks_df,
    hash_columns
):
    """
    Compare result sets without a unique key.

    Each row becomes a ROW_HASH.
    Comparison is based on hash frequency, which also handles duplicate rows.
    """

    pg = add_row_hash(
        postgres_df,
        hash_columns
    )

    dbx = add_row_hash(
        databricks_df,
        hash_columns
    )

    pg_hash_counts = (
        pg
        .groupby(
            "ROW_HASH",
            dropna=False
        )
        .size()
        .reset_index(
            name="POSTGRES_COUNT"
        )
    )

    dbx_hash_counts = (
        dbx
        .groupby(
            "ROW_HASH",
            dropna=False
        )
        .size()
        .reset_index(
            name="DATABRICKS_COUNT"
        )
    )

    merged = pd.merge(
        pg_hash_counts,
        dbx_hash_counts,
        on="ROW_HASH",
        how="outer"
    )

    merged[
        "POSTGRES_COUNT"
    ] = (
        merged[
            "POSTGRES_COUNT"
        ]
        .fillna(0)
        .astype(int)
    )

    merged[
        "DATABRICKS_COUNT"
    ] = (
        merged[
            "DATABRICKS_COUNT"
        ]
        .fillna(0)
        .astype(int)
    )

    merged[
        "COUNT_DIFFERENCE"
    ] = (
        merged[
            "DATABRICKS_COUNT"
        ]
        - merged[
            "POSTGRES_COUNT"
        ]
    )

    def status(row):

        pg_count = row[
            "POSTGRES_COUNT"
        ]

        dbx_count = row[
            "DATABRICKS_COUNT"
        ]

        if pg_count == dbx_count:
            return "MATCH"

        if (
            pg_count > 0
            and dbx_count == 0
        ):
            return "MISSING_IN_DATABRICKS"

        if (
            pg_count == 0
            and dbx_count > 0
        ):
            return "MISSING_IN_POSTGRES"

        return "DUPLICATE_COUNT_MISMATCH"

    merged[
        "RECON_STATUS"
    ] = merged.apply(
        status,
        axis=1
    )

    summary = (
        merged
        .groupby(
            "RECON_STATUS",
            dropna=False
        )
        .agg(
            hash_count=(
                "ROW_HASH",
                "count"
            ),
            postgres_rows=(
                "POSTGRES_COUNT",
                "sum"
            ),
            databricks_rows=(
                "DATABRICKS_COUNT",
                "sum"
            )
        )
        .reset_index()
    )

    mismatch_hashes = merged[
        merged[
            "RECON_STATUS"
        ] != "MATCH"
    ][
        "ROW_HASH"
    ]

    pg_unmatched = pg[
        pg[
            "ROW_HASH"
        ].isin(
            mismatch_hashes
        )
    ].copy()

    pg_unmatched[
        "SOURCE_SYSTEM"
    ] = "POSTGRES"

    dbx_unmatched = dbx[
        dbx[
            "ROW_HASH"
        ].isin(
            mismatch_hashes
        )
    ].copy()

    dbx_unmatched[
        "SOURCE_SYSTEM"
    ] = "DATABRICKS"

    unmatched_rows = pd.concat(
        [
            pg_unmatched,
            dbx_unmatched
        ],
        ignore_index=True
    )

    return {
        "mode":
            "NON_KEY_HASH",

        "postgres_with_hash":
            pg,

        "databricks_with_hash":
            dbx,

        "reconciliation":
            merged,

        "summary":
            summary,

        "unmatched_rows":
            unmatched_rows
    }


# ============================================================
# MAIN RECONCILIATION
# ============================================================

def reconcile(
    postgres_df,
    databricks_df
):
    key_columns = parse_list(
        KEY_COLUMNS
    )

    hash_columns = get_comparison_columns(
        postgres_df,
        databricks_df
    )

    postgres_only_columns = [
        column
        for column in postgres_df.columns
        if column not in databricks_df.columns
    ]

    databricks_only_columns = [
        column
        for column in databricks_df.columns
        if column not in postgres_df.columns
    ]

    print(
        "\nHash Columns:"
    )

    print(
        hash_columns
    )

    if key_columns:

        result = reconcile_with_keys(
            postgres_df,
            databricks_df,
            hash_columns,
            key_columns
        )

    else:

        result = reconcile_without_keys(
            postgres_df,
            databricks_df,
            hash_columns
        )

    result[
        "hash_columns"
    ] = hash_columns

    result[
        "postgres_only_columns"
    ] = postgres_only_columns

    result[
        "databricks_only_columns"
    ] = databricks_only_columns

    return result


# ============================================================
# EXECUTION
# ============================================================

print(
    "Executing PostgreSQL SQL..."
)

postgres_result = postgres_query(
    POSTGRES_SQL
)

print(
    f"PostgreSQL rows: "
    f"{len(postgres_result):,}"
)


print(
    "Executing Databricks SQL..."
)

databricks_result = databricks_query(
    DATABRICKS_SQL
)

print(
    f"Databricks rows: "
    f"{len(databricks_result):,}"
)


reconciliation = reconcile(
    postgres_result,
    databricks_result
)


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    f"RECONCILIATION MODE: "
    f"{reconciliation['mode']}"
)

print(
    "=" * 80
)


print(
    "\nRECONCILIATION SUMMARY"
)

safe_display(
    reconciliation[
        "summary"
    ]
)


print(
    "\nROW HASH RECONCILIATION"
)

safe_display(
    reconciliation[
        "reconciliation"
    ]
)


if (
    reconciliation[
        "mode"
    ] == "KEYED"
):

    print(
        "\nDETAILED COLUMN MISMATCHES"
    )

    safe_display(
        reconciliation[
            "detailed_mismatches"
        ]
    )

else:

    print(
        "\nUNMATCHED ROWS / DUPLICATE COUNT DIFFERENCES"
    )

    safe_display(
        reconciliation[
            "unmatched_rows"
        ]
    )


print(
    "\nColumns only in PostgreSQL:"
)

print(
    reconciliation[
        "postgres_only_columns"
    ]
)


print(
    "\nColumns only in Databricks:"
)

print(
    reconciliation[
        "databricks_only_columns"
    ]
)
