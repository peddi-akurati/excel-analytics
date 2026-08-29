"""
Simple PostgreSQL vs Databricks SQL Reconciliation
==================================================

Supports two reconciliation modes:

1) Row-hash reconciliation
   SUPPRESS_ROW_HASH_CHECK = False

   - KEY_COLUMNS provided:
       rows are aligned by key and row hashes are compared.
       detailed column mismatches are shown for hash-mismatch rows.

   - KEY_COLUMNS blank:
       full result sets are compared using row-hash frequency.
       duplicate row counts are handled.

2) Direct result-set comparison
   SUPPRESS_ROW_HASH_CHECK = True

   - No row hash is generated.
   - The complete common result-set columns are compared directly.
   - Records found only in PostgreSQL or only in Databricks are shown.

Requirements:
    pip install psycopg2-binary pandas

Designed to run inside Databricks where spark and display() are available.
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


# Optional unique/business key columns.
#
# Example:
# KEY_COLUMNS = "claim_id,member_id"
#
# Leave blank for non-key hash comparison:
# KEY_COLUMNS = ""
KEY_COLUMNS = ""


# Optional columns excluded from comparison.
#
# Example:
# IGNORE_COLUMNS = "created_ts,updated_ts"
IGNORE_COLUMNS = ""


# False -> row-hash reconciliation
# True  -> direct result-set comparison without row hash
SUPPRESS_ROW_HASH_CHECK = False


# ============================================================
# GENERIC HELPERS
# ============================================================

def parse_list(value):
    if not value:
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


def normalize_value(value):
    """
    Normalize values consistently across PostgreSQL and Databricks.
    """

    if value is None:
        return "<NULL>"

    try:
        if pd.isna(value):
            return "<NULL>"
    except Exception:
        pass

    if isinstance(value, Decimal):
        return format(
            value,
            "f"
        )

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


def postgres_query(sql_text):
    """
    Execute PostgreSQL SQL and return pandas DataFrame.
    """

    connection = psycopg2.connect(
        host=POSTGRES_CONFIG["host"],
        port=POSTGRES_CONFIG["port"],
        database=POSTGRES_CONFIG["database"],
        user=POSTGRES_CONFIG["user"],
        password=POSTGRES_CONFIG["password"]
    )

    try:
        return pd.read_sql_query(
            sql_text,
            connection
        )

    finally:
        connection.close()


def databricks_query(sql_text):
    """
    Execute Databricks SQL and return pandas DataFrame.

    Date/timestamp fields are converted to strings in Spark before
    toPandas() to reduce pandas timestamp-range issues.
    """

    dataframe = spark.sql(
        sql_text
    )

    select_expressions = []

    for field in dataframe.schema.fields:

        if field.dataType.typeName() in {
            "timestamp",
            "timestamp_ntz",
            "date"
        }:
            select_expressions.append(
                F.col(
                    field.name
                )
                .cast("string")
                .alias(
                    field.name
                )
            )

        else:
            select_expressions.append(
                F.col(
                    field.name
                )
            )

    return (
        dataframe
        .select(
            *select_expressions
        )
        .toPandas()
    )


def safe_display(pandas_df):
    """
    Display a pandas DataFrame safely in Databricks.
    """

    if (
        pandas_df is None
        or pandas_df.empty
    ):
        print(
            "No rows."
        )
        return

    safe_df = pandas_df.copy()

    for column in safe_df.columns:
        safe_df[
            column
        ] = safe_df[
            column
        ].map(
            lambda value:
                None
                if value is None
                else str(value)
        )

    display(
        spark.createDataFrame(
            safe_df
        )
    )


def get_common_columns(
    postgres_df,
    databricks_df
):
    """
    Return common columns excluding IGNORE_COLUMNS.
    """

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
            "No common columns are available for reconciliation."
        )

    return common_columns


def get_column_differences(
    postgres_df,
    databricks_df
):
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

    return (
        postgres_only_columns,
        databricks_only_columns
    )


# ============================================================
# ROW HASH HELPERS
# ============================================================

def calculate_row_hash(
    row,
    hash_columns
):
    """
    Create deterministic SHA-256 row hash.
    """

    parts = []

    for column in hash_columns:

        normalized_value = normalize_value(
            row[
                column
            ]
        )

        parts.append(
            f"{column}="
            f"{normalized_value}"
        )

    hash_input = "||".join(
        parts
    )

    return hashlib.sha256(
        hash_input.encode(
            "utf-8"
        )
    ).hexdigest()


def add_row_hash(
    dataframe,
    hash_columns
):
    result = dataframe.copy()

    result[
        "ROW_HASH"
    ] = result.apply(
        lambda row:
            calculate_row_hash(
                row,
                hash_columns
            ),
        axis=1
    )

    return result


# ============================================================
# MODE 1A - KEYED ROW HASH RECONCILIATION
# ============================================================

def reconcile_with_keys(
    postgres_df,
    databricks_df,
    hash_columns,
    key_columns
):
    """
    Align rows by KEY_COLUMNS and compare ROW_HASH values.
    """

    for key_column in key_columns:

        if key_column not in postgres_df.columns:
            raise ValueError(
                f"Key column '{key_column}' "
                f"missing in PostgreSQL result."
            )

        if key_column not in databricks_df.columns:
            raise ValueError(
                f"Key column '{key_column}' "
                f"missing in Databricks result."
            )

    postgres_hashed = add_row_hash(
        postgres_df,
        hash_columns
    )

    databricks_hashed = add_row_hash(
        databricks_df,
        hash_columns
    )

    postgres_hash_df = (
        postgres_hashed[
            key_columns
            + [
                "ROW_HASH"
            ]
        ]
        .rename(
            columns={
                "ROW_HASH":
                    "POSTGRES_ROW_HASH"
            }
        )
    )

    databricks_hash_df = (
        databricks_hashed[
            key_columns
            + [
                "ROW_HASH"
            ]
        ]
        .rename(
            columns={
                "ROW_HASH":
                    "DATABRICKS_ROW_HASH"
            }
        )
    )

    reconciliation_df = pd.merge(
        postgres_hash_df,
        databricks_hash_df,
        on=key_columns,
        how="outer",
        indicator=True
    )

    def derive_status(row):

        if row["_merge"] == "left_only":
            return "MISSING_IN_DATABRICKS"

        if row["_merge"] == "right_only":
            return "MISSING_IN_POSTGRES"

        if (
            row[
                "POSTGRES_ROW_HASH"
            ]
            == row[
                "DATABRICKS_ROW_HASH"
            ]
        ):
            return "MATCH"

        return "HASH_MISMATCH"

    reconciliation_df[
        "RECON_STATUS"
    ] = reconciliation_df.apply(
        derive_status,
        axis=1
    )

    reconciliation_df = reconciliation_df.drop(
        columns=[
            "_merge"
        ]
    )

    summary_df = (
        reconciliation_df
        .groupby(
            "RECON_STATUS",
            dropna=False
        )
        .size()
        .reset_index(
            name="row_count"
        )
    )

    # --------------------------------------------------------
    # Detailed mismatched columns
    # --------------------------------------------------------

    mismatch_key_df = reconciliation_df[
        reconciliation_df[
            "RECON_STATUS"
        ] == "HASH_MISMATCH"
    ][
        key_columns
    ]

    detailed_rows = []

    if not mismatch_key_df.empty:

        postgres_detail = pd.merge(
            mismatch_key_df,
            postgres_hashed,
            on=key_columns,
            how="left"
        )

        databricks_detail = pd.merge(
            mismatch_key_df,
            databricks_hashed,
            on=key_columns,
            how="left"
        )

        combined_detail = pd.merge(
            postgres_detail,
            databricks_detail,
            on=key_columns,
            how="inner",
            suffixes=(
                "__POSTGRES",
                "__DATABRICKS"
            )
        )

        for _, row in combined_detail.iterrows():

            key_values = {
                key_column:
                    row[
                        key_column
                    ]
                for key_column in key_columns
            }

            for column in hash_columns:

                if column in key_columns:
                    continue

                postgres_column = (
                    f"{column}"
                    f"__POSTGRES"
                )

                databricks_column = (
                    f"{column}"
                    f"__DATABRICKS"
                )

                if (
                    postgres_column
                    not in combined_detail.columns
                    or databricks_column
                    not in combined_detail.columns
                ):
                    continue

                postgres_value = normalize_value(
                    row[
                        postgres_column
                    ]
                )

                databricks_value = normalize_value(
                    row[
                        databricks_column
                    ]
                )

                if (
                    postgres_value
                    != databricks_value
                ):

                    detailed_rows.append(
                        {
                            **key_values,

                            "column_name":
                                column,

                            "postgres_value":
                                postgres_value,

                            "databricks_value":
                                databricks_value
                        }
                    )

    detailed_mismatch_df = pd.DataFrame(
        detailed_rows
    )

    return {
        "mode":
            "KEYED_ROW_HASH",

        "summary":
            summary_df,

        "reconciliation":
            reconciliation_df,

        "detailed_mismatches":
            detailed_mismatch_df,

        "postgres_with_hash":
            postgres_hashed,

        "databricks_with_hash":
            databricks_hashed
    }


# ============================================================
# MODE 1B - NON-KEY ROW HASH RECONCILIATION
# ============================================================

def reconcile_without_keys(
    postgres_df,
    databricks_df,
    hash_columns
):
    """
    Compare full result sets by ROW_HASH frequency.

    This works even when no unique key exists.
    """

    postgres_hashed = add_row_hash(
        postgres_df,
        hash_columns
    )

    databricks_hashed = add_row_hash(
        databricks_df,
        hash_columns
    )

    postgres_hash_counts = (
        postgres_hashed
        .groupby(
            "ROW_HASH",
            dropna=False
        )
        .size()
        .reset_index(
            name="POSTGRES_COUNT"
        )
    )

    databricks_hash_counts = (
        databricks_hashed
        .groupby(
            "ROW_HASH",
            dropna=False
        )
        .size()
        .reset_index(
            name="DATABRICKS_COUNT"
        )
    )

    reconciliation_df = pd.merge(
        postgres_hash_counts,
        databricks_hash_counts,
        on="ROW_HASH",
        how="outer"
    )

    reconciliation_df[
        "POSTGRES_COUNT"
    ] = (
        reconciliation_df[
            "POSTGRES_COUNT"
        ]
        .fillna(0)
        .astype(int)
    )

    reconciliation_df[
        "DATABRICKS_COUNT"
    ] = (
        reconciliation_df[
            "DATABRICKS_COUNT"
        ]
        .fillna(0)
        .astype(int)
    )

    reconciliation_df[
        "COUNT_DIFFERENCE"
    ] = (
        reconciliation_df[
            "DATABRICKS_COUNT"
        ]
        - reconciliation_df[
            "POSTGRES_COUNT"
        ]
    )

    def derive_status(row):

        postgres_count = row[
            "POSTGRES_COUNT"
        ]

        databricks_count = row[
            "DATABRICKS_COUNT"
        ]

        if (
            postgres_count
            == databricks_count
        ):
            return "MATCH"

        if (
            postgres_count > 0
            and databricks_count == 0
        ):
            return "MISSING_IN_DATABRICKS"

        if (
            postgres_count == 0
            and databricks_count > 0
        ):
            return "MISSING_IN_POSTGRES"

        return "DUPLICATE_COUNT_MISMATCH"

    reconciliation_df[
        "RECON_STATUS"
    ] = reconciliation_df.apply(
        derive_status,
        axis=1
    )

    summary_df = (
        reconciliation_df
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

    mismatch_hashes = reconciliation_df[
        reconciliation_df[
            "RECON_STATUS"
        ] != "MATCH"
    ][
        "ROW_HASH"
    ]

    postgres_unmatched = postgres_hashed[
        postgres_hashed[
            "ROW_HASH"
        ].isin(
            mismatch_hashes
        )
    ].copy()

    postgres_unmatched[
        "SOURCE_SYSTEM"
    ] = "POSTGRES"

    databricks_unmatched = databricks_hashed[
        databricks_hashed[
            "ROW_HASH"
        ].isin(
            mismatch_hashes
        )
    ].copy()

    databricks_unmatched[
        "SOURCE_SYSTEM"
    ] = "DATABRICKS"

    unmatched_rows_df = pd.concat(
        [
            postgres_unmatched,
            databricks_unmatched
        ],
        ignore_index=True
    )

    return {
        "mode":
            "NON_KEY_ROW_HASH",

        "summary":
            summary_df,

        "reconciliation":
            reconciliation_df,

        "unmatched_rows":
            unmatched_rows_df,

        "postgres_with_hash":
            postgres_hashed,

        "databricks_with_hash":
            databricks_hashed
    }


# ============================================================
# MODE 2 - DIRECT RESULT SET COMPARISON
# ============================================================

def reconcile_result_set_only(
    postgres_df,
    databricks_df,
    comparison_columns
):
    """
    Compare the complete result sets directly without creating ROW_HASH.

    The result set is treated as a multiset:
        identical rows are counted,
        duplicate occurrences are preserved,
        count differences are reported.
    """

    postgres_compare = (
        postgres_df[
            comparison_columns
        ]
        .copy()
    )

    databricks_compare = (
        databricks_df[
            comparison_columns
        ]
        .copy()
    )

    # Normalize every comparison value.
    for column in comparison_columns:

        postgres_compare[
            column
        ] = postgres_compare[
            column
        ].map(
            normalize_value
        )

        databricks_compare[
            column
        ] = databricks_compare[
            column
        ].map(
            normalize_value
        )

    # Group identical rows and compare occurrence count.
    postgres_counts = (
        postgres_compare
        .groupby(
            comparison_columns,
            dropna=False
        )
        .size()
        .reset_index(
            name="POSTGRES_COUNT"
        )
    )

    databricks_counts = (
        databricks_compare
        .groupby(
            comparison_columns,
            dropna=False
        )
        .size()
        .reset_index(
            name="DATABRICKS_COUNT"
        )
    )

    reconciliation_df = pd.merge(
        postgres_counts,
        databricks_counts,
        on=comparison_columns,
        how="outer"
    )

    reconciliation_df[
        "POSTGRES_COUNT"
    ] = (
        reconciliation_df[
            "POSTGRES_COUNT"
        ]
        .fillna(0)
        .astype(int)
    )

    reconciliation_df[
        "DATABRICKS_COUNT"
    ] = (
        reconciliation_df[
            "DATABRICKS_COUNT"
        ]
        .fillna(0)
        .astype(int)
    )

    reconciliation_df[
        "COUNT_DIFFERENCE"
    ] = (
        reconciliation_df[
            "DATABRICKS_COUNT"
        ]
        - reconciliation_df[
            "POSTGRES_COUNT"
        ]
    )

    def derive_status(row):

        postgres_count = row[
            "POSTGRES_COUNT"
        ]

        databricks_count = row[
            "DATABRICKS_COUNT"
        ]

        if (
            postgres_count
            == databricks_count
        ):
            return "MATCH"

        if (
            postgres_count > 0
            and databricks_count == 0
        ):
            return "MISSING_IN_DATABRICKS"

        if (
            postgres_count == 0
            and databricks_count > 0
        ):
            return "MISSING_IN_POSTGRES"

        return "DUPLICATE_COUNT_MISMATCH"

    reconciliation_df[
        "RECON_STATUS"
    ] = reconciliation_df.apply(
        derive_status,
        axis=1
    )

    summary_df = (
        reconciliation_df
        .groupby(
            "RECON_STATUS",
            dropna=False
        )
        .agg(
            result_set_pattern_count=(
                comparison_columns[0],
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

    detailed_difference_df = reconciliation_df[
        reconciliation_df[
            "RECON_STATUS"
        ] != "MATCH"
    ].copy()

    return {
        "mode":
            "DIRECT_RESULT_SET",

        "summary":
            summary_df,

        "reconciliation":
            reconciliation_df,

        "detailed_differences":
            detailed_difference_df
    }


# ============================================================
# MAIN RECONCILIATION ROUTER
# ============================================================

def reconcile(
    postgres_df,
    databricks_df
):
    key_columns = parse_list(
        KEY_COLUMNS
    )

    comparison_columns = get_common_columns(
        postgres_df,
        databricks_df
    )

    (
        postgres_only_columns,
        databricks_only_columns
    ) = get_column_differences(
        postgres_df,
        databricks_df
    )

    print(
        "\nComparison Columns:"
    )

    print(
        comparison_columns
    )

    # --------------------------------------------------------
    # Direct result-set comparison
    # --------------------------------------------------------

    if SUPPRESS_ROW_HASH_CHECK:

        result = reconcile_result_set_only(
            postgres_df,
            databricks_df,
            comparison_columns
        )

    # --------------------------------------------------------
    # Keyed row-hash comparison
    # --------------------------------------------------------

    elif key_columns:

        result = reconcile_with_keys(
            postgres_df,
            databricks_df,
            comparison_columns,
            key_columns
        )

    # --------------------------------------------------------
    # Non-key row-hash comparison
    # --------------------------------------------------------

    else:

        result = reconcile_without_keys(
            postgres_df,
            databricks_df,
            comparison_columns
        )

    result[
        "comparison_columns"
    ] = comparison_columns

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
    "\nRECONCILIATION DETAILS"
)

safe_display(
    reconciliation[
        "reconciliation"
    ]
)


# ------------------------------------------------------------
# Mode-specific detailed output
# ------------------------------------------------------------

if (
    reconciliation[
        "mode"
    ] == "KEYED_ROW_HASH"
):

    print(
        "\nDETAILED COLUMN MISMATCHES"
    )

    safe_display(
        reconciliation[
            "detailed_mismatches"
        ]
    )


elif (
    reconciliation[
        "mode"
    ] == "NON_KEY_ROW_HASH"
):

    print(
        "\nUNMATCHED ROWS"
    )

    safe_display(
        reconciliation[
            "unmatched_rows"
        ]
    )


elif (
    reconciliation[
        "mode"
    ] == "DIRECT_RESULT_SET"
):

    print(
        "\nDIRECT RESULT-SET DIFFERENCES"
    )

    safe_display(
        reconciliation[
            "detailed_differences"
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
