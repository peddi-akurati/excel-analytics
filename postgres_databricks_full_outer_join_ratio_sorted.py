"""
PostgreSQL vs Databricks Full Outer Join Reconciliation
=======================================================

Purpose
-------
1. Execute one SQL query in PostgreSQL.
2. Execute one SQL query in Databricks.
3. Perform a FULL OUTER JOIN using user-defined JOIN_COLUMNS.
4. Compare all other common columns side by side.
5. Show:
   - MATCH
   - MISMATCH
   - MISSING_IN_POSTGRES
   - MISSING_IN_DATABRICKS
6. Show detailed column-level mismatch information.

No row hashing is used anywhere.

Requirements
------------
pip install psycopg2-binary pandas

Designed to run inside Databricks where `spark` and `display()` exist.
"""

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


# Columns used for FULL OUTER JOIN.
#
# Example:
# JOIN_COLUMNS = "claim_id,member_id"
JOIN_COLUMNS = "claim_id,member_id"


# Optional columns excluded from value comparison.
#
# Join columns are automatically excluded from comparison.
#
# Example:
# IGNORE_COLUMNS = "created_ts,updated_ts"
IGNORE_COLUMNS = ""


# ============================================================
# HELPERS
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
    Normalize PostgreSQL and Databricks values for comparison.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
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
        return "true" if value else "false"

    return str(value).strip()


def values_equal(left_value, right_value):
    """
    Null-safe semantic equality.

    Treat numerically equivalent representations as equal, for example:
        4 == 4.0
        4 == Decimal("4.000")
        "4" == "4.0"
        "004.000" == 4

    Falls back to normalized string comparison for non-numeric values.
    """

    left_normalized = normalize_value(
        left_value
    )

    right_normalized = normalize_value(
        right_value
    )

    # Both null.
    if (
        left_normalized is None
        and right_normalized is None
    ):
        return True

    # One null and one non-null.
    if (
        left_normalized is None
        or right_normalized is None
    ):
        return False

    # --------------------------------------------------------
    # Numeric semantic comparison
    # --------------------------------------------------------

    def to_decimal_if_numeric(value):
        try:
            text = str(
                value
            ).strip()

            if text == "":
                return None

            # Explicitly avoid treating booleans as numeric.
            if text.lower() in {
                "true",
                "false"
            }:
                return None

            numeric_value = Decimal(
                text
            )

            if not numeric_value.is_finite():
                return None

            return numeric_value

        except Exception:
            return None

    left_numeric = to_decimal_if_numeric(
        left_normalized
    )

    right_numeric = to_decimal_if_numeric(
        right_normalized
    )

    if (
        left_numeric is not None
        and right_numeric is not None
    ):
        return (
            left_numeric
            == right_numeric
        )

    # --------------------------------------------------------
    # Non-numeric comparison
    # --------------------------------------------------------

    return (
        str(
            left_normalized
        ).strip()
        ==
        str(
            right_normalized
        ).strip()
    )


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

    Date/timestamp fields are cast to string before toPandas()
    to avoid pandas timestamp-range issues.
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
                F.col(field.name)
                .cast("string")
                .alias(field.name)
            )

        else:
            select_expressions.append(
                F.col(field.name)
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
    Display pandas DataFrame safely in Databricks.
    """

    if pandas_df is None or pandas_df.empty:
        print("No rows.")
        return

    safe_df = pandas_df.copy()

    for column in safe_df.columns:
        safe_df[column] = safe_df[column].map(
            lambda value:
                None if value is None else str(value)
        )

    display(
        spark.createDataFrame(
            safe_df
        )
    )



def numeric_ratio(postgres_value, databricks_value):
    """
    Return postgres_value / databricks_value when both values are numeric.

    Returns None when:
    - either value is null/non-numeric
    - Databricks value is zero
    """

    def to_decimal(value):
        normalized = normalize_value(value)

        if normalized is None:
            return None

        try:
            text = str(normalized).strip()

            if text == "":
                return None

            if text.lower() in {
                "true",
                "false"
            }:
                return None

            number = Decimal(text)

            if not number.is_finite():
                return None

            return number

        except Exception:
            return None

    pg_num = to_decimal(postgres_value)
    dbx_num = to_decimal(databricks_value)

    if (
        pg_num is None
        or dbx_num is None
        or dbx_num == 0
    ):
        return None

    try:
        return float(
            pg_num / dbx_num
        )
    except Exception:
        return None


# ============================================================
# VALIDATION
# ============================================================

def validate_inputs(
    postgres_df,
    databricks_df,
    join_columns
):
    if not join_columns:
        raise ValueError(
            "JOIN_COLUMNS must contain at least one column."
        )

    for column in join_columns:

        if column not in postgres_df.columns:
            raise ValueError(
                f"Join column '{column}' "
                f"is missing in PostgreSQL result."
            )

        if column not in databricks_df.columns:
            raise ValueError(
                f"Join column '{column}' "
                f"is missing in Databricks result."
            )


# ============================================================
# RECONCILIATION
# ============================================================

def reconcile(
    postgres_df,
    databricks_df
):
    join_columns = parse_list(
        JOIN_COLUMNS
    )

    ignore_columns = parse_list(
        IGNORE_COLUMNS
    )

    validate_inputs(
        postgres_df,
        databricks_df,
        join_columns
    )

    # --------------------------------------------------------
    # Determine columns
    # --------------------------------------------------------

    common_columns = [
        column
        for column in postgres_df.columns
        if column in databricks_df.columns
    ]

    comparison_columns = [
        column
        for column in common_columns
        if (
            column not in join_columns
            and column not in ignore_columns
        )
    ]

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

    # --------------------------------------------------------
    # Prepare working copies
    # --------------------------------------------------------

    postgres_work = postgres_df.copy()
    databricks_work = databricks_df.copy()

    # Preserve original join values for display.
    for column in join_columns:
        postgres_work[
            f"{column}__POSTGRES_ORIGINAL"
        ] = postgres_work[
            column
        ]

        databricks_work[
            f"{column}__DATABRICKS_ORIGINAL"
        ] = databricks_work[
            column
        ]

    # --------------------------------------------------------
    # Normalize JOIN columns to same datatype
    # --------------------------------------------------------

    def normalize_join_value(value):
        if value is None:
            return "<NULL>"

        try:
            if pd.isna(value):
                return "<NULL>"
        except Exception:
            pass

        # Handle 1.0 vs 1 consistently for numeric-looking keys.
        if isinstance(value, (int, float, Decimal)):
            try:
                numeric_value = Decimal(str(value))

                if numeric_value == numeric_value.to_integral_value():
                    return str(
                        numeric_value.to_integral_value()
                    )

                return format(
                    numeric_value.normalize(),
                    "f"
                )

            except Exception:
                pass

        return str(value).strip()

    for column in join_columns:
        postgres_work[
            column
        ] = postgres_work[
            column
        ].map(
            normalize_join_value
        )

        databricks_work[
            column
        ] = databricks_work[
            column
        ].map(
            normalize_join_value
        )

    # --------------------------------------------------------
    # Add source-presence indicators
    # --------------------------------------------------------

    postgres_work[
        "__POSTGRES_PRESENT"
    ] = True

    databricks_work[
        "__DATABRICKS_PRESENT"
    ] = True

    # --------------------------------------------------------
    # Rename all non-join columns
    # --------------------------------------------------------

    postgres_rename = {
        column:
            f"{column}__POSTGRES"
        for column in postgres_work.columns
        if (
            column not in join_columns
            and column != "__POSTGRES_PRESENT"
        )
    }

    databricks_rename = {
        column:
            f"{column}__DATABRICKS"
        for column in databricks_work.columns
        if (
            column not in join_columns
            and column != "__DATABRICKS_PRESENT"
        )
    }

    postgres_work = postgres_work.rename(
        columns=postgres_rename
    )

    databricks_work = databricks_work.rename(
        columns=databricks_rename
    )

    # --------------------------------------------------------
    # FULL OUTER JOIN
    # --------------------------------------------------------

    joined_df = pd.merge(
        postgres_work,
        databricks_work,
        on=join_columns,
        how="outer"
    )

    joined_df[
        "__POSTGRES_PRESENT"
    ] = joined_df[
        "__POSTGRES_PRESENT"
    ].fillna(False)

    joined_df[
        "__DATABRICKS_PRESENT"
    ] = joined_df[
        "__DATABRICKS_PRESENT"
    ].fillna(False)

    # --------------------------------------------------------
    # Build display-friendly join values
    # --------------------------------------------------------

    for column in join_columns:

        pg_original = (
            f"{column}"
            f"__POSTGRES_ORIGINAL"
            f"__POSTGRES"
        )

        dbx_original = (
            f"{column}"
            f"__DATABRICKS_ORIGINAL"
            f"__DATABRICKS"
        )

        if (
            pg_original in joined_df.columns
            and dbx_original in joined_df.columns
        ):
            joined_df[
                f"{column}__DISPLAY"
            ] = joined_df.apply(
                lambda row:
                    (
                        row[
                            pg_original
                        ]
                        if bool(
                            row[
                                "__POSTGRES_PRESENT"
                            ]
                        )
                        else row[
                            dbx_original
                        ]
                    ),
                axis=1
            )

    # --------------------------------------------------------
    # Row-level comparison
    # --------------------------------------------------------

    row_statuses = []
    mismatch_column_lists = []
    matched_column_counts = []
    mismatched_column_counts = []

    detailed_rows = []

    for _, row in joined_df.iterrows():

        postgres_present = bool(
            row[
                "__POSTGRES_PRESENT"
            ]
        )

        databricks_present = bool(
            row[
                "__DATABRICKS_PRESENT"
            ]
        )

        join_values = {}

        for column in join_columns:

            display_column = (
                f"{column}"
                f"__DISPLAY"
            )

            join_values[
                column
            ] = (
                row[
                    display_column
                ]
                if display_column in joined_df.columns
                else row[
                    column
                ]
            )

        # ----------------------------------------------------
        # Missing rows
        # ----------------------------------------------------

        if (
            postgres_present
            and not databricks_present
        ):

            row_statuses.append(
                "MISSING_IN_DATABRICKS"
            )

            mismatch_column_lists.append(
                None
            )

            matched_column_counts.append(
                0
            )

            mismatched_column_counts.append(
                len(
                    comparison_columns
                )
            )

            detailed_rows.append(
                {
                    **join_values,
                    "SOURCE":
                        "POSTGRES",
                    "column_name":
                        None,
                    "postgres_value":
                        None,
                    "databricks_value":
                        None,
                    "POSTGRES_DIV_DATABRICKS":
                        None,
                    "column_status":
                        None,
                    "row_status":
                        "MISSING_IN_DATABRICKS"
                }
            )

            continue

        if (
            databricks_present
            and not postgres_present
        ):

            row_statuses.append(
                "MISSING_IN_POSTGRES"
            )

            mismatch_column_lists.append(
                None
            )

            matched_column_counts.append(
                0
            )

            mismatched_column_counts.append(
                len(
                    comparison_columns
                )
            )

            detailed_rows.append(
                {
                    **join_values,
                    "SOURCE":
                        "DATABRICKS",
                    "column_name":
                        None,
                    "postgres_value":
                        None,
                    "databricks_value":
                        None,
                    "POSTGRES_DIV_DATABRICKS":
                        None,
                    "column_status":
                        None,
                    "row_status":
                        "MISSING_IN_POSTGRES"
                }
            )

            continue

        # ----------------------------------------------------
        # Compare all common non-key columns
        # ----------------------------------------------------

        mismatched_columns = []
        matched_count = 0

        for column in comparison_columns:

            postgres_column = (
                f"{column}"
                f"__POSTGRES"
            )

            databricks_column = (
                f"{column}"
                f"__DATABRICKS"
            )

            postgres_value = row[
                postgres_column
            ]

            databricks_value = row[
                databricks_column
            ]

            if values_equal(
                postgres_value,
                databricks_value
            ):

                matched_count += 1

            else:

                mismatched_columns.append(
                    column
                )

                detailed_rows.append(
                    {
                        **join_values,
                        "SOURCE":
                            "BOTH",
                        "column_name":
                            column,
                        "postgres_value":
                            normalize_value(
                                postgres_value
                            ),
                        "databricks_value":
                            normalize_value(
                                databricks_value
                            ),
                        "POSTGRES_DIV_DATABRICKS":
                            numeric_ratio(
                                postgres_value,
                                databricks_value
                            ),
                        "column_status":
                            "MISMATCH",
                        "row_status":
                            "MISMATCH"
                    }
                )

        if mismatched_columns:
            row_statuses.append(
                "MISMATCH"
            )
        else:
            row_statuses.append(
                "MATCH"
            )

        mismatch_column_lists.append(
            ",".join(
                mismatched_columns
            )
            if mismatched_columns
            else None
        )

        matched_column_counts.append(
            matched_count
        )

        mismatched_column_counts.append(
            len(
                mismatched_columns
            )
        )

    # --------------------------------------------------------
    # Add row comparison results
    # --------------------------------------------------------

    joined_df[
        "ROW_STATUS"
    ] = row_statuses

    joined_df[
        "MISMATCH_COLUMNS"
    ] = mismatch_column_lists

    joined_df[
        "MATCHED_COLUMN_COUNT"
    ] = matched_column_counts

    joined_df[
        "MISMATCHED_COLUMN_COUNT"
    ] = mismatched_column_counts

    joined_df[
        "SOURCE"
    ] = joined_df.apply(
        lambda row:
            (
                "BOTH"
                if (
                    bool(
                        row[
                            "__POSTGRES_PRESENT"
                        ]
                    )
                    and bool(
                        row[
                            "__DATABRICKS_PRESENT"
                        ]
                    )
                )
                else (
                    "POSTGRES"
                    if bool(
                        row[
                            "__POSTGRES_PRESENT"
                        ]
                    )
                    else "DATABRICKS"
                )
            ),
        axis=1
    )

    # Remove internal presence indicators.
    joined_df = joined_df.drop(
        columns=[
            "__POSTGRES_PRESENT",
            "__DATABRICKS_PRESENT"
        ]
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    summary_df = (
        joined_df
        .groupby(
            "ROW_STATUS",
            dropna=False
        )
        .size()
        .reset_index(
            name="row_count"
        )
    )

    detailed_df = pd.DataFrame(
        detailed_rows
    )

    if (
        detailed_df.empty
        or "column_status"
        not in detailed_df.columns
    ):
        column_summary_df = pd.DataFrame(
            columns=[
                "column_name",
                "mismatch_count"
            ]
        )
    else:
        mismatch_only = detailed_df[
            detailed_df[
                "column_status"
            ] == "MISMATCH"
        ]

        if mismatch_only.empty:
            column_summary_df = pd.DataFrame(
                columns=[
                    "column_name",
                    "mismatch_count"
                ]
            )
        else:
            column_summary_df = (
                mismatch_only
                .groupby(
                    "column_name",
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
            )

    total_summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "POSTGRES_RESULT_ROWS",
                "value":
                    len(
                        postgres_df
                    )
            },
            {
                "metric":
                    "DATABRICKS_RESULT_ROWS",
                "value":
                    len(
                        databricks_df
                    )
            },
            {
                "metric":
                    "JOINED_ROWS",
                "value":
                    len(
                        joined_df
                    )
            },
            {
                "metric":
                    "MATCH_ROWS",
                "value":
                    int(
                        (
                            joined_df[
                                "ROW_STATUS"
                            ]
                            == "MATCH"
                        ).sum()
                    )
            },
            {
                "metric":
                    "MISMATCH_ROWS",
                "value":
                    int(
                        (
                            joined_df[
                                "ROW_STATUS"
                            ]
                            == "MISMATCH"
                        ).sum()
                    )
            },
            {
                "metric":
                    "MISSING_IN_POSTGRES",
                "value":
                    int(
                        (
                            joined_df[
                                "ROW_STATUS"
                            ]
                            == "MISSING_IN_POSTGRES"
                        ).sum()
                    )
            },
            {
                "metric":
                    "MISSING_IN_DATABRICKS",
                "value":
                    int(
                        (
                            joined_df[
                                "ROW_STATUS"
                            ]
                            == "MISSING_IN_DATABRICKS"
                        ).sum()
                    )
            }
        ]
    )


    # --------------------------------------------------------
    # Ratio-sorted detailed mismatch datasets
    # --------------------------------------------------------

    if (
        detailed_df.empty
        or "POSTGRES_DIV_DATABRICKS"
        not in detailed_df.columns
    ):
        detailed_ratio_ascending_df = pd.DataFrame()
        detailed_ratio_descending_df = pd.DataFrame()

    else:
        ratio_only = detailed_df[
            detailed_df[
                "POSTGRES_DIV_DATABRICKS"
            ].notna()
        ].copy()

        detailed_ratio_ascending_df = (
            ratio_only
            .sort_values(
                "POSTGRES_DIV_DATABRICKS",
                ascending=True
            )
            .head(1000)
        )

        detailed_ratio_descending_df = (
            ratio_only
            .sort_values(
                "POSTGRES_DIV_DATABRICKS",
                ascending=False
            )
            .head(1000)
        )

    return {
        "join_columns":
            join_columns,

        "comparison_columns":
            comparison_columns,

        "postgres_only_columns":
            postgres_only_columns,

        "databricks_only_columns":
            databricks_only_columns,

        "total_summary":
            total_summary_df,

        "status_summary":
            summary_df,

        "column_mismatch_summary":
            column_summary_df,

        "full_outer_join_result":
            joined_df,

        "detailed_mismatches":
            detailed_df,

        "detailed_mismatches_ratio_ascending":
            detailed_ratio_ascending_df,

        "detailed_mismatches_ratio_descending":
            detailed_ratio_descending_df
    }


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
    f"PostgreSQL result rows: "
    f"{len(postgres_result):,}"
)


print(
    "\nExecuting Databricks SQL..."
)

databricks_result = databricks_query(
    DATABRICKS_SQL
)

print(
    f"Databricks result rows: "
    f"{len(databricks_result):,}"
)


print(
    "\nRunning FULL OUTER JOIN reconciliation..."
)

reconciliation = reconcile(
    postgres_result,
    databricks_result
)


# ============================================================
# DISPLAY OUTPUT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "OVERALL SUMMARY"
)

print(
    "=" * 80
)

safe_display(
    reconciliation[
        "total_summary"
    ]
)


print(
    "\nROW STATUS SUMMARY"
)

safe_display(
    reconciliation[
        "status_summary"
    ]
)


print(
    "\nCOLUMN MISMATCH SUMMARY"
)

safe_display(
    reconciliation[
        "column_mismatch_summary"
    ]
)


print(
    "\nFULL OUTER JOIN COMPARISON"
)

safe_display(
    reconciliation[
        "full_outer_join_result"
    ]
)


print(
    "\nDETAILED COLUMN MISMATCHES"
)

safe_display(
    reconciliation[
        "detailed_mismatches"
    ]
)



print(
    "\nDETAILED MISMATCHES - TOP 1000 BY POSTGRES/DATABRICKS RATIO ASCENDING"
)

safe_display(
    reconciliation[
        "detailed_mismatches_ratio_ascending"
    ]
)


print(
    "\nDETAILED MISMATCHES - TOP 1000 BY POSTGRES/DATABRICKS RATIO DESCENDING"
)

safe_display(
    reconciliation[
        "detailed_mismatches_ratio_descending"
    ]
)


print(
    "\nJoin columns:"
)

print(
    reconciliation[
        "join_columns"
    ]
)


print(
    "\nComparison columns:"
)

print(
    reconciliation[
        "comparison_columns"
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
