"""
PostgreSQL Table Validation Utility
===================================

Supported checks:
    Counts_Check
    Type_Check
    Sample_Check
    All_Check

Each enabled check is displayed as a pandas DataFrame
and written to a separate CSV file.

Requirements:
    pip install psycopg2-binary pandas
"""

import hashlib
import json
import os
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2 import sql


# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "your-postgres-host",
    "port": 5432,
    "database": "your_database",
    "user": "your_username",
    "password": "your_password"
}

SCHEMA_NAME = "public"
TABLE_NAME = "your_table"

# Allowed values:
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

# CSV output configuration
OUTPUT_DIR = "postgres_validation_outputs"

# True  -> output filename contains timestamp
# False -> same file gets overwritten on the next run
INCLUDE_TIMESTAMP_IN_FILENAME = True


# ============================================================
# GENERAL HELPERS
# ============================================================

def parse_column_list(value):
    """Convert a comma-separated string into a clean Python list."""

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


def get_connection():
    """Create PostgreSQL database connection."""

    return psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )


def validate_table_exists(conn, schema_name, table_name):
    """Validate that the requested table exists."""

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
            (schema_name, table_name)
        )
        exists = cursor.fetchone()[0]

    if not exists:
        raise ValueError(
            f"Table {schema_name}.{table_name} does not exist."
        )


def get_table_columns(conn, schema_name, table_name):
    """Return all table columns in ordinal order."""

    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(schema_name, table_name)
    )

    return df["column_name"].tolist()


# ============================================================
# CSV DOWNLOAD / EXPORT HELPERS
# ============================================================

def ensure_output_directory():
    """Create the CSV output directory if it does not exist."""

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


def sanitize_filename_part(value):
    """Convert a string into a filename-safe value."""

    value = str(value).strip()

    return "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in value
    )


def build_output_file_path(
    check_name,
    schema_name,
    table_name
):
    """
    Build a unique output CSV filename.

    Example:
      postgres_validation_outputs/
      mydb_public_claims_Counts_Check_20260815_105500.csv
    """

    ensure_output_directory()

    database_part = sanitize_filename_part(
        DB_CONFIG["database"]
    )

    schema_part = sanitize_filename_part(
        schema_name
    )

    table_part = sanitize_filename_part(
        table_name
    )

    check_part = sanitize_filename_part(
        check_name
    )

    timestamp_part = ""

    if INCLUDE_TIMESTAMP_IN_FILENAME:
        timestamp_part = (
            "_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

    filename = (
        f"{database_part}_"
        f"{schema_part}_"
        f"{table_part}_"
        f"{check_part}"
        f"{timestamp_part}.csv"
    )

    return os.path.join(
        OUTPUT_DIR,
        filename
    )


def save_dataframe_to_csv(
    dataframe,
    check_name,
    schema_name,
    table_name
):
    """
    Write a check result DataFrame to CSV.

    Returns:
        Full/relative path of generated CSV file.
    """

    output_file = build_output_file_path(
        check_name,
        schema_name,
        table_name
    )

    dataframe.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nCSV generated : {output_file}"
    )

    return output_file


# ============================================================
# 1. COUNTS CHECK
# ============================================================

def counts_check(
    conn,
    schema_name,
    table_name
):
    """Get total row count and save result to CSV."""

    query = sql.SQL("""
        SELECT COUNT(*) AS total_count
        FROM {}.{}
    """).format(
        sql.Identifier(schema_name),
        sql.Identifier(table_name)
    )

    with conn.cursor() as cursor:
        cursor.execute(query)
        total_count = cursor.fetchone()[0]

    result_df = pd.DataFrame([
        {
            "database": DB_CONFIG["database"],
            "schema": schema_name,
            "table": table_name,
            "total_count": total_count
        }
    ])

    print("\n" + "=" * 80)
    print("COUNTS CHECK")
    print("=" * 80)
    print(
        result_df.to_string(
            index=False
        )
    )

    csv_file = save_dataframe_to_csv(
        result_df,
        "Counts_Check",
        schema_name,
        table_name
    )

    return result_df, csv_file


# ============================================================
# 2. TYPE CHECK
# ============================================================

def type_check(
    conn,
    schema_name,
    table_name
):
    """Get column metadata and save result to CSV."""

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
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
    """

    result_df = pd.read_sql_query(
        query,
        conn,
        params=(
            schema_name,
            table_name
        )
    )

    print("\n" + "=" * 80)
    print("TYPE CHECK")
    print("=" * 80)
    print(
        result_df.to_string(
            index=False
        )
    )

    csv_file = save_dataframe_to_csv(
        result_df,
        "Type_Check",
        schema_name,
        table_name
    )

    return result_df, csv_file


# ============================================================
# HASH HELPERS
# ============================================================

def normalize_value(value):
    """Convert values into deterministic representations."""

    if pd.isna(value):
        return "<NULL>"

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

    if isinstance(
        value,
        (dict, list)
    ):
        return json.dumps(
            value,
            sort_keys=True,
            default=str
        )

    return str(value)


def generate_row_hash(
    row,
    hash_columns
):
    """Create SHA-256 hash for one row."""

    hash_components = []

    for column in hash_columns:

        normalized_value = normalize_value(
            row[column]
        )

        hash_components.append(
            f"{column}={normalized_value}"
        )

    hash_input = "||".join(
        hash_components
    )

    return hashlib.sha256(
        hash_input.encode("utf-8")
    ).hexdigest()


# ============================================================
# 3. SAMPLE CHECK
# ============================================================

def sample_check(
    conn,
    schema_name,
    table_name,
    unique_key_columns,
    date_column,
    ignore_columns,
    sample_size=1000
):
    """
    Pick latest N records using date column,
    calculate row hash, display result, and save CSV.
    """

    key_columns = parse_column_list(
        unique_key_columns
    )

    ignore_columns = parse_column_list(
        ignore_columns
    )

    table_columns = get_table_columns(
        conn,
        schema_name,
        table_name
    )

    if not table_columns:
        raise ValueError(
            f"No columns found for "
            f"{schema_name}.{table_name}"
        )

    # Validate unique key columns.
    missing_key_columns = [
        column
        for column in key_columns
        if column not in table_columns
    ]

    if missing_key_columns:
        raise ValueError(
            "Unique key columns not found: "
            + ", ".join(
                missing_key_columns
            )
        )

    # Validate date column.
    if not date_column:
        raise ValueError(
            "DATE_COLUMN must be supplied "
            "for Sample_Check."
        )

    if date_column not in table_columns:
        raise ValueError(
            f"Date column '{date_column}' "
            f"does not exist in "
            f"{schema_name}.{table_name}"
        )

    # Ignore invalid ignore-columns with warning.
    missing_ignore_columns = [
        column
        for column in ignore_columns
        if column not in table_columns
    ]

    if missing_ignore_columns:
        print(
            "\nWARNING - Ignore columns "
            "not found in the table:"
        )
        print(
            ", ".join(
                missing_ignore_columns
            )
        )

    valid_ignore_columns = [
        column
        for column in ignore_columns
        if column in table_columns
    ]

    # All table columns except ignored ones participate
    # in hash generation.
    hash_columns = [
        column
        for column in table_columns
        if column not in valid_ignore_columns
    ]

    if not hash_columns:
        raise ValueError(
            "No columns remain for row hashing "
            "after applying IGNORE_COLUMNS."
        )

    # Build deterministic ordering:
    # latest records by date, then unique key columns.
    order_by_parts = [
        sql.SQL(
            "{} DESC NULLS LAST"
        ).format(
            sql.Identifier(
                date_column
            )
        )
    ]

    for key_column in key_columns:

        if key_column != date_column:

            order_by_parts.append(
                sql.SQL(
                    "{} ASC NULLS LAST"
                ).format(
                    sql.Identifier(
                        key_column
                    )
                )
            )

    sample_query = sql.SQL("""
        SELECT *
        FROM {}.{}
        ORDER BY {}
        LIMIT %s
    """).format(
        sql.Identifier(
            schema_name
        ),
        sql.Identifier(
            table_name
        ),
        sql.SQL(", ").join(
            order_by_parts
        )
    )

    print("\n" + "=" * 80)
    print("SAMPLE CHECK")
    print("=" * 80)

    print(
        f"Table         : "
        f"{schema_name}.{table_name}"
    )
    print(
        f"Sample Size   : "
        f"{sample_size:,}"
    )
    print(
        f"Date Column   : "
        f"{date_column}"
    )
    print(
        f"Unique Keys   : "
        f"{key_columns}"
    )
    print(
        f"Ignored Cols  : "
        f"{valid_ignore_columns}"
    )
    print(
        f"Hash Columns  : "
        f"{len(hash_columns)}"
    )

    sample_df = pd.read_sql_query(
        sample_query.as_string(conn),
        conn,
        params=(sample_size,)
    )

    # Save an empty result as CSV as well.
    if sample_df.empty:

        print(
            "\nNo records found."
        )

        csv_file = save_dataframe_to_csv(
            sample_df,
            "Sample_Check",
            schema_name,
            table_name
        )

        return sample_df, csv_file

    # Generate hash.
    sample_df["ROW_HASH"] = sample_df.apply(
        lambda row: generate_row_hash(
            row,
            hash_columns
        ),
        axis=1
    )

    # Put keys, date, and hash first in output.
    first_columns = []

    for key_column in key_columns:

        if (
            key_column
            in sample_df.columns
            and key_column
            not in first_columns
        ):
            first_columns.append(
                key_column
            )

    if (
        date_column
        in sample_df.columns
        and date_column
        not in first_columns
    ):
        first_columns.append(
            date_column
        )

    first_columns.append(
        "ROW_HASH"
    )

    remaining_columns = [
        column
        for column in sample_df.columns
        if column not in first_columns
    ]

    sample_df = sample_df[
        first_columns
        + remaining_columns
    ]

    print("\nSample Result:")
    print(
        sample_df.to_string(
            index=False
        )
    )

    csv_file = save_dataframe_to_csv(
        sample_df,
        "Sample_Check",
        schema_name,
        table_name
    )

    return sample_df, csv_file


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_checks():

    valid_flags = {
        "counts_check",
        "type_check",
        "sample_check",
        "all_check"
    }

    check_flag = (
        CHECK_FLAG
        .strip()
        .lower()
    )

    if check_flag not in valid_flags:
        raise ValueError(
            f"Invalid CHECK_FLAG: {CHECK_FLAG}. "
            "Allowed values are: "
            "Counts_Check, Type_Check, "
            "Sample_Check, All_Check"
        )

    connection = None

    try:

        connection = get_connection()

        print(
            "\nSuccessfully connected "
            "to PostgreSQL."
        )

        print(
            f"Database    : "
            f"{DB_CONFIG['database']}"
        )

        print(
            f"Table       : "
            f"{SCHEMA_NAME}.{TABLE_NAME}"
        )

        print(
            f"Check       : "
            f"{CHECK_FLAG}"
        )

        print(
            f"CSV Folder  : "
            f"{OUTPUT_DIR}"
        )

        validate_table_exists(
            connection,
            SCHEMA_NAME,
            TABLE_NAME
        )

        results = {}
        generated_files = {}

        # ----------------------------------------------------
        # COUNTS CHECK
        # ----------------------------------------------------

        if check_flag in {
            "counts_check",
            "all_check"
        }:

            result_df, csv_file = counts_check(
                connection,
                SCHEMA_NAME,
                TABLE_NAME
            )

            results[
                "Counts_Check"
            ] = result_df

            generated_files[
                "Counts_Check"
            ] = csv_file

        # ----------------------------------------------------
        # TYPE CHECK
        # ----------------------------------------------------

        if check_flag in {
            "type_check",
            "all_check"
        }:

            result_df, csv_file = type_check(
                connection,
                SCHEMA_NAME,
                TABLE_NAME
            )

            results[
                "Type_Check"
            ] = result_df

            generated_files[
                "Type_Check"
            ] = csv_file

        # ----------------------------------------------------
        # SAMPLE CHECK
        # ----------------------------------------------------

        if check_flag in {
            "sample_check",
            "all_check"
        }:

            result_df, csv_file = sample_check(
                conn=connection,
                schema_name=SCHEMA_NAME,
                table_name=TABLE_NAME,
                unique_key_columns=UNIQUE_KEY_COLUMNS,
                date_column=DATE_COLUMN,
                ignore_columns=IGNORE_COLUMNS,
                sample_size=SAMPLE_SIZE
            )

            results[
                "Sample_Check"
            ] = result_df

            generated_files[
                "Sample_Check"
            ] = csv_file

        print("\n" + "=" * 80)
        print("VALIDATION COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print("\nGenerated CSV files:")

        for check_name, file_path in generated_files.items():
            print(
                f"  {check_name}: "
                f"{file_path}"
            )

        return {
            "results": results,
            "csv_files": generated_files
        }

    except Exception as error:

        print("\n" + "=" * 80)
        print("VALIDATION FAILED")
        print("=" * 80)

        print(
            f"Error: {error}"
        )

        raise

    finally:

        if connection is not None:

            connection.close()

            print(
                "\nPostgreSQL connection closed."
            )


# ============================================================
# START SCRIPT
# ============================================================

if __name__ == "__main__":

    validation_output = run_checks()
