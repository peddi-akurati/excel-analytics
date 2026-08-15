"""
PostgreSQL Table Validation Utility

Supported checks:
    Counts_Check
    Type_Check
    Sample_Check
    All_Check

Requirements:
    pip install psycopg2-binary pandas
"""

import hashlib
import json
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

# Allowed:
# Counts_Check, Type_Check, Sample_Check, All_Check
CHECK_FLAG = "All_Check"

# Sample Check configuration
UNIQUE_KEY_COLUMNS = "policy_id,claim_id"
DATE_COLUMN = "intimation_date"
IGNORE_COLUMNS = "created_timestamp,updated_timestamp"
SAMPLE_SIZE = 1000


# ============================================================
# HELPERS
# ============================================================

def parse_column_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).split(",") if x.strip()]


def get_connection():
    return psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )


def validate_table_exists(conn, schema_name, table_name):
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
        );
    """
    with conn.cursor() as cursor:
        cursor.execute(query, (schema_name, table_name))
        exists = cursor.fetchone()[0]

    if not exists:
        raise ValueError(f"Table {schema_name}.{table_name} does not exist.")


def get_table_columns(conn, schema_name, table_name):
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
    """
    df = pd.read_sql_query(
        query, conn, params=(schema_name, table_name)
    )
    return df["column_name"].tolist()


# ============================================================
# COUNTS CHECK
# ============================================================

def counts_check(conn, schema_name, table_name):
    query = sql.SQL("""
        SELECT COUNT(*) AS total_count
        FROM {}.{}
    """).format(
        sql.Identifier(schema_name),
        sql.Identifier(table_name)
    )

    with conn.cursor() as cursor:
        cursor.execute(query)
        count = cursor.fetchone()[0]

    result_df = pd.DataFrame([{
        "database": DB_CONFIG["database"],
        "schema": schema_name,
        "table": table_name,
        "total_count": count
    }])

    print("\n" + "=" * 80)
    print("COUNTS CHECK")
    print("=" * 80)
    print(result_df.to_string(index=False))

    return result_df


# ============================================================
# TYPE CHECK
# ============================================================

def type_check(conn, schema_name, table_name):
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
        params=(schema_name, table_name)
    )

    print("\n" + "=" * 80)
    print("TYPE CHECK")
    print("=" * 80)
    print(result_df.to_string(index=False))

    return result_df


# ============================================================
# HASH NORMALIZATION
# ============================================================

def normalize_value(value):
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
        return "true" if value else "false"

    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)

    return str(value)


def generate_row_hash(row, hash_columns):
    components = []
    for column in hash_columns:
        value = normalize_value(row[column])
        components.append(f"{column}={value}")

    hash_string = "||".join(components)
    return hashlib.sha256(hash_string.encode("utf-8")).hexdigest()


# ============================================================
# SAMPLE CHECK
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
    key_columns = parse_column_list(unique_key_columns)
    ignore_columns = parse_column_list(ignore_columns)

    table_columns = get_table_columns(conn, schema_name, table_name)

    if not table_columns:
        raise ValueError(f"No columns found for {schema_name}.{table_name}")

    missing_keys = [x for x in key_columns if x not in table_columns]
    if missing_keys:
        raise ValueError(
            "Unique key columns not found in table: " + ", ".join(missing_keys)
        )

    if not date_column:
        raise ValueError("DATE_COLUMN must be supplied for Sample_Check.")

    if date_column not in table_columns:
        raise ValueError(
            f"Date column '{date_column}' does not exist in "
            f"{schema_name}.{table_name}"
        )

    invalid_ignore_columns = [
        x for x in ignore_columns if x not in table_columns
    ]

    if invalid_ignore_columns:
        print(
            "\nWARNING: Ignore columns not present in table: "
            + ", ".join(invalid_ignore_columns)
        )

    ignore_columns = [
        x for x in ignore_columns if x in table_columns
    ]

    hash_columns = [
        column for column in table_columns
        if column not in ignore_columns
    ]

    if not hash_columns:
        raise ValueError(
            "No columns available for hashing after applying ignore list."
        )

    # Latest records by date. Keys provide deterministic ordering
    # when multiple records have the same date/timestamp.
    order_columns = [
        sql.SQL("{} DESC NULLS LAST").format(sql.Identifier(date_column))
    ]

    for key in key_columns:
        if key != date_column:
            order_columns.append(
                sql.SQL("{} ASC NULLS LAST").format(sql.Identifier(key))
            )

    query = sql.SQL("""
        SELECT *
        FROM {}.{}
        ORDER BY {}
        LIMIT %s
    """).format(
        sql.Identifier(schema_name),
        sql.Identifier(table_name),
        sql.SQL(", ").join(order_columns)
    )

    print("\n" + "=" * 80)
    print("SAMPLE CHECK")
    print("=" * 80)
    print(f"Table        : {schema_name}.{table_name}")
    print(f"Sample Size  : {sample_size:,}")
    print(f"Date Column  : {date_column}")
    print(f"Unique Keys  : {key_columns}")
    print(f"Ignore Cols  : {ignore_columns}")
    print(f"Hash Columns : {len(hash_columns)}")

    sample_df = pd.read_sql_query(
        query.as_string(conn),
        conn,
        params=(sample_size,)
    )

    if sample_df.empty:
        print("\nNo records found.")
        return sample_df

    sample_df["ROW_HASH"] = sample_df.apply(
        lambda row: generate_row_hash(row, hash_columns),
        axis=1
    )

    first_columns = []

    for column in key_columns:
        if column in sample_df.columns:
            first_columns.append(column)

    if (
        date_column in sample_df.columns
        and date_column not in first_columns
    ):
        first_columns.append(date_column)

    first_columns.append("ROW_HASH")

    remaining_columns = [
        x for x in sample_df.columns
        if x not in first_columns
    ]

    sample_df = sample_df[first_columns + remaining_columns]

    print("\nSample Result:")
    print(sample_df.to_string(index=False))

    return sample_df


# ============================================================
# MAIN
# ============================================================

def run_checks():
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
            "Allowed: Counts_Check, Type_Check, Sample_Check, All_Check"
        )

    conn = None

    try:
        conn = get_connection()

        print("\nSuccessfully connected to PostgreSQL.")
        print(f"Database : {DB_CONFIG['database']}")
        print(f"Table    : {SCHEMA_NAME}.{TABLE_NAME}")
        print(f"Check    : {CHECK_FLAG}")

        validate_table_exists(conn, SCHEMA_NAME, TABLE_NAME)

        results = {}

        if check_flag in {"counts_check", "all_check"}:
            results["Counts_Check"] = counts_check(
                conn, SCHEMA_NAME, TABLE_NAME
            )

        if check_flag in {"type_check", "all_check"}:
            results["Type_Check"] = type_check(
                conn, SCHEMA_NAME, TABLE_NAME
            )

        if check_flag in {"sample_check", "all_check"}:
            results["Sample_Check"] = sample_check(
                conn=conn,
                schema_name=SCHEMA_NAME,
                table_name=TABLE_NAME,
                unique_key_columns=UNIQUE_KEY_COLUMNS,
                date_column=DATE_COLUMN,
                ignore_columns=IGNORE_COLUMNS,
                sample_size=SAMPLE_SIZE
            )

        print("\n" + "=" * 80)
        print("VALIDATION COMPLETED SUCCESSFULLY")
        print("=" * 80)

        return results

    except Exception as error:
        print("\n" + "=" * 80)
        print("VALIDATION FAILED")
        print("=" * 80)
        print(f"Error: {error}")
        raise

    finally:
        if conn is not None:
            conn.close()
            print("\nPostgreSQL connection closed.")


if __name__ == "__main__":
    validation_results = run_checks()
