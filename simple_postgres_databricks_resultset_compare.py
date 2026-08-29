
# Configuration
SUPPRESS_ROW_HASH_CHECK = False   # True = Result set comparison only
KEY_COLUMNS = ""
IGNORE_COLUMNS = ""

# ... keep existing connection and SQL configuration ...

def reconcile_result_set_only(postgres_df, databricks_df):
    ignore_cols = [c.strip() for c in IGNORE_COLUMNS.split(",") if c.strip()]
    common_cols = [c for c in postgres_df.columns if c in databricks_df.columns and c not in ignore_cols]

    pg = postgres_df[common_cols].fillna("<NULL>").astype(str)
    dbx = databricks_df[common_cols].fillna("<NULL>").astype(str)

    # Row signature from complete result set (no hashing)
    pg["_signature"] = pg.apply(lambda r: "||".join(r.values), axis=1)
    dbx["_signature"] = dbx.apply(lambda r: "||".join(r.values), axis=1)

    pg_only = pg.loc[~pg["_signature"].isin(dbx["_signature"])].copy()
    dbx_only = dbx.loc[~dbx["_signature"].isin(pg["_signature"])].copy()

    pg_only["SOURCE_SYSTEM"] = "POSTGRES"
    dbx_only["SOURCE_SYSTEM"] = "DATABRICKS"

    details = pd.concat([pg_only, dbx_only], ignore_index=True).drop(columns=["_signature"])

    summary = pd.DataFrame({
        "Metric": [
            "Postgres Rows",
            "Databricks Rows",
            "Matched Rows",
            "Postgres Only Rows",
            "Databricks Only Rows"
        ],
        "Value": [
            len(pg),
            len(dbx),
            len(pg) - len(pg_only),
            len(pg_only),
            len(dbx_only)
        ]
    })

    return summary, details

# Main execution
if SUPPRESS_ROW_HASH_CHECK:
    summary_df, detail_df = reconcile_result_set_only(postgres_result, databricks_result)

    print("RESULT SET COMPARISON SUMMARY")
    display(spark.createDataFrame(summary_df.astype(str)))

    print("UNMATCHED RESULT SET RECORDS")
    display(spark.createDataFrame(detail_df.astype(str)))

else:
    # Existing row-hash reconciliation logic continues unchanged.
    reconciliation = reconcile(postgres_result, databricks_result)
