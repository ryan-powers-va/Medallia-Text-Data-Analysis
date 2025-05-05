import pandas as pd

# ── EDIT THESE TWO LINES ONLY ───────────────────────────────
INPUT_FILE  = "checkpoints/secondary_tagging/secondary_checkpoint_64.csv"
OUTPUT_FILE = "training_data/A11_output_processed.xlsx"
# ────────────────────────────────────────────────────────────

EXCLUDE_FAMILIES = {"Benefits (Non-Health Care)"}
THRESH = 5  # collapse if count ≤ THRESH

def collapse_sparse_tags(df):
    counts = (
        df.groupby(["Secondary_Tag_Family", "Secondary_Tag"])
          .size()
          .rename("cnt")
          .reset_index()
    )

    sparse = (
        (counts["cnt"] <= THRESH) &
        ~counts["Secondary_Tag_Family"].isin(EXCLUDE_FAMILIES)
    )
    sparse_set = set(
        tuple(x) for x in counts.loc[sparse, ["Secondary_Tag_Family", "Secondary_Tag"]].values
    )

    def _fix(row):
        key = (row["Secondary_Tag_Family"], row["Secondary_Tag"])
        if key in sparse_set:
            return f"Other {row['Secondary_Tag_Family']}"
        return row["Secondary_Tag"]

    df["Secondary_Tag"] = df.apply(_fix, axis=1)
    return df


if __name__ == "__main__":
    # Read CSV instead of Excel
    df = pd.read_csv(INPUT_FILE)
    df = collapse_sparse_tags(df)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved collapsed file → {OUTPUT_FILE}")
    print("Done! 🎉")