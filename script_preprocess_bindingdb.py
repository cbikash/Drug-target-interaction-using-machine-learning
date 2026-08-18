from src.data.preprocessing import preprocessing_bindingdb
from src.data.loading import load_bindingdb
import pandas as pd

df_chunks = load_bindingdb('data/raw/BindingDB_All.tsv')

cleaned_df = []

total_report = {
    "initial_rows": 0,
    "missing_required": 0,
    "invalid_or_censored_ki": 0,
    "non_positive_ki": 0,
    "invalid_smiles": 0,
    "invalid_protein": 0,
    "duplicates_removed_within_chunks": 0,
    "final_rows": 0,
    "total_duplicate": 0
}

for chunk_number, df in enumerate(df_chunks, start=1):
    print(
        f"Processing chunk {chunk_number} "
        f"({len(df):,} rows)"
    )

    clean_df, report = preprocessing_bindingdb(
        dataframe=df,
        smiles_column="Ligand SMILES",
        protein_column="BindingDB Target Chain Sequence 1",
        ki_column="Ki (nM)",
    )

    cleaned_df.append(clean_df)

    
    # Aggregate preprocessing statistics
    total_report[
        "initial_rows"
    ] += report.initial_rows

    total_report[
        "missing_required"
    ] += report.missing_required

    total_report[
        "invalid_or_censored_ki"
    ] += report.invalid_or_censored_ki

    total_report[
        "non_positive_ki"
    ] += report.non_positive_ki

    total_report[
        "invalid_smiles"
    ] += report.invalid_smiles

    total_report[
        "invalid_protein"
    ] += report.invalid_protein

    total_report[
        "duplicates_removed_within_chunks"
    ] += report.duplicates_removed

    total_report[
        "final_rows"
    ] += report.final_rows

# Combine cleaned chunks
final_df = pd.concat(
    cleaned_df,
    ignore_index=True,
)

# Remove duplicates across chunks

rows_before_deduplication = len(final_df)
final_df = final_df.drop_duplicates(
    subset=[
        "canonical_smiles",
        "sequence",
        "ki_nm",
    ],
    ignore_index=True,
)

duplicates_removed_across_chunks = (
    rows_before_deduplication
    - len(final_df)
)

# Update preprocessing report
total_report[
    "total_duplicates_removed"
] = (
    total_report.get(
        "duplicates_removed_within_chunks",
        0,
    )
    + duplicates_removed_across_chunks
)

total_report[
    "final_rows"
] = len(final_df)

# Save processed dataset
final_df.to_csv("data/processed/processed_bindingdb.csv")

report_df = pd.DataFrame(
    [total_report]
)

report_df.to_csv(
    "artifacts/metrics/preprocessing_report.csv",
    index=False,
)

print(report_df)