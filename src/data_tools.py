import pandas as pd


def inspect_dataset(file_path):
    df = pd.read_csv(file_path)
    profile = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "missing_values": df.isna().sum().to_dict(),
        "column_types": df.dtypes.apply(lambda x: str(x)).to_dict(),
    }
    return df, profile


def infer_target(df):
    # Simple heuristic: column with few unique values (<20%) and numeric or
    # categorical
    candidates = []
    for col in df.columns:
        n_unique = df[col].nunique()
        unique_ratio = n_unique / df.shape[0]

        # Strong Signal: Binary Target
        if n_unique == 2:
            candidates.append((col, "binary"))
        # Weak Signal: Low-cardinality numeric
        if unique_ratio < 0.05 and df[col].dtype != "object":
            candidates.append((col, "low_cardinality"))

    # Prefer Binary Targets
    for col, reason in candidates:
        if reason == "binary":
            return col

    # Fallback
    return candidates[0][0] if candidates else None
