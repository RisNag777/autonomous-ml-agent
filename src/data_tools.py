import pandas as pd


def inspect_dataset(file_path):
    df = pd.read_csv(file_path)
    profile = {
            "n_rows": df.shape[0],
            "n_columns": df.shape[1],
            "missing_values": df.isna().sum().to_dict(),
            "column_types": df.dtypes.apply(lambda x: str(x)).to_dict()
            }
    return df, profile


def infer_target(df):
    # Simple heuristic: column with few unique values (<20%) and numeric or
    # categorical
    candidates = []
    for col in df.columns:
        unique_ratio = df[col].nunique() / df.shape[0]
        if unique_ratio < 0.2 and df[col].dtype != "object":
            candidates.append(col)
    # Return the first candidate as default
    return candidates[0] if candidates else None
