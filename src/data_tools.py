import pandas as pd


def inspect_dataset(file_path):
    df = pd.read_csv(file_path)
    profile = {
            "n_rows": df.shape[0],
            "n_columns": df.shape[1],
            "missing_values": df.isna().sum.to_dict(),
            "column_types": df.dtypes.apply(lambda x: str(x)).to_dict()
            }
    return df, profile
