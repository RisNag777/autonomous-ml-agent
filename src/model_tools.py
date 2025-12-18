from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def train_baseline_model(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    model.fit(X_train, y_train)
    return model, X_val, y_val
