import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
)


def evaluate_classification_model(model, X_val, y_val):
    y_probs = model.predict_proba(X_val)[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_val, y_probs),
        "f1": f1_score(y_val, y_preds),
        "confusion_matrix": confusion_matrix(y_val, y_preds).tolist(),
        "positive_rate": float(np.mean(y_val)),
    }
    return metrics
