from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import numpy as np


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
