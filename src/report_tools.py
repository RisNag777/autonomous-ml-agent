import joblib
import json
import os


def save_artifacts(model, metrics, iteration):
    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/metrics", exist_ok=True)

    model_path = f"artifacts/models/model_iter_{iteration}.pkl"
    metrics_path = f"artifacts/metrics/metrics_iter_{iteration}.json"

    joblib.dump(model, model_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return model_path, metrics_path
