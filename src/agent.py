from src.evaluation_tools import evaluate_classification_model
from src.model_tools import train_baseline_model
from src.report_tools import save_artifacts


class AgentState:
    def __init__(self):
        self.dataset_profile = {}
        self.target_column = None
        self.problem_type = None  # Classification or Regression
        self.models_tried = []
        self.best_model = None
        self.best_metric = None
        self.issues_detected = []
        self.iteration = 0


def run_baseline_agent(state, df):
    model, X_val, y_val = train_baseline_model(df, state.target_column)
    metrics = evaluate_classification_model(model, X_val, y_val)

    state.models_tried.append("logistic_regression")
    state.best_model = model
    state.best_metric = metrics["roc_auc"]
    state.iteration += 1

    save_artifacts(model, metrics, state.iteration)
    return metrics
