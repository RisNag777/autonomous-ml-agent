from src.evaluation_tools import evaluate_classification_model
from src.model_tools import train_baseline_model
from src.report_tools import save_artifacts

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class AgentState:
    def __init__(self):
        self.dataset_profile = {}
        self.target_column = None
        self.problem_type = "classification"
        self.models_tried = []
        self.best_model = None
        self.best_metric = None
        self.iteration = 0
        self.improvement = 0
        self.no_improvement_count = 0
        self.stop_reason = None
        self.decisions = []


def run_iteration(state, df, model_type="logistic"):
    """Run one iteration of model training and evaluation"""

    if model_type == "logistic":
        model, X_val, y_val = train_baseline_model(df, state.target_column)
        model_name = "Logistic Regression"
    elif model_type == "tree":
        # Tree-based stronger model
        X = df.drop(columns=[state.target_column])
        y = df[state.target_column]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        )
        model.fit(X_train, y_train)
        model_name = "Random Forest"
    else:
        raise ValueError("Unknown Model Type")

    # Evaluate
    metrics = evaluate_classification_model(model, X_val, y_val)

    # Save artifacts
    save_artifacts(model, metrics, state.iteration + 1)

    # Update agent state
    state.models_tried.append(model_name)
    state.iteration += 1

    # Check improvement
    if state.best_metric is None:
        state.best_metric = metrics["roc_auc"]
        state.best_model = model
        state.improvement = 0
    else:
        state.improvement = metrics["roc_auc"] - state.best_metric
        if state.improvement > 0:
            state.best_metric = metrics["roc_auc"]
            state.best_model = model
        else:
            state.no_improvement_count += 1

    # Record decision
    state.decisions.append(
        {
            "iteration": state.iteration,
            "model": model_name,
            "roc_auc": metrics["roc_auc"],
            "improvement": state.improvement,
            "no_improvement_count": state.no_improvement_count,
        }
    )

    return metrics


def check_stop(state, max_iterations=3, min_improvement=0.01):
    """Determine if agent should stop iterating"""
    if state.iteration >= max_iterations:
        state.stop_reason = "Max iterations reached"
        return True
    if state.no_improvement_count >= 2:
        state.stop_reason = "No improvement in 2 consecutive iterations"
        return True
    if state.improvement < min_improvement:
        state.stop_reason = "Improvement below threshold"
        return True
    return False


def run_agent(df, state):
    """Run the agent until stop conditions are met"""

    # Phase 1: Baseline
    metrics = run_iteration(state, df, model_type="logistic")
    print(
        f"Iteration {state.iteration} - Logistic Regression ROC AUC: "
        + f"{metrics['roc_auc']:.4f}"
    )

    while not check_stop(state):
        # Phase 2: Stronger Model
        metrics = run_iteration(state, df, model_type="tree")
        print(
            f"Iteration {state.iteration} - Random Forest ROC AUC: "
            + f"{metrics['roc_auc']:.4f}"
        )

    print(
        f"Agent stopped after {state.iteration} iterations. Reason: "
        + f"{state.stop_reason}"
    )
    print(
        f"Best model: {state.models_tried[-1]} with ROC AUC: "
        + f"{state.best_metric:.4f}"
    )
