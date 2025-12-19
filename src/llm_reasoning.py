def build_agent_summary(state):
    """
    Converts AgentState into a structured text summary for LLM
    consumption
    """

    summary = []

    summary.append("DATASET SUMMARY:")
    for k, v in state.dataset_profile.items():
        summary.append(f"- {k}: {v}")

    summary.append("\nMODEL ITERATIONS:")
    for d in state.decisions:
        summary.append(
            f"Iteration {d['iteration']}: "
            f"Model = {d['model']}, "
            f"ROC_AUC = {d['roc_auc']:.4f}, "
            f"F1 = {d['f1']:.4f}, "
            f"Recall = {d['recall']:.4f}, "
            f"Train Time = {d['train_time_sec']}s"
        )

    summary.append("\nFINAL DECISION:")
    summary.append(f" - Stop Reason: {state.stop_reason}")
    summary.append(f" - Best ROC AUC: {state.best_metric:.4f}")
    return "\n".join(summary)
