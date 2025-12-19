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


def build_llm_prompt(agent_summary):
    return f"""
    You are a senior data scientist reviewing the behavior of an autonomous
    machine learning agent.

    Below is a structured summary of the agent's actions and results.

    Your task:
    1. Explain what the agent did
    2. Justify why or why not model escalation occurred
    3. Explain why the agent stopped
    4. Provide 2-3 concrete suggestions for improvements

    AGENT SUMMARY:
    {agent_summary}

    Write clearly, concisely and professionally
    """


def generate_agent_report(llm_client, prompt):
    """
    llm_client is a callable that takes a prompt and returns text.
    This keeps the agent LLM-agnostic
    """
    return llm_client(prompt)


def dummy_llm(prompt):
    return (
        "The agent correctly identified a highly imbalanced "
        "classification problem. A logistic regression baseline achieved "
        "strong ROC AUC but poor recall, triggering escalation to a more "
        "expressive model. The agent stopped after marginal gains were "
        "observed. Future improvements include threshold tuning, feature "
        "engineering, and cost-sensitive optimization."
    )
