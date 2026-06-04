import json
from backend.agents.state import AgentState
from backend.agents.llm_helper import get_llm, add_agent_log

def supervisor_node(state: AgentState) -> dict:
    """
    Decides which agent to route to based on state.
    """
    llm = get_llm()
    logs = list(state.get("logs", []))
    
    add_agent_log(logs, "Supervisor", "Reviewing current system state and choosing the next agent node...")
    
    # Check current progress
    metadata = state.get("metadata", {})
    train_results = state.get("train_results", {})
    eval_report = state.get("evaluation_report", None)
    
    next_node = None
    status = state.get("status", "started")
    
    if not metadata:
        next_node = "data_agent"
        status = "preprocessing"
        add_agent_log(logs, "Supervisor", "No data metadata found. Directing workflow to Data Agent.")
    elif not train_results:
        next_node = "model_agent"
        status = "training"
        add_agent_log(logs, "Supervisor", "Preprocessing finished. Directing workflow to Model Agent.")
    elif not eval_report:
        next_node = "evaluator_agent"
        status = "evaluating"
        add_agent_log(logs, "Supervisor", "Model training finished. Directing workflow to Evaluator Agent.")
    else:
        next_node = "end"
        status = "completed"
        add_agent_log(logs, "Supervisor", "All workflow stages completed. Ending run.")
        
    return {
        "active_agent": "Supervisor",
        "status": status,
        "logs": logs,
        "active_target": next_node
    }
