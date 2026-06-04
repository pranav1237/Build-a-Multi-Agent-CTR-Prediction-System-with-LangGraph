from langgraph.graph import StateGraph, END
from backend.agents.state import AgentState
from backend.agents.supervisor import supervisor_node
from backend.agents.data_agent import data_agent_node
from backend.agents.model_agent import model_agent_node
from backend.agents.evaluator_agent import evaluator_agent_node

# Initialize the State Graph
workflow = StateGraph(AgentState)

# Add agent nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("data_agent", data_agent_node)
workflow.add_node("model_agent", model_agent_node)
workflow.add_node("evaluator_agent", evaluator_agent_node)

# Set initial entry point
workflow.set_entry_point("supervisor")

# Define supervisor routing function
def route_next(state: AgentState):
    """
    Reads the supervisor's decision from the state and routes to the correct node.
    """
    target = state.get("active_target")
    if target == "data_agent":
        return "data_agent"
    elif target == "model_agent":
        return "model_agent"
    elif target == "evaluator_agent":
        return "evaluator_agent"
    else:
        return "end"

# Wire conditional edges from supervisor
workflow.add_conditional_edges(
    "supervisor",
    route_next,
    {
        "data_agent": "data_agent",
        "model_agent": "model_agent",
        "evaluator_agent": "evaluator_agent",
        "end": END
    }
)

# Connect worker nodes back to supervisor
workflow.add_edge("data_agent", "supervisor")
workflow.add_edge("model_agent", "supervisor")
workflow.add_edge("evaluator_agent", "supervisor")

# Compile the graph
app = workflow.compile()
