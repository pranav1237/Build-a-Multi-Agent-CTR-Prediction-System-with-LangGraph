from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    # Data management
    dataset_path: str
    target_column: str
    exclude_columns: List[str]
    metadata: Dict[str, Any]
    
    # User preferences / guidance
    user_instructions: Optional[str]
    
    # State tracking
    active_agent: str
    status: str # 'started', 'preprocessing', 'training', 'evaluating', 'completed', 'failed'
    logs: List[Dict[str, Any]] # list of agent thoughts/logs: {"agent": str, "message": str, "timestamp": str}
    
    # Agent configurations & outputs
    preprocessing_config: Dict[str, Any]
    model_selection: List[str]
    train_results: Dict[str, Any]
    best_model_type: Optional[str]
    evaluation_report: Optional[str]
    output_dir: str
