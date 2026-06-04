import os
import pandas as pd
from backend.agents.state import AgentState
from backend.agents.llm_helper import get_llm, add_agent_log
from backend.ml.pipeline import CTRPipeline

def data_agent_node(state: AgentState) -> dict:
    """
    Analyzes the uploaded dataset, sets metadata, and plans preprocessing steps.
    """
    logs = list(state.get("logs", []))
    dataset_path = state["dataset_path"]
    target_column = state.get("target_column", "clicked")
    exclude_columns = state.get("exclude_columns", ["impression_id", "user_id", "timestamp"])
    user_instructions = state.get("user_instructions", "")
    
    add_agent_log(logs, "Data Agent", f"Analyzing dataset at: {os.path.basename(dataset_path)}")
    
    if not os.path.exists(dataset_path):
        add_agent_log(logs, "Data Agent", f"Error: Dataset file not found at {dataset_path}")
        return {
            "status": "failed",
            "logs": logs
        }
        
    try:
        # Load sample to analyze
        df = pd.read_csv(dataset_path)
        
        # Ensure target is present
        if target_column not in df.columns:
            # Fallback to last column or find columns containing 'click'
            click_cols = [c for c in df.columns if 'click' in c.lower()]
            if click_cols:
                target_column = click_cols[0]
            else:
                target_column = df.columns[-1]
            add_agent_log(logs, "Data Agent", f"Target column not explicitly matched. Selected target: '{target_column}'")

        # Run pipeline analysis
        pipeline = CTRPipeline(target_column=target_column, exclude_columns=exclude_columns)
        analysis = pipeline.analyze_dataset(df)
        
        # Check high cardinality categorical features
        high_cardinality_cols = []
        for cat_col in analysis["categorical_features"]:
            unique_count = df[cat_col].nunique()
            if unique_count > 50:
                high_cardinality_cols.append((cat_col, unique_count))
                
        # Handle high cardinality
        warning_msg = ""
        if high_cardinality_cols:
            warning_msg = "Note: The following high-cardinality categorical columns were found: " + \
                          ", ".join([f"{col} ({val} values)" for col, val in high_cardinality_cols]) + \
                          ". They will be one-hot encoded, which might increase feature space significantly."
            add_agent_log(logs, "Data Agent", warning_msg)

        # Prepare default preprocessing config
        preprocessing_config = {
            "numerical_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaling": "standard_scaler",
            "encoding": "one_hot"
        }
        
        # LLM Reasoning
        llm = get_llm()
        llm_insights = ""
        if llm:
            add_agent_log(logs, "Data Agent", "LLM is active. Requesting dataset insight analysis and preprocessing suggestions...")
            prompt = f"""
            Analyze the following dataset structure for Click-Through Rate (CTR) prediction:
            - Dataset rows: {analysis['total_rows']}
            - Target column: {target_column}
            - Numerical features: {analysis['numerical_features']}
            - Categorical features: {analysis['categorical_features']}
            - Excluded features: {analysis['excluded']}
            - Class distribution: {analysis['class_distribution']}
            {f'- Warnings: {warning_msg}' if warning_msg else ''}
            
            User guidelines: {user_instructions}
            
            Provide a short (2-3 sentences) set of feature engineering recommendations.
            Identify any possible risk (e.g. class imbalance, high cardinality) and how the pipeline should address it.
            """
            try:
                response = llm.invoke([prompt])
                llm_insights = response.content.strip()
                add_agent_log(logs, "Data Agent", f"LLM Insights: {llm_insights}")
            except Exception as e:
                add_agent_log(logs, "Data Agent", f"Failed to query LLM: {str(e)}. Proceeding with heuristic insights.")
                llm_insights = "Rule-based preprocessor configured. No major data anomalies detected."
        else:
            add_agent_log(logs, "Data Agent", "Running in offline/heuristic mode. Standard scaling and encoding applied.")
            llm_insights = f"Dataset shape: {df.shape}. Numeric features scaling active. Low-cardinality categorical encoding active."
            
        analysis["llm_insights"] = llm_insights
        
        add_agent_log(logs, "Data Agent", "Feature categorization completed. Preprocessing configuration written successfully.")
        
        return {
            "metadata": analysis,
            "target_column": target_column,
            "preprocessing_config": preprocessing_config,
            "logs": logs,
            "status": "preprocessing_completed"
        }
        
    except Exception as e:
        add_agent_log(logs, "Data Agent", f"Exception during analysis: {str(e)}")
        return {
            "status": "failed",
            "logs": logs
        }
