import os
import pandas as pd
from backend.agents.state import AgentState
from backend.agents.llm_helper import get_llm, add_agent_log
from backend.ml.pipeline import CTRPipeline

def model_agent_node(state: AgentState) -> dict:
    """
    Selects candidate algorithms, trains them, and evaluates baseline metrics.
    """
    logs = list(state.get("logs", []))
    dataset_path = state["dataset_path"]
    target_column = state["target_column"]
    exclude_columns = state.get("exclude_columns", ["impression_id", "user_id", "timestamp"])
    user_instructions = state.get("user_instructions", "")
    output_dir = state.get("output_dir", "./output")
    
    # Models to train (can be customized by user)
    # Default list
    model_types = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    
    add_agent_log(logs, "Model Agent", f"Initializing model training pipeline. Candidate models: {', '.join(model_types)}")
    
    try:
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Instantiate pipeline
        pipeline = CTRPipeline(target_column=target_column, exclude_columns=exclude_columns)
        
        # Train and evaluate models
        add_agent_log(logs, "Model Agent", "Running feature encoding and scaling transforms on train data...")
        add_agent_log(logs, "Model Agent", f"Starting training loop for {len(model_types)} classifiers. This may take a few seconds...")
        
        results = pipeline.train_and_evaluate(df, model_types=model_types)
        
        # Find best model based on AUC
        best_auc = -1
        best_model_type = None
        for m_type, m_res in results.items():
            if "metrics" in m_res:
                auc = m_res["metrics"]["auc"]
                add_agent_log(logs, "Model Agent", f"Model '{m_type}' training complete. Validation AUC: {auc:.4f}, LogLoss: {m_res['metrics']['logloss']:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    best_model_type = m_type
            elif "error" in m_res:
                add_agent_log(logs, "Model Agent", f"Model '{m_type}' failed training: {m_res['error']}")
                
        if not best_model_type:
            raise ValueError("All models failed training.")
            
        add_agent_log(logs, "Model Agent", f"Best performing model: '{best_model_type}' (AUC: {best_auc:.4f})")
        
        # Save pipeline and best model
        pipeline.save_pipeline(best_model_type, output_dir)
        
        # Get LLM commentary on the models selection
        llm = get_llm()
        llm_insights = ""
        if llm:
            add_agent_log(logs, "Model Agent", "Querying LLM for model selection commentary and model performance insights...")
            prompt = f"""
            Compare the performance of the following models trained on our Click-Through Rate (CTR) prediction task:
            { {m: r['metrics'] for m, r in results.items() if 'metrics' in r} }
            
            Best model: {best_model_type}
            
            Provide a short explanation (2-3 sentences) comparing the strengths and weaknesses of these models for CTR. Why did the tree-based or linear model perform better/worse here?
            """
            try:
                response = llm.invoke([prompt])
                llm_insights = response.content.strip()
                add_agent_log(logs, "Model Agent", f"LLM Insights: {llm_insights}")
            except Exception as e:
                add_agent_log(logs, "Model Agent", f"LLM query failed: {str(e)}")
                llm_insights = "Model training complete. Standard performance metrics computed successfully."
        else:
            llm_insights = f"Logistic Regression, RF, XGBoost, and LightGBM models trained. The best validation AUC was achieved by {best_model_type}."

        results["llm_insights"] = llm_insights
        
        add_agent_log(logs, "Model Agent", "Training pipeline completed. Model weights and preprocessors saved to disk.")
        
        return {
            "train_results": results,
            "best_model_type": best_model_type,
            "logs": logs,
            "status": "training_completed"
        }
        
    except Exception as e:
        add_agent_log(logs, "Model Agent", f"Exception during model training: {str(e)}")
        return {
            "status": "failed",
            "logs": logs
        }
