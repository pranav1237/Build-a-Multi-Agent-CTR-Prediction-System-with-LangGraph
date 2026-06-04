from backend.agents.state import AgentState
from backend.agents.llm_helper import get_llm, add_agent_log

def evaluator_agent_node(state: AgentState) -> dict:
    """
    Evaluates trained models, analyzes feature importances, and writes a final markdown report.
    """
    logs = list(state.get("logs", []))
    train_results = state["train_results"]
    best_model_type = state["best_model_type"]
    user_instructions = state.get("user_instructions", "")
    
    add_agent_log(logs, "Evaluator Agent", f"Starting model validation and feature importance analysis. Best model is {best_model_type}.")
    
    try:
        # Extract metrics
        best_metrics = train_results[best_model_type]["metrics"]
        best_feat_imp = train_results[best_model_type].get("feature_importance", {})
        
        # Build comparison table
        comparison_table = "| Model | AUC-ROC | Log Loss | Accuracy | Precision | Recall |\n|---|---|---|---|---|---|\n"
        for model_name, res in train_results.items():
            if model_name == "llm_insights":
                continue
            if "metrics" in res:
                metrics = res["metrics"]
                comparison_table += f"| {model_name.replace('_', ' ').title()} | {metrics['auc']:.4f} | {metrics['logloss']:.4f} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} |\n"
        
        # Build feature importance list
        feat_imp_str = ""
        for feat, imp in list(best_feat_imp.items())[:10]:
            feat_imp_str += f"- **{feat}**: {imp:.4f}\n"
            
        llm = get_llm()
        report = ""
        
        if llm:
            add_agent_log(logs, "Evaluator Agent", "Querying LLM to write a comprehensive data science evaluation report...")
            
            prompt = f"""
            Write a professional, comprehensive data science evaluation report based on the following CTR prediction experiment:
            
            Model Comparison Table:
            {comparison_table}
            
            Best Model: {best_model_type}
            
            Top Feature Importances (Best Model):
            {feat_imp_str}
            
            User Guidelines: {user_instructions}
            
            The report must be in Markdown format and include these sections:
            1. **Executive Summary**: High-level recap of the goal and the champion model.
            2. **Model Performance Evaluation**: Analysis of why the champion model performed best (mention AUC-ROC and LogLoss).
            3. **Feature Importance & Business Insights**: Explain what the top features tell us about user behavior, and what business/ad-placement decisions should be made.
            4. **Recommendations for Future Iterations**: Next steps to improve click rates (e.g. data collection, feature engineering ideas).
            
            Keep the report professional, actionable, and structured with clear headers.
            """
            try:
                response = llm.invoke([prompt])
                report = response.content.strip()
                add_agent_log(logs, "Evaluator Agent", "LLM successfully compiled evaluation report.")
            except Exception as e:
                add_agent_log(logs, "Evaluator Agent", f"LLM generation failed: {str(e)}. Falling back to templated report.")
                report = None
        
        if not report:
            # Heuristic Markdown report
            report = f"""# CTR Model Evaluation Report

## Executive Summary
This report analyzes the performance of multiple machine learning classifiers trained to predict user Click-Through Rate (CTR). Based on extensive validation testing, the **{best_model_type.replace('_', ' ').title()}** model has been selected as the champion classifier due to its superior predictive performance.

## Model Performance Evaluation
The candidate models were evaluated on key classification and probability metrics. In CTR modeling, **AUC-ROC** measures how well the model ranks clicks vs non-clicks, while **Log Loss** measures the calibration of the predicted probabilities (essential for ad bidding).

{comparison_table}

The champion model **{best_model_type.replace('_', ' ').title()}** achieved an **AUC-ROC of {best_metrics['auc']:.4f}** and a **Log Loss of {best_metrics['logloss']:.4f}**.

## Feature Importance & Business Insights
Understanding what drives clicks is critical for ad targeting and user experience optimization. The top 10 most influential features in our champion model are:

{feat_imp_str}

### Key Business Insights:
- **Feature Influence**: Features with high importance weights represent the strongest levers for predicting user engagement. Target advertising efforts towards demographics or context categories associated with these top features.
- **Data Calibration**: With a calibrated Log Loss of {best_metrics['logloss']:.4f}, these probabilities can be directly utilized in downstream expected-value formulas (e.g., bid valuation = bid price * predicted CTR).

## Recommendations for Future Iterations
1. **Engage in Feature Crosses**: Create interaction features (e.g., user age bucket * device type) to capture non-linear relationships.
2. **Collect Behavioral History**: Add sequential user behavior data (e.g., time elapsed since last click) to model user fatigue.
3. **Hyperparameter Tuning**: Run systematic grid or Bayesian search sweeps to squeeze extra performance from the {best_model_type.replace('_', ' ').title()} model.
"""
            add_agent_log(logs, "Evaluator Agent", "Heuristic report compiled successfully.")
            
        add_agent_log(logs, "Evaluator Agent", "All model evaluation tasks complete.")
        
        return {
            "evaluation_report": report,
            "logs": logs,
            "status": "completed"
        }
    except Exception as e:
        add_agent_log(logs, "Evaluator Agent", f"Exception during evaluation: {str(e)}")
        return {
            "status": "failed",
            "logs": logs
        }
