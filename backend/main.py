import os
import uuid
import threading
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from backend.agents.graph import app as graph_app
from backend.agents.llm_helper import add_agent_log
from backend.ml.pipeline import CTRPipeline

server = FastAPI(
    title="Multi-Agent CTR Prediction System",
    description="LangGraph-powered Click-Through Rate prediction workflow and analytics dashboard"
)

# Enable CORS for local dev flexibility
server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shared runs state
runs: Dict[str, Dict[str, Any]] = {}

class PredictionInput(BaseModel):
    features: Dict[str, Any]

@server.get("/api-key-status")
def get_api_key_status():
    """
    Returns whether a Gemini or Google API key is configured.
    """
    has_key = "GEMINI_API_KEY" in os.environ or "GOOGLE_API_KEY" in os.environ
    return {"has_key": has_key}

@server.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Uploads a CSV dataset, validates it, and returns a preview of the columns and top rows.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    try:
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Read a preview of the CSV
        df = pd.read_csv(filepath)
        row_count = len(df)
        columns = list(df.columns)
        
        # Replace NaN values with empty string or None for JSON compatibility
        preview_data = df.head(10).replace({pd.NA: None}).where(pd.notnull(df.head(10)), None).to_dict(orient="records")
        
        return {
            "dataset_path": filepath,
            "filename": file.filename,
            "total_rows": row_count,
            "columns": columns,
            "preview": preview_data
        }
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")

@server.post("/generate-sample")
def generate_sample(num_rows: int = Form(2500)):
    """
    Triggers generating a synthetic dataset and returns a preview.
    """
    from backend.utils.sample_data import generate_sample_ctr_data
    filename = f"synthetic_ctr_{uuid.uuid4().hex[:8]}.csv"
    filepath = os.path.join(UPLOAD_DIR, filename)
    try:
        df = generate_sample_ctr_data(num_rows=num_rows, output_path=filepath)
        preview_data = df.head(10).replace({pd.NA: None}).where(pd.notnull(df.head(10)), None).to_dict(orient="records")
        return {
            "dataset_path": filepath,
            "filename": "synthetic_ctr_data.csv",
            "total_rows": len(df),
            "columns": list(df.columns),
            "preview": preview_data
        }
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Failed to generate synthetic data: {str(e)}")

def run_agent_workflow(run_id: str, inputs: dict, api_key: Optional[str] = None):
    """
    Background worker thread to execute the LangGraph workflow.
    """
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        
    try:
        # Use LangGraph's stream to capture state transitions and intermediate outputs
        for event in graph_app.stream(inputs, stream_mode="updates"):
            # event is a dictionary mapping node names to state updates
            for node_name, state_update in event.items():
                runs[run_id]["state"].update(state_update)
                # Keep logs list updated
                if "logs" in state_update:
                    runs[run_id]["state"]["logs"] = state_update["logs"]
                runs[run_id]["state"]["active_agent"] = node_name.replace("_", " ").title()
                
        runs[run_id]["completed"] = True
        runs[run_id]["state"]["status"] = "completed"
        runs[run_id]["state"]["active_agent"] = "None"
    except Exception as e:
        print(f"Error in background run {run_id}: {str(e)}")
        runs[run_id]["completed"] = True
        runs[run_id]["error"] = str(e)
        runs[run_id]["state"]["status"] = "failed"
        
        # Append error log
        curr_logs = runs[run_id]["state"].get("logs", [])
        add_agent_log(curr_logs, "System Error", f"Workflow execution halted: {str(e)}")
        runs[run_id]["state"]["logs"] = curr_logs

@server.post("/run")
def start_pipeline(
    dataset_path: str = Form(...),
    target_column: str = Form("clicked"),
    exclude_columns: str = Form(""),
    user_instructions: str = Form(""),
    api_key: str = Form(""),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Spawns the LangGraph agent prediction pipeline in the background.
    """
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="Dataset file not found. Upload data first.")
        
    run_id = str(uuid.uuid4())
    
    # Process excluded columns
    ex_cols = [c.strip() for c in exclude_columns.split(",") if c.strip()]
    if not ex_cols:
        ex_cols = ["impression_id", "user_id", "timestamp"]
        
    initial_state = {
        "dataset_path": dataset_path,
        "target_column": target_column,
        "exclude_columns": ex_cols,
        "metadata": {},
        "user_instructions": user_instructions if user_instructions.strip() else None,
        "active_agent": "Supervisor",
        "status": "started",
        "logs": [],
        "preprocessing_config": {},
        "model_selection": [],
        "train_results": {},
        "best_model_type": None,
        "evaluation_report": None,
        "output_dir": OUTPUT_DIR
    }
    
    runs[run_id] = {
        "state": initial_state,
        "completed": False,
        "error": None
    }
    
    # Run the graph asynchronously in a separate thread
    thread = threading.Thread(
        target=run_agent_workflow,
        args=(run_id, initial_state, api_key if api_key.strip() else None),
        daemon=True
    )
    thread.start()
    
    return {"run_id": run_id, "status": "started"}

@server.get("/status/{run_id}")
def get_run_status(run_id: str):
    """
    Retrieves the current execution progress, active node, and agent logs.
    """
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run ID not found.")
        
    run = runs[run_id]
    return {
        "completed": run["completed"],
        "error": run["error"],
        "state": run["state"]
    }

@server.post("/predict")
def predict_ctr(input_data: PredictionInput):
    """
    Evaluates click-through probability for a single inference record.
    """
    try:
        pipeline = CTRPipeline()
        best_model_type = pipeline.load_pipeline(OUTPUT_DIR)
        prob = pipeline.predict_probability(input_data.features, best_model_type)
        return {
            "model_type": best_model_type,
            "click_probability": prob,
            "prediction": 1 if prob >= 0.5 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@server.get("/download/model")
def download_model():
    """
    Downloads the trained champion model binary.
    """
    model_path = os.path.join(OUTPUT_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="No model has been trained yet.")
    return FileResponse(path=model_path, filename="ctr_champion_model.joblib", media_type="application/octet-stream")

# Health endpoint for deployment/runtime checks
@server.get("/health")
def health_check():
    return {"status": "ok", "service": "multi-agent-ctr-prediction"}

# Serve the existing frontend from the same FastAPI deployment.
# Keep explicit file routes so the Vercel Python runtime can always resolve
# the SPA entrypoint and its browser assets without relying on a separate host.
frontend_dir = os.path.join(os.path.dirname(BASE_DIR), "frontend")

if os.path.isdir(frontend_dir):
    frontend_index = os.path.join(frontend_dir, "index.html")
    frontend_app_js = os.path.join(frontend_dir, "app.js")
    frontend_style_css = os.path.join(frontend_dir, "style.css")

    @server.get("/", include_in_schema=False)
    def read_root():
        return FileResponse(frontend_index, media_type="text/html")

    @server.get("/app.js", include_in_schema=False)
    def frontend_javascript():
        return FileResponse(frontend_app_js, media_type="application/javascript")

    @server.get("/style.css", include_in_schema=False)
    def frontend_stylesheet():
        return FileResponse(frontend_style_css, media_type="text/css")

    server.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    @server.get("/", include_in_schema=False)
    def read_root():
        return HTMLResponse("<h2>Frontend directory not found. Please create 'frontend/' and index.html.</h2>")
