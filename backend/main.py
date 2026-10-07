import os
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

server = FastAPI(
    title="Multi-Agent CTR Prediction System",
    description="LangGraph-powered Click-Through Rate prediction workflow and analytics dashboard",
)

# Keep production static assets outside FastAPI middleware so Vercel can
# promote StaticFiles to its CDN. CORS is only needed for local development.
if not os.environ.get("VERCEL"):
    server.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Vercel Functions have a read-only deployment filesystem. Use /tmp for
# generated uploads and model artifacts. Local development keeps the familiar
# backend/data and backend/output directories.
if os.environ.get("VERCEL"):
    RUNTIME_DIR = os.path.join("/tmp", "ctr-prediction")
else:
    RUNTIME_DIR = os.path.join(os.path.dirname(BASE_DIR), ".runtime")

UPLOAD_DIR = os.path.join(RUNTIME_DIR, "data")
OUTPUT_DIR = os.path.join(RUNTIME_DIR, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

runs: Dict[str, Dict[str, Any]] = {}


class PredictionInput(BaseModel):
    features: Dict[str, Any]


@server.get("/health")
def health_check():
    return {"status": "ok", "service": "multi-agent-ctr-prediction"}


@server.get("/api-key-status")
def get_api_key_status():
    has_key = "GEMINI_API_KEY" in os.environ or "GOOGLE_API_KEY" in os.environ
    return {"has_key": has_key}


@server.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and validate a CSV dataset."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    import pandas as pd

    filename = f"{uuid.uuid4()}_{os.path.basename(file.filename)}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    try:
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)

        df = pd.read_csv(filepath)
        preview = (
            df.head(10)
            .where(pd.notnull(df.head(10)), None)
            .to_dict(orient="records")
        )

        return {
            "dataset_path": filepath,
            "filename": file.filename,
            "total_rows": len(df),
            "columns": list(df.columns),
            "preview": preview,
            "csv_data": df.to_csv(index=False),
        }
    except Exception as exc:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process CSV file: {exc}",
        ) from exc


@server.post("/generate-sample")
def generate_sample(num_rows: int = Form(2500)):
    """Generate a synthetic CTR dataset."""
    from backend.utils.sample_data import generate_sample_ctr_data

    if num_rows < 10 or num_rows > 100_000:
        raise HTTPException(
            status_code=400,
            detail="num_rows must be between 10 and 100000.",
        )

    filename = f"synthetic_ctr_{uuid.uuid4().hex[:8]}.csv"
    filepath = os.path.join(UPLOAD_DIR, filename)

    try:
        df = generate_sample_ctr_data(num_rows=num_rows, output_path=filepath)
        preview = (
            df.head(10)
            .where(df.head(10).notna(), None)
            .to_dict(orient="records")
        )
        return {
            "dataset_path": filepath,
            "filename": "synthetic_ctr_data.csv",
            "total_rows": len(df),
            "columns": list(df.columns),
            "preview": preview,
        }
    except Exception as exc:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate synthetic data: {exc}",
        ) from exc


def run_agent_workflow(run_id: str, inputs: dict, api_key: Optional[str] = None):
    """Run the LangGraph workflow only when a pipeline request is made."""
    # Keep heavy LangGraph/ML imports out of cold-start paths such as GET /.
    from backend.agents.graph import app as graph_app
    from backend.agents.llm_helper import add_agent_log

    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    try:
        for event in graph_app.stream(inputs, stream_mode="updates"):
            for node_name, state_update in event.items():
                runs[run_id]["state"].update(state_update)
                if "logs" in state_update:
                    runs[run_id]["state"]["logs"] = state_update["logs"]
                runs[run_id]["state"]["active_agent"] = (
                    node_name.replace("_", " ").title()
                )

        runs[run_id]["completed"] = True
        runs[run_id]["state"]["status"] = "completed"
        runs[run_id]["state"]["active_agent"] = "None"
    except Exception as exc:
        print(f"Error in background run {run_id}: {exc}")
        runs[run_id]["completed"] = True
        runs[run_id]["error"] = str(exc)
        runs[run_id]["state"]["status"] = "failed"

        current_logs = runs[run_id]["state"].get("logs", [])
        add_agent_log(
            current_logs,
            "System Error",
            f"Workflow execution halted: {exc}",
        )
        runs[run_id]["state"]["logs"] = current_logs


@server.post("/run")
async def start_pipeline(
    dataset_path: str = Form(""),
    target_column: str = Form("clicked"),
    exclude_columns: str = Form(""),
    user_instructions: str = Form(""),
    api_key: str = Form(""),
    model_types: str = Form(""),
    dataset_file: UploadFile | None = File(None),
    dataset_csv: str = Form(""),
):
    """Run the LangGraph workflow inside the current Vercel request."""
    if dataset_file is not None and dataset_file.filename:
        if not dataset_file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        filename = f"{uuid.uuid4()}_{os.path.basename(dataset_file.filename)}"
        dataset_path = os.path.join(UPLOAD_DIR, filename)
        with open(dataset_path, "wb") as f:
            f.write(await dataset_file.read())
    elif dataset_csv.strip():
        filename = f"dataset_{uuid.uuid4().hex[:8]}.csv"
        dataset_path = os.path.join(UPLOAD_DIR, filename)
        with open(dataset_path, "w", encoding="utf-8", newline="") as f:
            f.write(dataset_csv)
    elif not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="Dataset is missing. Upload the CSV or generate the sample again.")

    run_id = str(uuid.uuid4())
    ex_cols = [c.strip() for c in exclude_columns.split(",") if c.strip()]
    if not ex_cols:
        ex_cols = ["impression_id", "user_id", "timestamp"]
    selected_models = [m.strip() for m in model_types.split(",") if m.strip()]
    allowed_models = {"logistic_regression", "random_forest", "xgboost", "lightgbm"}
    selected_models = [m for m in selected_models if m in allowed_models]
    if not selected_models:
        selected_models = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    run_output_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_output_dir, exist_ok=True)
    initial_state = {
        "dataset_path": dataset_path,
        "target_column": target_column,
        "exclude_columns": ex_cols,
        "metadata": {},
        "user_instructions": user_instructions.strip() or None,
        "model_types": selected_models,
        "active_agent": "Supervisor",
        "status": "started",
        "logs": [],
        "preprocessing_config": {},
        "model_selection": selected_models,
        "train_results": {},
        "best_model_type": None,
        "evaluation_report": None,
        "output_dir": run_output_dir,
    }
    runs[run_id] = {"state": initial_state, "completed": False, "error": None}
    run_agent_workflow(run_id, initial_state, api_key.strip() or None)
    return {
        "run_id": run_id,
        "status": runs[run_id]["state"]["status"],
        "completed": runs[run_id]["completed"],
        "error": runs[run_id]["error"],
        "state": runs[run_id]["state"],
    }

@server.get("/status/{run_id}")
def get_run_status(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run ID not found.")

    run = runs[run_id]
    return {
        "completed": run["completed"],
        "error": run["error"],
        "state": run["state"],
    }


@server.post("/predict")
def predict_ctr(input_data: PredictionInput):
    """Run single-record inference using the saved champion model."""
    try:
        from backend.ml.pipeline import CTRPipeline

        pipeline = CTRPipeline()
        best_model_type = pipeline.load_pipeline(OUTPUT_DIR)
        probability = pipeline.predict_probability(
            input_data.features,
            best_model_type,
        )
        return {
            "model_type": best_model_type,
            "click_probability": probability,
            "prediction": 1 if probability >= 0.5 else 0,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {exc}",
        ) from exc


@server.get("/download/model")
def download_model():
    from fastapi.responses import FileResponse

    model_path = os.path.join(OUTPUT_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=400,
            detail="No model has been trained yet.",
        )
    return FileResponse(
        path=model_path,
        filename="ctr_champion_model.joblib",
        media_type="application/octet-stream",
    )


# The frontend is part of this same deployment. API routes above are declared
# first, so this root mount handles / and static assets without intercepting APIs.
frontend_dir = os.path.join(os.path.dirname(BASE_DIR), "frontend")
if os.path.isdir(frontend_dir):
    server.mount(
        "/",
        StaticFiles(directory=frontend_dir, html=True),
        name="frontend",
    )
