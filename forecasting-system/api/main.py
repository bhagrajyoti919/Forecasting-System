from fastapi import FastAPI, UploadFile, File
import shutil
from forecasting.forecast import generate_forecast
from services.pipeline_service import run_complete_pipeline

app = FastAPI(
    title="Sales Forecasting API",
    version="1.0"
)

@app.get("/health")
def health_check():
    return {"status": "API is running"}

@app.get("/forecast/{state_name}")
def forecast(state_name: str):
    return generate_forecast(state_name)

@app.post("/train")
def train_pipeline(file: UploadFile = File(...)):
    # Save uploaded file locally
    file_path = f"data/raw/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run the complete data and training pipeline
    result = run_complete_pipeline(file_path)
    return result
