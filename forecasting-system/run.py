import os
from services.pipeline_service import run_complete_pipeline

if __name__ == "__main__":
    data_path = "data/raw/sales_data.xlsx"
    
    if os.path.exists(data_path):
        print(f"Starting end-to-end pipeline with {data_path}...")
        result = run_complete_pipeline(data_path)
        print(result["message"])
    else:
        print(f"Error: {data_path} not found. Place your data in data/raw/.")
