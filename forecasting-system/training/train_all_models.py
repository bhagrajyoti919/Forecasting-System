from services.pipeline_service import run_complete_pipeline

def train_all():
    
    file_path = "data/raw/sales_data.xlsx"
    print("Executing full training pipeline...")
    return run_complete_pipeline(file_path)

if __name__ == "__main__":
    train_all()
