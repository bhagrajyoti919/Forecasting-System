import pandas as pd
from preprocessing.load_data import load_dataset
from preprocessing.clean_data import clean_dataset
from preprocessing.split_data import split_dataset
from models.arima_model import train_sarima
from models.prophet_model import train_prophet
from models.xgboost_model import train_xgboost
from models.lstm_model import train_lstm
from feature_engineering.build_features import build_features

def run_complete_pipeline(file_path):
    
    print("\nStarting Complete Pipeline...\n")

    # Load and Clean
    df = load_dataset(file_path)
    cleaned_df = clean_dataset(df)
    cleaned_df.to_csv("data/processed/cleaned_sales.csv", index=False)

    # Feature Engineering
    feature_df = build_features(cleaned_df)
    feature_df.to_csv("data/processed/featured_sales.csv", index=False)

    # Train/Validation Split
    train_df, validation_df = split_dataset(feature_df)
    train_df.to_csv("data/processed/train.csv", index=False)
    validation_df.to_csv("data/processed/validation.csv", index=False)

    # Train all models
    train_sarima(train_df, validation_df)
    train_prophet(train_df, validation_df)
    train_xgboost(train_df, validation_df)
    train_lstm(train_df, validation_df)

    # Compare results and select best model per state
    import evaluation.compare_models
    print("\nPipeline Completed Successfully.\n")

    return {
        "status": "success",
        "message": "Dataset processed and models trained successfully"
    }
