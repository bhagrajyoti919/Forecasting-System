import pandas as pd
import numpy as np
import warnings
import joblib
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

def train_xgboost(train_df, validation_df):
   
    print("\nStarting XGBoost Training...\n")
    results = []
    states = train_df['state'].unique()

    feature_columns = [
        'lag_1', 'lag_7', 'lag_30', 
        'rolling_mean_4', 'rolling_std_4', 
        'month', 'week_of_year', 'quarter', 'year', 'is_holiday'
    ]

    for state in states:
        print(f"Training XGBoost for: {state}")

        train_state = train_df[train_df['state'] == state].copy()
        validation_state = validation_df[validation_df['state'] == state].copy()

        X_train, y_train = train_state[feature_columns], train_state['total']
        X_validation, y_validation = validation_state[feature_columns], validation_state['total']

        # Initialize and train model
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            objective='reg:squarederror',
            random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        mae = mean_absolute_error(y_validation, predictions)
        rmse = np.sqrt(mean_squared_error(y_validation, predictions))

        # Save model
        model_path = f"saved_models/xgboost/{state}.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        results.append({
            'state': state,
            'mae': mae,
            'rmse': rmse
        })
        print(f"{state} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    results_df = pd.DataFrame(results)
    print("\nXGBoost Training Completed\n")
    print(results_df.head())

    results_df.to_csv("evaluation/xgboost_results.csv", index=False)
    return results_df

if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv")
    validation_df = pd.read_csv("data/processed/validation.csv")

    train_xgboost(train_df, validation_df)
