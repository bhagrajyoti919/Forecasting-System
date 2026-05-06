import pandas as pd
import numpy as np
import warnings
import joblib
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

def train_sarima(train_df, validation_df):
   
    print("\nStarting SARIMA Training...\n")
    results = []
    states = train_df['state'].unique()

    for state in states:
        print(f"Training SARIMA for: {state}")

        # Filter state data
        train_state = train_df[train_df['state'] == state]
        validation_state = validation_df[validation_df['state'] == state]

        y_train = train_state['total']
        y_validation = validation_state['total']

       
        model = SARIMAX(
            y_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 52),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)

        # Forecast next 8 weeks
        forecast = model_fit.forecast(steps=8)

        # Calculate metrics
        mae = mean_absolute_error(y_validation, forecast)
        rmse = np.sqrt(mean_squared_error(y_validation, forecast))

        # Save trained model
        model_path = f"saved_models/arima/{state}.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_fit, model_path)

        results.append({
            'state': state,
            'mae': mae,
            'rmse': rmse
        })
        print(f"{state} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    results_df = pd.DataFrame(results)
    print("\nSARIMA Training Completed\n")
    print(results_df.head())

    results_df.to_csv("evaluation/sarima_results.csv", index=False)
    return results_df

if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv")
    validation_df = pd.read_csv("data/processed/validation.csv")

    train_sarima(train_df, validation_df)
