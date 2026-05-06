import pandas as pd
import numpy as np
import warnings
import joblib
import os
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

def train_prophet(train_df, validation_df):
   
    print("\nStarting Prophet Training...\n")
    results = []
    states = train_df['state'].unique()

    for state in states:
        print(f"Training Prophet for: {state}")

        train_state = train_df[train_df['state'] == state].copy()
        validation_state = validation_df[validation_df['state'] == state].copy()

        # Format data for Prophet
        prophet_train = train_state[['date', 'total']].rename(
            columns={'date': 'ds', 'total': 'y'}
        )

        # Initialize and train model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model.fit(prophet_train)

        # Forecast
        future = model.make_future_dataframe(periods=8, freq='W')
        forecast = model.predict(future)
        predictions = forecast.tail(8)['yhat'].values

        actual = validation_state['total'].values

        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))

        # Save model
        model_path = f"saved_models/prophet/{state}.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        results.append({
            'state': state,
            'mae': mae,
            'rmse': rmse
        })
        print(f"{state} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    results_df = pd.DataFrame(results)
    print("\nProphet Training Completed\n")
    print(results_df.head())

    results_df.to_csv("evaluation/prophet_results.csv", index=False)
    return results_df

if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv")
    validation_df = pd.read_csv("data/processed/validation.csv")

    train_df['date'] = pd.to_datetime(train_df['date'])
    validation_df['date'] = pd.to_datetime(validation_df['date'])

    train_prophet(train_df, validation_df)
