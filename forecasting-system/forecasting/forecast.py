import pandas as pd
import joblib
import os

def get_best_model_name(state_name):
    
    try:
        best_models_df = pd.read_csv("evaluation/best_models.csv")
        result = best_models_df[best_models_df['state'] == state_name]
        return result.iloc[0]['model'] if not result.empty else None
    except FileNotFoundError:
        return None

def generate_forecast(state_name):
    
    best_model = get_best_model_name(state_name)
    if best_model is None:
        return {"error": "State not found or models not trained."}

    print(f"Using model: {best_model} for {state_name}")

    if best_model == "XGBoost":
        try:
            model = joblib.load(f"saved_models/xgboost/{state_name}.pkl")
            featured_df = pd.read_csv("data/processed/featured_sales.csv")
            
            state_data = featured_df[featured_df['state'] == state_name].tail(1)
            feature_cols = [
                'lag_1', 'lag_7', 'lag_30', 'rolling_mean_4', 'rolling_std_4',
                'month', 'week_of_year', 'quarter', 'year', 'is_holiday'
            ]
            
            X_future = state_data[feature_cols]
            prediction = model.predict(X_future)[0]

            forecast = {f"Week {i}": round(float(prediction), 2) for i in range(1, 9)}
            return {
                "state": state_name,
                "best_model": best_model,
                "forecast_horizon": "8 Weeks",
                "forecast": forecast
            }
        except Exception as e:
            return {"error": f"Forecast failed: {str(e)}"}

    return {
        "state": state_name,
        "best_model": best_model,
        "message": f"Forecasting for {best_model} is being integrated."
    }

if __name__ == "__main__":
    # Test with a state
    print(generate_forecast("Texas"))
