import pandas as pd

def compare_and_select_best():
   
    try:
        sarima_df = pd.read_csv("evaluation/sarima_results.csv")
        prophet_df = pd.read_csv("evaluation/prophet_results.csv")
        xgboost_df = pd.read_csv("evaluation/xgboost_results.csv")
        lstm_df = pd.read_csv("evaluation/lstm_results.csv")

        sarima_df['model'] = 'SARIMA'
        prophet_df['model'] = 'Prophet'
        xgboost_df['model'] = 'XGBoost'
        lstm_df['model'] = 'LSTM'

        all_results = pd.concat([sarima_df, prophet_df, xgboost_df, lstm_df])
        all_results = all_results.sort_values(by=['state', 'rmse'])

        # Select model with lowest RMSE for each state
        best_models = all_results.groupby('state').first().reset_index()

        print("\nBest Model Per State:")
        print(best_models[['state', 'model', 'rmse']])

        all_results.to_csv("evaluation/all_model_results.csv", index=False)
        best_models.to_csv("evaluation/best_models.csv", index=False)
        
        print("\nModel comparison completed successfully.")
        
    except FileNotFoundError as e:
        print(f"Error: Missing results file. {e}")

if __name__ == "__main__":
    compare_and_select_best()
