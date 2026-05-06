import pandas as pd
import numpy as np
import warnings
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

warnings.filterwarnings("ignore")

def create_sequences(data, sequence_length=8):
   
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_lstm(train_df, validation_df):
    """
    Train LSTM model for each state.
    """
    print("\nStarting LSTM Training...\n")
    results = []
    states = train_df['state'].unique()

    for state in states:
        print(f"Training LSTM for: {state}")

        train_state = train_df[train_df['state'] == state]
        validation_state = validation_df[validation_df['state'] == state]

        train_values = train_state['total'].values.reshape(-1, 1)
        validation_values = validation_state['total'].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_values)
        validation_scaled = scaler.transform(validation_values)

        X_train, y_train = create_sequences(train_scaled)

        # Prepare validation data by including last 8 points of training
        combined_data = np.concatenate((train_scaled[-8:], validation_scaled), axis=0)
        X_validation, y_validation = create_sequences(combined_data)

        if len(X_train) == 0 or len(X_validation) == 0:
            print(f"Skipping {state}: Not enough data.")
            continue

        # Reshape to (samples, time_steps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], 1)

        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Train
        model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=0)

        # Predict and inverse scale
        predictions = model.predict(X_validation, verbose=0)
        predictions = scaler.inverse_transform(predictions)
        y_val_actual = scaler.inverse_transform(y_validation.reshape(-1, 1))

        mae = mean_absolute_error(y_val_actual, predictions)
        rmse = np.sqrt(mean_squared_error(y_val_actual, predictions))

        results.append({
            'state': state,
            'mae': mae,
            'rmse': rmse
        })
        print(f"{state} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    results_df = pd.DataFrame(results)
    print("\nLSTM Training Completed\n")
    print(results_df.head())

    results_df.to_csv("evaluation/lstm_results.csv", index=False)
    return results_df

if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv")
    validation_df = pd.read_csv("data/processed/validation.csv")

    train_lstm(train_df, validation_df)
