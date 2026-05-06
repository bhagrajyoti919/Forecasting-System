import pandas as pd

def split_dataset(df, forecast_horizon=8):
    print("\nStarting Train/Test Split...\n")

    df = df.sort_values(by=['state', 'date'])
    train_data = []
    validation_data = []

    states = df['state'].unique()
    for state in states:
        state_df = df[df['state'] == state]
        
        train = state_df.iloc[:-forecast_horizon]
        validation = state_df.iloc[-forecast_horizon:]

        train_data.append(train)
        validation_data.append(validation)

    train_df = pd.concat(train_data)
    validation_df = pd.concat(validation_data)

    train_df.reset_index(drop=True, inplace=True)
    validation_df.reset_index(drop=True, inplace=True)

    print(f"Train Shape: {train_df.shape}")
    print(f"Validation Shape: {validation_df.shape}")
    print(f"\nTrain Range: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Validation Range: {validation_df['date'].min()} to {validation_df['date'].max()}")

    return train_df, validation_df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/featured_sales.csv")
    df['date'] = pd.to_datetime(df['date'])

    train_df, validation_df = split_dataset(df)

    train_df.to_csv("data/processed/train.csv", index=False)
    validation_df.to_csv("data/processed/validation.csv", index=False)
    print("\nTrain and Validation datasets saved successfully.")
