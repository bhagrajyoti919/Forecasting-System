import pandas as pd
from preprocessing.load_data import load_dataset

def clean_dataset(df):
    
    print("\nStarting Data Cleaning...\n")

    # Remove duplicates
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate Rows Found: {duplicate_count}")
    df = df.drop_duplicates()

    # Handle missing values
    print("\nMissing Values Check:")
    print(df.isnull().sum())
    df = df.dropna(subset=['date', 'total'])

    # Aggregate sales state-wise by date
    df = df.groupby(['state', 'date'])['total'].sum().reset_index()
    df = df.sort_values(by=['state', 'date'])

    # Ensure continuous date frequency for each state
    cleaned_data = []
    states = df['state'].unique()

    for state in states:
        state_df = df[df['state'] == state].copy()
        state_df.set_index('date', inplace=True)

        detected_frequency = pd.infer_freq(state_df.index)
        if detected_frequency is None:
            detected_frequency = 'W-SAT'

        full_dates = pd.date_range(
            start=state_df.index.min(),
            end=state_df.index.max(),
            freq=detected_frequency
        )

        state_df = state_df.reindex(full_dates)
        state_df['state'] = state
        state_df['total'] = state_df['total'].ffill()
        
        state_df.reset_index(inplace=True)
        state_df.rename(columns={'index': 'date'}, inplace=True)
        cleaned_data.append(state_df)

    final_df = pd.concat(cleaned_data)
    final_df = final_df.sort_values(by=['state', 'date'])
    final_df.reset_index(drop=True, inplace=True)

    print("\nCleaning Completed Successfully\n")
    print(final_df.head())
    print(f"\nFinal Shape: {final_df.shape}")

    return final_df

if __name__ == "__main__":
    file_path = "data/raw/sales_data.xlsx"
    df = load_dataset(file_path)
    cleaned_df = clean_dataset(df)

    cleaned_df.to_csv("data/processed/cleaned_sales.csv", index=False)
    print("\nCleaned dataset saved successfully.")
