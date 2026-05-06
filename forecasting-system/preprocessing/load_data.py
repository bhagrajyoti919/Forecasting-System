import pandas as pd

def load_dataset(file_path):
    
    df = pd.read_excel(file_path)

    print("\nDataset Loaded Successfully\n")
    print(df.head())

    df.columns = df.columns.str.lower().str.strip()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    df.reset_index(drop=True, inplace=True)

    print("\nDataset Information\n")
    print(df.info())

    return df

if __name__ == "__main__":
    file_path = "data/raw/sales_data.xlsx"
    df = load_dataset(file_path)
    print(f"\nFinal Dataset Shape: {df.shape}")
