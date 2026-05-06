import pandas as pd
import holidays

def build_features(df):
 
    print("\nStarting Feature Engineering...\n")

    df = df.sort_values(by=['state', 'date'])

    # Lag features
    df['lag_1'] = df.groupby('state')['total'].shift(1)
    df['lag_7'] = df.groupby('state')['total'].shift(7)
    df['lag_30'] = df.groupby('state')['total'].shift(30)

    # Rolling statistics
    df['rolling_mean_4'] = df.groupby('state')['total'].transform(lambda x: x.rolling(window=4).mean())
    df['rolling_std_4'] = df.groupby('state')['total'].transform(lambda x: x.rolling(window=4).std())

    # Date-based features
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year

    # Holiday flags (US)
    us_holidays = holidays.US()
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in us_holidays else 0)

    # Drop rows with NaN values created by lag/rolling
    df = df.dropna().reset_index(drop=True)

    print("\nFeature Engineering Completed\n")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_sales.csv")
    df['date'] = pd.to_datetime(df['date'])

    featured_df = build_features(df)
    featured_df.to_csv("data/processed/featured_sales.csv", index=False)
    print("\nFeatured dataset saved successfully.")
