import pandas as pd
import json
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings

def load_and_prepare_data(json_file):
    print("Loading and cleaning JSON data...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert JSON to a Pandas DataFrame
    df = pd.DataFrame(data)
    
    # Keep only what we need
    if 'date' not in df.columns or 'total_amount' not in df.columns:
        print("Error: Required columns not found in data.")
        return None

    df = df[['date', 'total_amount']].dropna()
    
    # Clean Amounts: Ensure it's a number
    df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
    
    # Clean Dates: Convert to datetime
    # to hide the Pandas format warning for a cleaner terminal
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
# --- The Date Sanity Check ---
    # The SROIE dataset was published in 2019. Anything after 2019 is an OCR error!
    df = df[(df['date'] >= '2015-01-01') & (df['date'] <= '2019-12-31')]
    
    # --- The Amount Sanity Check (Adjusted for MYR) ---
    # Throw out any OCR errors that read phone numbers or barcodes as prices
    df = df[df['total_amount'] < 20000]
    
    # Drop rows where the date or amount was completely unreadable
    df = df.dropna(subset=['date', 'total_amount'])
    
    if df.empty:
        print("Error: No valid data left after cleaning!")
        return None

    # Aggregate: Sum up expenses per day
    daily_expenses = df.groupby('date')['total_amount'].sum().reset_index()
    
    # Rename for Prophet
    daily_expenses.rename(columns={'date': 'ds', 'total_amount': 'y'}, inplace=True)
    
    print(f"Success! Loaded {len(daily_expenses)} unique days of historical data.")
    return daily_expenses

def train_and_forecast(df, periods=30):
    print("Initializing Prophet Time-Series model...")
    # Initialize Prophet (we turn off daily seasonality because we are predicting day-by-day)
    model = Prophet(daily_seasonality=False, yearly_seasonality=True)
    model.fit(df)
    
    print(f"Forecasting expenses for the next {periods} days...")
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    # Plot the forecast
    fig = model.plot(forecast)
    plt.title('Expense Forecast Model (Cleaned Data)')
    plt.xlabel('Date')
    plt.ylabel('Total Spent (MYR)')
    plt.tight_layout()
    plt.show()
    
    return forecast

if __name__ == "__main__":
    json_path = "../data/extracted_receipts.json"
    
    df_clean = load_and_prepare_data(json_path)
    
    if df_clean is not None:
        forecast_data = train_and_forecast(df_clean, periods=30)