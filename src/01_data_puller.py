import os
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import pytz

"""
yfinance restricts intraday data (like 1-minute and 5-minute intervals) to only the last 60 days. 
Since our backtest requires data from 2021-01-01 to 2024-12-31, yfinance is not viable for our intraday needs. 
While different markets can have different data sources (ICT strategies do not rely on cross-market correlation), 
we CANNOT use different sources for different timeframes of the same asset. Doing so would cause volume 
and price mismatches that would ruin the machine learning model's predictions.

To solve this, we use the MetaTrader5 Python API to download 1-minute historical data directly from the local MT5 terminal. 
It handles missing historical segments dynamically—if an asset's data only starts after 2021, it will process whatever 
is available up to the end of 2024. It explicitly extracts 'tick_volume' (since CFDs report 0 real volume) and tracks the 
'spread' for accurate trading cost simulation.

After retrieving the 1-minute datasets, this script loads them into pandas DataFrames, filters the date range, 
and mathematically builds the 5m, 15m, 1H, and 1D datasets sequentially (1m -> 5m -> 15m -> 1H -> 1D). 
It parses the datetime into separate 'Date' and 'Time' columns, ensuring all timeframes strictly reflect the candle open time.
"""

ASSETS = {
    'EURUSD': 'EURUSD',
    'Gold': 'XAUUSD',
    'Silver': 'XAGUSD',
    'SP500': 'US500', 
    'Nasdaq': 'USTEC'
}

START_YEAR, START_MONTH, START_DAY = 2021, 1, 1
END_YEAR, END_MONTH, END_DAY = 2024, 12, 31


AGGREGATION_DICT = {
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum',
    'Spread': 'mean' 
}

os.makedirs('data/raw', exist_ok=True)

if not mt5.initialize():
    print(f"MT5 initialization failed. Error code: {mt5.last_error()}")
    quit()

timezone = pytz.timezone("Etc/UTC")
utc_from = datetime(START_YEAR, START_MONTH, START_DAY, tzinfo=timezone)
utc_to = datetime(END_YEAR, END_MONTH, END_DAY, 23, 59, 59, tzinfo=timezone)

for name, symbol in ASSETS.items():
    print(f"\nProcessing {name} ({symbol})")
    
    selected = mt5.symbol_select(symbol, True)

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, utc_from, utc_to)

        
    df = pd.DataFrame(rates)
    df['Datetime'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('Datetime', inplace=True)
    
    
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume',
        'spread': 'Spread'
    }, inplace=True)
    
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Spread']]
    df.sort_index(inplace=True)
    
    
    df = df.loc['2021-01-01':'2024-12-31']
    
    if df.empty:
        print(f"Error: Dataset is empty after filtering.")
        continue
        
    df.dropna(inplace=True)
    print(f"Success: Found {len(df)} rows. Range: {df.index[0].date()} to {df.index[-1].date()}")
    
    def save_dataset(dataframe, filename):
        out_df = dataframe.copy()
        
        
        out_df['Spread'] = out_df['Spread'].round().astype(int)
        
        out_df['Date'] = out_df.index.date
        out_df['Time'] = out_df.index.time
        out_df = out_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Spread', 'Date', 'Time']]
        out_df.to_csv(f"data/raw/{filename}", index=False)




    print(f"building timeframes for {name}")

    
    df_1m = df.copy()
    save_dataset(df_1m, f"{name}_1M.csv")
    
    df_5m = df_1m.resample('5min', closed='left', label='left').agg(AGGREGATION_DICT).dropna()
    save_dataset(df_5m, f"{name}_5M.csv")
    
    df_15m = df_5m.resample('15min', closed='left', label='left').agg(AGGREGATION_DICT).dropna()
    save_dataset(df_15m, f"{name}_15M.csv")
    
    df_1h = df_15m.resample('1h', closed='left', label='left').agg(AGGREGATION_DICT).dropna()
    save_dataset(df_1h, f"{name}_1H.csv")
    
    df_1d = df_1h.resample('1D', closed='left', label='left').agg(AGGREGATION_DICT).dropna()
    save_dataset(df_1d, f"{name}_1D.csv")
    
    print(f"Finished {name}.")

mt5.shutdown()