import os
import pytz
from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5

if not mt5.initialize():
    print(f"MT5 Initialization failed. Error code: {mt5.last_error()}")
    quit()

symbol_map = {
    "EURUSD": "EURUSD",
    "XAUUSD": "Gold",
    "XAGUSD": "Silver",
    "US500": "SP500",
    "USTEC": "Nasdaq"
}

timezone = pytz.timezone("Etc/UTC")
utc_from = datetime(2025, 1, 1, tzinfo=timezone)
utc_to = datetime(2026, 7, 1, 23, 59, 59, tzinfo=timezone)

os.makedirs('data/oos', exist_ok=True)

print("MT5 Initialized Successfully. Beginning data extraction...")

for sym, name in symbol_map.items():
    print(f"Requesting 1M data for {sym}...")
    rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M1, utc_from, utc_to)
    
    if rates is None:
        print(f"Failed to get data for {sym}. Error code: {mt5.last_error()}")
        continue
        
    if len(rates) > 0:
        df = pd.DataFrame(rates)
        
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df['Date'] = df['time'].dt.date
        df['Time'] = df['time'].dt.time
        
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume',
            'spread': 'Spread'
        }, inplace=True)
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Spread', 'Date', 'Time']]
        
        df.to_csv(f'data/oos/{name}_1M.csv', index=False)
        print(f"Saved {len(df)} rows for {name}")
    else:
        print(f"No data returned for {sym}.")

mt5.shutdown()
print("Data extraction complete. MT5 connection closed.")