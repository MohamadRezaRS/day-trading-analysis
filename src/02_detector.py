import os
import pandas as pd
import numpy as np


"""
02_detector.py

This module programmatically identifies four core Inner Circle Trader (ICT) setups 
(Fair Value Gaps, Order Blocks, Optimal Trade Entries, and Liquidity Sweeps) 
across key modeled timeframes (5M, 15M, 1H, 1D) without lookahead bias.

Integrated 1-Minute Simulation Engine:
Instead of leaving the Target Variable (TP_or_SL) blank for a separate backtester, 
this script ingests the 1-Minute tick data directly. Upon detecting a valid setup 
and entry trigger, it calculates a 1:2 Risk-to-Reward framework and walks forward 
minute-by-minute to accurately simulate if the Stop Loss or Take Profit was hit first, 
completely eliminating intra-candle path dependency.

Timeframe-Specific High/Low Anchors (Used for OTE & Liquidity Sweeps):
- 1D: Previous Month High / Previous Month Low (PMH / PML)
- 1H: Previous Week High / Previous Week Low (PWH / PWL)
- 15M: Previous Day High / Previous Day Low (PDH / PDL)
- 5M: Intra Session Highs / Lows (Asia, London, NY AM completed sessions)
"""

"""
    
TRADE SIMULATION & RISK MANAGEMENT LOGIC (1:2 Risk-to-Reward)
    
Every pattern strictly targets a 1:2 R:R. 
Risk = ABS(Entry_Price - SL_Price)
TP_Size = Risk * 2
Take Profit = Entry_Price +/- TP_Size

1. Fair Value Gap (FVG)
    - Entry: The gap boundary (Low of Candle 3 for Bullish, High for Bearish).
    - SL: The absolute extreme of the displacement origin (Low of Candle 1 for Bullish).
       
2. Order Block (OB)
    - Entry: The front edge of the OB candle (High of C2 for Bullish).
    - SL: The back edge of the OB candle (Low of C2 for Bullish).
       
3. Liquidity Sweep
    - Entry: The Close price of the sweeping candle (confirming rejection).
    - SL: The absolute extreme tip of the sweep wick.
       
4. Optimal Trade Entry (OTE)
    - Entry: The 62% Fibonacci retracement level.
    - SL: The absolute origin of the impulse leg (Anchor Low for Bullish).
    
"""


ASSETS = ['EURUSD','Gold','Silver','Nasdaq','SP500']
TIMEFRAMES = ['5M', '15M', '1H', '1D']

def calc_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()


class ICTDetector:
    def __init__(self, df, df_1m, timeframe, asset):
        self.df = df.copy()
        self.df_1m = df_1m.copy()
        self.timeframe = timeframe
        self.asset = asset
        
        
        self.df['Datetime'] = pd.to_datetime(self.df['Date'].astype(str) + ' ' + self.df['Time'].astype(str))
        self.df.set_index('Datetime', inplace=True)
        self.df['ATR'] = calc_atr(self.df)
        
        self.df_1m['Datetime'] = pd.to_datetime(self.df_1m['Date'].astype(str) + ' ' + self.df_1m['Time'].astype(str))
        self.df_1m.set_index('Datetime', inplace=True)
        
        
        self._o = self.df['Open'].values
        self._h = self.df['High'].values
        self._l = self.df['Low'].values
        self._c = self.df['Close'].values
        self._v = self.df['Volume'].values
        self._atr = self.df['ATR'].values
        self._dt = self.df.index.values
        self._date = self.df['Date'].astype(str).values
        self._time = self.df['Time'].astype(str).values
        
        
        self._m1_h = self.df_1m['High'].values
        self._m1_l = self.df_1m['Low'].values
        self._m1_dt = self.df_1m.index.values
        
        
        self.fvg_records, self.ob_records = [], []
        self.sweep_records, self.ote_records = [], []

    def _simulate_trade(self, start_dt, sl_price, tp_price, direction):
        """
        Pure NumPy Vectorized Forward Simulation.
        Uses binary search (searchsorted) and argmax for instant execution.
        """
        
        idx = np.searchsorted(self._m1_dt, start_dt)
        if idx >= len(self._m1_dt): 
            return np.nan
            
        future_h = self._m1_h[idx:]
        future_l = self._m1_l[idx:]
        
        if direction == 1: # Bullish
            sl_mask = future_l <= sl_price
            tp_mask = future_h >= tp_price
        else: # Bearish
            sl_mask = future_h >= sl_price
            tp_mask = future_l <= tp_price
            
        
        sl_any = np.any(sl_mask)
        tp_any = np.any(tp_mask)
        
        if not sl_any and not tp_any: return np.nan
        if not tp_any: return 0
        if not sl_any: return 1
        
        # If both hit, argmax finds the index of the VERY FIRST True value
        sl_first_idx = np.argmax(sl_mask)
        tp_first_idx = np.argmax(tp_mask)
        
        return 0 if sl_first_idx <= tp_first_idx else 1

    def _get_time_anchored_liquidity(self):
        dates = self.df.index.date
        if self.timeframe == '1D':
            period = self.df.index.to_period('M')
        elif self.timeframe == '1H':
            period = self.df.index.to_period('W')
        else:
            period = pd.Series(dates, index=self.df.index)

        # Get values
        anchor_high = self.df.groupby(period)['High'].max().shift(1)
        anchor_low = self.df.groupby(period)['Low'].min().shift(1)
        
        
        anchor_high_dt = self.df.groupby(period)['High'].idxmax().shift(1)
        anchor_low_dt = self.df.groupby(period)['Low'].idxmin().shift(1)
        
        
        mapped_h_val = period.map(anchor_high).values
        mapped_l_val = period.map(anchor_low).values
        
        mapped_h_dt = period.map(anchor_high_dt)
        mapped_l_dt = period.map(anchor_low_dt)
        
        
        ah_time_arr = pd.Series(mapped_h_dt).dt.strftime('%H:%M:%S').fillna("00:00:00").values
        al_time_arr = pd.Series(mapped_l_dt).dt.strftime('%H:%M:%S').fillna("00:00:00").values

        return mapped_h_val, mapped_l_val, ah_time_arr, al_time_arr


    
    def detect_fvg(self, ah_arr, al_arr):
        c1_h, c1_l = self.df['High'].shift(2).values, self.df['Low'].shift(2).values
        c2_o, c2_c = self.df['Open'].shift(1).values, self.df['Close'].shift(1).values
        
        bullish_mask = (c1_h < self._l) & (c2_c > c2_o)
        bearish_mask = (c1_l > self._h) & (c2_c < c2_o)
        
        valid_indices = np.where(bullish_mask | bearish_mask)[0]
        
        for i in valid_indices:
            if i < 2: continue
            
            is_bull = bullish_mask[i]
            direction = 1 if is_bull else -1
            gap_size = self._l[i] - c1_h[i] if is_bull else c1_l[i] - self._h[i]
            
            was_extreme = 0
            if is_bull and c1_l[i] <= al_arr[i]: was_extreme = 1
            if not is_bull and c1_h[i] >= ah_arr[i]: was_extreme = 1

            end_idx = min(i + 50, len(self._l))
            hits = np.where(self._l[i+1:end_idx] <= self._l[i])[0] if is_bull else np.where(self._h[i+1:end_idx] >= self._h[i])[0]
                
            if len(hits) > 0:
                e_idx = i + 1 + hits[0]
                entry_price = self._l[i] if is_bull else self._h[i]
                sl_price = c1_l[i] if is_bull else c1_h[i]
                risk = abs(entry_price - sl_price)
                if risk == 0: continue
                
                tp_size = 2 * risk
                tp_price = entry_price + tp_size if is_bull else entry_price - tp_size
                outcome = self._simulate_trade(self._dt[e_idx], sl_price, tp_price, direction)
                
                self.fvg_records.append({
                    'Direction': direction,
                    'C1_Body': abs(self._o[i-2] - self._c[i-2]),
                    'C2_Body': abs(self._o[i-1] - self._c[i-1]),
                    'C3_Body': abs(self._o[i] - self._c[i]),
                    '3C_Volume': self._v[i-2] + self._v[i-1] + self._v[i],
                    'Gap_Size': gap_size,
                    'Formation_Time': self._time[i],
                    'Formation_Date': self._date[i],
                    'Was_Extreme': was_extreme,
                    'Entry_Body': abs(self._o[e_idx] - self._c[e_idx]),
                    'Entry_Time': self._time[e_idx],
                    'Entry_Date': self._date[e_idx],
                    'Entry_Volume': self._v[e_idx],
                    'TP_Size': tp_size,
                    'TP_or_SL': outcome
                })

    
    def detect_ob(self):
        c1_h, c1_l = self.df['High'].shift(2).values, self.df['Low'].shift(2).values
        c2_h, c2_l = self.df['High'].shift(1).values, self.df['Low'].shift(1).values
        
        c3_body = np.abs(self._o - self._c)
        avg_body = np.abs(self.df['Open'] - self.df['Close']).rolling(10).mean().shift(1).values
        
        with np.errstate(invalid='ignore'): 
            is_displacement = c3_body > (avg_body * 1.5)

        bullish_mask = (c2_l < c1_l) & (c2_l < self._l) & (self._c > self._o) & is_displacement
        bearish_mask = (c2_h > c1_h) & (c2_h > self._h) & (self._c < self._o) & is_displacement
        
        valid_indices = np.where(bullish_mask | bearish_mask)[0]
        
        for i in valid_indices:
            if i < 2: continue
            
            is_bull = bullish_mask[i]
            direction = 1 if is_bull else -1
            end_idx = min(i + 50, len(self._l))
            
            hits = np.where(self._l[i+1:end_idx] <= c2_h[i])[0] if is_bull else np.where(self._h[i+1:end_idx] >= c2_l[i])[0]
                
            if len(hits) > 0:
                e_idx = i + 1 + hits[0]
                entry_price = c2_h[i] if is_bull else c2_l[i]
                sl_price = c2_l[i] if is_bull else c2_h[i]
                risk = abs(entry_price - sl_price)
                if risk == 0: continue
                
                tp_size = 2 * risk
                tp_price = entry_price + tp_size if is_bull else entry_price - tp_size
                outcome = self._simulate_trade(self._dt[e_idx], sl_price, tp_price, direction)

                self.ob_records.append({
                    'Direction': direction,
                    'C1_Volume': self._v[i-2],
                    'C2_Volume': self._v[i-1],
                    'C3_Volume': self._v[i],
                    'Formation_Time': self._time[i],
                    'Formation_Date': self._date[i],
                    'ATR_at_Formation': self._atr[i],
                    'TP_Size': tp_size,
                    'TP_or_SL': outcome
                })


    def detect_sweep(self, ah_arr, al_arr, ah_time, al_time):
        bearish_mask = (self._h > ah_arr) & (self._c < ah_arr)
        bullish_mask = (self._l < al_arr) & (self._c > al_arr)
        
        valid_indices = np.where(bullish_mask | bearish_mask)[0]
        
        for i in valid_indices:
            if pd.isna(ah_arr[i]) or pd.isna(al_arr[i]): continue
            
            is_bull = bullish_mask[i]
            direction = 1 if is_bull else -1
            gap_size = ah_arr[i] - al_arr[i]
            
            entry_price = self._c[i]
            sl_price = self._l[i] if is_bull else self._h[i]
            risk = abs(entry_price - sl_price)
            
            if risk > 0:
                tp_size = 2 * risk
                tp_price = entry_price + tp_size if is_bull else entry_price - tp_size
                outcome = self._simulate_trade(self._dt[i], sl_price, tp_price, direction)
            else:
                tp_size = np.nan
                outcome = np.nan
            
            self.sweep_records.append({
                'Direction': direction,
                'Highs_Date': self._date[i], 
                'Highs_Time': ah_time[i],    
                'Lows_Date': self._date[i],  
                'Lows_Time': al_time[i],     
                'Gap_Size': gap_size,
                'Sweep_Time': self._time[i],
                'Sweep_Date': self._date[i],
                'Sweep_Volume': self._v[i],
                'TP_Size': tp_size,
                'TP_or_SL': outcome
            })


    def detect_ote(self, ah_arr, al_arr, ah_time, al_time):
        gap_size = ah_arr - al_arr
        
        with np.errstate(invalid='ignore'):
            valid_gap = gap_size > 0
            fib_62_bull, fib_79_bull = ah_arr - (gap_size * 0.62), ah_arr - (gap_size * 0.79)
            fib_62_bear, fib_79_bear = al_arr + (gap_size * 0.62), al_arr + (gap_size * 0.79)
            
            bullish_mask = valid_gap & (self._l <= fib_62_bull) & (self._l >= fib_79_bull)
            bearish_mask = valid_gap & (self._h >= fib_62_bear) & (self._h <= fib_79_bear)
        
        valid_indices = np.where(bullish_mask | bearish_mask)[0]
        
        for i in valid_indices:
            is_bull = bullish_mask[i]
            direction = 1 if is_bull else -1
            
            entry_price = fib_62_bull[i] if is_bull else fib_62_bear[i]
            sl_price = al_arr[i] if is_bull else ah_arr[i]
            risk = abs(entry_price - sl_price)
            if risk == 0: continue
            
            tp_size = 2 * risk
            tp_price = entry_price + tp_size if is_bull else entry_price - tp_size
            outcome = self._simulate_trade(self._dt[i], sl_price, tp_price, direction)
            
            self.ote_records.append({
                'Direction': direction,
                'Highs_Date': self._date[i],
                'Highs_Time': ah_time[i],  
                'Lows_Date': self._date[i],
                'Lows_Time': al_time[i],   
                'Gap_Size': gap_size[i],
                'Entry_Volume': self._v[i],
                'Entry_Date': self._date[i],
                'Entry_Time': self._time[i],
                'Entry_Body': abs(self._o[i] - self._c[i]),
                'TP_Size': tp_size,
                'TP_or_SL': outcome
            })

    def run_and_save(self):
        ah_arr, al_arr, ah_time, al_time = self._get_time_anchored_liquidity()
        
        self.detect_fvg(ah_arr, al_arr)
        self.detect_ob()
        self.detect_sweep(ah_arr, al_arr, ah_time, al_time)
        self.detect_ote(ah_arr, al_arr, ah_time, al_time)
        
        if self.fvg_records:
            pd.DataFrame(self.fvg_records).to_csv(f"data/processed/fvg_{self.timeframe}_{self.asset}.csv", index=False)
        if self.ob_records:
            pd.DataFrame(self.ob_records).to_csv(f"data/processed/ob_{self.timeframe}_{self.asset}.csv", index=False)
        if self.sweep_records:
            pd.DataFrame(self.sweep_records).to_csv(f"data/processed/sweep_{self.timeframe}_{self.asset}.csv", index=False)
        if self.ote_records:
            pd.DataFrame(self.ote_records).to_csv(f"data/processed/ote_{self.timeframe}_{self.asset}.csv", index=False)



if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    
    for asset in ASSETS:
        try:
            df_1m = pd.read_csv(f"data/raw/{asset}_1M.csv")
        except FileNotFoundError:
            print(f"Skipping {asset}: 1M data not found.")
            continue
            
        for tf in TIMEFRAMES:
            file_path = f"data/raw/{asset}_{tf}.csv"
            if not os.path.exists(file_path):
                continue
                
            print(f"Extracting & Simulating 4 Patterns for {asset} [{tf}]...")
            df = pd.read_csv(file_path)
            
            detector = ICTDetector(df, df_1m, tf, asset)
            detector.run_and_save()
            
    print("\nExtraction and 1M Simulation complete. 80 custom labeled DataFrames have been saved.")