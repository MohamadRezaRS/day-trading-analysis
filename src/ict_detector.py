import pandas as pd
import numpy as np


   
class ICTDetector:
    def __init__(self, df, df_1m, timeframe, asset):
        self.df = df.copy()
        self.df_1m = df_1m.copy()
        self.timeframe = timeframe
        self.asset = asset
        
        self.df['Datetime'] = pd.to_datetime(self.df['Date'].astype(str) + ' ' + self.df['Time'].astype(str))
        self.df.set_index('Datetime', inplace=True)
        self.df['ATR'] = self.df['High'].rolling(14).max() - self.df['Low'].rolling(14).min()
        
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
        
        self._m1_dt = self.df_1m.index.values
        self._m1_h = self.df_1m['High'].values
        self._m1_l = self.df_1m['Low'].values
        self._m1_spread = self.df_1m['Spread'].values
        
        self.fvg_records = []
        self.ob_records = []
        self.sweep_records = []
        self.ote_records = []

    def _simulate_trade(self, start_dt, entry_price, sl_price, tp_price, direction):
        idx = np.searchsorted(self._m1_dt, start_dt)
        if idx >= len(self._m1_dt): 
            return np.nan, None, np.nan, np.nan
            
        future_h = self._m1_h[idx:]
        future_l = self._m1_l[idx:]
        future_dt = self._m1_dt[idx:]
        future_spread = self._m1_spread[idx:]
        
        entry_mask = future_l <= entry_price if direction == 1 else future_h >= entry_price
            
        if not np.any(entry_mask):
            return np.nan, None, np.nan, np.nan
            
        entry_idx = np.argmax(entry_mask)
        entry_time = future_dt[entry_idx]
        entry_spread = future_spread[entry_idx]
        
        active_h = future_h[entry_idx:]
        active_l = future_l[entry_idx:]
        active_dt = future_dt[entry_idx:]
        
        sl_mask = active_l <= sl_price if direction == 1 else active_h >= sl_price
        tp_mask = active_h >= tp_price if direction == 1 else active_l <= tp_price
            
        sl_any = np.any(sl_mask)
        tp_any = np.any(tp_mask)
        
        if not sl_any and not tp_any: return np.nan, entry_time, np.nan, entry_spread
        if not tp_any: return -1.0, entry_time, active_dt[np.argmax(sl_mask)], entry_spread
        if not sl_any: return 2.0, entry_time, active_dt[np.argmax(tp_mask)], entry_spread
        
        sl_first_idx = np.argmax(sl_mask)
        tp_first_idx = np.argmax(tp_mask)
        
        if sl_first_idx <= tp_first_idx:
            return -1.0, entry_time, active_dt[sl_first_idx], entry_spread
        else:
            return 2.0, entry_time, active_dt[tp_first_idx], entry_spread

    def _get_time_anchored_liquidity(self):
        dates = self.df.index.date
        if self.timeframe == '1D':
            period = self.df.index.to_period('M')
        elif self.timeframe == '1H':
            period = self.df.index.to_period('W')
        else:
            period = pd.Series(dates, index=self.df.index)

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
        ah_date_arr = pd.Series(mapped_h_dt).dt.strftime('%Y-%m-%d').fillna("1970-01-01").values
        al_date_arr = pd.Series(mapped_l_dt).dt.strftime('%Y-%m-%d').fillna("1970-01-01").values

        return mapped_h_val, mapped_l_val, ah_time_arr, al_time_arr, ah_date_arr, al_date_arr
    
    def detect_fvg(self, ah_arr, al_arr):
        c1_h, c1_l = pd.Series(self._h).shift(2).values, pd.Series(self._l).shift(2).values
        c2_o, c2_c = pd.Series(self._o).shift(1).values, pd.Series(self._c).shift(1).values
        
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
                outcome, entry_time, exit_time, entry_spread = self._simulate_trade(self._dt[e_idx], entry_price, sl_price, tp_price, direction)
                
                self.fvg_records.append({
                    'Direction': direction,
                    'C1_Body': abs(self._o[i-2] - self._c[i-2]),
                    'C2_Body': abs(self._o[i-1] - self._c[i-1]),
                    'C3_Body': abs(self._o[i] - self._c[i]),
                    '3C_Volume': self._v[i-2] + self._v[i-1] + self._v[i],
                    'Gap_Size': gap_size,
                    'Was_Extreme': was_extreme,
                    'Entry_Body': abs(self._o[e_idx] - self._c[e_idx]),
                    'Entry_Volume': self._v[e_idx],
                    'TP_Size': tp_size,
                    'Outcome': outcome,
                    'Entry_Time': entry_time,
                    'Exit_Time': exit_time,
                    'Formation_Date': self._date[i],
                    'Formation_Time': self._time[i],
                    'Entry_Price': entry_price,
                    'TP_Price': tp_price,
                    'SL_Price': sl_price,
                    'Entry_Spread': entry_spread
                })

    def detect_ob(self):
        c1_h, c1_l = pd.Series(self._h).shift(2).values, pd.Series(self._l).shift(2).values
        c2_h, c2_l = pd.Series(self._h).shift(1).values, pd.Series(self._l).shift(1).values
        
        c3_body = np.abs(self._o - self._c)
        avg_body = pd.Series(np.abs(self._o - self._c)).rolling(10).mean().shift(1).values
        
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
                outcome, entry_time, exit_time, entry_spread = self._simulate_trade(self._dt[e_idx], entry_price, sl_price, tp_price, direction)

                self.ob_records.append({
                    'Direction': direction,
                    'C1_Volume': self._v[i-2],
                    'C2_Volume': self._v[i-1],
                    'C3_Volume': self._v[i],
                    'ATR_at_Formation': self._atr[i],
                    'Formation_Time': self._time[i],
                    'Formation_Date': self._date[i],
                    'TP_Size': tp_size,
                    'Outcome': outcome,
                    'Entry_Time': entry_time,
                    'Exit_Time': exit_time,
                    'Entry_Price': entry_price,
                    'SL_Price': sl_price,
                    'TP_Price': tp_price,
                    'Entry_Spread': entry_spread
                })

    def detect_sweep(self, ah_arr, al_arr):
        bearish_mask = (self._h > ah_arr) & (self._c < ah_arr)
        bullish_mask = (self._l < al_arr) & (self._c > al_arr)
        
        valid_indices = np.where(bullish_mask | bearish_mask)[0]
        
        for i in valid_indices:
            if pd.isna(ah_arr[i]) or pd.isna(al_arr[i]): continue
            
            is_bull = bullish_mask[i]
            direction = 1 if is_bull else -1
            
            entry_price = self._c[i]
            sl_price = self._l[i] if is_bull else self._h[i]
            risk = abs(entry_price - sl_price)
            
            if risk > 0:
                tp_size = 2 * risk
                tp_price = entry_price + tp_size if is_bull else entry_price - tp_size
                outcome, entry_time, exit_time, entry_spread = self._simulate_trade(self._dt[i], entry_price, sl_price, tp_price, direction)
                
                self.sweep_records.append({
                    'Direction': direction,
                    'Entry_Price': entry_price,
                    'SL_Price': sl_price,
                    'TP_Price': tp_price,
                    'Entry_Time': entry_time,
                    'Exit_Time': exit_time,
                    'Entry_Spread': entry_spread,
                    'Outcome': outcome
                })

    def detect_ote(self, ah_arr, al_arr, ah_time, al_time, ah_date, al_date):
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
            outcome, entry_time, exit_time, entry_spread = self._simulate_trade(self._dt[i], entry_price, sl_price, tp_price, direction)
            
            self.ote_records.append({
                'Direction': direction,
                'Highs_Date': ah_date[i],
                'Highs_Time': ah_time[i],  
                'Lows_Date': al_date[i],
                'Lows_Time': al_time[i],   
                'Gap_Size': gap_size[i],
                'Entry_Volume': self._v[i],
                'Entry_Time': entry_time,
                'Exit_Time': exit_time,
                'Entry_Price': entry_price,
                'SL_Price': sl_price,
                'TP_Price': tp_price,
                'Entry_Spread': entry_spread,
                'Entry_Body': abs(self._o[i] - self._c[i]),
                'TP_Size': tp_size,
                'Outcome': outcome
            })

    def run_detection(self):
        ah_arr, al_arr, ah_time, al_time, ah_date, al_date = self._get_time_anchored_liquidity()
        
        self.detect_fvg(ah_arr, al_arr)
        self.detect_ob()
        self.detect_sweep(ah_arr, al_arr)
        self.detect_ote(ah_arr, al_arr, ah_time, al_time, ah_date, al_date)
        
        return {
            'fvg': pd.DataFrame(self.fvg_records),
            'ob': pd.DataFrame(self.ob_records),
            'sweep': pd.DataFrame(self.sweep_records),
            'ote': pd.DataFrame(self.ote_records)
        }