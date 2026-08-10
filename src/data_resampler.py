import pandas as pd


import pandas as pd

class DataResampler:
    def __init__(self, df_1m):
        self.df_1m = df_1m.copy()
        
        if self.df_1m.index.name != 'Datetime':
            self.df_1m['Datetime'] = pd.to_datetime(self.df_1m['Date'].astype(str) + ' ' + self.df_1m['Time'].astype(str))
            self.df_1m.set_index('Datetime', inplace=True)
            
        self.agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Spread': 'mean'
        }

    def _format_columns(self, df):
        out_df = df.copy()
        out_df['Spread'] = out_df['Spread'].round().astype(int)
        out_df['Date'] = out_df.index.date
        out_df['Time'] = out_df.index.time
        return out_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Spread', 'Date', 'Time']]

    def generate_timeframes(self):
        timeframes = {}
        timeframes['1M'] = self._format_columns(self.df_1m)
        
        df_5m = self.df_1m.resample('5min', closed='left', label='left').agg(self.agg_dict).dropna()
        timeframes['5M'] = self._format_columns(df_5m)
        
        df_15m = df_5m.resample('15min', closed='left', label='left').agg(self.agg_dict).dropna()
        timeframes['15M'] = self._format_columns(df_15m)
        
        df_1h = df_15m.resample('1h', closed='left', label='left').agg(self.agg_dict).dropna()
        timeframes['1H'] = self._format_columns(df_1h)
        
        df_1d = df_1h.resample('1D', closed='left', label='left').agg(self.agg_dict).dropna()
        timeframes['1D'] = self._format_columns(df_1d)
        
        return timeframes
