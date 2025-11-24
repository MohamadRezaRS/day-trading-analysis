import joblib
import os
import numpy as np
import pandas as pd
from pattern_detector import PatternDetector
from setup_maker import SetupMaker

class XAUUSD_M30_Processor:
    def __init__(self):
        self.detector = PatternDetector()
        self.setup = SetupMaker()

        self.model_paths = {
            "bearish fvg": "models/xauusd_fvg_bearish_M30_GradientBoosting.pkl",
            "bullish orderblock": "models/xauusd_orderblock_bullish_M30_SVR.pkl",
            "bearish orderblock": "models/xauusd_orderblock_bearish_M30_SVR.pkl"
        }
        self.scaler_paths = {
            "bearish fvg": "scalers/scaler_xauusd_fvg_bearish_M30.pkl",
            "bullish orderblock": "scalers/scaler_xauusd_orderblock_bullish_M30.pkl",
            "bearish orderblock": "scalers/scaler_xauusd_orderblock_bearish_M30.pkl"
        }

        self.models = {}
        self.scalers = {}
        self._load_models_and_scalers()

    def _load_models_and_scalers(self):
        for direction in ["bullish orderblock", "bearish orderblock","bearish fvg"]:
            model_path = self.model_paths[direction]
            scaler_path = self.scaler_paths[direction]

            if os.path.exists(model_path):
                self.models[direction] = joblib.load(model_path)
            if os.path.exists(scaler_path):
                self.scalers[direction] = joblib.load(scaler_path)

    def process_live_candles(self, candles):
        result = self.detector.detect(candles)
        if result in ["bullish orderblock", "bearish orderblock","bearish fvg"]:
            direction = result.split()[0]
            return {
                "pattern": result,
                "direction": direction,
                "candles": candles
            }
        return "no pattern detected"
    

    def process_trigger(self, candles, pattern, noisy_day=None, is_highest_day=None, is_highest_week=None, entry_date=None,session_code=None,direction=None):
        
        if pattern == "bearish fvg":
            return self._process_fvg(candles, pattern,entry_date)

        return self._process_trigger(candles, pattern, noisy_day, is_highest_day, is_highest_week, session_code, direction)
    
    def _process_fvg(self, candles, pattern,entry_date):
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]

        # Index mapping
        HIGH, LOW, CLOSE, TIMESTAMP = 2, 3, 4, 0

        highs = [float(c1[HIGH]), float(c2[HIGH]), float(c3[HIGH])]
        lows = [float(c1[LOW]), float(c2[LOW]), float(c3[LOW])]

        candle_size = max(highs) - min(lows)
        gap_size = float(c1[LOW]) - float(c3[HIGH])
        percentage = gap_size / float(c3[CLOSE]) if float(c3[CLOSE]) != 0 else 0
        weekday = pd.to_datetime(c3[TIMESTAMP]).weekday()
        entry_day=entry_date
        volume=sum([float(candle[5]) for candle in candles[-3:]])
        scaler = self.scalers[pattern]
        candle_min, gap_min, pct_min,volume_min = scaler.data_min_[:4]
        candle_max, gap_max, pct_max,volume_max = scaler.data_max_[:4]

        # Manually scale features
        scaled_candle = (candle_size - candle_min) / (candle_max - candle_min)
        scaled_gap = (gap_size - gap_min) / (gap_max - gap_min)
        scaled_pct = (percentage - pct_min) / (pct_max - pct_min)
        scaled_volume=(volume - volume_min) / (volume_max - volume_min)
        vector = {"candle_size":scaled_candle,"gap_size": scaled_gap, "percentage":scaled_pct,
                  "weekday": int(weekday),"entry_weekday": int(entry_day),"volume":scaled_volume}

        # Predict — no unscaling needed
        prediction = self.models[pattern].predict(pd.DataFrame([vector]))[0]

        signal = self.setup.make_signal(
            pattern="fvg",
            direction="bearish",
            prediction=prediction,
            current_price=float(c3[CLOSE]),
            candle=candles,
            timeframe="M15"
        )
        return signal    


    def _rocess_trigger(self, candles, noisy_day, is_highest_day, is_highest_week, session_code, direction):


        c1, c2 = candles[-2], candles[-1]

        # Index mapping
        TIMESTAMP = 0
        VOLUME = 5
        CLOSE = 4

        weekday = pd.to_datetime(int(c2[TIMESTAMP]), unit='s').weekday()
        volume = float(c1[VOLUME]) + float(c2[VOLUME])

        # Extract direction-specific min/max
        scaler = self.scalers[direction]
        volume_index =list(scaler.feature_names_in_).index("volume")
        volume_min = scaler.data_min_[volume_index]
        volume_max = scaler.data_max_[volume_index]
        target_index =list(scaler.feature_names_in_).index("target")
        target_min = scaler.data_min_[target_index]
        target_max = scaler.data_max_[target_index]

        # Manually scale volume
        scaled_volume = (volume - volume_min) / (volume_max - volume_min)

        # Build vector
        if direction == "bullish":
            vector = [ int(noisy_day), int(is_highest_day), int(is_highest_week), scaled_volume, int(session_code)]
        else:
            vector = [int(noisy_day), int(is_highest_day), int(is_highest_week), scaled_volume, int(session_code), int(weekday)]

        # Predict and unscale
        scaled_prediction = self.models[direction].predict([vector])[0]
        unscaled_prediction = scaled_prediction * (target_max - target_min) + target_min

        signal = self.setup.make_signal(
            pattern=f"{direction} orderblock",
            direction=direction,
            prediction=scaled_prediction,
            current_price=float(c2[CLOSE]),
            candle=candles,
            timeframe="M30"
        )
        return signal