import joblib
import os
import numpy as np
import pandas as pd
from test_pattern_detector import PatternDetector
from test_setup_maker import SetupMaker

class XAUUSD_H1_Processor:
    def __init__(self):
        self.detector = PatternDetector()
        self.setup = SetupMaker()

        self.model_paths = {
            "bullish": "models/xauusd_orderblock_bullish_H1_SVR.pkl",
            "bearish": "models/xauusd_orderblock_bearish_H1_GradientBoosting.pkl"
        }
        self.scaler_paths = {
            "bullish": "scalers/scaler_xauusd_orderblock_bullish_H1.pkl",
            "bearish": "scalers/scaler_xauusd_orderblock_bearish_H1.pkl"
        }

        self.models = {}
        self.scalers = {}
        self._load_models_and_scalers()

    def _load_models_and_scalers(self):
        for direction in ["bullish", "bearish"]:
            model_path = self.model_paths[direction]
            scaler_path = self.scaler_paths[direction]

            if os.path.exists(model_path):
                self.models[direction] = joblib.load(model_path)
            if os.path.exists(scaler_path):
                self.scalers[direction] = joblib.load(scaler_path)

    def process_live_candles(self, candles):
        result = self.detector.detect(candles)
        if result in ["bullish orderblock", "bearish orderblock"]:
            direction = result.split()[0]
            return {
                "pattern": "orderblock",
                "direction": direction,
                "candles": candles
            }
        return "no pattern detected"

    def process_trigger(self, candles,pattern, noisy_day, is_highest_day, is_highest_week, session_code, direction):
        

        c1, c2 = candles[-2], candles[-1]

        # Index mapping
        TIMESTAMP = 0
        VOLUME = 5
        CLOSE = 4

        volume = float(c1[VOLUME]) + float(c2[VOLUME])
        scaler = self.scalers[direction]

        # Extract direction-specific min/max
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
            weekday = pd.to_datetime(c2[TIMESTAMP]).weekday()
            vector = {"noisy_day":int(noisy_day),"is_highest_day": int(is_highest_day),"is_highest_week": int(is_highest_week),
                      "volume":scaled_volume,"session": int(session_code),"weekday": int(weekday)}
        else:
            vector = {"noisy_day":int(noisy_day),"is_highest_day": int(is_highest_day),"is_highest_week": int(is_highest_week),
                                "volume":scaled_volume,"session": int(session_code)}
        # Predict and unscale
        scaled_prediction = float(self.models[direction].predict(pd.DataFrame([vector])))
        #unscaled_prediction = (scaled_prediction * (target_max - target_min) ) + target_min

        signal = self.setup.make_signal(
            pattern=f"{direction} orderblock",
            direction=direction,
            prediction=scaled_prediction,
            current_price=float(c2[CLOSE]),
            candle=candles,
            timeframe="H1"
        )
        return signal