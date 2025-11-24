import joblib
import os
import numpy as np
import pandas as pd
from test_pattern_detector import PatternDetector
from test_setup_maker import SetupMaker

class XAUUSD_M15_Processor:
    def __init__(self):
        self.detector = PatternDetector()
        self.setup = SetupMaker()

        self.model_paths = {
            "bearish fvg": "models/xauusd_fvg_bearish_M15_GradientBoosting.pkl",
            "bullish engulfing": "models/xauusd_engulfing_bullish_M15_SVR.pkl",
            "bullish orderblock": "models/xauusd_orderblock_bullish_M15_SVR.pkl",
            "bearish orderblock": "models/xauusd_orderblock_bearish_M15_SVR.pkl"
        }

        self.scaler_paths = {
            "bearish fvg": "scalers/scaler_xauusd_fvg_bearish_M15.pkl",
            "bullish engulfing": "scalers/scaler_xauusd_engulfing_bullish_M15.pkl",
            "bullish orderblock": "scalers/scaler_xauusd_orderblock_bullish_M15.pkl",
            "bearish orderblock": "scalers/scaler_xauusd_orderblock_bearish_M15.pkl"
        }

        self.models = {}
        self.scalers = {}
        self._load_models_and_scalers()

    def _load_models_and_scalers(self):
        for pattern in self.model_paths:
            if os.path.exists(self.model_paths[pattern]):
                self.models[pattern] = joblib.load(self.model_paths[pattern])
            if os.path.exists(self.scaler_paths[pattern]):
                self.scalers[pattern] = joblib.load(self.scaler_paths[pattern])

    def process_live_candles(self, candles):
        result = self.detector.detect(candles)
        if result in self.models:
            direction = result.split()[0]
            #if result != "bearish orderblock":
            return {
                    "direction": direction ,
                    "pattern": result,
                    "candles": candles
            }
        return "no pattern detected"

    def process_trigger(self, candles, pattern, noisy_day=None, is_highest_day=None, is_highest_week=None, entry_date=None,session_code=None,direction=None):
        
        if pattern == "bearish fvg":
            return self._process_fvg(candles, pattern,entry_date)

        return self._process_trigger(candles, pattern, noisy_day, is_highest_day, is_highest_week, session_code)

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
        scaler = self.scalers[pattern]
        candle_min, gap_min, pct_min = scaler.data_min_[:3]
        candle_max, gap_max, pct_max = scaler.data_max_[:3]

        # Manually scale features
        scaled_candle = (candle_size - candle_min) / (candle_max - candle_min)
        scaled_gap = (gap_size - gap_min) / (gap_max - gap_min)
        scaled_pct = (percentage - pct_min) / (pct_max - pct_min)

        vector = {"candle_size":scaled_candle,"gap_size": scaled_gap,"percentage": scaled_pct,"entry_weekday": int(entry_day)}

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

    def _process_trigger(self, candles, pattern, noisy_day, is_highest_day, is_highest_week, session_code):
        c1, c2 = candles[-2], candles[-1]

        # Index mapping
        TIMESTAMP = 0
        VOLUME = 5
        CLOSE = 4

        volume = float(c1[VOLUME]) + float(c2[VOLUME])
        scaler = self.scalers[pattern]
        volume_index =list(scaler.feature_names_in_).index("volume")
        volume_min = scaler.data_min_[volume_index]
        volume_max = scaler.data_max_[volume_index]
        target_index =list(scaler.feature_names_in_).index("target")
        target_min = scaler.data_min_[target_index]
        target_max = scaler.data_max_[target_index]


        scaled_volume = (volume - volume_min) / (volume_max - volume_min)
        weekday = pd.to_datetime(c2[TIMESTAMP]).weekday()

        if pattern == "bullish engulfing":
            vector = {"noisy_day":int(noisy_day),"is_highest_day": int(is_highest_day),"is_highest_week": int(is_highest_week),
                      "volume":scaled_volume,"session": int(session_code),"weekday": int(weekday)}
        elif pattern == "bullish orderblock":
            vector = {"noisy_day": int(noisy_day),"is_highest_day": int(is_highest_day),"is_highest_week": int(is_highest_week),
                      "volume":scaled_volume,"session": int(session_code),"weekday":int(weekday)}
        elif pattern == "bearish orderblock":
            vector = {"noisy_day": int(noisy_day),"is_highest_day": int(is_highest_day),"is_highest_week": int(is_highest_week),
                      "volume":scaled_volume,"session": int(session_code)}
        else:
            return "no_trade"

        scaled_prediction = float(self.models[pattern].predict(pd.DataFrame([vector])))
        #unscaled_prediction = (scaled_prediction * (target_max - target_min)) + target_min

        signal = self.setup.make_signal(
            pattern=pattern,
            direction=pattern.split()[0],
            prediction=scaled_prediction,
            current_price=float(c2[CLOSE]),
            candle=candles,
            timeframe="M15"
        )
        
        return signal