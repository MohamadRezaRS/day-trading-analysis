import os
import pandas as pd
import numpy as np
import joblib

class MLPredictor:
    def __init__(self, models_dir='../models', scalers_dir='../scalers'):
        self.models = {}
        self.scalers = {}
        self._load_assets(models_dir, scalers_dir)

    def _load_assets(self, models_dir, scalers_dir):
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('_xgb.pkl'):
                    key = file.replace('_xgb.pkl', '')
                    self.models[key] = joblib.load(os.path.join(models_dir, file))
                    
        if os.path.exists(scalers_dir):
            for file in os.listdir(scalers_dir):
                if file.endswith('_scaler.pkl'):
                    key = file.replace('_scaler.pkl', '')
                    self.scalers[key] = joblib.load(os.path.join(scalers_dir, file))

    def _engineer_features(self, df):
        out_df = df.copy()
        
        target_time_cols = ['Formation_Time', 'Entry_Time', 'Formation_Date', 'Entry_Date']
        
        for col in target_time_cols:
            if col in out_df.columns:
                prefix = col.replace('_Time', '').replace('_Date', '')
                temp_dt = pd.to_datetime(out_df[col].astype(str), errors='coerce')
                
                day_of_week = temp_dt.dt.dayofweek
                out_df[f'{prefix}_DayOfWeek'] = day_of_week
                out_df[f'{prefix}_DayOfWeek_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
                out_df[f'{prefix}_DayOfWeek_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
                
                hour = temp_dt.dt.hour
                out_df[f'{prefix}_Hour'] = hour
                out_df[f'{prefix}_Hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
                out_df[f'{prefix}_Hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

        date_cols = [col for col in out_df.columns if 'Date' in col and col not in target_time_cols]
        time_cols = [col for col in out_df.columns if 'Time' in col and col not in target_time_cols + ['Exit_Time']]
        
        for col in date_cols:
            temp_date = pd.to_datetime(out_df[col].astype(str), errors='coerce')
            prefix = col.replace('Date', '').strip('_')
            day_of_week = temp_date.dt.dayofweek
            out_df[f'{prefix}_DayOfWeek'] = day_of_week
            out_df[f'{prefix}_DayOfWeek_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
            out_df[f'{prefix}_DayOfWeek_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
            
        for col in time_cols:
            temp_time = pd.to_datetime(out_df[col].astype(str), errors='coerce')
            prefix = col.replace('Time', '').strip('_')
            hour = temp_time.dt.hour
            out_df[f'{prefix}_Hour'] = hour
            out_df[f'{prefix}_Hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
            out_df[f'{prefix}_Hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
            
        return out_df

    def process_dataframes(self, df_dict):
        processed_dfs = {}
        for key, df_raw in df_dict.items():
            if key not in self.models or key not in self.scalers or df_raw.empty:
                continue
                
            model = self.models[key]
            scaler = self.scalers[key]
            df_engineered = self._engineer_features(df_raw)
            
            # 1. Ask XGBoost exactly what 16 features it needs and in what order
            if hasattr(model, 'get_booster'):
                model_features = model.get_booster().feature_names
            else:
                # Fallback if not standard xgboost API
                model_features = df_engineered.select_dtypes(include=[np.number]).columns.tolist()

            # 2. Build a dataframe with exactly those 16 features
            df_model_input = df_engineered.reindex(columns=model_features, fill_value=0.0).copy()
            df_model_input.fillna(0, inplace=True)
            
            # 3. Ask the scaler which subset of columns it is supposed to scale (the 10 features)
            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = list(scaler.feature_names_in_)
                
                # Scale ONLY those 10 columns and update them in the model input dataframe
                scaled_subset = scaler.transform(df_model_input[scaler_features])
                df_model_input.loc[:, scaler_features] = scaled_subset
            
            # 4. Predict using the full 16 features!
            predictions = model.predict_proba(df_model_input)[:, 1]
            
            df_final = df_engineered.copy()
            df_final['Prediction'] = predictions
            processed_dfs[key] = df_final
            
        return processed_dfs