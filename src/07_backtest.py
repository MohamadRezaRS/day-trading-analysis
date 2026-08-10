import os
import pandas as pd
import numpy as np
from data_resampler import DataResampler
from ict_detector import ICTDetector
from ml_predictor import MLPredictor
from account_manager import EventAccountManager, EventTradePool, fast_simulate
import sys
import os
import importlib

sys.path.append(os.path.abspath('../src'))


import ml_predictor
import ict_detector
import data_resampler
import account_manager


importlib.reload(ml_predictor)
importlib.reload(ict_detector)
importlib.reload(data_resampler)
importlib.reload(account_manager)


def run_pipeline(
    data_dir='../data/oos',
    models_dir='../models',
    scalers_dir='../scalers',
    cache_dir='../data/processed_cache',
    starting_balance=10000.0,
    max_risk_per_trade=0.01,
    max_total_risk=0.03,
    allowed_markets=['EURUSD', 'Gold', 'Silver', 'SP500', 'Nasdaq'],
    allowed_patterns=['fvg', 'ob', 'sweep', 'ote'],
    scenario=1
):
    os.makedirs(cache_dir, exist_ok=True)
    df_1m_dict = {}
    
    for market in allowed_markets:
        file_path = os.path.join(data_dir, f"{market}_1M.csv")
        if os.path.exists(file_path):
            df_1m = pd.read_csv(file_path)
            if not df_1m.empty:
                df_1m['Datetime'] = pd.to_datetime(df_1m['Date'].astype(str) + ' ' + df_1m['Time'].astype(str))
                df_1m.set_index('Datetime', inplace=True)
                df_1m_dict[market] = df_1m

    raw_detected_dfs = {}
    ml_detected_dfs = {}

    raw_files = [f for f in os.listdir(cache_dir) if f.startswith('raw_') and f.endswith('.csv')]
    
    if len(raw_files) >= 60:
        for f in raw_files:
            path = os.path.join(cache_dir, f)
            temp_df = pd.read_csv(path)
            if not temp_df.empty:
                for col in ['Entry_Time', 'Exit_Time', 'Formation_Date', 'Entry_Date']:
                    if col in temp_df.columns:
                        temp_df[col] = pd.to_datetime(temp_df[col], errors='coerce')
                orig_key = f.replace("raw_", "").replace(".csv", "")
                raw_detected_dfs[orig_key] = temp_df
    else:
        for market, df_1m in df_1m_dict.items():
            resampler = DataResampler(df_1m)
            timeframes_dict = resampler.generate_timeframes()
            
            for tf_name, df_tf in timeframes_dict.items():
                if tf_name == '1D': 
                    continue
                
                detector = ICTDetector(df_tf, df_1m, tf_name, market)
                detected_dict = detector.run_detection()
                
                for pat_name, df_pat in detected_dict.items():
                    if df_pat is not None and not df_pat.empty:
                        df_pat['Market'] = market
                        df_pat['Strategy'] = pat_name
                        df_pat['Timeframe'] = tf_name
                        key = f"{pat_name}_{tf_name}_{market}"
                        raw_detected_dfs[key] = df_pat
                        df_pat.to_csv(os.path.join(cache_dir, f"raw_{key}.csv"), index=False)

    predictor = MLPredictor(models_dir=models_dir, scalers_dir=scalers_dir)
    ml_keys_needed = list(predictor.models.keys())
    
    ml_files_found = [f for f in os.listdir(cache_dir) if f.startswith('ml_') and f.endswith('.csv')]
    
    if len(ml_files_found) == len(ml_keys_needed) and len(ml_keys_needed) > 0:
        for f in ml_files_found:
            path = os.path.join(cache_dir, f)
            temp_df = pd.read_csv(path)
            if not temp_df.empty:
                for col in ['Entry_Time', 'Exit_Time', 'Formation_Date', 'Entry_Date']:
                    if col in temp_df.columns:
                        temp_df[col] = pd.to_datetime(temp_df[col], errors='coerce')
                orig_key = f.replace("ml_", "").replace(".csv", "")
                ml_detected_dfs[orig_key] = temp_df
    else:
        ml_processed = predictor.process_dataframes(raw_detected_dfs)
        for key, df in ml_processed.items():
            ml_detected_dfs[key] = df
            df.to_csv(os.path.join(cache_dir, f"ml_{key}.csv"), index=False)

    if scenario in [1, 2]:
        target_dict = raw_detected_dfs
    else:
        target_dict = ml_detected_dfs

    valid_dfs = [df for df in target_dict.values() if df is not None and not df.empty]
    if not valid_dfs:
        return np.empty((0, 2))
        
    master_df = pd.concat(valid_dfs, ignore_index=True)
    master_df = master_df[master_df['Strategy'].isin(allowed_patterns)]

    if master_df.empty:
        return np.empty((0, 2))

    scenario_max_risk = max_risk_per_trade if scenario != 4 else 0.02
    scenario_total_risk = max_total_risk if scenario != 4 else 0.07

    manager = EventAccountManager(
        start_balance=starting_balance,
        max_risk_per_trade=scenario_max_risk,
        max_total_risk=scenario_total_risk,
        allowed_markets=allowed_markets,
        scenario=scenario
    )
    pool = EventTradePool()
    
    raw_events = []
    
    print(f"Building timeline for {len(master_df)} setups in Scenario {scenario}...")
    for i, (_, row) in enumerate(master_df.iterrows()):
        if i % 100000 == 0 and i > 0:
            print(f"  ...processed {i} setups...")
            
        setup = row.to_dict()
        
        entry_p = setup.get('Entry_Price', 0)
        sl_p = setup.get('SL_Price', 0)
        
        if pd.isna(entry_p) or pd.isna(sl_p) or abs(entry_p - sl_p) < 1e-5:
            continue
            
        en_t = pd.to_datetime(setup.get('Entry_Time'))
        ex_t = pd.to_datetime(setup.get('Exit_Time'))
        
        if pd.isna(en_t) or pd.isna(ex_t):
            continue
            
        setup['Setup_Time'] = en_t
        
        if scenario in [3, 4] and 'Prediction' in setup and pd.notna(setup['Prediction']):
            prob = setup['Prediction']
            if 0.20 <= prob < 0.65:
                continue 
                
            if prob < 0.20:
                setup['Direction'] *= -1
                dist = abs(entry_p - sl_p)
                setup['TP_Price'] = entry_p + (dist * 2 * setup['Direction'])
                setup['SL_Price'] = entry_p - (dist * setup['Direction'])
                
                mkt = setup['Market']
                if mkt in df_1m_dict:
                    m1_df = df_1m_dict[mkt]
                    
                    form_raw = setup.get('Formation_Time')
                    sim_start_time = pd.to_datetime(form_raw) if not pd.isna(form_raw) else en_t
                        
                    en_time, ex_time, out = fast_simulate(
                        m1_df.index.values, m1_df['High'].values,
                        m1_df['Low'].values, m1_df['Spread'].values,
                        sim_start_time, 
                        entry_p, setup['SL_Price'], setup['TP_Price'], setup['Direction']
                    )
                    if ex_time is None:
                        continue
                    en_t, ex_t, setup['Outcome'] = pd.to_datetime(en_time), pd.to_datetime(ex_time), out
                    setup['Entry_Time'], setup['Exit_Time'] = en_t, ex_t

        raw_events.append((en_t, 'ENTRY', setup))
        raw_events.append((ex_t, 'EXIT', setup))

    if not raw_events:
        return np.empty((0, 2))

    raw_events.sort(key=lambda x: (x[0], 0 if x[1] == 'EXIT' else 1))
    events = [{'Time': t, 'Type': typ, 'Setup': stp} for t, typ, stp in raw_events]

    df_1m_arrays = {
        mkt: (df.index.values.astype('datetime64[ns]'), df['Close'].values) 
        for mkt, df in df_1m_dict.items()
    }

    print(f"Simulating {len(events)} chronological events...")
    for i, event in enumerate(events):
        if manager.is_blown:
            print(f"  └─ Account hit 30% drawdown at event {i}. Halting scenario to save time.")
            break
            
        if i % 200000 == 0 and i > 0:
            print(f"  ...simulated {i} events...")
            
        curr_time = event['Time']
        setup = event['Setup']
        
        if event['Type'] == 'EXIT':
            manager.handle_exit(setup, curr_time, df_1m_dict, df_1m_arrays)
        elif event['Type'] == 'ENTRY':
            if scenario in [1, 3]:
                manager.handle_entry(setup, curr_time, df_1m_dict, df_1m_arrays)
            elif scenario in [2, 4]:
                if pool.process_signal(setup, curr_time):
                    manager.handle_entry(setup, curr_time, df_1m_dict, df_1m_arrays)

    if not manager.equity_curve:
        return np.empty((0, 2))
        
    df_eq = pd.DataFrame(manager.equity_curve).drop_duplicates('Datetime', keep='last')
    timestamps = df_eq['Datetime'].astype(np.int64).values / 10**9 
    balances = df_eq['Balance'].values
    return np.column_stack((timestamps, balances))