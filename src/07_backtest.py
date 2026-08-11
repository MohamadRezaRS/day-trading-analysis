import os
import pandas as pd
import numpy as np
from data_resampler import DataResampler
from ict_detector import ICTDetector
from ml_predictor import MLPredictor
from account_manager import EventAccountManager, EventTradePool
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
    scenario=1,
    use_sweep=True
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
    master_df = master_df[master_df['Market'].isin(allowed_markets)]

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
    
    
    for i, (_, row) in enumerate(master_df.iterrows()):
        
            
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
        is_inverse = False
        
        if setup.get('Strategy') == 'sweep':
            if not use_sweep:
                continue
            is_inverse = True

        if scenario in [3, 4] and 'Prediction' in setup and pd.notna(setup['Prediction']):
            prob = setup['Prediction']
            if 0.20 <= prob < 0.65:
                continue 
                
            if prob < 0.20:
                is_inverse = True

        setup['is_inverse'] = is_inverse

        if is_inverse:
            orig_dir = setup['Direction']
            setup['Direction'] = orig_dir * -1
            orig_sl = setup['SL_Price']
            orig_tp = setup.get('TP_Price', entry_p + (abs(entry_p - orig_sl) * 2 * orig_dir))
            setup['SL_Price'] = orig_tp
            setup['TP_Price'] = orig_sl

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

    
    for i, event in enumerate(events):
        if manager.is_blown:
            print(f"Account hit 30% drawdown at event {i}. Halting scenario to save time.")
            break
            
        
            
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

    tp_ratio = (manager.winning_trades / manager.total_trades * 100) if manager.total_trades > 0 else 0
    print(f"Scenario {scenario} Finished | Trades: {manager.total_trades} | Win Rate: {tp_ratio:.2f}%")

    if not manager.equity_curve:
        return np.empty((0, 2))
        
    df_eq = pd.DataFrame(manager.equity_curve).drop_duplicates('Datetime', keep='last')
    timestamps = df_eq['Datetime'].astype(np.int64).values / 10**9 
    balances = df_eq['Balance'].values
    return np.column_stack((timestamps, balances))