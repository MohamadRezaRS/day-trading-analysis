import pandas as pd
import numpy as np

MARKET_SPECS = {
    'Gold': {'contract_size': 100},
    'Silver': {'contract_size': 5000},
    'EURUSD': {'contract_size': 100000},
    'SP500': {'contract_size': 10},
    'Nasdaq': {'contract_size': 1}
}

class EventTradePool:
    def __init__(self):
        self.active_trades = []
        
    def process_signal(self, setup, curr_time):
        self.active_trades = [t for t in self.active_trades if pd.to_datetime(t['Exit_Time']) > curr_time]
        
        for t in self.active_trades:
            if t['Market'] == setup['Market']:
                return False
                
        self.active_trades.append(setup)
        return True

class EventAccountManager:
    def __init__(self, start_balance, max_risk_per_trade, max_total_risk, allowed_markets, scenario):
        self.start_balance = start_balance
        self.balance = start_balance
        self.min_balance = start_balance * 0.70
        self.is_blown = False
        
        self.max_risk_per_trade = max_risk_per_trade
        self.max_total_risk = max_total_risk
        self.allowed_markets = allowed_markets
        self.scenario = scenario
        
        self.active_trades = []
        self.equity_curve = [{'Datetime': pd.Timestamp('2000-01-01'), 'Balance': start_balance}]
        
        self.total_trades = 0
        self.winning_trades = 0
        
    def _get_unrealized_pnl(self, curr_time, df_1m_dict, df_1m_arrays):
        unrealized_pnl = 0.0
        active_risk_amount = 0.0
        
        for t in self.active_trades:
            mkt = t['Market']
            if mkt not in df_1m_arrays:
                continue
                
            times, closes = df_1m_arrays[mkt]
            idx = np.searchsorted(times, np.datetime64(curr_time))
            if idx >= len(times):
                idx = len(times) - 1
                
            current_price = closes[idx]
            entry_price = t['Entry_Price']
            sl_price = t['SL_Price']
            direction = t.get('Direction', 1)
            risk_amt = t.get('Risk_Amount', 0.0)
            
            denom = abs(sl_price - entry_price)
            if denom < 1e-6:
                continue
                
            dist_pct = ((current_price - entry_price) * direction) / denom
            dist_pct = np.clip(dist_pct, -1.5, 3.0) 
            
            add_pnl = dist_pct * risk_amt
            if np.isfinite(add_pnl):
                unrealized_pnl += add_pnl
                
            if np.isfinite(risk_amt):
                active_risk_amount += risk_amt
                
        return np.nan_to_num(unrealized_pnl), np.nan_to_num(active_risk_amount)

    def handle_entry(self, setup, curr_time, df_1m_dict, df_1m_arrays):
        if self.is_blown:
            return
            
        unrealized_pnl, active_risk_amount = self._get_unrealized_pnl(curr_time, df_1m_dict, df_1m_arrays)
        current_equity = max(self.balance + unrealized_pnl, 0.01)
        free_margin = max(current_equity - active_risk_amount, 0.01)
        
        risk_pct = self.max_risk_per_trade
        
        if self.scenario == 4 and 'Prediction' in setup:
            prob = setup['Prediction']
            if prob < 0.20:
                w = 1.0 - prob
            else:
                w = prob
                
            kelly = w - ((1.0 - w) / 2.0)
            half_kelly = kelly / 2.0
            risk_pct = np.clip(half_kelly, 0.001, self.max_risk_per_trade)
            
        if (active_risk_amount / current_equity) + risk_pct > self.max_total_risk:
            return
            
        risk_amount = np.nan_to_num(free_margin * risk_pct)
        if risk_amount <= 0:
            return
            
        setup['Risk_Amount'] = risk_amount
        self.active_trades.append(setup)

    def handle_exit(self, setup, curr_time, df_1m_dict, df_1m_arrays):
        if self.is_blown:
            return
            
        target_id = setup.get('ID', None)
        matched_trade = None
        
        for i, t in enumerate(self.active_trades):
            if t.get('ID') == target_id or (t['Market'] == setup['Market'] and t['Entry_Time'] == setup['Entry_Time']):
                matched_trade = self.active_trades.pop(i)
                break
                
        if not matched_trade:
            return

        risk_amt = matched_trade.get('Risk_Amount', 0.0)
        if not np.isfinite(risk_amt):
            risk_amt = 0.0
            
        outcome = setup.get('Outcome', -1.0)
        is_inverse = setup.get('is_inverse', False)
        slippage_pct = np.random.uniform(0.0, 0.02)
        
        if is_inverse:
            is_win = (outcome == -1.0)
            reward_mult = 0.5
        else:
            is_win = (outcome == 2.0)
            reward_mult = 2.0
            
        if is_win:
            pnl = (risk_amt * reward_mult) - (risk_amt * slippage_pct)
            self.winning_trades += 1
        else:
            pnl = -risk_amt - (risk_amt * slippage_pct)
            
        pnl = np.nan_to_num(pnl)
        
        self.balance += pnl
        self.total_trades += 1
        self.equity_curve.append({'Datetime': curr_time, 'Balance': self.balance})
        
        if self.balance <= self.min_balance:
            self.balance = self.min_balance
            self.is_blown = True