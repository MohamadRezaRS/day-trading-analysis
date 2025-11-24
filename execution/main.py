"""

run this a few second after a 4h candle closes

"""


import MetaTrader5 as mt5
import threading
import time
from datetime import datetime, timedelta
from xauusd_H1_processor import XAUUSD_H1_Processor
from xauusd_M30_processor import XAUUSD_M30_Processor
from xauusd_H4_processor import XAUUSD_H4_Processor
from xauusd_M15_processor import XAUUSD_M15_Processor
from datetime import datetime, timedelta



# Constants
BROKER_OFFSET = -2
NOISY_THRESHOLD = 11450
symbol = "XAUUSD"

session_map = {
    "Sydney": 0,
    "Tokyo": 1,
    "London": 2,
    "New York": 3
}

mt5.initialize()
if not mt5.initialize():
        print(" MT5 initialization failed")

# Utility functions
def get_current_price(symbol):
    tick = mt5.symbol_info_tick(symbol)
    return float(tick.ask) if tick else None

def get_last_daily_candle(symbol):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 1)
    return tuple(map(float, rates[0])) if rates else None

def get_last_weekly_candle(symbol):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_W1, 0, 1)
    return tuple(map(float, rates[0])) if rates else None

def get_entry_context(timestamp):
    dt = datetime.fromtimestamp(timestamp)+ timedelta(hours=BROKER_OFFSET)
    hour = dt.hour
    if 0 <= hour < 9:
        session = "Tokyo"
    elif 9 <= hour < 14:
        session = "London"
    elif 14 <= hour < 21:
        session = "New York"
    else:
        session = "Sydney"
    return dt.weekday(), session_map.get(session, -1)

# Noisy day tracker


class NoisyDayTracker:
    def __init__(self, symbol):
        self.symbol = symbol
        self.noisy_day = False
        self.last_evaluation_time = None
        self.market_open_hour = 1  # broker time
        self.evaluation_hour = 5   # 4 hours after market open

    def update(self):
        now = datetime.now() + timedelta(hours=BROKER_OFFSET)
        current_hour = now.hour

        # Reset after 20 hours from last evaluation
        if self.last_evaluation_time and now - self.last_evaluation_time >= timedelta(hours=20):
            self.noisy_day = False
            self.last_evaluation_time = None

        # If no evaluation yet today and it's time to evaluate
        if self.last_evaluation_time is None and current_hour == self.evaluation_hour:
            candle = get_last_daily_candle(self.symbol)
            if candle:
                volume = float(candle[5])
                self.noisy_day = volume > NOISY_THRESHOLD
                self.last_evaluation_time = now

        return self.noisy_day

# FVG tracker
class FVGTracker:
    def __init__(self):
        self.active_fvgs = []

    def add_fvg(self, timeframe, candles, entry_price):
        entry_time = candles[-1][0]
        self.active_fvgs.append({
            "timeframe": timeframe,
            "candles": candles,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "triggered": False
        })

    def check_triggers(self, current_price, processors):
        results = []
        for fvg in self.active_fvgs:
            if not fvg["triggered"] and current_price > fvg["entry_price"]:
                processor = processors.get(fvg["timeframe"])
                if processor:
                    entry_time = fvg["entry_time"]
                    entry_date, entry_session = get_entry_context(entry_time)
                    signal = processor.process_trigger(
                        candles=fvg["candles"],
                        entry_date=entry_date,
                        entry_session=entry_session
                    )
                    fvg["triggered"] = True
                    results.append((fvg["timeframe"], signal))
        return results

# Processor threads
def run_H4(symbol, processor, fvg_tracker):
    while True:
        candles = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 4)
        
        if  len(candles) >= 4:
            candles = [tuple(map(float, c)) for c in candles[:3]]
            signal = processor.process_live_candles(candles)
            if signal != "no_trade" and signal!="no pattern detected":
                entry_price = float(candles[-1][4])
                fvg_tracker.add_fvg("H4", candles, entry_price)
                print("FVG registered at H4")
            else:
                print("no trade at H4")
        time.sleep(4 * 3600)

def run_H1(symbol, processor, noisy_tracker):
    while True:
        candles = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 4)
        if  len(candles) >= 4:
            candles = [tuple(map(float, c)) for c in candles[:3]]
            live = processor.process_live_candles(candles)
            if live != "no_trade" and live!="no pattern detected":
                direction = live["direction"]
                c2 = live["candles"][-1]
                session_code = get_entry_context(c2[0])[1]
                noisy_day = noisy_tracker.update()
                highs = [c[2] for c in live["candles"]]
                lows = [c[3] for c in live["candles"]]
                daily = get_last_daily_candle(symbol)
                weekly = get_last_weekly_candle(symbol)
                if direction == "bullish":
                    max_high = max(highs)
                    is_highest_day = max_high >= daily[2]
                    is_highest_week = max_high >= weekly[2]
                else:
                    min_low = min(lows)
                    is_highest_day = min_low <= daily[3]
                    is_highest_week = min_low <= weekly[3]
                signal = processor.process_trigger(
                    candles=live["candles"],
                    noisy_day=noisy_day,
                    is_highest_day=is_highest_day,
                    is_highest_week=is_highest_week,
                    session_code=session_code,
                    direction=direction
                )
                print("H1 signal:", signal)
                execute_trade(signal)
            else:
                print("no trade at H1")
        time.sleep(3600)

def run_M30(symbol, processor, noisy_tracker,fvg_tracker):
    while True:
        candles = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, 0, 4)
        if len(candles) >= 4:
            candles = [tuple(map(float, c)) for c in candles[:3]]
            live = processor.process_live_candles(candles)
            if live != "no_trade" and live!="no pattern detected":
                direction = live["direction"]
                pattern=signal["pattern"]
                c2 = live["candles"][-1]
                session_code = get_entry_context(c2[0])[1]
                noisy_day = noisy_tracker.update()
                highs = [c[2] for c in live["candles"]]
                lows = [c[3] for c in live["candles"]]
                daily = get_last_daily_candle(symbol)
                weekly = get_last_weekly_candle(symbol)
                if pattern == "bearish fvg":
                    entry_price = float(c2[4])
                    fvg_tracker.add_fvg("M30", live["candles"], entry_price)
                    print("FVG registered at M30")
                else:
                    if direction == "bullish":
                        max_high = max(highs)
                        is_highest_day = max_high >= daily[2]
                        is_highest_week = max_high >= weekly[2]
                    else:
                        min_low = min(lows)
                        is_highest_day = min_low <= daily[3]
                        is_highest_week = min_low <= weekly[3]
                    signal = processor.process_trigger(
                        candles=live["candles"],
                        noisy_day=noisy_day,
                        is_highest_day=is_highest_day,
                        is_highest_week=is_highest_week,
                        session_code=session_code,
                        direction=direction
                    )
                    print("M30 signal:", signal)
                    execute_trade(signal)
            else:
                print("no trade at M30")
        time.sleep(1800)

def run_M15(symbol, processor, noisy_tracker, fvg_tracker):
    while True:
        candles = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 4)
        if  len(candles) >= 4:
            candles = [tuple(map(float, c)) for c in candles[:3]]
            live = processor.process_live_candles(candles)
            if live != "no_trade" and live!="no pattern detected":
                pattern = live["pattern"]
                last_candle = live["candles"][-1]
                session_code = get_entry_context(last_candle[0])[1]
                noisy_day = noisy_tracker.update()
                if pattern == "bearish fvg":
                    entry_price = float(last_candle[4])
                    fvg_tracker.add_fvg("M15", live["candles"], entry_price)
                    print("FVG registered at M15")
                else:
                    highs = [c[2] for c in live["candles"]]
                    lows = [c[3] for c in live["candles"]]
                    daily = get_last_daily_candle(symbol)
                    weekly = get_last_weekly_candle(symbol)
                    if pattern.startswith("bullish"):
                        max_high = max(highs)
                        is_highest_day = max_high >= daily[2]
                        is_highest_week = max_high >= weekly[2]
                    else:
                        min_low = min(lows)
                        is_highest_day = min_low <= daily[3]
                        is_highest_week = min_low <= weekly[3]
                    signal = processor.process_trigger(
                        candles=live["candles"],
                        pattern=pattern,
                        noisy_day=noisy_day,
                        is_highest_day=is_highest_day,
                        is_highest_week=is_highest_week,
                        session_code=session_code
                    )
                    print("M15 signal:", signal)
                    execute_trade(signal)
            else:
                print("no trade at M15")
        time.sleep(900)

def run_fvg_tracker_loop(fvg_tracker, processors, symbol):
    while True:
        current_price = get_current_price(symbol)
        triggered = fvg_tracker.check_triggers(current_price, processors)
        for tf, signal in triggered:
            print(f"Triggered FVG at {tf}: {signal}")
            execute_trade(signal)
        time.sleep(5)
        

def execute_trade(signal):
    if signal == "no trade bad r:r" or signal=="no registered pattern" or not isinstance(signal, dict):
        print("No valid signal to execute.")
        
        return

    symbol = "XAUUSD"

    # Get account balance
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to retrieve account info.")
        return

    balance = account_info.balance

    order_type = mt5.ORDER_TYPE_BUY if signal["order_type"] == "buy" else mt5.ORDER_TYPE_SELL
    entry_price = get_current_price(symbol=symbol)
    tp = signal["tp"]
    sl = signal["sl"]
    direction = signal["direction"]
    pattern = signal["pattern"]

    timeframe = signal["timeframe"]
    loss=abs(signal["sl"] - signal["entry"]) # e.g 5
    risk_percentage=0

    if signal["timeframe"] in ["M30","H1"] :
        risk_percentage = 0.003
    else : 
        risk_percentage = 0.001
    risk = risk_percentage * balance # e.g 12
    
    volume = max((risk // loss) * 0.01, 0.01) # e.g 0.02
    
    if volume <= 0:
        print("Calculated volume is zero. Check account balance.")
        return

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": entry_price,
        "sl": sl,
        "tp": tp,
        "deviation": 100,
        "magic": 123456,
        "comment": f"{pattern} {direction} {timeframe}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    if (signal["order_type"]=="buy" and not sl <= entry_price <= tp  )or (signal["order_type"]=="sell" and not tp <= entry_price <= sl) :  
        print ("price is not in the entry range")
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Trade failed: {result.retcode} - {result.comment}")
    else:
        print(f"Trade executed: {direction.upper()} {volume} lots at {entry_price} | TP: {tp} | SL: {sl}")


def start_engine(symbol):
    processors = {
        "H4": XAUUSD_H4_Processor(),
        "H1": XAUUSD_H1_Processor(),
        "M30": XAUUSD_M30_Processor(),
        "M15": XAUUSD_M15_Processor()
    }

    noisy_tracker = NoisyDayTracker(symbol)
    fvg_tracker = FVGTracker()

    #threading.Thread(target=run_H4, args=(symbol, processors["H4"], fvg_tracker), daemon=True).start()
    threading.Thread(target=run_H1, args=(symbol, processors["H1"], noisy_tracker), daemon=True).start()
    threading.Thread(target=run_M30, args=(symbol, processors["M30"], noisy_tracker), daemon=True).start()
    threading.Thread(target=run_M15, args=(symbol, processors["M15"], noisy_tracker, fvg_tracker), daemon=True).start()
    threading.Thread(target=run_fvg_tracker_loop, args=(fvg_tracker, processors, symbol), daemon=True).start()

    print("Trading engine started. Monitoring signals and FVG triggers...")
    while True:
        time.sleep(60)

if __name__ == "__main__":
    start_engine("XAUUSD")
    