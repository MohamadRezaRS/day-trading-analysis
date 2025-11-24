import pandas as pd, matplotlib.pyplot as plt
from datetime import datetime, timedelta
from test_xauusd_M15_processor import XAUUSD_M15_Processor
from test_xauusd_M30_processor import XAUUSD_M30_Processor
from test_xauusd_H1_processor import XAUUSD_H1_Processor
from test_xauusd_H4_processor import XAUUSD_H4_Processor


START_BALANCE = 1000
NOISY_THRESHOLD = 11450
BROKER_OFFSET = -2


def get_entry_context(timestamp):
    dt = timestamp + timedelta(hours=BROKER_OFFSET)
    hour = dt.hour
    if 0 <= hour < 9: session = "Tokyo"
    elif 9 <= hour < 14: session = "London"
    elif 14 <= hour < 21: session = "New York"
    else: session = "Sydney"
    return dt.weekday(), {"Sydney":0,"Tokyo":1,"London":2,"New York":3}[session]


def simulate_tp_sl(signal, future_candles, commission, mode):
    
    tp, sl = signal["tp"], signal["sl"]
    order_type = signal["order_type"]

    if order_type == "buy":
        tp += commission
        sl += commission
    else:
        tp -= commission
        sl -= commission

    if order_type == "buy" and tp <= sl:
        return "invalid" , None
    if order_type == "sell" and tp >= sl:
        return "invalid" , None

    for candle in future_candles:
        high, low = candle[2], candle[3]
        hit_tp = hit_sl = False

        if order_type == "buy":
            hit_tp = high >= tp
            hit_sl = low <= sl
        else:
            hit_tp = low <= tp
            hit_sl = high >= sl

        if hit_tp and hit_sl:
            if mode == "best":
                return "TP" , candle[0]
            else:
                return "SL" , candle[0]
        elif hit_tp:
            return "TP" , candle[0]
        elif hit_sl:
            return "SL" , candle[0]

    return "open" , None


class NoisyDayTracker:
    def __init__(self, intraday_df):
        self.df = intraday_df
        self.last_checked_date = None
        self.noisy_day = False
    def update(self, current_time):
        current_date = current_time.date()
        if current_date == self.last_checked_date:
            return self.noisy_day
        day_candles = self.df[self.df['time'].dt.date == current_date]
        first_4 = day_candles.head(4)
        volume_sum = first_4['volume'].sum()
        self.noisy_day = volume_sum > NOISY_THRESHOLD
        self.last_checked_date = current_date
        return self.noisy_day


class FVGTracker:
    def __init__(self): self.fvgs = []
    def add(self, timeframe, candles, entry_price):
        self.fvgs.append({
            "timeframe": timeframe,
            "candles": candles,
            "entry_price": entry_price,
            "triggered": False,
            "entry_time": candles[-1][0]
        })
    def check_triggers(self, current_price, processor):
        triggered_signals = []
        for fvg in self.fvgs:
            if not fvg["triggered"] and current_price > fvg["entry_price"]:
                entry_date, session_code = get_entry_context(fvg["entry_time"])
                signal = processor.process_trigger(
                    candles=fvg["candles"],
                    pattern="bearish fvg",
                    entry_date=entry_date,
                    session_code=session_code
                )
                if isinstance(signal, dict):
                    fvg["triggered"] = True
                    triggered_signals.append(signal)
        return triggered_signals


def execute_trade_dual(signal, entry_price, balances, ledgers, balance_histories, future_candles):
    loss = abs(signal["sl"] - signal["entry"])
    risk_percentage = 0.003 if signal["timeframe"] in ["M30", "H1"] else 0.001

    for mode in ["best", "worst"]:
        balance = balances[mode]
        risk = risk_percentage * balance
        volume = max((risk // loss) * 0.01, 0.01)

        if signal["order_type"] == "buy" and not signal["sl"] <= entry_price <= signal["tp"]:
            continue
        if signal["order_type"] == "sell" and not signal["tp"] <= entry_price <= signal["sl"]:
            continue

        commission = 0.15
        result , resolution_time= simulate_tp_sl(signal, future_candles, commission, mode)
        if result in ["open", "invalid"]:
            continue

        pnl = (signal["tp"] - entry_price if result == "TP" else signal["sl"] - entry_price)
        pnl *= volume * 100 if signal["order_type"] == "buy" else -volume * 100
        balance += pnl

        ledgers[mode].append({
            "time": signal["candles"][-1][0],
            "pattern": signal["pattern"],
            "timeframe": signal["timeframe"],
            "direction": signal["direction"],
            "order_type": signal["order_type"],
            "result": result,
            "pnl": pnl,
            "volume": volume,
            "tp": signal["tp"],
            "sl": signal["sl"],
            "entry_price": entry_price
        })

        balance_histories[mode].append((resolution_time, balance))
        balances[mode] = balance


def run_tf_simulation(tf, df, daily_df, processor, start_date):
    balances = {"best": START_BALANCE, "worst": START_BALANCE}
    balance_histories = {"best": [], "worst": []}
    ledgers = {"best": [], "worst": []}
    
    noisy_tracker = NoisyDayTracker(df)
    fvg_tracker = FVGTracker()

    for i in range(2, len(df)-1):
        now = pd.to_datetime(df.iloc[i]['time'])
        current_price = df.iloc[i]['close']
        noisy_day = noisy_tracker.update(now)
        future_candles = df.iloc[i+1:].values.tolist()
        candles = df.iloc[i-2:i+1].values.tolist()

        signal = processor.process_live_candles(candles)
        if signal in ["no_trade", "no pattern detected"]:
            continue

        highs = [c[2] for c in signal["candles"]]
        lows = [c[3] for c in signal["candles"]]
        direction = signal["direction"]
        pattern = signal["pattern"]

        c3 = df.iloc[i]
        c3_date = c3["time"].date()
        c3_time = c3["time"]
        c3_week = c3_time.isocalendar().week

        same_day = df[(df["time"].dt.date == c3_date) & (df["time"] <= c3_time)]
        same_week = df[(df["time"].dt.isocalendar().week == c3_week) & (df["time"] <= c3_time)]

        if "fvg" in pattern:
            if direction == "bullish":
                is_highest_day = any(h >= same_day["high"].max() for h in highs)
                is_highest_week = any(h >= same_week["high"].max() for h in highs)
            else:
                is_highest_day = any(l <= same_day["low"].min() for l in lows)
                is_highest_week = any(l <= same_week["low"].min() for l in lows)
        else:
            if direction == "bullish":
                is_highest_day = any(l <= same_day["low"].min() for l in lows)
                is_highest_week = any(l <= same_week["low"].min() for l in lows)
            else:
                is_highest_day = any(h >= same_day["high"].max() for h in highs)
                is_highest_week = any(h >= same_week["high"].max() for h in highs)

        session_code = get_entry_context(signal["candles"][-1][0])[1]

        if pattern == "bearish fvg":
            fvg_tracker.add(tf, signal["candles"], signal["candles"][-1][4])
        else:
            final_signal = processor.process_trigger(
                candles=signal["candles"],
                pattern=pattern,
                noisy_day=noisy_day,
                is_highest_day=is_highest_day,
                is_highest_week=is_highest_week,
                session_code=session_code,
                direction=direction
            )
            if isinstance(final_signal, dict):
                execute_trade_dual(
                    final_signal, current_price,
                    balances, ledgers, balance_histories, future_candles
                )

        for fvg_signal in fvg_tracker.check_triggers(current_price, processor):
            execute_trade_dual(
                fvg_signal, current_price,
                balances, ledgers, balance_histories, future_candles
            )

        if balances["best"] <= 0 and balances["worst"] <= 0:
            break

    return {
        "best": (ledgers["best"], balance_histories["best"]),
        "worst": (ledgers["worst"], balance_histories["worst"])
    }


def visualize(results, label):
    plt.figure(figsize=(14, 6))

    final_balances = {}

    for mode, (trades, balance_history) in results.items():
        if not trades or 'pnl' not in pd.DataFrame(trades).columns:
            print(f"No trades executed for {label} [{mode}]")
            continue

        df = pd.DataFrame(trades)
        df['pnl'] = df['pnl'].astype(float)

        bh_df = pd.DataFrame(balance_history, columns=["time", "balance"])
        bh_df["time"] = pd.to_datetime(bh_df["time"])

        color = "green" if mode == "best" else "red"
        plt.plot(bh_df["time"], bh_df["balance"], label=f"{mode}-case", color=color)

        final_balances[mode] = bh_df["balance"].iloc[-1]

        for tf in df['timeframe'].unique():
            tf_df = df[df['timeframe'] == tf]
            tf_df['tp_distance'] = (tf_df['tp'] - tf_df['entry_price']).abs()
            tf_df['sl_distance'] = (tf_df['sl'] - tf_df['entry_price']).abs()

            summary = tf_df.groupby('pattern').agg(
                total_trades=('pattern', 'count'),
                tp_count=('result', lambda x: (x == 'TP').sum()),
                sl_count=('result', lambda x: (x == 'SL').sum()),
                total_pnl=('pnl', 'sum'),
                avg_tp_distance=('tp_distance', 'mean'),
                avg_sl_distance=('sl_distance', 'mean')
            ).reset_index()

            total_row = pd.DataFrame([{
                'pattern': 'TOTAL',
                'total_trades': summary['total_trades'].sum(),
                'tp_count': summary['tp_count'].sum(),
                'sl_count': summary['sl_count'].sum(),
                'total_pnl': summary['total_pnl'].sum(),
                'avg_tp_distance': summary['avg_tp_distance'].mean(),
                'avg_sl_distance': summary['avg_sl_distance'].mean()
            }])

            print(f"\n=== {label} Summary ({mode}-case) ===")
            print(pd.concat([summary, total_row]))

    # Final plot formatting
    plt.title(f"{label} Balance Over Time (Best vs Worst Case)")
    plt.xlabel("Time")
    plt.ylabel("Balance")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()

    # Annotate final balances
    y_max = max(final_balances.values())
    y_min = min(final_balances.values())
    y_text = y_max + (y_max - y_min) * 0.05

    for i, (mode, value) in enumerate(final_balances.items()):
        color = "green" if mode == "best" else "red"
        plt.text(
            plt.gca().get_xlim()[0],
            y_text - i * (y_max - y_min) * 0.07,
            f"{mode.capitalize()} Final Balance: {value:.2f}",
            fontsize=10,
            color=color,
            weight="bold"
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    start_date = datetime(2025, 1, 30)

    def load(path):
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        return df[df['time'] >= start_date].reset_index(drop=True)

    m15_df = load("backtesting/XAUUSD_M15.csv")
    m30_df = load("backtesting/XAUUSD_M30.csv")
    h1_df = load("backtesting/XAUUSD_H1.csv")
    h4_df = load("backtesting/XAUUSD_H4.csv")

    m15_results = run_tf_simulation("M15", m15_df, None, XAUUSD_M15_Processor(), start_date)
    m30_results = run_tf_simulation("M30", m30_df, None, XAUUSD_M30_Processor(), start_date)
    h1_results = run_tf_simulation("H1", h1_df, None, XAUUSD_H1_Processor(), start_date)
    #h4_results = run_tf_simulation("H4", h4_df, None, XAUUSD_H4_Processor(), start_date)

    visualize(m15_results, "M15")
    visualize(m30_results, "M30")
    visualize(h1_results, "H1")
    #visualize(h4_results, "H4")