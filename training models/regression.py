import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Inputs
symbol = input("Enter symbol (e.g., eurusd): ").lower()
pattern = input("Enter pattern (e.g., engulfing): ").lower()
base_path = f"ml datasets/{symbol}/{pattern}"
scaler_path = "scalers"

# Models to test
models = {
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "LinearRegression": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
    "XGBoost": XGBRegressor()
}

for filename in os.listdir(base_path):
    if not filename.endswith(".csv"):
        continue

    path = os.path.join(base_path, filename)
    df = pd.read_csv(path)

    if "target" not in df.columns:
        continue

    # Extract direction and timeframe
    parts = filename.replace(".csv", "").split("_")
    if len(parts) < 4:
        print(f"Skipping malformed filename: {filename}")
        continue
    direction = parts[1]
    timeframe = parts[-1]

    # Load scaler
    scaler_name = f"scaler_{symbol}_{pattern}_{direction}_{timeframe}.pkl"
    scaler_path_full = os.path.join(scaler_path, scaler_name)
    if not os.path.exists(scaler_path_full):
        print(f"Missing scaler for {filename}")
        continue
    scaler = joblib.load(scaler_path_full)

    # Unscale target
    try:
        target_index = list(scaler.feature_names_in_).index("target")
        target_min = scaler.data_min_[target_index]
        target_max = scaler.data_max_[target_index]
        target_scaled = df["target"].values
        target_unscaled = (target_scaled  * (target_max - target_min) ) + target_min
    except Exception as e:
        print(f"Error unscaling target in {filename}: {e}")
        continue

    # Split data
    split_idx = int(len(df) * 0.9)
    X = df.drop(columns=["target"])
    y = target_unscaled

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    best_mse= float("inf")
    best_model = None
    best_name = None
    accurate_found = False

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # TP logic
        tp_count = 0
        for pred, true in zip(preds, y_test):
            if  pred < true: # underestimate 
                                    
                tp_count += 1

        tp_ratio = tp_count / len(y_test)
        mse = mean_squared_error(y_test, preds)

        if tp_ratio <= 0.5:
            print(f"{filename} — {name}: Skipped (TP Ratio={tp_ratio:.2f}, MSE={mse:.2f})")
            continue

        print(f"{filename} — {name}: TP Ratio = {tp_ratio:.2f}, MSE = {mse:.2f}")
        accurate_found = True

        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_name = name

    # Save best model if valid
    if accurate_found and best_model:
        model_filename = f"{symbol}_{pattern}_{direction}_{timeframe}_{best_name}.pkl"
        joblib.dump(best_model, model_filename)
        print(f" Saved best model: {model_filename}")
    else:
        print(f"No accurate model found for {direction.upper()} {timeframe}")