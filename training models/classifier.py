import pandas as pd
import os
import joblib
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score
)

# Inputs
symbol = input("Enter symbol (e.g., eurusd): ").lower()
pattern = input("Enter pattern (e.g., fvg): ").lower()
base_path = f"ml datasets/{symbol}/{pattern}"

# Models to test
models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

# Store results
results = defaultdict(list)

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

    # Split data
    split_idx = int(len(df) * 0.9)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Check balance
    pos_ratio = sum(y_test == 1) / len(y_test)
    neg_ratio = sum(y_test == 0) / len(y_test)
    balanced = (pos_ratio >= 0.3) and (neg_ratio >= 0.3)

    print(f"\n=== Dataset: {filename} | Balanced: {balanced} (Pos={pos_ratio:.2f}, Neg={neg_ratio:.2f}) ===")

    candidate_models = []  # store models eligible for saving

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        cm = confusion_matrix(y_test, preds)

        if balanced:
            # Balanced dataset → Accuracy + Weighted F1
            acc = accuracy_score(y_test, preds)
            f1w = f1_score(y_test, preds, average="weighted")
            print(f"{name}: Accuracy={acc:.3f}, F1(weighted)={f1w:.3f}")

            # Store candidate if F1 >= 0.55
            if f1w >= 0.55:
                candidate_models.append((acc, f1w, name, model))

        else:
            # Imbalanced dataset → Precision, Recall, F1 (minority class), PR-AUC
            precision = precision_score(y_test, preds, pos_label=1, zero_division=0)
            recall = recall_score(y_test, preds, pos_label=1, zero_division=0)
            f1m = f1_score(y_test, preds, pos_label=1, zero_division=0)
            pr_auc = average_precision_score(y_test, probs) if probs is not None else None
            roc_auc = roc_auc_score(y_test, probs) if probs is not None else None

            pr_auc_str = f"{pr_auc:.3f}" if pr_auc is not None else "None"
            roc_auc_str = f"{roc_auc:.3f}" if roc_auc is not None else "None"

            print(
                f"{name}: Precision(SL)={precision:.3f}, Recall(SL)={recall:.3f}, "
                f"F1(SL)={f1m:.3f}, PR-AUC={pr_auc_str}, ROC-AUC={roc_auc_str}"
            )

    # Save best model for balanced datasets
    if balanced and candidate_models:
        # Pick model with highest accuracy among candidates
        best_acc, best_f1, best_name, best_model = max(candidate_models, key=lambda x: x[0])
        model_filename = f"{symbol}_{pattern}_{direction}_{timeframe}_{best_name}.pkl"
        joblib.dump(best_model, model_filename)
        print(f"Saved best balanced model: {model_filename} (Accuracy={best_acc:.3f}, F1={best_f1:.3f})")
    elif balanced:
        print("No balanced model met F1 ≥ 0.55 threshold.")