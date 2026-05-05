"""
Disease Prediction Model Training Script  (v2 – sklearn, no Prophet/Stan)
==========================================================================
Uses GradientBoostingRegressor with time-series feature engineering.
No C++ compiler, no Stan, no external dependencies beyond sklearn.

Predicts per disease:
  * case_count        - monthly new cases
  * avg_los           - average length of stay (days per case)
  * bed_days          - total bed-days demanded (cases x avg_los)

Horizons: 1 month ahead, 12 months ahead, 36 months ahead

Run:
    python train_model.py

Output:
    ./models/forecast_summary.json   <- used by api.py
    ./models/*.pkl                   <- one model file per disease x metric
"""

import os, json, warnings, joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH   = "disease_monthly_summary_powerbi.csv"
MODELS_DIR  = "models"
MIN_MONTHS  = 24
TOP_N       = 100
HORIZONS    = {"1_month": 1, "1_year": 12, "3_years": 36}
FEATURES    = ["year_num","month_num","trend","sin12","cos12","sin6","cos6","lag1","lag3","lag12"]


def load_data(path):
    df = pd.read_csv(path)
    df["ds"]       = pd.to_datetime(df["year_month"] + "-01")
    df["bed_days"] = df["case_records"] * df["avg_length_of_stay_days"]
    return df


def select_diseases(df):
    stats = (
        df.groupby("disease_name")
          .agg(n_months=("ds","count"), total_cases=("case_records","sum"))
          .reset_index()
    )
    eligible = stats[stats["n_months"] >= MIN_MONTHS]
    return eligible.sort_values("total_cases", ascending=False).head(TOP_N)["disease_name"].tolist()


def build_features(series, target):
    s = series[["ds", target]].dropna().sort_values("ds").reset_index(drop=True)
    s["year_num"]  = s["ds"].dt.year
    s["month_num"] = s["ds"].dt.month
    s["trend"]     = np.arange(len(s))
    s["sin12"]     = np.sin(2 * np.pi * s["month_num"] / 12)
    s["cos12"]     = np.cos(2 * np.pi * s["month_num"] / 12)
    s["sin6"]      = np.sin(2 * np.pi * s["month_num"] / 6)
    s["cos6"]      = np.cos(2 * np.pi * s["month_num"] / 6)
    s["lag1"]      = s[target].shift(1)
    s["lag3"]      = s[target].shift(3)
    s["lag12"]     = s[target].shift(12)
    return s.dropna().reset_index(drop=True)


def train_model(s, target):
    X = s[FEATURES].values
    y = s[target].values
    split = max(len(X) - 12, int(len(X) * 0.85))
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=4, subsample=0.8, random_state=42)
    model.fit(X[:split], y[:split])
    metrics = {}
    if split < len(X):
        preds = model.predict(X[split:]).clip(min=0)
        mae  = mean_absolute_error(y[split:], preds)
        rmse = np.sqrt(mean_squared_error(y[split:], preds))
        mape = float(np.mean(np.abs((y[split:] - preds) / (y[split:] + 1e-6))) * 100)
        metrics = {"MAE": round(mae,3), "RMSE": round(rmse,3), "MAPE_%": round(mape,2)}
    model.fit(X, y)
    return model, metrics


def generate_forecasts(model, s, target):
    max_h      = max(HORIZONS.values())
    last_date  = s["ds"].max()
    last_trend = int(s["trend"].max())
    history    = list(s[target].values)
    all_preds  = []

    for i in range(1, max_h + 1):
        fd = last_date + pd.DateOffset(months=i)
        m  = fd.month
        n  = len(history)
        lag1  = history[-1]  if n >= 1  else 0
        lag3  = history[-3]  if n >= 3  else 0
        lag12 = history[-12] if n >= 12 else 0
        row  = np.array([[fd.year, m, last_trend+i,
                          np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12),
                          np.sin(2*np.pi*m/6),  np.cos(2*np.pi*m/6),
                          lag1, lag3, lag12]])
        yhat = float(model.predict(row).clip(min=0)[0])
        rolling_std = np.std(history[-24:]) if len(history) >= 24 else np.std(history)
        ci = 1.5 * rolling_std * (1 + i / max_h * 0.5)
        all_preds.append({"ds": fd.strftime("%Y-%m-%d"),
                          "yhat": round(yhat,2),
                          "yhat_lower": round(max(0,yhat-ci),2),
                          "yhat_upper": round(yhat+ci,2)})
        history.append(yhat)

    return {label: all_preds[:h] for label, h in HORIZONS.items()}


def train_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("="*60)
    print("  Disease Prediction Training  (sklearn v2 - no Prophet)")
    print("="*60)

    print("\n[1/3] Loading data...")
    df = load_data(DATA_PATH)
    print(f"      {len(df):,} rows  |  {df['disease_name'].nunique():,} unique diseases")

    print("\n[2/3] Selecting diseases...")
    diseases = select_diseases(df)
    print(f"      {len(diseases)} diseases selected")

    print("\n[3/3] Training models (3 per disease x 100 diseases = 300 total)...\n")

    targets = [
        ("case_records",             "case_count"),
        ("avg_length_of_stay_days",  "avg_los"),
        ("bed_days",                 "bed_days"),
    ]

    summary   = {}
    total_ok  = 0
    total_err = 0

    for disease in tqdm(diseases, desc="Training"):
        sub      = df[df["disease_name"] == disease].sort_values("ds")
        safe_key = "".join(c if c.isalnum() or c in "-_ " else "_" for c in disease[:80])

        disease_info = {
            "disease_name":     disease,
            "disease_category": sub["disease_category"].iloc[-1],
            "n_months_history": len(sub),
            "models": {},
        }

        for raw_target, label in targets:
            try:
                s             = build_features(sub, raw_target)
                model, metrics = train_model(s, raw_target)
                forecasts      = generate_forecasts(model, s, raw_target)
                model_path     = os.path.join(MODELS_DIR, f"{safe_key}__{label}.pkl")
                joblib.dump(model, model_path)
                disease_info["models"][label] = {
                    "target": raw_target, "metrics": metrics,
                    "forecasts": forecasts, "model_file": model_path,
                }
                total_ok += 1
            except Exception as e:
                disease_info["models"][label] = {"error": str(e)}
                total_err += 1

        summary[disease] = disease_info

    summary_path = os.path.join(MODELS_DIR, "forecast_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Successful models : {total_ok} / {total_ok + total_err}")
    print(f"  Errors            : {total_err}")
    print(f"  Saved to          : ./{MODELS_DIR}/")
    print(f"\n  Sample MAPE scores (case_count):")
    for name, info in list(summary.items())[:5]:
        m = info["models"].get("case_count", {})
        if "metrics" in m:
            print(f"    {name[:55]:<55} {m['metrics'].get('MAPE_%')}%")
    print("\n  Next step -> run:  uvicorn api:app --port 8000")
    return summary


if __name__ == "__main__":
    train_all()