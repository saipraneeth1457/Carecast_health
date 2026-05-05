"""
Power BI Python Script Connector
==================================
Paste this script inside Power BI Desktop:
  Home → Transform Data → New Source → Python Script

It fetches predictions from the FastAPI service and returns
a flat DataFrame that Power BI can use directly.

Steps:
  1. Start the API:  uvicorn api:app --port 8000
  2. Paste this script in Power BI's Python script source
  3. Select the 'dataset' table from the navigator
"""

import requests
import pandas as pd

API_BASE = "http://localhost:8000"

# ── 1. Summary table (one row per disease, all horizons) ──────────────────
resp = requests.get(f"{API_BASE}/summary_table", timeout=30)
resp.raise_for_status()
dataset = pd.DataFrame(resp.json())

# ── 2. Bed demand forecast (monthly, top 50 diseases) ─────────────────────
resp2 = requests.get(
    f"{API_BASE}/bed_demand",
    params={"horizon": "3_years", "top_n": 50},
    timeout=30
)
bed_demand_df = pd.DataFrame(resp2.json())
bed_demand_df["date"] = pd.to_datetime(bed_demand_df["date"])

# ── 3. Case count forecast for all diseases (1-year horizon) ──────────────
# Get list of diseases first
diseases = [d["disease_name"] for d in requests.get(f"{API_BASE}/diseases").json()]

batch_resp = requests.post(
    f"{API_BASE}/predict/batch",
    json={"diseases": diseases, "horizon": "1_year"},
    timeout=60
)
case_forecast_df = pd.DataFrame(batch_resp.json())
case_forecast_df["date"] = pd.to_datetime(case_forecast_df["date"])

# Power BI exposes all DataFrames as separate tables in the navigator.
# dataset, bed_demand_df, case_forecast_df  ← select these in Power BI
