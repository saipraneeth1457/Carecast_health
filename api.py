"""
Disease Prediction API  –  FastAPI
=====================================
Serves trained Prophet model forecasts to Power BI (or any HTTP client).

Endpoints:
  GET  /diseases               → list of all diseases with trained models
  GET  /predict/{disease_name} → full forecast for a disease
  POST /predict/batch          → forecasts for multiple diseases at once
  GET  /bed_demand             → aggregated bed-demand forecast across diseases
  GET  /health                 → health check

Run locally:
  uvicorn api:app --reload --port 8000

Power BI connection:
  Use "Web" connector → http://localhost:8000/predict/<disease_name>
  Or use Python script connector to call /predict/batch
"""

import json, os, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODELS_DIR    = "models"
SUMMARY_PATH  = os.path.join(MODELS_DIR, "forecast_summary.json")
DASHBOARD_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
# ────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Disease Prediction API",
    description="Predict future case counts, avg length of stay, and bed demand per disease.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-computed forecast summary at startup
_summary: dict = {}


@app.on_event("startup")
def load_summary():
    global _summary
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH) as f:
            _summary = json.load(f)
        print(f"✓ Loaded forecasts for {len(_summary)} diseases")
    else:
        print("⚠ No forecast_summary.json found. Run train_model.py first.")


# ── SCHEMAS ─────────────────────────────────────────────────────────────────

class BatchRequest(BaseModel):
    diseases: list[str]
    horizon:  str = "1_year"   # "1_month" | "1_year" | "3_years"


# ── HELPERS ─────────────────────────────────────────────────────────────────

def _get_disease(name: str) -> dict:
    """Case-insensitive disease lookup."""
    for k, v in _summary.items():
        if k.lower() == name.lower():
            return v
    raise HTTPException(status_code=404, detail=f"Disease '{name}' not found.")


def _flatten_forecast(disease_info: dict, horizon: str) -> list[dict]:
    """Return flat list of forecast rows (one per month) for Power BI."""
    rows = []
    disease_name = disease_info["disease_name"]
    category     = disease_info["disease_category"]

    for model_key, model_data in disease_info.get("models", {}).items():
        if "error" in model_data:
            continue
        forecasts = model_data.get("forecasts", {}).get(horizon, [])
        for entry in forecasts:
            rows.append({
                "disease_name":     disease_name,
                "disease_category": category,
                "metric":           model_key,
                "date":             entry["ds"],
                "predicted":        entry["yhat"],
                "lower_bound":      entry["yhat_lower"],
                "upper_bound":      entry["yhat_upper"],
            })
    return rows


# ── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "diseases_loaded": len(_summary)}


@app.get("/diseases")
def list_diseases(category: Optional[str] = None):
    """List all diseases that have trained models."""
    result = []
    for name, info in _summary.items():
        if category and info.get("disease_category", "").lower() != category.lower():
            continue
        result.append({
            "disease_name":     name,
            "disease_category": info.get("disease_category"),
            "n_months_history": info.get("n_months_history"),
        })
    return result


@app.get("/predict/{disease_name}")
def predict_disease(
    disease_name: str,
    horizon: str = Query("1_year", description="1_month | 1_year | 3_years"),
    flat: bool   = Query(True,    description="Return flat rows (best for Power BI)"),
):
    """
    Return forecast for a single disease.
    Power BI usage:
      Web connector → http://localhost:8000/predict/End-stage renal disease (disorder)?horizon=3_years
    """
    info = _get_disease(disease_name)
    if flat:
        return _flatten_forecast(info, horizon)

    # Full nested response
    result = {
        "disease_name":     info["disease_name"],
        "disease_category": info["disease_category"],
        "horizon":          horizon,
        "forecasts": {}
    }
    for model_key, model_data in info.get("models", {}).items():
        if "error" not in model_data:
            result["forecasts"][model_key] = {
                "predictions": model_data["forecasts"].get(horizon, []),
                "metrics":     model_data.get("metrics", {}),
            }
    return result


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    """
    Batch endpoint – forecasts for multiple diseases.
    Returns flat list → easy to load in Power BI via Python script connector.
    """
    rows = []
    for disease in req.diseases:
        try:
            info = _get_disease(disease)
            rows.extend(_flatten_forecast(info, req.horizon))
        except HTTPException:
            pass   # skip unknown diseases silently
    return rows


@app.get("/bed_demand")
def bed_demand(
    horizon:  str = Query("1_year",  description="1_month | 1_year | 3_years"),
    category: Optional[str] = Query(None, description="Filter by disease_category"),
    top_n:    int = Query(20,        description="Return top N diseases by predicted bed-days"),
):
    """
    Aggregate bed-day demand forecast across diseases.
    Returns one row per disease per month showing total predicted bed-days.
    Ideal for hospital capacity planning dashboards.
    """
    rows = []
    for name, info in _summary.items():
        if category and info.get("disease_category", "").lower() != category.lower():
            continue
        bed_model = info.get("models", {}).get("bed_days", {})
        if "error" in bed_model or not bed_model:
            continue
        forecasts = bed_model.get("forecasts", {}).get(horizon, [])
        for entry in forecasts:
            rows.append({
                "disease_name":     name,
                "disease_category": info["disease_category"],
                "date":             entry["ds"],
                "predicted_bed_days": entry["yhat"],
                "lower":              entry["yhat_lower"],
                "upper":              entry["yhat_upper"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return []

    # Rank by total predicted bed-days over the horizon
    totals = df.groupby("disease_name")["predicted_bed_days"].sum().nlargest(top_n).index
    df = df[df["disease_name"].isin(totals)]
    return df.to_dict(orient="records")


@app.get("/summary_table")
def summary_table():
    """
    One-row-per-disease summary with 1-month, 1-year, and 3-year totals.
    Perfect for a Power BI overview table.
    """
    rows = []
    for name, info in _summary.items():
        row = {
            "disease_name":     name,
            "disease_category": info.get("disease_category"),
            "months_history":   info.get("n_months_history"),
        }
        for model_key in ["case_count", "avg_los", "bed_days"]:
            m = info.get("models", {}).get(model_key, {})
            for horizon in ["1_month", "1_year", "3_years"]:
                preds = m.get("forecasts", {}).get(horizon, [])
                col_prefix = f"{model_key}_{horizon}"
                if preds:
                    vals = [p["yhat"] for p in preds]
                    row[f"{col_prefix}_total"] = round(sum(vals), 2)
                    row[f"{col_prefix}_avg"]   = round(np.mean(vals), 2)
                else:
                    row[f"{col_prefix}_total"] = None
                    row[f"{col_prefix}_avg"]   = None

            # Add accuracy metrics
            if "metrics" in m:
                row[f"{model_key}_MAE"]    = m["metrics"].get("MAE")
                row[f"{model_key}_MAPE_%"] = m["metrics"].get("MAPE_%")
        rows.append(row)
    return rows


# ── DASHBOARD ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the web dashboard."""
    if not os.path.exists(DASHBOARD_HTML):
        return HTMLResponse(content="<h1>Dashboard HTML not found</h1><p>Please ensure dashboard.html exists in the same directory.</p>", status_code=404)
    with open(DASHBOARD_HTML, encoding="utf-8") as f:
        return f.read()


# ── CHATBOT ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


# Build a context string from loaded forecasts so Gemini knows the data
def _build_dashboard_context() -> str:
    if not _summary:
        return "No forecast data is currently loaded."
    diseases_list = list(_summary.keys())
    categories = list(set(info.get("disease_category", "Unknown") for info in _summary.values()))
    # Top 10 by bed demand
    bed_ranking = []
    for name, info in _summary.items():
        bd = info.get("models", {}).get("bed_days", {})
        if "error" not in bd and bd:
            preds_1y = bd.get("forecasts", {}).get("1_year", [])
            total = sum(p["yhat"] for p in preds_1y) if preds_1y else 0
            bed_ranking.append((name, round(total)))
    bed_ranking.sort(key=lambda x: x[1], reverse=True)
    top10 = bed_ranking[:10]
    top10_str = "\n".join(f"  {i+1}. {n}: {v:,} predicted bed-days (1-year)" for i, (n, v) in enumerate(top10))
    return (
        f"Dashboard currently tracks {len(diseases_list)} diseases across {len(categories)} categories.\n"
        f"Categories: {', '.join(sorted(categories))}\n"
        f"Top 10 diseases by predicted bed demand (1-year):\n{top10_str}\n"
        f"All diseases: {', '.join(diseases_list[:30])}{'...' if len(diseases_list) > 30 else ''}"
    )


SYSTEM_PROMPT = """
You are the Disease Forecasting Dashboard AI Assistant.
You help users understand the dashboard, its predictions, and hospital capacity planning.

About the dashboard:
- It uses machine learning (GradientBoostingRegressor) trained on historical hospital encounter data.
- It predicts 3 metrics per disease: case count, average length of stay (days), and total bed-days.
- Forecasts are available for 3 horizons: 1 month, 1 year, and 3 years ahead.
- The model captures trend, seasonality, and recent patterns using time-series feature engineering.
- Confidence ranges show prediction uncertainty — narrow = confident, wide = uncertain.
- There are 100 diseases tracked, each with 3 separate prediction models (300 total).
- The dashboard has 3 charts (case count, bed demand, avg length of stay) and filters by category/disease.

Be concise, friendly, and helpful. Use bullet points. If asked about specific diseases, use the data context provided.
"""


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """AI chatbot endpoint using Google Gemini with model fallback."""
    import time
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        return {"reply": "The Gemini API key is not configured. Please add your key to the .env file. Get a free key at https://aistudio.google.com"}

    # Try multiple models in case one has exhausted its free-tier quota
    MODELS_TO_TRY = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"]
    context = _build_dashboard_context()
    prompt = f"{SYSTEM_PROMPT}\n\nCurrent dashboard data:\n{context}\n\nUser question: {req.message}"

    import google.generativeai as genai
    genai.configure(api_key=api_key)

    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return {"reply": response.text}
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                time.sleep(3)
                continue
            elif "404" in err_str:
                continue  # model not available, try next
            else:
                return {"reply": f"Sorry, something went wrong: {err_str}"}

    return {"reply": f"All models are currently rate-limited. Please wait about 60 seconds and try again. (Error: quota exceeded)"}
