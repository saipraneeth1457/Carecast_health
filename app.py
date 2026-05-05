"""
Disease Forecasting Streamlit Application
==========================================
Run:  streamlit run app.py
"""

import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit.components.v1 as components
from streamlit.errors import StreamlitSecretNotFoundError

load_dotenv()

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareCast Health Insights",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME COLOURS ────────────────────────────────────────────────────────────
PRIMARY   = "#1F4E79"
SECONDARY = "#2E75B6"
ACCENT    = "#C55A11"
SUCCESS   = "#375623"
WARNING   = "#FFC000"
DANGER    = "#C00000"
LIGHT_BG  = "#F0F4F8"

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F7F9FC; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px 20px;
        border-left: 5px solid #2E75B6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 8px;
    }
    .metric-card.orange { border-left-color: #C55A11; }
    .metric-card.green  { border-left-color: #375623; }
    .metric-card.red    { border-left-color: #C00000; }

    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1F2937;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 12px;
        color: #9CA3AF;
        margin-top: 4px;
    }

    .alert-box {
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 14px;
        line-height: 1.5;
    }
    .alert-high {
        background: #FEF2F2;
        border-left: 4px solid #C00000;
        color: #7F1D1D;
    }
    .alert-medium {
        background: #FFFBEB;
        border-left: 4px solid #FFC000;
        color: #78350F;
    }
    .alert-info {
        background: #EFF6FF;
        border-left: 4px solid #2E75B6;
        color: #1E3A5F;
    }

    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #1F4E79;
        padding-bottom: 6px;
        border-bottom: 2px solid #DBEAFE;
        margin-bottom: 16px;
        margin-top: 8px;
    }

    .rank-badge {
        display: inline-block;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #1F4E79;
        color: white;
        font-size: 13px;
        font-weight: 700;
        text-align: center;
        line-height: 28px;
        margin-right: 8px;
    }
    .rank-badge.gold   { background: #B45309; }
    .rank-badge.silver { background: #6B7280; }
    .rank-badge.bronze { background: #92400E; }

    .disease-row {
        background: white;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        display: flex;
        align-items: center;
    }

    div[data-testid="stSidebarContent"] {
        background: #1F4E79;
    }
    div[data-testid="stSidebarContent"] p, 
    div[data-testid="stSidebarContent"] h1, 
    div[data-testid="stSidebarContent"] h2, 
    div[data-testid="stSidebarContent"] h3, 
    div[data-testid="stSidebarContent"] span {
        color: white !important;
    }
    /* Ensure selectbox text is dark for readability on white background */
    .stSelectbox div[data-baseweb="select"] * {
        color: #1F4E79 !important;
    }
    div[data-testid="stSidebarContent"] .stSelectbox label,
    div[data-testid="stSidebarContent"] .stRadio label {
        color: #CBD5E1 !important;
        font-size: 13px;
    }

    .page-title {
        font-size: 30px;
        font-weight: 800;
        color: #1F4E79;
        margin-bottom: 2px;
    }
    .page-subtitle {
        font-size: 14px;
        color: #6B7280;
        margin-bottom: 20px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #EFF6FF;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 6px 18px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #1F4E79;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ── FLOATING CHAT IFRAME CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Keep the chat component pinned at top-right */
    div[data-testid="stHtml"]:last-of-type {
        position: fixed !important;
        top: 70px !important;
        right: 0 !important;
        width: 420px !important;
        height: 570px !important;
        z-index: 999999 !important;
        pointer-events: none !important;
    }
    div[data-testid="stHtml"]:last-of-type iframe {
        pointer-events: all !important;
        background: transparent !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ── AI CHATBOT CONFIG ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the CareCast Health Insights AI Assistant.
You help users understand the dashboard, its predictions, and hospital capacity planning.

About the dashboard:
- It uses machine learning (GradientBoostingRegressor/Prophet) trained on historical hospital encounter data.
- It predicts 3 metrics per disease: case count, average length of stay (days), and total bed-days.
- Forecasts are available for 3 horizons: 1 month, 1 year, and 3 years ahead.
- The model captures trend, seasonality, and recent patterns using time-series feature engineering.
- Confidence ranges show prediction uncertainty — narrow = confident, wide = uncertain.
- There are 100 diseases tracked, each with 3 separate prediction models (300 total).
- The dashboard has interactive charts, filters, and automated alerts for high-demand diseases.

Be concise, friendly, and helpful. Use bullet points. If asked about specific diseases, use the data context provided.
"""

def build_chat_context(data: dict) -> str:
    """Build a context string for the LLM based on current dashboard data."""
    if not data: return "No forecast data is currently loaded."
    diseases_list = list(data.keys())
    categories = list(set(info.get("disease_category", "Unknown") for info in data.values()))
    
    # Top 5 by bed demand for context
    bed_ranking = []
    for name, info in data.items():
        bd = info.get("models", {}).get("bed_days", {})
        if "error" not in bd and bd:
            preds_1y = bd.get("forecasts", {}).get("1_year", [])
            total = sum(p["yhat"] for p in preds_1y) if preds_1y else 0
            bed_ranking.append((name, round(total)))
    bed_ranking.sort(key=lambda x: x[1], reverse=True)
    top5 = bed_ranking[:5]
    top5_str = "\n".join(f"  - {n}: {v:,} predicted bed-days (1-year)" for n, v in top5)

    return (
        f"Dashboard tracks {len(diseases_list)} diseases across {len(categories)} categories.\n"
        f"Categories: {', '.join(sorted(categories)[:15])}...\n"
        f"Top 5 diseases by predicted bed demand (1-year):\n{top5_str}\n"
    )


# ── DATA LOADING ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_forecast_data():
    """Load forecast_summary.json — works offline (no API needed)."""
    paths = [
        Path("models/forecast_summary.json"),
        Path("forecast_summary.json"),
    ]
    for p in paths:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def build_summary_df(data: dict) -> pd.DataFrame:
    """Flatten forecast_summary.json into one row per disease."""
    rows = []
    for name, info in data.items():
        row = {
            "disease_name":     name,
            "disease_category": info.get("disease_category", "Unknown"),
            "months_history":   info.get("n_months_history", 0),
        }
        for mk in ["case_count", "avg_los", "bed_days"]:
            m = info.get("models", {}).get(mk, {})
            if "error" in m or not m:
                for hz in ["1_month", "1_year", "3_years"]:
                    row[f"{mk}_{hz}_total"] = None
                    row[f"{mk}_{hz}_avg"]   = None
                row[f"{mk}_MAPE_%"] = None
                continue
            for hz, h in [("1_month", 1), ("1_year", 12), ("3_years", 36)]:
                preds = m.get("forecasts", {}).get(hz, [])
                vals  = [p["yhat"] for p in preds] if preds else []
                row[f"{mk}_{hz}_total"] = round(sum(vals), 2)    if vals else None
                row[f"{mk}_{hz}_avg"]   = round(np.mean(vals), 2) if vals else None
            row[f"{mk}_MAPE_%"] = m.get("metrics", {}).get("MAPE_%")
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_forecast_series(data: dict, disease: str, metric: str, horizon: str):
    """Return (dates, yhats, lowers, uppers) for a disease/metric/horizon."""
    info      = data.get(disease, {})
    model     = info.get("models", {}).get(metric, {})
    forecasts = model.get("forecasts", {}).get(horizon, [])
    if not forecasts:
        return [], [], [], []
    dates  = [f["ds"]         for f in forecasts]
    yhats  = [f["yhat"]        for f in forecasts]
    lowers = [f["yhat_lower"]  for f in forecasts]
    uppers = [f["yhat_upper"]  for f in forecasts]
    return dates, yhats, lowers, uppers


@st.cache_data(show_spinner=False)
def compute_trend_growth(data: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate YoY growth: compare avg of first 6 forecast months vs last 6
    for case_count over the 3-year horizon.
    """
    rows = []
    for name, info in data.items():
        m = info.get("models", {}).get("case_count", {})
        if "error" in m or not m:
            continue
        preds = m.get("forecasts", {}).get("3_years", [])
        if len(preds) < 12:
            continue
        vals   = [p["yhat"] for p in preds]
        early  = np.mean(vals[:6])
        late   = np.mean(vals[-6:])
        growth = ((late - early) / (early + 1e-6)) * 100
        rows.append({
            "disease_name":     name,
            "disease_category": info.get("disease_category", "Unknown"),
            "growth_%":         round(growth, 1),
            "early_avg":        round(early, 1),
            "late_avg":         round(late, 1),
        })
    return pd.DataFrame(rows).sort_values("growth_%", ascending=False)


@st.cache_data(show_spinner=False)
def compute_trend_growth_for_horizon(data: dict, horizon: str) -> pd.DataFrame:
    """
    Compute growth % for the selected horizon using case_count forecasts.
    Growth is measured from first to last forecast point for each disease.
    """
    rows = []
    for name, info in data.items():
        m = info.get("models", {}).get("case_count", {})
        if "error" in m or not m:
            continue
        preds = m.get("forecasts", {}).get(horizon, [])
        if len(preds) < 2:
            continue

        vals = [p["yhat"] for p in preds]
        first = vals[0]
        last = vals[-1]
        growth = ((last - first) / (abs(first) + 1e-6)) * 100
        rows.append({
            "disease_name": name,
            "disease_category": info.get("disease_category", "Unknown"),
            "growth_%": round(growth, 1),
            "first_point": round(first, 1),
            "last_point": round(last, 1),
        })

    if not rows:
        return pd.DataFrame(columns=["disease_name", "disease_category", "growth_%", "first_point", "last_point"])
    return pd.DataFrame(rows).sort_values("growth_%", ascending=False)


# ── CHART HELPERS ────────────────────────────────────────────────────────────

def forecast_chart(dates, yhats, lowers, uppers, title, y_label, color=SECONDARY):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=uppers + lowers[::-1],
        fill="toself",
        fillcolor=f"rgba(46,117,182,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence range",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=lowers,
        line=dict(color="rgba(46,117,182,0.35)", width=1, dash="dot"),
        name="Lower bound", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=uppers,
        line=dict(color="rgba(46,117,182,0.35)", width=1, dash="dot"),
        name="Upper bound", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=yhats,
        line=dict(color=color, width=2.5),
        name="Predicted",
        mode="lines+markers",
        marker=dict(size=5),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=PRIMARY)),
        yaxis_title=y_label,
        xaxis_title="Date",
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F3F4F6")
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6")
    return fig


def top_n_bar(df_top, x_col, y_col, title, color_col=None, n=10):
    df_plot = df_top.head(n).copy()
    colors  = px.colors.sequential.Blues[3:]
    if color_col:
        cats   = df_plot[color_col].unique().tolist()
        palette = px.colors.qualitative.Set2
        cmap   = {c: palette[i % len(palette)] for i, c in enumerate(cats)}
        bar_colors = df_plot[color_col].map(cmap).tolist()
    else:
        bar_colors = colors[: len(df_plot)]

    fig = go.Figure(go.Bar(
        x=df_plot[x_col],
        y=df_plot[y_col].apply(lambda v: v[:45] + "…" if isinstance(v, str) and len(v) > 45 else v),
        orientation="h",
        marker_color=bar_colors,
        text=df_plot[x_col].apply(lambda v: f"{v:,.1f}"),
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=PRIMARY)),
        height=max(320, n * 38),
        margin=dict(l=10, r=60, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F3F4F6")
    return fig


def donut_chart(labels, values, title):
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=px.colors.qualitative.Set2),
        textinfo="percent+label",
        textfont_size=12,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=PRIMARY)),
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


# ── SIDEBAR ──────────────────────────────────────────────────────────────────

def render_sidebar(data, df_summary, df_trend):
    with st.sidebar:
        st.markdown("## 🏥 CareCast Health Insights")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["📊 Overview", "🔍 Disease Explorer", "🏆 Top Diseases", "📈 Growth & Trends", "🚨 Alerts"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Filter by Category**")
        categories = ["All"] + sorted(df_summary["disease_category"].dropna().unique().tolist())
        selected_cat = st.selectbox("Category", categories, label_visibility="collapsed")

        st.markdown("**Forecast Horizon**")
        horizon = st.selectbox(
            "Horizon", ["1_month", "1_year", "3_years"],
            format_func=lambda x: {"1_month": "1 Month", "1_year": "1 Year", "3_years": "3 Years"}[x],
            label_visibility="collapsed",
        )

        st.markdown("---")
        total_d = len(df_summary)
        ok_d    = df_summary["case_count_1_year_total"].notna().sum()
        st.markdown(f"**{total_d}** diseases tracked")
        st.markdown(f"**{ok_d}** models ready")

    return page, selected_cat, horizon



def render_floating_chat(data):
    """Render a beautiful floating chat bubble + popup window using pure HTML/CSS/JS."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    except StreamlitSecretNotFoundError:
        api_key = os.getenv("GEMINI_API_KEY", "")
    ctx = build_chat_context(data).replace("'", "\\'").replace("\n", " ")[:2000]
    prompt_prefix = SYSTEM_PROMPT.replace("'", "\\'").replace("\n", " ")

    chat_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: transparent; overflow: hidden; font-family: 'Inter', sans-serif; }}

    .chat-bubble {{
        position: fixed; top: 24px; right: 24px;
        width: 56px; height: 56px; border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        border: none; cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 20px rgba(99,102,241,0.5);
        z-index: 99999; transition: transform 0.2s;
    }}
    .chat-bubble:hover {{ transform: scale(1.1); }}
    .chat-bubble svg {{ width: 26px; height: 26px; fill: white; }}

    .chat-window {{
        position: fixed; top: 90px; right: 24px;
        width: 360px; height: 460px;
        background: #0f172a;
        border: 1px solid #334155; border-radius: 16px;
        display: none; flex-direction: column;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        z-index: 99998; overflow: hidden;
        animation: slideUp 0.3s ease;
    }}
    .chat-window.open {{ display: flex; }}
    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .chat-header {{
        padding: 14px 16px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white; display: flex;
        justify-content: space-between; align-items: center;
    }}
    .chat-header h4 {{ margin: 0; font-size: 14px; font-weight: 600; }}
    .chat-header .close-btn {{
        background: rgba(255,255,255,0.2); border: none;
        color: white; width: 28px; height: 28px;
        border-radius: 50%; cursor: pointer;
        font-size: 16px; display: flex;
        align-items: center; justify-content: center;
        transition: background 0.2s;
    }}
    .chat-header .close-btn:hover {{ background: rgba(255,255,255,0.35); }}

    .chat-messages {{
        flex: 1; padding: 14px; overflow-y: auto;
        display: flex; flex-direction: column; gap: 10px;
        background: #0f172a;
    }}
    .chat-messages::-webkit-scrollbar {{ width: 4px; }}
    .chat-messages::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 4px; }}

    .msg {{
        max-width: 85%; padding: 10px 14px;
        border-radius: 12px; font-size: 13px;
        line-height: 1.5; color: #f1f5f9;
        word-wrap: break-word;
    }}
    .msg.bot {{ background: #1e293b; align-self: flex-start; border-bottom-left-radius: 4px; }}
    .msg.user {{ background: #6366f1; align-self: flex-end; border-bottom-right-radius: 4px; }}

    .chat-input-area {{
        padding: 12px; border-top: 1px solid #1e293b;
        display: flex; gap: 8px; background: #111827;
    }}
    .chat-input-area input {{
        flex: 1; padding: 10px 14px; border-radius: 10px;
        border: 1px solid #334155; background: #1e293b;
        color: #f1f5f9; outline: none; font-size: 13px;
        font-family: 'Inter', sans-serif;
    }}
    .chat-input-area input::placeholder {{ color: #64748b; }}
    .chat-input-area input:focus {{ border-color: #6366f1; }}
    .chat-input-area button {{
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border: none; border-radius: 10px; color: white;
        padding: 0 16px; cursor: pointer; font-weight: 600;
        font-size: 13px; transition: opacity 0.2s;
    }}
    .chat-input-area button:hover {{ opacity: 0.85; }}
    </style>

    <button class="chat-bubble" onclick="toggleChat()" id="chatBubble">
        <svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/></svg>
    </button>

    <div class="chat-window" id="chatWin">
        <div class="chat-header">
            <h4>🤖 CareCast Assistant</h4>
            <button class="close-btn" onclick="toggleChat()">&times;</button>
        </div>
        <div class="chat-messages" id="chatMsgs">
            <div class="msg bot">Hi! I'm the CareCast Assistant. You can ask me about hospital demand forecasts, model accuracy, or specific diseases. 
            <br><br><b>Suggested Questions:</b>
            <ul style="margin-left: 15px; margin-top: 5px;">
                <li>What is the purpose of this dashboard?</li>
                <li>Which models are used for forecasting?</li>
                <li>What metrics are predicted?</li>
                <li>What horizons can I view?</li>
                <li>Which diseases have the highest bed demand?</li>
                <li>How many diseases are tracked?</li>
                <li>How accurate are the models (MAPE)?</li>
                <li>How should I read the confidence ranges?</li>
                <li>Does the model handle seasonality?</li>
                <li>What are the top 5 diseases by case count?</li>
            </ul>
            </div>
        </div>
        <div class="chat-input-area">
            <input type="text" id="chatInp" placeholder="Ask about trends..." onkeypress="if(event.key==='Enter')sendChat()">
            <button onclick="sendChat()">Send</button>
        </div>
    </div>

    <script>
    function toggleChat() {{
        document.getElementById('chatWin').classList.toggle('open');
    }}

    async function sendChat() {{
        const inp = document.getElementById('chatInp');
        const msg = inp.value.trim();
        if(!msg) return;
        inp.value = '';

        const msgs = document.getElementById('chatMsgs');
        msgs.innerHTML += '<div class="msg user">' + msg + '</div>';
        msgs.scrollTop = msgs.scrollHeight;

        const typing = document.createElement('div');
        typing.className = 'msg bot';
        typing.innerText = 'Thinking...';
        msgs.appendChild(typing);
        msgs.scrollTop = msgs.scrollHeight;

        const apiKey = '{api_key}';
        const models = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash'];
        let success = false;

        for (const modelName of models) {{
            try {{
                const url = 'https://generativelanguage.googleapis.com/v1beta/models/' + modelName + ':generateContent?key=' + apiKey;
                const resp = await fetch(url, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        contents: [{{ parts: [{{ text: '{prompt_prefix} Context: {ctx} User Question: ' + msg }}] }}]
                    }})
                }});
                const data = await resp.json();
                if (data.candidates && data.candidates[0] && data.candidates[0].content) {{
                    typing.innerText = data.candidates[0].content.parts[0].text;
                    success = true;
                    break;
                }}
            }} catch (e) {{
                continue;
            }}
        }}

        if (!success) {{
            typing.innerText = 'Sorry, the AI is currently busy. Please try again in a moment.';
        }}
        msgs.scrollTop = msgs.scrollHeight;
    }}
    </script>
    """
    components.html(chat_html, height=560)


# ── METRIC CARD HTML ─────────────────────────────────────────────────────────

def metric_card(label, value, sub="", style=""):
    return f"""
    <div class="metric-card {style}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """


def alert_box(text, level="info"):
    return f'<div class="alert-box alert-{level}">{text}</div>'


# ── PAGES ────────────────────────────────────────────────────────────────────

def metric_columns_for_horizon(horizon: str):
    return (
        f"case_count_{horizon}_total",
        f"bed_days_{horizon}_total",
        f"avg_los_{horizon}_avg",
    )


def page_overview(data, df_summary, df_trend, horizon):
    st.markdown('<div class="page-title">🏥 CareCast Health Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">ML-based hospital demand insights for cases, bed utilization, and length of stay</div>', unsafe_allow_html=True)

    cases_col, beds_col, los_col = metric_columns_for_horizon(horizon)

    # ── KPI Row ───────────────────────────────────────────────────────────
    total_cases = df_summary[cases_col].sum()
    total_beds = df_summary[beds_col].sum()
    avg_los = df_summary[los_col].mean()
    high_demand = (df_summary[cases_col] > df_summary[cases_col].quantile(0.9)).sum()
    top_disease = df_summary.nlargest(1, cases_col)["disease_name"].values[0]
    top_disease_short = top_disease[:30] + "…" if len(top_disease) > 30 else top_disease
    avg_mape         = df_summary["case_count_MAPE_%"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card(f"Total Cases ({horizon.replace('_', ' ').title()})", f"{int(total_cases):,}", "Across filtered diseases"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(f"Total Bed-Days ({horizon.replace('_', ' ').title()})", f"{int(total_beds):,}", "Hospital capacity demand", "orange"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(f"Avg Length of Stay ({horizon.replace('_', ' ').title()})", f"{avg_los:.1f} days", "Average across filtered diseases", "green"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Model Accuracy", f"{100 - avg_mape:.0f}%", f"Avg across 100 diseases (MAPE: {avg_mape:.1f}%)", "red"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Top diseases bar + category donut ──────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown(f'<div class="section-header">Top 10 Diseases by Cases ({horizon.replace("_", " ").title()})</div>', unsafe_allow_html=True)
        top10 = df_summary.nlargest(10, cases_col)[["disease_name", cases_col, "disease_category"]].dropna()
        fig = top_n_bar(top10, cases_col, "disease_name", "", color_col="disease_category", n=10)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Cases by Category</div>', unsafe_allow_html=True)
        cat_totals = df_summary.groupby("disease_category")[cases_col].sum().dropna().nlargest(8)
        fig2 = donut_chart(cat_totals.index.tolist(), cat_totals.values.tolist(), "")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Fastest Growing</div>', unsafe_allow_html=True)
        top3_growth = df_trend.head(3)
        for _, row in top3_growth.iterrows():
            name_short = row["disease_name"][:38] + "…" if len(row["disease_name"]) > 38 else row["disease_name"]
            arrow = "🔺" if row["growth_%"] > 0 else "🔻"
            st.markdown(f"**{arrow} {name_short}**  `+{row['growth_%']:.1f}%`", unsafe_allow_html=False)

    # ── Row 3: Bed demand top 5 ───────────────────────────────────────────
    st.markdown(f'<div class="section-header">Top 5 Diseases by Bed Demand ({horizon.replace("_", " ").title()})</div>', unsafe_allow_html=True)
    top5_beds = df_summary.nlargest(5, beds_col)[["disease_name", beds_col, los_col, cases_col, "disease_category"]].dropna()

    cols = st.columns(5)
    colors_list = ["#1F4E79", "#2E75B6", "#C55A11", "#375623", "#7B3F00"]
    for i, (_, row) in enumerate(top5_beds.iterrows()):
        with cols[i]:
            name_s = row["disease_name"][:28] + "…" if len(row["disease_name"]) > 28 else row["disease_name"]
            st.markdown(f"""
            <div style="background:white;border-radius:10px;padding:14px;border-top:4px solid {colors_list[i]};box-shadow:0 2px 6px rgba(0,0,0,0.07);text-align:center">
                <div style="font-size:11px;color:#6B7280;font-weight:600;text-transform:uppercase;margin-bottom:6px">{name_s}</div>
                <div style="font-size:22px;font-weight:700;color:{colors_list[i]}">{int(row[beds_col]):,}</div>
                <div style="font-size:11px;color:#9CA3AF">bed-days ({horizon.replace("_", " ")})</div>
                <div style="font-size:12px;color:#374151;margin-top:6px">LOS: <b>{row[los_col]:.1f}d</b></div>
            </div>
            """, unsafe_allow_html=True)


def page_disease_explorer(data, df_summary, horizon, selected_cat):
    st.markdown('<div class="page-title">🔍 Disease Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Select any disease to see its detailed forecast</div>', unsafe_allow_html=True)

    # Disease selector
    col_sel, col_hz = st.columns([3, 1])
    with col_sel:
        diseases_available = sorted([
            n for n, info in data.items()
            if "error" not in info.get("models", {}).get("case_count", {})
            and (selected_cat == "All" or info.get("disease_category", "Unknown") == selected_cat)
        ])
        if not diseases_available:
            st.warning("No diseases found for this category. Change the category filter.")
            return
        default_idx = diseases_available.index("End-stage renal disease (disorder)") if "End-stage renal disease (disorder)" in diseases_available else 0
        selected = st.selectbox("Select Disease", diseases_available, index=default_idx)

    info = data.get(selected, {})
    cat  = info.get("disease_category", "Unknown")
    hist = info.get("n_months_history", 0)

    # Disease info row
    c1, c2, c3, c4 = st.columns(4)
    cc_model = info.get("models", {}).get("case_count", {})
    bd_model = info.get("models", {}).get("bed_days", {})
    los_model = info.get("models", {}).get("avg_los", {})

    mape_cc  = cc_model.get("metrics", {}).get("MAPE_%", "N/A") if "metrics" in cc_model else "N/A"
    mape_bd  = bd_model.get("metrics", {}).get("MAPE_%", "N/A") if "metrics" in bd_model else "N/A"

    cc_vals = [p["yhat"] for p in cc_model.get("forecasts", {}).get(horizon, [])]
    bd_vals = [p["yhat"] for p in bd_model.get("forecasts", {}).get(horizon, [])]
    los_vals = [p["yhat"] for p in los_model.get("forecasts", {}).get(horizon, [])]
    cc_total = sum(cc_vals) if cc_vals else 0
    bd_total = sum(bd_vals) if bd_vals else 0
    los_avg = np.mean(los_vals) if los_vals else 0
    hz_label = horizon.replace("_", " ").title()

    with c1: st.markdown(metric_card("Category", cat, f"{hist} months of history"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card(f"Forecast Cases ({hz_label})", f"{int(cc_total):,}", f"Model accuracy: {100 - mape_cc:.0f}%" if mape_cc != "N/A" else ""), unsafe_allow_html=True)
    with c3: st.markdown(metric_card(f"Bed-Days Demand ({hz_label})", f"{int(bd_total):,}", f"MAPE: {mape_bd}%", "orange"), unsafe_allow_html=True)
    with c4: st.markdown(metric_card(f"Avg Length of Stay ({hz_label})", f"{los_avg:.1f} days", "Selected forecast horizon", "green"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Forecast charts
    col_a, col_b = st.columns(2)

    with col_a:
        dates, yhats, lowers, uppers = get_forecast_series(data, selected, "case_count", horizon)
        if dates:
            fig = forecast_chart(dates, yhats, lowers, uppers, "Monthly Case Count Forecast", "Predicted Cases", SECONDARY)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No case count forecast available for this disease.")

    with col_b:
        dates, yhats, lowers, uppers = get_forecast_series(data, selected, "bed_days", horizon)
        if dates:
            fig = forecast_chart(dates, yhats, lowers, uppers, "Monthly Bed-Days Demand Forecast", "Predicted Bed-Days", ACCENT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No bed demand forecast available.")

    # LOS chart full width
    dates, yhats, lowers, uppers = get_forecast_series(data, selected, "avg_los", horizon)
    if dates:
        fig = forecast_chart(dates, yhats, lowers, uppers, "Average Length of Stay Forecast", "Days per Case", SUCCESS)
        st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    with st.expander("📋 View Raw Forecast Numbers"):
        if cc_model.get("forecasts", {}).get(horizon):
            preds = cc_model["forecasts"][horizon]
            bd_preds = bd_model.get("forecasts", {}).get(horizon, [{}] * len(preds))
            rows = []
            for i, (p, b) in enumerate(zip(preds, bd_preds)):
                rows.append({
                    "Date":            p["ds"],
                    "Predicted Cases": round(p["yhat"], 1),
                    "Cases Lower":     round(p["yhat_lower"], 1),
                    "Cases Upper":     round(p["yhat_upper"], 1),
                    "Bed-Days":        round(b.get("yhat", 0), 1),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


def page_top_diseases(data, df_summary, selected_cat, horizon):
    st.markdown('<div class="page-title">🏆 Top Diseases</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Rankings by case volume, bed demand, and length of stay</div>', unsafe_allow_html=True)

    df = df_summary.copy()
    if selected_cat != "All":
        df = df[df["disease_category"] == selected_cat]

    cases_col, beds_col, los_col = metric_columns_for_horizon(horizon)
    tab1, tab2, tab3 = st.tabs(["📊 Top 3 Quick View", "🔟 Top 10 Detailed", "📋 Full Table"])

    with tab1:
        st.markdown(f"### Top 3 by Cases ({horizon.replace('_', ' ').title()})")
        top3 = df.nlargest(3, cases_col).reset_index(drop=True)
        badge_classes = ["gold", "silver", "bronze"]
        badge_labels  = ["🥇", "🥈", "🥉"]
        cols = st.columns(3)
        for i, (_, row) in enumerate(top3.iterrows()):
            with cols[i]:
                name_s = row["disease_name"][:35] + "…" if len(row["disease_name"]) > 35 else row["disease_name"]
                los_v = row[los_col]
                beds_v = row[beds_col]
                st.markdown(f"""
                <div style="background:white;border-radius:12px;padding:20px;box-shadow:0 3px 10px rgba(0,0,0,0.09);text-align:center;min-height:200px">
                    <div style="font-size:36px;margin-bottom:8px">{badge_labels[i]}</div>
                    <div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:12px">{name_s}</div>
                    <div style="font-size:26px;font-weight:700;color:#1F4E79">{int(row[cases_col]):,}</div>
                    <div style="font-size:11px;color:#9CA3AF;margin-bottom:10px">predicted cases ({horizon.replace("_", " ")})</div>
                    <div style="font-size:12px;color:#6B7280">🛏 {int(beds_v):,} bed-days &nbsp;|&nbsp; ⏱ {los_v:.1f}d LOS</div>
                    <div style="margin-top:8px">
                        <span style="background:#EFF6FF;color:#1F4E79;font-size:11px;padding:3px 10px;border-radius:12px;font-weight:600">{row['disease_category']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### Top 3 by Bed Demand ({horizon.replace('_', ' ').title()})")
        top3_beds = df.nlargest(3, beds_col).reset_index(drop=True)
        cols2 = st.columns(3)
        for i, (_, row) in enumerate(top3_beds.iterrows()):
            with cols2[i]:
                name_s = row["disease_name"][:35] + "…" if len(row["disease_name"]) > 35 else row["disease_name"]
                st.markdown(f"""
                <div style="background:white;border-radius:12px;padding:16px;box-shadow:0 3px 10px rgba(0,0,0,0.09);text-align:center">
                    <div style="font-size:28px;margin-bottom:6px">{badge_labels[i]}</div>
                    <div style="font-size:12px;font-weight:600;color:#374151;margin-bottom:8px">{name_s}</div>
                    <div style="font-size:22px;font-weight:700;color:#C55A11">{int(row[beds_col]):,}</div>
                    <div style="font-size:11px;color:#9CA3AF">bed-days ({horizon.replace("_", " ")})</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        col_left, col_right = st.columns(2)
        with col_left:
            top10_cases = df.nlargest(10, cases_col)[["disease_name", cases_col, "disease_category"]].dropna()
            fig = top_n_bar(top10_cases, cases_col, "disease_name", f"Top 10 by Forecast Cases ({horizon.replace('_', ' ').title()})", color_col="disease_category", n=10)
            st.plotly_chart(fig, use_container_width=True)
        with col_right:
            top10_beds = df.nlargest(10, beds_col)[["disease_name", beds_col, "disease_category"]].dropna()
            fig2 = top_n_bar(top10_beds, beds_col, "disease_name", f"Top 10 by Bed-Days Demand ({horizon.replace('_', ' ').title()})", color_col="disease_category", n=10)
            st.plotly_chart(fig2, use_container_width=True)

        # Top 10 LOS
        top10_los = df.nlargest(10, los_col)[["disease_name", los_col, "disease_category"]].dropna()
        fig3 = top_n_bar(top10_los, los_col, "disease_name", f"Top 10 by Avg Length of Stay ({horizon.replace('_', ' ').title()})", color_col="disease_category", n=10)
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        display_df = df[["disease_name", "disease_category", cases_col, beds_col, los_col, "case_count_MAPE_%"]].copy()
        display_df.columns = ["Disease", "Category", "Cases", "Bed-Days", "Avg LOS (days)", "MAPE %"]
        display_df = display_df.sort_values("Cases", ascending=False).reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, height=500)


def page_growth_trends(data, df_summary, df_trend, selected_cat, horizon):
    st.markdown('<div class="page-title">📈 Growth & Trends</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">Diseases with the fastest predicted growth over the selected horizon ({horizon.replace("_", " ").title()})</div>', unsafe_allow_html=True)

    df_t = compute_trend_growth_for_horizon(data, horizon)
    if selected_cat != "All":
        df_t = df_t[df_t["disease_category"] == selected_cat]

    if df_t.empty:
        st.info("Not enough forecast points to compute growth for this horizon. Try 1 Year or 3 Years.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Top 10 Fastest Growing Diseases</div>', unsafe_allow_html=True)
        top_growing = df_t[df_t["growth_%"] > 0].head(10)
        if not top_growing.empty:
            fig = go.Figure(go.Bar(
                x=top_growing["growth_%"],
                y=top_growing["disease_name"].apply(lambda x: x[:40] + "…" if len(x) > 40 else x),
                orientation="h",
                marker=dict(
                    color=top_growing["growth_%"],
                    colorscale="Oranges",
                    showscale=False,
                ),
                text=top_growing["growth_%"].apply(lambda v: f"+{v:.1f}%"),
                textposition="outside",
            ))
            fig.update_layout(
                height=400, margin=dict(l=10, r=60, t=20, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(autorange="reversed"),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#F3F4F6", title=f"Growth % ({horizon.replace('_', ' ').title()} first vs last point)")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Top 10 Declining Diseases</div>', unsafe_allow_html=True)
        declining = df_t[df_t["growth_%"] < 0].tail(10).sort_values("growth_%")
        if not declining.empty:
            fig2 = go.Figure(go.Bar(
                x=declining["growth_%"],
                y=declining["disease_name"].apply(lambda x: x[:40] + "…" if len(x) > 40 else x),
                orientation="h",
                marker=dict(
                    color=declining["growth_%"],
                    colorscale="Blues_r",
                    showscale=False,
                ),
                text=declining["growth_%"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            ))
            fig2.update_layout(
                height=400, margin=dict(l=10, r=60, t=20, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(autorange="reversed"),
            )
            fig2.update_xaxes(showgrid=True, gridcolor="#F3F4F6", title="Growth %")
            st.plotly_chart(fig2, use_container_width=True)

    # Scatter: growth vs volume
    st.markdown('<div class="section-header">Growth Rate vs Case Volume — Bubble Chart</div>', unsafe_allow_html=True)
    cases_col, beds_col, _ = metric_columns_for_horizon(horizon)
    merged = df_t.merge(df_summary[["disease_name", cases_col, beds_col]], on="disease_name", how="left").dropna()
    fig3 = px.scatter(
        merged.head(60),
        x="growth_%",
        y=cases_col,
        size=beds_col,
        color="disease_category",
        hover_name="disease_name",
        labels={"growth_%": f"Predicted Growth % ({horizon.replace('_', ' ')})", cases_col: f"Forecast Cases ({horizon.replace('_', ' ')})"},
        title="Growth Rate vs Volume — bubble size = bed-days",
        height=420,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    fig3.update_xaxes(showgrid=True, gridcolor="#F3F4F6", zeroline=True, zerolinecolor="#E5E7EB")
    fig3.update_yaxes(showgrid=True, gridcolor="#F3F4F6")
    st.plotly_chart(fig3, use_container_width=True)


def page_alerts(data, df_summary, df_trend, selected_cat, horizon):
    st.markdown('<div class="page-title">🚨 Alerts & Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Automatically detected patterns that require attention</div>', unsafe_allow_html=True)

    # Thresholds
    cases_col, beds_col, los_col = metric_columns_for_horizon(horizon)
    base = df_summary.copy()
    if selected_cat != "All":
        base = base[base["disease_category"] == selected_cat]
    trend_base = df_trend.copy()
    if selected_cat != "All":
        trend_base = trend_base[trend_base["disease_category"] == selected_cat]

    p90_cases = base[cases_col].quantile(0.90)
    p90_beds = base[beds_col].quantile(0.90)
    p75_los = base[los_col].quantile(0.75)
    p90_growth = trend_base["growth_%"].quantile(0.90) if not trend_base.empty else 50

    high_demand = base[base[cases_col] >= p90_cases].sort_values(cases_col, ascending=False)
    high_beds = base[base[beds_col] >= p90_beds].sort_values(beds_col, ascending=False)
    long_stay = base[base[los_col] >= p75_los].sort_values(los_col, ascending=False).head(10)
    fast_growing = trend_base[trend_base["growth_%"] >= p90_growth].head(8)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<div class="section-header">🔴 High Case Volume Alerts ({len(high_demand)})</div>', unsafe_allow_html=True)
        st.markdown(alert_box(f"<b>Threshold:</b> {int(p90_cases):,}+ cases ({horizon.replace('_', ' ')}, top 10%)", "high"), unsafe_allow_html=True)
        for _, row in high_demand.head(8).iterrows():
            name_s = row["disease_name"][:50] + "…" if len(row["disease_name"]) > 50 else row["disease_name"]
            st.markdown(alert_box(
                f"⚠️ <b>{name_s}</b><br>"
                f"Category: {row['disease_category']} &nbsp;|&nbsp; "
                f"Forecast: <b>{int(row[cases_col]):,}</b> cases ({horizon.replace('_', ' ')}) &nbsp;|&nbsp; "
                f"LOS: <b>{row[los_col]:.1f}</b> days",
                "high"
            ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">🟡 Long Stay Diseases ({len(long_stay)})</div>', unsafe_allow_html=True)
        st.markdown(alert_box(f"<b>Threshold:</b> {p75_los:.1f}+ days average LOS ({horizon.replace('_', ' ')}, top 25%)", "medium"), unsafe_allow_html=True)
        for _, row in long_stay.iterrows():
            name_s = row["disease_name"][:50] + "…" if len(row["disease_name"]) > 50 else row["disease_name"]
            st.markdown(alert_box(
                f"🛏️ <b>{name_s}</b><br>"
                f"Avg LOS: <b>{row[los_col]:.1f} days</b> &nbsp;|&nbsp; "
                f"Bed-days: {int(row[beds_col]):,}",
                "medium"
            ), unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="section-header">🔴 High Bed Demand Alerts ({len(high_beds)})</div>', unsafe_allow_html=True)
        st.markdown(alert_box(f"<b>Threshold:</b> {int(p90_beds):,}+ bed-days ({horizon.replace('_', ' ')}, top 10%)", "high"), unsafe_allow_html=True)
        for _, row in high_beds.head(8).iterrows():
            name_s = row["disease_name"][:50] + "…" if len(row["disease_name"]) > 50 else row["disease_name"]
            st.markdown(alert_box(
                f"🏥 <b>{name_s}</b><br>"
                f"Bed-days: <b>{int(row[beds_col]):,}</b> &nbsp;|&nbsp; "
                f"Cases: {int(row[cases_col]):,} &nbsp;|&nbsp; "
                f"LOS: {row[los_col]:.1f}d",
                "high"
            ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if not fast_growing.empty:
            st.markdown(f'<div class="section-header">📈 Rapidly Growing Diseases ({len(fast_growing)})</div>', unsafe_allow_html=True)
            st.markdown(alert_box(f"<b>Threshold:</b> {p90_growth:.0f}%+ predicted growth over 3 years (top 10%)", "info"), unsafe_allow_html=True)
            for _, row in fast_growing.iterrows():
                name_s = row["disease_name"][:50] + "…" if len(row["disease_name"]) > 50 else row["disease_name"]
                st.markdown(alert_box(
                    f"📈 <b>{name_s}</b><br>"
                    f"Category: {row['disease_category']} &nbsp;|&nbsp; "
                    f"Predicted growth: <b>+{row['growth_%']:.1f}%</b> over 3 years",
                    "info"
                ), unsafe_allow_html=True)

    # Summary counts
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Alert Summary</div>', unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1: st.markdown(metric_card("High Case Volume", str(len(high_demand)), "diseases require attention", "red"), unsafe_allow_html=True)
    with sc2: st.markdown(metric_card("High Bed Demand", str(len(high_beds)), "diseases at capacity risk", "red"), unsafe_allow_html=True)
    with sc3: st.markdown(metric_card("Long Stay Diseases", str(len(long_stay)), "above 75th percentile LOS", "orange"), unsafe_allow_html=True)
    with sc4: st.markdown(metric_card("Rapid Growth", str(len(fast_growing)), "diseases growing >top 10%", "orange"), unsafe_allow_html=True)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    with st.spinner("Loading forecast data…"):
        data       = load_forecast_data()
        df_summary = build_summary_df(data)
        df_trend   = compute_trend_growth(data, df_summary)

    if not data:
        st.error("❌ No forecast data found. Please run `python train_model.py` first and make sure `models/forecast_summary.json` exists in the same folder as this app.")
        st.stop()

    page, selected_cat, horizon = render_sidebar(data, df_summary, df_trend)

    df_filtered = df_summary.copy()
    if selected_cat != "All":
        df_filtered = df_summary[df_summary["disease_category"] == selected_cat]

    df_trend_filtered = df_trend.copy()
    if selected_cat != "All":
        df_trend_filtered = df_trend[df_trend["disease_category"] == selected_cat]


    if page == "📊 Overview":
        page_overview(data, df_filtered, df_trend_filtered, horizon)
    elif page == "🔍 Disease Explorer":
        page_disease_explorer(data, df_summary, horizon, selected_cat)
    elif page == "🏆 Top Diseases":
        page_top_diseases(data, df_summary, selected_cat, horizon)
    elif page == "📈 Growth & Trends":
        page_growth_trends(data, df_summary, df_trend, selected_cat, horizon)
    elif page == "🚨 Alerts":
        page_alerts(data, df_summary, df_trend, selected_cat, horizon)

    # Always render the floating chat bubble (in main area, not sidebar)
    render_floating_chat(data)


if __name__ == "__main__":
    main()
