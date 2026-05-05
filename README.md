# Disease Forecasting Dashboard
### Complete Setup & User Guide

---

## What This Dashboard Does

This dashboard uses a machine learning model trained on hospital encounter data to predict:

- **How long patients stay** in hospital per disease (average length of stay in days)
- **How many new cases** each disease will produce in the next 1 month, 1 year, and 3 years
- **How many bed-days** each disease will demand from the hospital over the next 3 years

The dashboard has **3 charts** and **2 filters** that work together. Click any disease or category in the filters and all charts update instantly.

---

## What You Need Before Starting

Please install the following on your PC before anything else.

### 1. Python 3.10
Download from: https://www.python.org/downloads/
- During installation tick the box that says **"Add Python to PATH"**
- To check it installed correctly open Command Prompt and type: `python --version`
- You should see something like `Python 3.10.x`

### 2. Power BI Desktop (free)
Download from: https://powerbi.microsoft.com/desktop
- Sign in with a Microsoft account (free to create if you don't have one)

### 3. The Project Files
You should have received a folder called `disease_prediction` containing:

```
disease_prediction/
├── train_model.py
├── api.py
├── powerbi_connector.py
├── requirements.txt
├── Disease_Forecasting_Dashboard.pbix     ← Power BI dashboard file
├── disease_monthly_summary_powerbi.csv    ← your data file (200MB+)
└── models/
    ├── forecast_summary.json
    └── (100+ .pkl model files)
```

> **Important:** If the `models/` folder is empty or missing, you will need to run Step 3 (Train the models) below before the dashboard will work.

---

## First-Time Setup (Do This Once)

### Step 1 — Install Python packages

Open **Command Prompt** (search "cmd" in the Start menu).

Navigate to the project folder by typing:
```
cd C:\Users\YourName\Desktop\disease_prediction
```
Replace `YourName\Desktop` with wherever you saved the folder.

Then run:
```
pip install -r requirements.txt
```
Wait for it to finish. This installs all the libraries the model needs.

---

### Step 2 — Configure Power BI to use Python

1. Open **Power BI Desktop**
2. Click **File** → **Options and settings** → **Options**
3. Click **Python scripting** in the left panel
4. Make sure the Python path shown matches where you installed Python
   - It should look like: `C:\Users\YourName\AppData\Local\Programs\Python\Python310`
   - If it shows a different path, click **Other** and paste the correct path
5. Click **OK**

---

### Step 3 — Train the models (first time only, takes ~5 minutes)

> **Skip this step if the `models/` folder already contains .pkl files and forecast_summary.json**

In Command Prompt (still in the project folder) run:
```
python train_model.py
```

You will see a progress bar. When it finishes it will print:
```
DONE!
Successful models : 300 / 300
Errors            : 0
```

This creates all the prediction models. You only need to do this once unless the data changes.

---

## Running the Dashboard (Every Time)

You need to do **two things** every time you want to use the dashboard.

### Step A — Start the prediction API

Open Command Prompt, navigate to the project folder, and run:
```
uvicorn api:app --host 0.0.0.0 --port 8000
```

When you see this message the API is ready:
```
INFO:     Application startup complete.
```

**Leave this Command Prompt window open.** Do not close it while using the dashboard. If you close it the charts will stop working.

---

### Step B — Open the dashboard in Power BI

1. Open **Power BI Desktop**
2. Click **File** → **Open report** → **Browse**
3. Navigate to the project folder and open `Disease_Forecasting_Dashboard.pbix`
4. If Power BI asks you to refresh the data click **Refresh**
5. The dashboard loads with all 3 charts populated

---

## Using the Dashboard

### The 3 Charts

**Chart 1 — Average Hospital Stay Duration by Disease (Days)**
- A horizontal bar chart
- Shows which diseases keep patients in hospital the longest
- Longer bar = more bed days used per patient per visit
- Use this to identify which diseases put the most pressure on bed capacity

**Chart 2 — Hospital Bed Demand Forecast — Next 3 Years**
- A line chart showing total bed-days demanded per month
- The middle line is the prediction
- The upper and lower lines are the confidence range (best case / worst case)
- Use this for hospital capacity planning over the next 3 years

**Chart 3 — Monthly Case Count Forecast with Confidence Range**
- A line chart showing how many new cases are expected each month
- The middle line is the predicted case count
- The upper and lower lines show the uncertainty range
- Use this to plan staffing and resource allocation

---

### The 2 Filters (Slicers)

Both filters are on the left side of the dashboard. They work together.

**Filter by Disease Category**
- Click a category (e.g. Renal, Cancer, Cardiovascular) to filter all 3 charts to only that category
- Click again to deselect

**Filter by Disease**
- Scroll through the list and click any specific disease
- All 3 charts instantly update to show data for that disease only
- To select multiple diseases hold **Ctrl** and click each one
- To clear the filter click the eraser icon that appears in the top right of the slicer

**Tip:** Use both filters together — first click a category to shorten the disease list, then click a specific disease.

---

## Shutting Down

When you are done:
1. Close Power BI Desktop (save if prompted)
2. Go to the Command Prompt running the API
3. Press **Ctrl + C** to stop it
4. Close the Command Prompt window

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Charts show no data / blank | Make sure the API is running (Step A). Open a new Command Prompt and run `uvicorn api:app --host 0.0.0.0 --port 8000` |
| "Module not found" error when starting API | Run `pip install -r requirements.txt` again in the project folder |
| Power BI says "Unable to connect" | The API is not running. Complete Step A first |
| "No module named requests" in Power BI | Open Command Prompt and run: `C:\Users\YourName\AppData\Local\Programs\Python\Python310\python.exe -m pip install requests pandas numpy` |
| Charts load but show wrong numbers | Click **Home → Refresh** in Power BI to pull fresh predictions from the API |
| Training fails with errors | Make sure `disease_monthly_summary_powerbi.csv` is in the same folder as `train_model.py` |
| Power BI opens but Python script fails | Go to File → Options → Python scripting and confirm the path matches your Python installation |

---

## Understanding the Predictions

The model was trained on your historical hospital encounter data going back decades. It uses the following logic to predict the future:

- **Trend** — is this disease increasing or decreasing over the years?
- **Seasonality** — does this disease spike in certain months every year?
- **Recent patterns** — what happened in the last 1, 3, and 12 months?

The **confidence range** (upper and lower lines on the charts) tells you how certain the model is. A narrow range means the model is confident. A wide range means there is more uncertainty, usually because the disease has irregular patterns.

The model covers the **top 100 diseases** by case volume in your data. Each disease has 3 separate prediction models:
- One for case count
- One for average length of stay
- One for total bed-days demanded

---

## Refreshing the Data

If you receive a new data file (updated `disease_monthly_summary_powerbi.csv`):

1. Replace the old CSV file in the project folder with the new one
2. Open Command Prompt in the project folder
3. Run `python train_model.py` to retrain the models on the new data (takes ~5 minutes)
4. Restart the API: `uvicorn api:app --host 0.0.0.0 --port 8000`
5. Open Power BI and click **Refresh**

Your dashboard will now reflect the latest data.

---

## Contact & Support

For any technical issues with this dashboard please contact the development team.

---

*Dashboard built with Python · scikit-learn · FastAPI · Power BI Desktop*
