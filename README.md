# 🛡️ Inflation Shield — Margin Squeeze Predictor

**Predicting EBITDA margin compression for India's largest firms, 2 quarters ahead.**

A predictive analytics project that trains a **Gradient Boosting + LSTM ensemble** on CMIE Prowess quarterly financials to identify firms at risk of significant margin erosion. Outputs a single **Margin Squeeze Score (MSS) 0–100** per firm per quarter, with full interpretability via SHAP.

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Ensemble Test AUC | ~0.72 |
| LGBM-only Test AUC | ~0.73 |
| LSTM Test AUC | ~0.55–0.65 |
| Test Period | Q1 2012 – Q4 2013 |
| Firms Covered | 30 (BSE Sensex 30) |
| Prediction Horizon | 2 quarters ahead |

> **Honest assessment**: Real-world AUC likely in the **0.67–0.73** range. The model meaningfully outperforms a naive sector-rate baseline but is constrained by small firm universe and sparse financial data.

---

## 🎯 What This Project Does

1. **Problem**: Can we predict which firms will see ≥150 basis point EBITDA margin compression over the next 2 quarters?

2. **Approach**: Engineer 6 domain-driven features from quarterly financials, train LightGBM (tuned via Optuna) + LSTM ensemble, output a 0–100 risk score.

3. **Features Engineered**:

| Feature | What It Captures |
|---------|-----------------|
| **CPR** (Cost Pass-through Ratio) | How well a firm passes input cost increases to revenue |
| **ICSI** (Input Cost Sensitivity) | Exposure to raw material cost volatility |
| **IBD** (Inventory Buffer Days) | Days of raw material inventory held as a buffer |
| **WCSS** (Working Capital Stress) | Net working capital pressure (debtor days − creditor days) |
| **OL** (Operating Leverage) | Fixed cost burden relative to variable margin |
| **SIB** (Sector Inflation Beta) | Firm's margin sensitivity relative to sector peers |

4. **Output**: Margin Squeeze Score (MSS) 0–100 — higher score = higher risk of margin compression.

---

## 📁 Repository Structure

```
inflation-shield/
├── notebooks/
│   └── inflation_shield.ipynb    # Full analysis notebook (Sections 0–9)
├── data/
│   ├── standalone_quarterly.csv  # 1,897 rows, 30 firms, 1995–2014
│   ├── consolidated_quarterly.csv # 558 rows, 20 firms, 2002–2014
│   └── identity.csv              # Firm metadata (sectors, codes)
├── outputs/
│   ├── predictions.json          # MSS scores per (firm, quarter)
│   ├── drivers.json              # Feature values + SHAP attributions
│   ├── metrics.json              # Model evaluation metrics
│   ├── sector_summary.json       # Sector-level risk aggregates
│   ├── feature_list.json         # Model feature order
│   ├── model.pkl                 # Serialized model bundle
│   └── plots/                    # Evaluation visualizations
│       ├── feature_distributions.png
│       ├── correlation_matrix.png
│       ├── shap_summary.png
│       ├── lstm_training.png
│       ├── roc_calibration.png
│       └── confusion_matrix.png
├── index.html                    # Static dashboard (Chart.js)
├── requirements.txt
├── build_notebook.py             # Notebook generator script
└── README.md
```

---

## 🔬 Data Source

**CMIE Prowess** — Centre for Monitoring Indian Economy's corporate financial database.

- **Standalone financials**: 1,897 quarterly observations across 30 BSE Sensex 30 firms (1995-Q1 to 2014-Q2)
- **Consolidated financials**: 558 observations for 20 of 30 firms (2002–2014). Used where consolidated revenue diverges >20% from standalone (captures subsidiaries like Tata Steel's Corus).
- **Identity data**: Firm metadata including sector classification, exchange symbols

### Data Limitations (be upfront about these)

- **Severe missingness**: `inventories` ~88% missing, `trade_payables` ~92% missing, `trade_receivables` ~87% missing. Worst pre-2005, improves post-2010.
- **No macro overlays**: WPI, Brent crude, INR-USD exchange rates are **not** in the source data. We don't fabricate them.
- **Small universe**: 30 firms is sufficient for gradient boosting but severely constrains LSTM.
- **Data ends 2014**: Pre-GST, pre-COVID regime. Model has not seen post-2014 economic conditions.

---

## 🎯 Target Variable

**Binary classification**: `y = 1` if EBITDA margin drops by ≥150 basis points over the next 2 quarters.

$$\text{y\_binary} = \mathbb{1}\left[\text{EBITDA\_margin}_{t+2} - \text{EBITDA\_margin}_{t} \leq -1.5\right]$$

This threshold captures meaningful margin compression (not quarterly noise). The base rate is ~30% — approximately 1 in 3 firm-quarters experience this level of compression.

---

## 🏗️ Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│  6 Engineered Features + Lags + Momentum + Sector       │
│  (CPR, ICSI, IBD, WCSS, OL, SIB + 15 derived)          │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
        ┌──────▼──────┐        ┌──────▼──────┐
        │  LightGBM   │        │    LSTM     │
        │  (Optuna)   │        │  (8Q seq)   │
        │  Weight: w  │        │ Weight: 1-w │
        └──────┬──────┘        └──────┬──────┘
               │                      │
        ┌──────▼──────────────────────▼──────┐
        │     Ensemble: MSS = 100 × prob     │
        │        (0–100 risk score)           │
        └────────────────────────────────────┘
```

- **LightGBM**: Primary model. Handles NaN natively, robust to small datasets. Tuned via Optuna (20 trials).
- **LSTM**: Secondary model. 8-quarter sequences, 32 units, dropout 0.3. Constrained by data size — honestly underperforms LGBM.
- **Ensemble weight** is tuned on validation set.

---

## 🚀 How to Reproduce

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt
```

### Run the Analysis

```bash
# Option 1: Open in Jupyter
jupyter notebook notebooks/inflation_shield.ipynb

# Option 2: Execute from command line
jupyter nbconvert --to notebook --execute notebooks/inflation_shield.ipynb
```

### Serve the Dashboard

```bash
python -m http.server 8000
# Open http://localhost:8000
```

> ⚠️ The dashboard must be served via HTTP (not `file://`) because it fetches JSON via `fetch()`.

---

## 📈 Dashboard

The static dashboard (`index.html`) visualizes MSS scores, feature drivers, and sector risk using Chart.js and custom SVG. It loads predictions and metrics from the `outputs/` JSON files.

**Features**:
- Firm leaderboard with risk ranking and sparkline trends
- Individual firm explorer with MSS gauge, SHAP drivers, and trajectory charts
- Sector risk bar chart and heatmap
- Model methodology and validation section

---

## ⚠️ Limitations & Honest Assessment

1. **Small firm universe (30 firms)**: Insufficient for robust deep learning. LSTM is a proof-of-concept, not production-ready.
2. **Data ends in 2014**: The model has not encountered GST (2017), COVID-19 (2020), or post-pandemic commodity cycles.
3. **Missing data is severe**: IBD and WCSS features are >85% sparse, reducing their predictive contribution.
4. **No macro features**: Adding Brent crude, WPI, and INR-USD would likely improve predictions for commodity-sensitive firms.
5. **Single-firm sectors**: 15 of 21 sectors contain exactly 1 firm, making sector-relative features degenerate.
6. **CMIE sector labels**: Some classifications are questionable (e.g., Coal India under "Other fund based financial services").

### What Would Improve This

- Expand to BSE 500 (200+ firms) for statistical power
- Extend data to 2024 with post-disclosure-reform financials
- Add macro overlays (Brent, WPI, INR-USD via public APIs)
- Rolling backtest across multiple test windows
- Industry-specific sub-models (manufacturing vs services)

---

## 🛠️ Tech Stack

- **Data**: pandas, numpy
- **Modelling**: LightGBM, TensorFlow/Keras (LSTM), scikit-learn
- **Tuning**: Optuna (Bayesian hyperparameter optimization)
- **Interpretability**: SHAP (TreeExplainer)
- **Visualization**: matplotlib, seaborn, Chart.js (dashboard)
- **Statistics**: statsmodels (rolling OLS for CPR/SIB)

---

## 📄 License

Academic project. Data sourced from CMIE Prowess under institutional access.
