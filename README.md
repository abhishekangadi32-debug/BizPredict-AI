# BizPredict AI

**An AI-powered business intelligence and predictive analytics platform for small retailers.**

BizPredict AI turns raw retail sales data into decisions: it cleans whatever CSV you give it,
explains what it did, forecasts future sales with a validated machine-learning model, and turns
that forecast into safety-stock and reorder-point recommendations.

It is not a mocked dashboard. Every number on screen is computed from the dataset you load, and
every figure the data cannot support is labelled as unavailable instead of being invented.

---

## 1. Introduction

A small retailer usually has sales data but not the analytics team to interpret it. Spreadsheets
answer "what happened last month" but not "what will next month look like, how confident am I,
and how much stock should I hold to serve 95% of demand during a 7-day lead time?".

BizPredict AI closes that gap with a modular Python application and an interactive Streamlit
dashboard across five pages: Executive Overview, Sales Analytics, AI Sales Forecasting,
Inventory Intelligence and Data Explorer.

## 2. Features

**Data ingestion**
- Loads the bundled public Pentaho sample dataset (2,823 order lines, 2003–2005).
- Loads the bundled 156,000-row merged weekly retail dataset (20 columns).
- Accepts any uploaded CSV whose schema it can recognise.
- Automatic encoding detection (UTF-8, UTF-8-BOM, CP1252, Latin-1), delimiter sniffing,
  and header normalisation.
- Schema-agnostic column mapping: `ORDERDATE`/`date`, `SALES`/`weekly_sales`,
  `QUANTITYORDERED`/`quantity`, `PRODUCTLINE`/`department`/`category`, and more.
- Clear, actionable error messages when a required column is missing — including the list of
  columns that *were* found.

**Preprocessing**

- Multi-format date parsing with a day-first fallback that only engages when it rescues more rows.
- Duplicate removal, invalid-date removal, non-numeric and negative sales removal — each counted.
- Sales derived from `quantity × price` only when the sales value is genuinely absent, and flagged.
- Transactional data aggregated to daily totals; weekly data summed per date without double counting.
- Calendar features: year, quarter, month, ISO week, day, day-of-week, weekend flag.
- Full preprocessing statistics surfaced in the UI.

**Analytics**

- Revenue, orders, average order value, units sold, top category, average daily sales.
- Unique order counting when an order ID exists; line-item counting (clearly labelled) when it does not.
- Daily / weekly / monthly revenue, category and store breakdowns, distribution, seasonality,
  weekday profile, year × month heat map, holiday vs regular comparison, growth and volatility.

**Forecasting**

- Prophet as the primary model (trend, yearly/weekly seasonality, holiday regressors).
- Holt-Winters exponential smoothing and a linear-trend baseline as automatic fallbacks.
- Strictly chronological train/test split — no future observation ever enters training.
- Horizon slider from 7 to 60 days, with prediction intervals.
- MAE, RMSE and MAPE (with zero-actual handling), plus WAPE and bias for robustness.
- Actual-vs-predicted chart, residual chart, model comparison table, forecast CSV download.

**Inventory**

- Safety stock, reorder point, stockout probability, days of cover, recommended order quantity.
- Configurable lead time (1–90 days) and service level (80%–99.9%).
- Optional current-stock input, globally or per category.

**Interface**

- Light and dark themes, responsive layout, KPI cards, interactive Plotly charts.
- Clean sidebar navigation across five pages.

## 3. System architecture

```
                ┌──────────────────────────┐
                │   Data ingestion         │  data_loader.py
                │   CSV / URL / upload     │  encoding + delimiter detection
                └────────────┬─────────────┘
                             │ raw DataFrame
                ┌────────────▼─────────────┐
                │   Schema discovery       │  detect_schema()
                │   alias → canonical map  │  explicit errors if required missing
                └────────────┬─────────────┘
                             │ unified records
                ┌────────────▼─────────────┐
                │   Validation & cleaning  │  preprocessing.py
                │   dates, numerics, dupes │  auditable statistics
                └────────────┬─────────────┘
                             │ cleaned records + period series
                ┌────────────▼─────────────┐
                │   EDA / analytics        │  analytics.py
                │   KPIs, trends, segments │
                └────────────┬─────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
 ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
 │ Forecasting    │ │ Evaluation     │ │ Inventory      │
 │ Prophet → HW → │ │ MAE/RMSE/MAPE  │ │ safety stock,  │
 │ linear trend   │ │ chronological  │ │ ROP, risk      │
 └────────┬───────┘ └────────┬───────┘ └────────┬───────┘
          └──────────────────┼──────────────────┘
                             ▼
                ┌──────────────────────────┐
                │   Streamlit dashboard    │  app.py
                │   5 pages, 2 themes      │
                └──────────────────────────┘
```

Each layer is a separate module with no Streamlit dependency, so the pipeline can be tested
from a terminal:

```bash
python tests/smoke_test.py
```

## 4. Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ (tested on 3.12) |
| Data handling | pandas, NumPy |
| Machine learning | Prophet (Stan backend), statsmodels (Holt-Winters) |
| Statistics | SciPy (normal quantile for the service-level factor) |
| Visualisation | Plotly |
| Application | Streamlit |
| Testing | Plain-Python assertion harness (`tests/smoke_test.py`) |

## 5. Folder structure

```
BizPredict-AI/
├── app.py                      # Streamlit dashboard (UI only)
├── config.py                   # every tunable constant
├── requirements.txt
├── README.md
├── data/
│   ├── sales_data_sample.csv           # bundled default dataset
│   ├── retail_sales_complete_dataset.csv  # optional 156k merged dataset
│   └── README.md
├── modules/
│   ├── __init__.py
│   ├── data_loader.py          # ingestion + schema discovery
│   ├── preprocessing.py        # cleaning, aggregation, features
│   ├── analytics.py            # KPIs and descriptive analytics
│   ├── forecasting.py          # Prophet / Holt-Winters / linear
│   ├── evaluation.py           # MAE, RMSE, MAPE, WAPE
│   └── inventory.py            # safety stock, ROP, stockout risk
├── assets/
│   └── logo.png
├── outputs/
│   └── README.md               # generated exports land here
├── docs/
│   ├── Project_Report.md       # academic mini-project report
│   ├── Presentation.md         # 8-slide deck
│   └── Viva_Questions.md       # 20 viva Q&A
└── tests/
    └── smoke_test.py           # end-to-end pipeline verification
```

## 6. Installation

Python 3.10 or newer is required. Prophet pulls in a Stan backend; on most systems the wheel
installs cleanly. If Prophet cannot be installed, the app still runs and automatically falls back
to Holt-Winters.

### Windows (PowerShell or CMD)

```bat
cd BizPredict-AI
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
cd BizPredict-AI
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> If your machine already has an Anaconda installation, you can skip the virtual environment and
> install the requirements into the base environment instead.

## 7. Running the application

```bash
streamlit run app.py
```

The dashboard opens at <http://localhost:8501>. To use a different port:

```bash
streamlit run app.py --server.port 8502
```

Verify the data pipeline without starting the UI:

```bash
python tests/smoke_test.py
```

## 8. Dataset information

### 8.1 Default dataset

Public Pentaho sample (`data/sales_data_sample.csv`), 2,823 order lines, 25 columns,
January 2003 – May 2005.

Columns used: `ORDERDATE`, `SALES`, `QUANTITYORDERED`, `PRICEEACH`, `PRODUCTLINE`,
`ORDERNUMBER`, `STATUS`.

> **Important characteristic:** this file is a *sample*. Only 252 of the 877 calendar days in its
> range contain any order. Empty days are therefore treated as **not observed** and dropped rather
> than filled with zeros — zero-filling would fabricate 625 zero-sales days and destroy every
> downstream metric. The app states this in the Data Explorer and on the forecasting page.

The same dataset can be fetched from the original URL, which is configured in `config.py`.

### 8.2 Merged weekly dataset

`data/retail_sales_complete_dataset.csv` — 156,000 rows × 20 columns, 156 consecutive weeks
(2022-01-01 to 2024-12-21), 50 stores × 20 departments × 156 weeks.

Columns include `store_id`, `department`, `date`, `weekly_sales`, `sales_is_holiday`,
`holiday_name`, `temperature`, `fuel_price`, `markdown_1..5`, `cpi`, `unemployment`, `season`,
`store_type`, `store_size`, `region`, `feature_is_holiday`.

Mapped automatically: `date → date`, `weekly_sales → sales`, `department → category`,
`store_id → store`, `sales_is_holiday → is_holiday`, `holiday_name → holiday_name`.
`feature_is_holiday` is left unmapped (it duplicates the sales-level flag) and shown only for reference.

### 8.3 Your own CSV

Any CSV with a recognisable date column and a sales column works. Matching ignores case, spaces,
underscores and punctuation. To support a new header, add it to `COLUMN_ALIASES` in `config.py` —
no other file needs to change.

## 9. Machine learning methodology

1. **Aggregation.** Records are aggregated to one observation per period (day or week).
2. **Aggregation choice for forecasting.** The forecasting page lets you model daily, weekly or
   monthly totals. Weekly is the default for daily data because sparse, high-variance daily sales
   make weekly totals considerably more stable.
3. **Chronological split.** The newest `TEST_FRACTION` (20%) of periods is held out. The model is
   fitted on the older part only, then scored on the hold-out — no leakage.
4. **Model ladder.** Prophet → Holt-Winters → linear trend. Each fallback records the reason.
5. **Refit.** After evaluation, the chosen model is refitted on the full history to produce the
   forward forecast, because the hold-out window is now part of the known past.
6. **Intervals.** Prophet produces its own uncertainty interval. Holt-Winters and the linear
   baseline use residual spread scaled by `√h`, which is an approximation and is labelled as such.
7. **Flooring.** Predictions are floored at zero: negative sales are not physical.

### Metrics

```
MAE  = (1/n) · Σ |yᵢ − ŷᵢ|
RMSE = √( (1/n) · Σ (yᵢ − ŷᵢ)² )
MAPE = (100/n) · Σ |(yᵢ − ŷᵢ) / yᵢ|
WAPE = 100 · Σ|yᵢ − ŷᵢ| / Σ|yᵢ|
```

**MAPE limitation.** MAPE is undefined when an actual value is zero. Periods with zero actual
sales are excluded from the MAPE denominator, and the number excluded is shown next to the metric.
WAPE is reported alongside because it remains valid when zeros are present. On datasets with many
zero periods, ranking models by MAPE alone is misleading.

## 10. Inventory formulas

```
Safety Stock  = Z × σ_d × √L
Reorder Point = (d̄ × L) + Safety Stock
Stockout risk = 1 − Φ( (S − d̄·L) / (σ_d·√L) )
```

where:

- `Z` — service-level factor, `Φ⁻¹(service level)`
- `σ_d` — standard deviation of daily demand
- `d̄` — average daily demand
- `L` — lead time in days
- `S` — on-hand stock

For weekly datasets, `d̄ = mean_weekly / 7` and `σ_d = σ_weekly / √7`. The second relation assumes
independent daily demand within a week; it is an approximation and is stated in the UI.

**Data limitation, stated honestly:** neither bundled dataset contains inventory balances or
supplier lead times. Safety stock and reorder points are therefore *estimates* from demand
history plus the lead time and service level you supply. Reorder decisions and stockout
probabilities appear only after you enter current stock.

## 11. Limitations

- **No inventory or supplier data.** Safety stock and ROP are demand-based estimates.
- **The default dataset is a sparse sample.** Only 252 of 877 days have transactions; forecast
  error on it is therefore high (measured MAPE ≈ 47% at weekly granularity). The 156k weekly
  dataset produces much tighter forecasts (measured MAPE ≈ 1.7%).
- **Prophet cannot know the future.** It extrapolates past structure. It does not know about a new
  competitor, a price change, a promotion it has never seen, or a supply disruption.
- **Univariate.** The forecast uses the aggregated sales series (plus holiday flags when present).
  Price, markdowns, CPI and fuel price are loaded but not used as regressors.
- **Category-level forecasts are not produced.** Forecasting is on the aggregated series and per
  modelling granularity; inventory planning uses per-category historical demand.
- **Currency is a label only.** Values are displayed with a configurable symbol; no exchange-rate
  conversion is performed.
- **Streamlit session state.** Uploaded files live in browser session memory and are cleared when
  the session ends.

## 12. Future enhancements

- Hierarchical forecasting per store/department with reconciliation.
- Promotion and markdown effects as Prophet regressors or a gradient-boosted model on lag features.
- Probabilistic inventory optimisation (newsvendor / service-level cost trade-off) and EOQ.
- Automated data refresh and scheduled report generation.
- Authentication, multi-tenant workspaces and persistent storage.
- Model monitoring: drift detection and automatic retraining triggers.
- Export to PDF/Excel executive packs.

## 13. Testing the application

A scripted walkthrough is in `docs/Project_Report.md` (section 17) and repeated below.

| Feature | How to check |
|---|---|
| Default dataset loads | Open the app; the Executive Overview header should show 2,823 records and 252 daily periods |
| Merged dataset loads | Sidebar → *Merged weekly dataset (156k)*; header shows 156,000 records, weekly series, 156 periods |
| Upload | Sidebar → *Upload my own CSV*; any CSV with a date and a sales column |
| Schema error | Upload a CSV without a date/sales column; a red error lists the columns found |
| KPI correctness | Total Revenue on the Overview equals the sum of the Revenue trend values |
| Forecast | Forecasting page → horizon 30 → *Generate forecast*; metric cards, interval band and CSV download appear |
| Forecast leakage check | The forecast line starts after the last observed date; the evaluation window is strictly later than training |
| MAPE zero handling | On data containing zero-sales periods, the warning states how many periods MAPE excluded |
| Inventory | Set lead time 7 and service level 95%; toggle *I have current stock figures* and enter a value; status, risk and order quantity appear |
| Downloads | Every page exposes a CSV download that opens correctly |
| Theme | Sidebar → Dark; cards, charts and axes restyle |

## 14. Academic material

- [`docs/Project_Report.md`](docs/Project_Report.md) — full mini-project report
- [`docs/Presentation.md`](docs/Presentation.md) — 8-slide deck with speaker notes
- [`docs/Viva_Questions.md`](docs/Viva_Questions.md) — 20 viva questions with answers

## 15. License and attribution

Built as an engineering mini-project. Datasets belong to their original publishers
(Pentaho sample data; the merged dataset supplied by the project author). Prophet is released
under the MIT license by Meta.