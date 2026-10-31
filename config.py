"""
BizPredict AI - central configuration.

Every tunable constant of the application lives here so that the modules stay
free of magic values. Nothing in this file fabricates data: the defaults only
describe *how* real data is read, cleaned and modelled.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
OUTPUTS_DIR = BASE_DIR / "outputs"

#: Local copy of the public Pentaho sample dataset (used when the network is
#: unavailable, so the demo never silently fails).
DEFAULT_DATASET_LOCAL = DATA_DIR / "sales_data_sample.csv"

#: The public retail dataset requested by the project brief.
DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/ambientelivre/iguana/master/iserver/"
    "lib/pentaho/biserver-ce/data/csv/sales_data_sample.csv"
)

#: Optional bundled copy of the 156k-row merged weekly retail dataset.
MERGED_DATASET_LOCAL = DATA_DIR / "retail_sales_complete_dataset.csv"

# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
#: Encodings tried in order when reading a CSV.
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

#: Guard rail so a stray huge file cannot exhaust memory during a demo.
MAX_ROWS = 2_000_000

#: Guard rail for files uploaded through the dashboard (bytes).
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# --------------------------------------------------------------------------- #
# Column mapping
# --------------------------------------------------------------------------- #
#: Canonical field -> accepted header spellings (already normalised: lowercase,
#: alphanumerics only). Extend these lists to support new schemas; no other file
#: needs to change.
COLUMN_ALIASES = {
    "date": [
        "orderdate", "date", "order_date", "salesdate", "saledate",
        "transactiondate", "weekdate", "week_date", "datetime", "timestamp",
        "period", "month_year",
    ],
    "sales": [
        "sales", "weekly_sales", "revenue", "total", "totalsales", "total_sales",
        "amount", "gross_sales", "netsales", "net_sales", "sales_amount",
    ],
    "units": [
        "quantityordered", "quantity", "qty", "units", "unitssold", "units_sold",
        "order_quantity", "quantity_sold", "volumesold",
    ],
    "unit_price": [
        "priceeach", "price", "unitprice", "unit_price", "price_per_unit",
        "sellingprice", "msrp",
    ],
    "category": [
        "productline", "product_line", "department", "dept", "category",
        "productcategory", "product_category", "product_type", "segment",
    ],
    "store": [
        "store", "store_id", "storeid", "store_number", "outlet", "shop",
        "branch", "shop_id",
    ],
    "order_id": [
        "ordernumber", "order_id", "orderid", "order_number", "invoice",
        "invoiceno", "invoice_id", "transaction_id", "transactionid", "receipt",
    ],
    "status": ["status", "order_status", "orderstatus", "state"],
    "is_holiday": [
        "sales_is_holiday", "is_holiday", "isholiday", "holiday_flag",
        "is_holiday_flag", "holiday",
    ],
    "holiday_name": ["holiday_name", "holidayname", "festival", "event"],
}

#: Row-level holiday flags that describe the *transaction* rather than the
#: calendar. Preferred over calendar-level flags when both exist.
PREFERRED_HOLIDAY_FLAGS = ("sales_is_holiday", "is_holiday", "isholiday", "holiday")

#: Non-canonical numeric/contextual columns that are interesting for EDA.
EXTRA_CONTEXT_COLUMNS = (
    "temperature", "fuel_price", "cpi", "unemployment", "markdown_1",
    "markdown_2", "markdown_3", "markdown_4", "markdown_5", "store_size",
)

#: Descriptive extras surfaced in the Data Explorer if present.
EXTRA_LABEL_COLUMNS = ("region", "store_type", "season", "city", "country",
                       "territory", "dealsize", "customer_name")

#: Columns that must be resolved for the application to work at all.
REQUIRED_FIELDS = ("date", "sales")

# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
#: Rows with a sales value below this are treated as invalid (returns/nulls).
MIN_VALID_SALES = 0.0

#: Whether days/weeks with no transactions are materialised as zero-sales
#: periods. This is a *documented* modelling assumption (no transaction in the
#: period means no sale), not an invention of records.
FILL_MISSING_PERIODS_WITH_ZERO = True

#: Upper bound on how many periods may be zero-filled. A dataset that is missing
#: more than this share of its calendar (for example a sampled extract that keeps
#: only some months) is treated as *partially observed*: empty periods are then
#: dropped instead of being asserted as zero-sales days, which would fabricate
#: spiky demand. The decision and the count are always reported in the UI.
ZERO_FILL_MAX_SHARE = 0.40

#: Maximum share of a column that may be missing before it is dropped.
MAX_MISSING_SHARE = 0.95

#: Quantile used to flag (never delete) extreme sales values.
OUTLIER_QUANTILE = 0.999

# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
#: Horizon offered in the UI, expressed in *days* and converted to native
#: periods for weekly datasets.
FORECAST_HORIZON_MIN_DAYS = 7
FORECAST_HORIZON_MAX_DAYS = 60
FORECAST_HORIZON_DEFAULT_DAYS = 30

#: Share of the chronological history held out for evaluation.
TEST_FRACTION = 0.2

#: Minimum history required before a forecast is attempted.
MIN_HISTORY_PERIODS = 20

#: Prophet settings.
PROPHET_YEARLY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = True
PROPHET_INTERVAL_WIDTH = 0.80
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.05

#: Exponential-smoothing (Holt-Winters) fallback settings.
HW_SEASONAL_PERIODS = 7
HW_MIN_PERIODS_FOR_SEASONALITY = 28

#: MAPE is undefined when the actual value is zero. Actuals at or below this
#: threshold are excluded from MAPE and the count is reported to the user.
MAPE_ZERO_THRESHOLD = 1e-9

# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
#: Service level -> z. Used when scipy is unavailable and to document the
#: values the UI offers.
SERVICE_LEVELS = {
    0.80: 0.842,
    0.85: 1.036,
    0.90: 1.282,
    0.95: 1.645,
    0.975: 1.960,
    0.99: 2.326,
}
DEFAULT_SERVICE_LEVEL = 0.95
DEFAULT_LEAD_TIME_DAYS = 7
LEAD_TIME_MIN = 1
LEAD_TIME_MAX = 90

#: A SKU is "at risk" when its stockout probability exceeds this value.
STOCKOUT_RISK_THRESHOLD = 0.50
#: Reorder when projected on-hand at the end of lead time falls below this
#: share of the reorder point.
REORDER_TRIGGER_RATIO = 1.0

# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #
APP_TITLE = "BizPredict AI"
APP_TAGLINE = "Business Intelligence & Predictive Analytics for Small Retailers"
APP_VERSION = "1.0.0"

#: Currency symbol used for display only; the source data is not re-based.
CURRENCY_SYMBOL = "$"

#: Colour tokens shared by the CSS and the Plotly templates.
PALETTE = {
    "primary": "#4F46E5",
    "primary_soft": "#818CF8",
    "accent": "#06B6D4",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "neutral": "#64748B",
}
CHART_SEQUENCE = [
    "#4F46E5", "#06B6D4", "#10B981", "#F59E0B",
    "#EF4444", "#8B5CF6", "#14B8A6", "#F472B6",
]

PAGES = (
    "Executive Overview",
    "Sales Analytics",
    "AI Sales Forecasting",
    "Inventory Intelligence",
    "Data Explorer",
)