"""
Validation, cleaning and feature engineering.

The output is a single unified model regardless of which dataset was loaded:

* ``clean``  - one row per original record, with canonical columns;
* ``series`` - one row per time period (day or week) with aggregated sales.

Nothing is invented. Periods with no transactions are materialised as zero only
because :data:`config.FILL_MISSING_PERIODS_WITH_ZERO` is a *declared* assumption,
and the count of such periods is always reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from modules.data_loader import DatasetProfile, LoadedData, SchemaError, normalise_header

CANONICAL_COLUMNS = [
    "date", "sales", "units", "unit_price", "category",
    "store", "order_id", "status", "is_holiday", "holiday_name",
]


# --------------------------------------------------------------------------- #
# Date parsing
# --------------------------------------------------------------------------- #
def parse_dates_smart(values: pd.Series, column: str) -> pd.Series:
    """Parse a mixed-format date column, choosing day-first only when it helps.

    Retail exports mix ``2/24/2003 0:00``, ``2022-01-01`` and ``24/02/2003``.
    Pandas' ``format="mixed"`` handles the first two; the day-first variant is
    only adopted when it demonstrably rescues more rows, so a mostly
    month-first file is never silently reinterpreted.
    """
    month_first = pd.to_datetime(values, errors="coerce", format="mixed")
    if month_first.notna().mean() >= 0.9:
        return month_first

    try:
        day_first = pd.to_datetime(values, errors="coerce", dayfirst=True)
    except (ValueError, TypeError):
        return month_first

    if day_first.notna().sum() > month_first.notna().sum():
        return day_first
    return month_first


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessingStats:
    """Every number the Data Explorer needs to justify the cleaning."""

    rows_loaded: int = 0
    columns_detected: int = 0
    duplicate_rows_removed: int = 0
    invalid_date_rows_removed: int = 0
    invalid_sales_rows_removed: int = 0
    negative_sales_rows_removed: int = 0
    sales_derived_from_units_and_price: int = 0
    missing_cells_before: int = 0
    missing_cells_filled: int = 0
    missing_cells_remaining: int = 0
    columns_dropped_for_missingness: List[str] = field(default_factory=list)
    final_rows: int = 0
    periods_created: int = 0
    zero_sales_periods: int = 0
    unobserved_periods_dropped: int = 0
    zero_fill_applied: bool = True
    outlier_rows_flagged: int = 0
    aggregation: str = ""
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def rows_removed_total(self) -> int:
        return (
            self.duplicate_rows_removed
            + self.invalid_date_rows_removed
            + self.invalid_sales_rows_removed
            + self.negative_sales_rows_removed
        )

    def as_dict(self) -> Dict[str, object]:
        """Flat, display-friendly view used by the dashboard."""
        return {
            "Records loaded": f"{self.rows_loaded:,}",
            "Columns detected": self.columns_detected,
            "Duplicate records removed": f"{self.duplicate_rows_removed:,}",
            "Invalid dates removed": f"{self.invalid_date_rows_removed:,}",
            "Invalid / missing sales removed": f"{self.invalid_sales_rows_removed:,}",
            "Negative sales removed": f"{self.negative_sales_rows_removed:,}",
            "Sales derived (units × price)": f"{self.sales_derived_from_units_and_price:,}",
            "Missing cells before cleaning": f"{self.missing_cells_before:,}",
            "Missing cells filled": f"{self.missing_cells_filled:,}",
            "Missing cells remaining": f"{self.missing_cells_remaining:,}",
            "Records after cleaning": f"{self.final_rows:,}",
            "Time periods modelled": f"{self.periods_created:,}",
            "Genuine zero-sales periods": f"{self.zero_sales_periods:,}",
            "Empty periods dropped (not observed)": f"{self.unobserved_periods_dropped:,}",
            "Extreme values flagged": f"{self.outlier_rows_flagged:,}",
            "Aggregation": self.aggregation,
            "Date range": (
                f"{self.date_min} → {self.date_max}"
                if self.date_min and self.date_max else "—"
            ),
        }


@dataclass
class PreparedData:
    """The unified model every other module consumes."""

    profile: DatasetProfile
    clean: pd.DataFrame
    series: pd.DataFrame
    stats: PreprocessingStats
    freq: str                 # pandas offset alias for the native grid
    freq_label: str           # "daily" | "weekly"
    granularities: List[str]  # subset of ["Daily", "Weekly", "Monthly"]
    holiday_frame: Optional[pd.DataFrame] = None
    notes: List[str] = field(default_factory=list)

    # -- convenience accessors -------------------------------------------------
    @property
    def latest_sales(self) -> float:
        return float(self.series["sales"].iloc[-1]) if len(self.series) else 0.0

    @property
    def total_sales(self) -> float:
        return float(self.series["sales"].sum())

    def periods_for_days(self, days: int) -> int:
        """Convert a day-based horizon into native periods."""
        return max(1, int(round(days / 7))) if self.freq_label == "weekly" else int(days)

    def day_equivalent_period_length(self) -> float:
        """Length of one native period in days (7 for weekly data)."""
        return 7.0 if self.freq_label == "weekly" else 1.0


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce text such as '1,234.50' or '$99.00' into floats."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9eE.,\-+]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_records(loaded: LoadedData, profile: DatasetProfile) -> tuple[pd.DataFrame, PreprocessingStats]:
    """Validate and clean the raw rows, returning canonical records."""
    schema = profile.schema
    if not schema.is_valid:
        raise SchemaError(
            "Cannot clean a dataset that is missing: "
            + ", ".join(schema.missing_required)
        )

    stats = PreprocessingStats(
        rows_loaded=loaded.row_count,
        columns_detected=loaded.column_count,
    )
    raw = loaded.frame
    stats.missing_cells_before = int(raw.isna().sum().sum())

    # -- project onto the canonical model ------------------------------------
    frame = pd.DataFrame(index=raw.index)
    for canonical in CANONICAL_COLUMNS:
        header = schema.mapping.get(canonical)
        if header is not None:
            frame[canonical] = raw[header]

    # Preserve the untouched context columns for the explorer.
    for col in profile.context_columns + profile.label_columns:
        if col in raw.columns and col not in frame.columns:
            frame[f"ctx_{col}"] = raw[col]

    # -- dates ---------------------------------------------------------------
    frame["date"] = parse_dates_smart(frame["date"], schema.mapping["date"])
    bad_dates = frame["date"].isna()
    stats.invalid_date_rows_removed = int(bad_dates.sum())
    frame = frame.loc[~bad_dates].copy()

    # -- numeric sales -------------------------------------------------------
    frame["sales"] = _to_numeric(frame["sales"])
    if "units" in frame:
        frame["units"] = _to_numeric(frame["units"])
    if "unit_price" in frame:
        frame["unit_price"] = _to_numeric(frame["unit_price"])

    missing_sales = frame["sales"].isna()
    if missing_sales.any() and {"units", "unit_price"}.issubset(frame.columns):
        derived = frame.loc[missing_sales, "units"] * frame.loc[missing_sales, "unit_price"]
        usable = derived.notna()
        frame.loc[missing_sales & usable, "sales"] = derived[usable]
        stats.sales_derived_from_units_and_price = int(usable.sum())
        if stats.sales_derived_from_units_and_price:
            stats.notes.append(
                f"{stats.sales_derived_from_units_and_price:,} records had no sales "
                "value; it was derived as quantity × unit price and those rows are "
                "marked as derived in the dataset."
            )
        frame.loc[:, "sales_is_derived"] = missing_sales & usable

    still_missing = frame["sales"].isna()
    stats.invalid_sales_rows_removed = int(still_missing.sum())
    frame = frame.loc[~still_missing].copy()

    negative = frame["sales"] < config.MIN_VALID_SALES
    stats.negative_sales_rows_removed = int(negative.sum())
    if stats.negative_sales_rows_removed:
        stats.notes.append(
            f"{stats.negative_sales_rows_removed:,} records had negative sales "
            "(returns/adjustments) and were excluded from demand modelling."
        )
    frame = frame.loc[~negative].copy()

    # -- duplicates ----------------------------------------------------------
    subset = [c for c in ("date", "sales", "units", "category", "store", "order_id")
              if c in frame.columns]
    before = len(frame)
    frame = frame.drop_duplicates(subset=subset, keep="first")
    stats.duplicate_rows_removed = int(before - len(frame))

    # -- fill secondary gaps -------------------------------------------------
    filled = 0
    for col in ("units", "unit_price"):
        if col in frame.columns and frame[col].isna().any():
            median = frame[col].median()
            if pd.notna(median):
                count = int(frame[col].isna().sum())
                frame[col] = frame[col].fillna(median)
                filled += count
    if filled:
        stats.missing_cells_filled += filled
        stats.notes.append(
            f"{filled:,} missing unit/price cells were filled with the column "
            "median so that per-unit analysis stays complete."
        )

    # -- holiday signal ------------------------------------------------------
    if "is_holiday" in frame.columns:
        frame["is_holiday"] = (
            frame["is_holiday"].astype(str).str.strip().str.lower()
            .isin({"1", "1.0", "true", "t", "yes", "y"})
        )
    if "holiday_name" in frame.columns:
        frame["holiday_name"] = frame["holiday_name"].astype("string")

    frame["date"] = frame["date"].dt.normalize()
    stats.missing_cells_remaining = int(frame.isna().sum().sum())
    stats.final_rows = len(frame)
    stats.date_min = frame["date"].min().date().isoformat()
    stats.date_max = frame["date"].max().date().isoformat()
    return frame.sort_values("date").reset_index(drop=True), stats


# --------------------------------------------------------------------------- #
# Temporal aggregation
# --------------------------------------------------------------------------- #
def _period_grid(dates: pd.Series, frequency: str) -> pd.DatetimeIndex:
    """Build the complete calendar of periods between first and last date.

    Daily data uses a 1-day grid. Weekly data keeps the *observed* weekday as
    the anchor so a Saturday-dated file is never shifted onto a Sunday.
    """
    if dates.empty:
        return pd.DatetimeIndex([])
    start, end = dates.min(), dates.max()
    if frequency == "weekly":
        anchor = int(pd.Series(dates.dt.weekday).mode().iloc[0])
        offset = (anchor - start.weekday()) % 7
        start = start + pd.Timedelta(days=offset)
        return pd.date_range(start, end, freq="7D")
    return pd.date_range(start, end, freq="D")


def build_series(clean: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate records into a gapless period series.

    Weekly files already contain one row per store-department-week, so summing
    rows that share a date produces the chain-wide weekly total exactly once -
    no record is counted twice.
    """
    grouped = clean.groupby("date", as_index=False).agg(
        sales=("sales", "sum"),
        records=("sales", "size"),
    )
    if "units" in clean.columns:
        grouped["units"] = clean.groupby("date")["units"].sum(min_count=1).values
    if "order_id" in clean.columns:
        orders = clean.dropna(subset=["order_id"]).groupby("date")["order_id"].nunique()
        grouped["orders"] = grouped["date"].map(orders).fillna(0).astype(int)
    if "is_holiday" in clean.columns:
        flags = clean.groupby("date")["is_holiday"].max()
        grouped["is_holiday"] = grouped["date"].map(flags).fillna(False).astype(bool)
    if "holiday_name" in clean.columns:
        names = (
            clean.dropna(subset=["holiday_name"])
            .groupby("date")["holiday_name"]
            .agg(lambda s: ", ".join(sorted({str(v) for v in s if str(v).strip()})))
        )
        grouped["holiday_name"] = grouped["date"].map(names)

    grid = _period_grid(clean["date"], frequency)
    grouped = grouped.set_index("date").reindex(grid)
    grouped.index.name = "date"
    # A period with no records was never observed; it is not the same as a period
    # in which the business genuinely sold nothing.
    grouped["observed"] = grouped["records"].notna()

    fill_zero = config.FILL_MISSING_PERIODS_WITH_ZERO
    grouped["sales"] = grouped["sales"].fillna(0.0) if fill_zero else grouped["sales"]
    grouped["records"] = grouped["records"].fillna(0).astype(int)
    if "units" in grouped:
        grouped["units"] = grouped["units"].fillna(0.0) if fill_zero else grouped["units"]
    if "orders" in grouped:
        grouped["orders"] = grouped["orders"].fillna(0).astype(int)
    if "is_holiday" in grouped:
        grouped["is_holiday"] = grouped["is_holiday"].fillna(False).astype(bool)

    series = grouped.reset_index()
    series["sales"] = series["sales"].astype(float)
    return series


def add_time_features(series: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar features used by the charts and the fallback model."""
    out = series.copy()
    dt = out["date"].dt
    out["year"] = dt.year
    out["quarter"] = dt.quarter
    out["month"] = dt.month
    out["month_name"] = dt.strftime("%b %Y")
    out["iso_week"] = dt.isocalendar().week.astype(int)
    out["day"] = dt.day
    out["day_of_week"] = dt.dayofweek          # 0 = Monday
    out["day_name"] = dt.day_name()
    out["is_weekend"] = out["day_of_week"].isin([5, 6])
    out["period_index"] = np.arange(len(out))
    out["is_zero_sales"] = out["sales"] <= 0
    out["rolling_7"] = out["sales"].rolling(7, min_periods=1).mean()
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def prepare(loaded: LoadedData, profile: DatasetProfile) -> PreparedData:
    """Run the whole preprocessing pipeline."""
    clean, stats = clean_records(loaded, profile)
    if clean.empty:
        raise SchemaError(
            "No usable records remained after cleaning. Check that the date and "
            "sales columns contain valid values."
        )

    frequency = profile.frequency
    series = build_series(clean, frequency)

    # -- zero-fill policy ----------------------------------------------------
    # Zero-filling is only defensible when the file is a continuous record. A
    # sampled extract (for example several complete months per year) has empty
    # calendar slots that are simply not observed; labelling them zero-sales days
    # would invent high-variance demand and wreck every downstream metric.
    empty_periods = series[series["sales"] <= 0]
    empty_share = len(empty_periods) / len(series) if len(series) else 0.0
    if (
        config.FILL_MISSING_PERIODS_WITH_ZERO
        and empty_share > config.ZERO_FILL_MAX_SHARE
    ):
        unobserved_mask = (series["sales"] <= 0) & (~series["observed"])
        stats.unobserved_periods_dropped = int(unobserved_mask.sum())
        series = series.loc[~unobserved_mask].copy()
        stats.zero_fill_applied = False
        stats.notes.append(
            f"{stats.unobserved_periods_dropped:,} of {len(series) + stats.unobserved_periods_dropped:,} "
            "calendar periods contain no transactions at all. Because more than "
            f"{config.ZERO_FILL_MAX_SHARE * 100:.0f}% of the calendar would have to "
            "be filled with zeros, those periods are treated as *not observed* and "
            "dropped instead. They are not reported as zero-sales periods."
        )

    stats.zero_sales_periods = int((series["sales"] <= 0).sum())

    series = add_time_features(series)
    stats.periods_created = len(series)
    stats.aggregation = (
        "Weekly records summed per date (one observation per store-department-week)."
        if frequency == "weekly"
        else "Transactions aggregated to daily totals."
    )

    # Flag (do not remove) extreme values so the user knows they exist.
    if len(series) >= 10:
        threshold = series["sales"].quantile(config.OUTLIER_QUANTILE)
        stats.outlier_rows_flagged = int((series["sales"] > threshold).sum())

    granularities = ["Weekly", "Monthly"]
    if frequency == "daily":
        granularities.insert(0, "Daily")

    holiday_frame = None
    if "is_holiday" in series.columns and bool(series["is_holiday"].any()):
        holiday_frame = series.loc[series["is_holiday"], ["date", "holiday_name"]].copy()
        holiday_frame["holiday"] = (
            holiday_frame["holiday_name"].fillna("Holiday").astype(str)
            .replace({"": "Holiday"})
        )
        holiday_frame = holiday_frame[["date", "holiday"]].rename(columns={"date": "ds"})

    notes = list(stats.notes)
    if frequency == "weekly":
        notes.append(
            "Weekly data: each native period is 7 days. Forecast horizons entered "
            "in days are converted to weeks, and daily charts are unavailable."
        )

    return PreparedData(
        profile=profile,
        clean=clean,
        series=series,
        stats=stats,
        freq="7D" if frequency == "weekly" else "D",
        freq_label=frequency,
        granularities=granularities,
        holiday_frame=holiday_frame,
        notes=notes,
    )


def resample_series(series: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Re-aggregate the native series for the Sales Analytics page."""
    if granularity == "Daily":
        return series.copy()

    rule = "W-SUN" if granularity == "Weekly" else "MS"
    out = (
        series.set_index("date")
        .resample(rule)
        .agg(sales=("sales", "sum"), records=("records", "sum"))
        .reset_index()
    )
    out = out[out["sales"].notna()]
    if granularity == "Weekly":
        out["label"] = out["date"].dt.strftime("Wk of %d %b %Y")
    else:
        out["label"] = out["date"].dt.strftime("%b %Y")
    return out.reset_index(drop=True)