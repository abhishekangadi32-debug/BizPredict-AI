"""
Business analytics: KPIs, breakdowns and descriptive summaries.

Everything here is computed from the cleaned records. When a metric needs a
column that the loaded dataset does not contain (unique orders, units, holiday
flags, stores), the metric reports *why* it is unavailable instead of showing a
made-up number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from modules.data_loader import normalise_header
from modules.preprocessing import PreparedData, resample_series

CURRENCY = getattr(config, "CURRENCY_SYMBOL", "$")


def money(value: Optional[float], compact: bool = False) -> str:
    """Format a currency value, optionally in K/M shorthand for cards."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if compact:
        magnitude = abs(value)
        if magnitude >= 1e9:
            return f"{CURRENCY}{value / 1e9:,.2f}B"
        if magnitude >= 1e6:
            return f"{CURRENCY}{value / 1e6:,.2f}M"
        if magnitude >= 1e3:
            return f"{CURRENCY}{value / 1e3:,.1f}K"
    return f"{CURRENCY}{value:,.0f}"


def number(value: Optional[float], decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:,.{decimals}f}"


# --------------------------------------------------------------------------- #
# Executive KPIs
# --------------------------------------------------------------------------- #
@dataclass
class ExecutiveKPIs:
    total_revenue: float
    total_units: Optional[float]
    unit_basis: str
    order_count: int
    order_basis: str
    average_order_value: Optional[float]
    period_count: int
    days_covered: int
    average_daily_sales: float
    top_category: Optional[str]
    top_category_revenue: float
    top_category_share: float
    growth_pct: Optional[float]
    growth_window: str
    peak_period_date: Optional[str]
    peak_period_sales: float
    zero_periods: int
    period_word: str = "period"
    category_card_label: str = "Top Product Line"
    notes: List[str] = field(default_factory=list)

    def cards(self) -> List[Tuple[str, str, str]]:
        """(label, value, caption) triples for the overview cards."""
        plural = self.period_word if self.period_count == 1 else f"{self.period_word}s"
        return [
            ("Total Revenue", money(self.total_revenue, compact=True),
             f"across {self.period_count:,} {plural}"),
            ("Total Orders", f"{self.order_count:,}", self.order_basis),
            ("Average Order Value", money(self.average_order_value, compact=True),
             "revenue ÷ orders" if self.order_basis.startswith("unique") else "revenue ÷ records"),
            ("Total Units Sold", number(self.total_units), self.unit_basis),
            (self.category_card_label, self.top_category or "n/a",
             f"{self.top_category_share:.1f}% of revenue" if self.top_category else "not available"),
            ("Average Daily Sales", money(self.average_daily_sales, compact=True),
             f"over {self.days_covered:,} days"),
        ]


def executive_kpis(prepared: PreparedData) -> ExecutiveKPIs:
    """Compute the six headline KPIs from real data only."""
    series = prepared.series
    clean = prepared.clean
    profile = prepared.profile

    total_revenue = float(series["sales"].sum())
    period_count = int(len(series))
    days_covered = int(
        (series["date"].max() - series["date"].min()).days + 1
    ) if period_count else 0
    notes: List[str] = []

    # -- orders: only real identifiers count as orders ------------------------
    if "order_id" in clean.columns and clean["order_id"].notna().any():
        order_count = int(clean["order_id"].nunique())
        order_basis = "unique order IDs"
    elif "records" in series.columns:
        order_count = int(series["records"].sum())
        order_basis = "line items (no order ID column)"
        notes.append(
            "The dataset has no order identifier, so 'orders' is the number of "
            "records/line items, not distinct orders. Average order value is "
            "therefore an average per line item."
        )
    else:
        order_count = period_count
        order_basis = "aggregated periods"
        notes.append("No order-level column exists; periods are counted instead.")

    aov = total_revenue / order_count if order_count else None

    # -- units ---------------------------------------------------------------
    total_units: Optional[float] = None
    if "units" in clean.columns and clean["units"].notna().any():
        total_units = float(clean["units"].sum())
        unit_basis = "summed quantity column"
    else:
        unit_basis = "no quantity column"
        notes.append("The dataset has no quantity column, so units sold is unavailable.")

    # -- category ------------------------------------------------------------
    top_category = None
    top_revenue = 0.0
    top_share = 0.0
    if profile.has_category:
        perf = category_performance(prepared)
        if not perf.empty:
            top_category = str(perf.iloc[0]["category"])
            top_revenue = float(perf.iloc[0]["revenue"])
            top_share = float(perf.iloc[0]["share_pct"])

    # -- growth --------------------------------------------------------------
    growth_pct: Optional[float] = None
    window = "last 4 vs previous 4 periods"
    recent_n = 4
    if period_count >= recent_n * 2:
        recent = series["sales"].tail(recent_n).sum()
        previous = series["sales"].iloc[-recent_n * 2:-recent_n].sum()
        if previous > 0:
            growth_pct = float(100 * (recent - previous) / previous)

    peak_idx = series["sales"].idxmax() if period_count else None
    peak_date = (
        series.loc[peak_idx, "date"].date().isoformat() if peak_idx is not None else None
    )
    peak_sales = float(series.loc[peak_idx, "sales"]) if peak_idx is not None else 0.0

    category_header = normalise_header(prepared.profile.schema.mapping.get("category", ""))
    category_card_label = (
        "Top Department" if category_header in {"dept", "department"}
        else "Top Product Line" if category_header == "productline"
        else "Top Category"
    )
    if top_category is not None and category_card_label == "Top Department" \
            and str(top_category).isdigit():
        # Department IDs are numbers in the weekly dataset; label them clearly.
        top_category = f"Department {top_category}"

    return ExecutiveKPIs(
        total_revenue=total_revenue,
        total_units=total_units,
        unit_basis=unit_basis,
        order_count=order_count,
        order_basis=order_basis,
        average_order_value=aov,
        period_count=period_count,
        days_covered=days_covered,
        average_daily_sales=total_revenue / days_covered if days_covered else 0.0,
        top_category=top_category,
        top_category_revenue=top_revenue,
        top_category_share=top_share,
        growth_pct=growth_pct,
        growth_window=window,
        peak_period_date=peak_date,
        peak_period_sales=peak_sales,
        zero_periods=int((series["sales"] <= 0).sum()),
        period_word="week" if prepared.freq_label == "weekly" else "day",
        category_card_label=category_card_label,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Breakdowns
# --------------------------------------------------------------------------- #
def category_performance(prepared: PreparedData) -> pd.DataFrame:
    """Revenue, units and share by product line / department."""
    clean = prepared.clean
    if "category" not in clean.columns:
        return pd.DataFrame(
            columns=["category", "revenue", "units", "records", "share_pct", "avg_ticket"]
        )

    frame = clean.dropna(subset=["category"]).copy()
    if frame.empty:
        return pd.DataFrame(
            columns=["category", "revenue", "units", "records", "share_pct", "avg_ticket"]
        )

    agg = {"revenue": ("sales", "sum"), "records": ("sales", "size")}
    if "units" in frame.columns:
        agg["units"] = ("units", "sum")
    grouped = frame.groupby(frame["category"].astype(str)).agg(**agg).reset_index()
    grouped = grouped.rename(columns={grouped.columns[0]: "category"})

    total = grouped["revenue"].sum()
    grouped["share_pct"] = np.where(total > 0, 100 * grouped["revenue"] / total, 0.0)
    grouped["avg_ticket"] = grouped["revenue"] / grouped["records"].replace(0, np.nan)
    if "units" in grouped.columns:
        grouped["avg_price"] = grouped["revenue"] / grouped["units"].replace(0, np.nan)
    return grouped.sort_values("revenue", ascending=False).reset_index(drop=True)


def store_performance(prepared: PreparedData) -> pd.DataFrame:
    """Revenue by store, when the dataset has a store column."""
    clean = prepared.clean
    columns = ["store", "revenue", "records", "share_pct"]
    if "store" not in clean.columns:
        return pd.DataFrame(columns=columns)
    frame = clean.dropna(subset=["store"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        frame.groupby(frame["store"].astype(str))
        .agg(revenue=("sales", "sum"), records=("sales", "size"))
        .reset_index()
    )
    grouped = grouped.rename(columns={grouped.columns[0]: "store"})
    total = grouped["revenue"].sum()
    grouped["share_pct"] = np.where(total > 0, 100 * grouped["revenue"] / total, 0.0)
    return grouped.sort_values("revenue", ascending=False).reset_index(drop=True)


def revenue_trend(prepared: PreparedData, granularity: str) -> pd.DataFrame:
    """Revenue at the requested granularity (Daily/Weekly/Monthly)."""
    if granularity not in prepared.granularities:
        granularity = prepared.granularities[-1]
    frame = resample_series(prepared.series, granularity)
    if "label" not in frame.columns:
        frame["label"] = frame["date"].dt.strftime("%d %b %Y")
    frame["moving_average"] = frame["sales"].rolling(3, min_periods=1).mean()
    return frame


def holiday_comparison(prepared: PreparedData) -> Optional[Dict[str, object]]:
    """Average period sales on holiday vs non-holiday periods."""
    series = prepared.series
    if "is_holiday" not in series.columns or series["is_holiday"].nunique() < 2:
        return None
    holiday = series.loc[series["is_holiday"], "sales"]
    regular = series.loc[~series["is_holiday"], "sales"]
    if holiday.empty or regular.empty:
        return None
    return {
        "holiday_mean": float(holiday.mean()),
        "regular_mean": float(regular.mean()),
        "holiday_periods": int(len(holiday)),
        "regular_periods": int(len(regular)),
        "lift_pct": (
            float(100 * (holiday.mean() - regular.mean()) / regular.mean())
            if regular.mean() > 0 else None
        ),
        "holiday_total": float(holiday.sum()),
        "regular_total": float(regular.sum()),
    }


def seasonality_profile(prepared: PreparedData) -> pd.DataFrame:
    """Average sales by month, month name and day of week."""
    series = prepared.series
    out = series.copy()
    out["month"] = out["date"].dt.month
    out["month_name_short"] = out["date"].dt.strftime("%b")
    by_month = (
        out.groupby(["month", "month_name_short"], as_index=False)["sales"]
        .mean()
        .sort_values("month")
    )
    return by_month


def weekday_profile(prepared: PreparedData) -> Optional[pd.DataFrame]:
    """Average sales by day of week - only meaningful for daily data."""
    if prepared.freq_label != "daily":
        return None
    series = prepared.series
    out = (
        series.groupby(["day_of_week", "day_name"], as_index=False)["sales"]
        .mean()
        .sort_values("day_of_week")
    )
    return out


def monthly_heatmap(prepared: PreparedData) -> pd.DataFrame:
    """Year × month revenue pivot used for the heat map."""
    series = prepared.series
    out = series.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.strftime("%b")
    pivot = out.pivot_table(
        index="year", columns="month", values="sales", aggfunc="sum"
    )
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return pivot[[c for c in order if c in pivot.columns]]


def distribution_summary(prepared: PreparedData) -> Dict[str, float]:
    """Descriptive statistics of period sales."""
    sales = prepared.series["sales"]
    return {
        "count": float(sales.count()),
        "mean": float(sales.mean()),
        "median": float(sales.median()),
        "std": float(sales.std(ddof=1)) if len(sales) > 1 else 0.0,
        "min": float(sales.min()),
        "max": float(sales.max()),
        "q25": float(sales.quantile(0.25)),
        "q75": float(sales.quantile(0.75)),
        "total": float(sales.sum()),
    }


def best_and_worst(prepared: PreparedData, top_n: int = 3) -> Dict[str, pd.DataFrame]:
    """Best and worst performing categories by revenue."""
    perf = category_performance(prepared)
    if perf.empty:
        return {"best": perf, "worst": perf}
    return {
        "best": perf.head(top_n),
        "worst": perf.tail(top_n).sort_values("revenue"),
    }


def period_extremes(prepared: PreparedData) -> Dict[str, object]:
    """Highest and lowest revenue periods."""
    series = prepared.series
    if series.empty:
        return {}
    best = series.loc[series["sales"].idxmax()]
    worst = series.loc[series["sales"].idxmin()]
    return {
        "best_date": best["date"].date().isoformat(),
        "best_sales": float(best["sales"]),
        "worst_date": worst["date"].date().isoformat(),
        "worst_sales": float(worst["sales"]),
    }


# --------------------------------------------------------------------------- #
# Narrative summary
# --------------------------------------------------------------------------- #
def business_summary(prepared: PreparedData, kpis: ExecutiveKPIs) -> List[str]:
    """Data-driven sentences. No claim is made that the data cannot support."""
    sentences: List[str] = []
    is_weekly = prepared.freq_label == "weekly"
    freq_word = "week" if is_weekly else "day"
    observed_noun = (
        f"{kpis.period_count:,} weeks" if is_weekly
        else f"{kpis.period_count:,} observed trading days"
    )

    sentences.append(
        f"The dataset spans {kpis.days_covered:,} calendar days "
        f"({prepared.stats.date_min} to {prepared.stats.date_max}) and contains "
        f"{observed_noun}, generating {money(kpis.total_revenue)} in total revenue."
    )
    sentences.append(
        f"Average revenue per {freq_word} with sales is "
        f"{money(kpis.total_revenue / kpis.period_count if kpis.period_count else 0)}, "
        f"or {money(kpis.average_daily_sales)} per calendar day."
    )

    if kpis.top_category:
        noun = (
            "department" if kpis.category_card_label == "Top Department"
            else "product line" if kpis.category_card_label == "Top Product Line"
            else "category"
        )
        sentences.append(
            f"{kpis.top_category} is the strongest {noun} with "
            f"{money(kpis.top_category_revenue)} "
            f"({kpis.top_category_share:.1f}% of total revenue)."
        )

    if kpis.growth_pct is not None:
        direction = "grown" if kpis.growth_pct >= 0 else "declined"
        sentences.append(
            f"Revenue has {direction} {abs(kpis.growth_pct):.1f}% when comparing "
            f"the last four periods with the four before them."
        )

    if kpis.peak_period_date:
        sentences.append(
            f"The single strongest period was {kpis.peak_period_date} with "
            f"{money(kpis.peak_period_sales)} in sales."
        )

    if kpis.zero_periods:
        sentences.append(
            f"{kpis.zero_periods:,} of {kpis.period_count:,} periods recorded zero "
            "sales; they are kept so that demand variability is not understated."
        )

    holiday = holiday_comparison(prepared)
    if holiday and holiday.get("lift_pct") is not None:
        sentences.append(
            f"Holiday periods average {money(holiday['holiday_mean'])} versus "
            f"{money(holiday['regular_mean'])} on regular periods "
            f"({holiday['lift_pct']:+.1f}%)."
        )

    sentences.extend(kpis.notes[-1:])  # surface the order-count caveat if present
    return sentences


def insights(prepared: PreparedData) -> List[Tuple[str, str]]:
    """(severity, message) pairs for the 'insights' panel."""
    out: List[Tuple[str, str]] = []
    series = prepared.series

    if len(series) < config.MIN_HISTORY_PERIODS:
        out.append((
            "warning",
            f"Only {len(series)} periods of history are available. Forecasts need at "
            f"least {config.MIN_HISTORY_PERIODS} to be meaningful.",
        ))

    if series["sales"].tail(4).mean() and series["sales"].head(4).mean():
        change = 100 * (
            series["sales"].tail(4).mean() - series["sales"].head(4).mean()
        ) / series["sales"].head(4).mean()
        if abs(change) >= 10:
            out.append((
                "success" if change > 0 else "danger",
                f"Revenue is {abs(change):.1f}% {'higher' if change > 0 else 'lower'} "
                "in the most recent four periods than in the first four.",
            ))

    cv = series["sales"].std() / series["sales"].mean() if series["sales"].mean() else 0
    if cv > 0.75:
        out.append((
            "warning",
            f"Demand is highly volatile (coefficient of variation {cv:.2f}); safety "
            "stock will be large relative to average demand.",
        ))

    perf = category_performance(prepared)
    if not perf.empty and perf["share_pct"].iloc[0] > 45:
        out.append((
            "warning",
            f"'{perf['category'].iloc[0]}' alone contributes "
            f"{perf['share_pct'].iloc[0]:.1f}% of revenue - a concentration risk.",
        ))

    if prepared.stats.zero_sales_periods:
        share = 100 * prepared.stats.zero_sales_periods / max(len(series), 1)
        if share > 10:
            out.append((
                "warning",
                f"{share:.0f}% of periods have zero sales. Percentage-based metrics "
                "will be unreliable for this dataset.",
            ))

    return out