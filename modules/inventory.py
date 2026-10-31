"""
Inventory intelligence: safety stock, reorder point and stockout risk.

Formulas (as specified in the brief):

    Safety Stock = Z x sigma_d x sqrt(L)
    Reorder Point = (Average Daily Demand x Lead Time) + Safety Stock

where ``Z`` is the service-level factor, ``sigma_d`` the standard deviation of
daily demand and ``L`` the lead time in days.

The datasets carry **no inventory balances and no supplier lead times**. Both are
therefore supplied by the user; until they are, the application reports demand
estimates only and explicitly labels reorder figures as estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from modules.preprocessing import PreparedData


# --------------------------------------------------------------------------- #
# Service level factor
# --------------------------------------------------------------------------- #
def z_factor(service_level: float) -> float:
    """Normal inverse CDF of the service level.

    scipy is used when available; otherwise the factor is linearly interpolated
    between the tabulated values in :data:`config.SERVICE_LEVELS`.
    """
    level = float(min(max(service_level, 0.5), 0.9999))
    try:
        from scipy.stats import norm
        return float(norm.ppf(level))
    except Exception:  # noqa: BLE001 - scipy is optional
        levels = sorted(config.SERVICE_LEVELS)
        zs = [config.SERVICE_LEVELS[l] for l in levels]
        return float(np.interp(level, levels, zs))


def stockout_probability(current_stock: float, mean_demand: float, sigma_demand: float) -> float:
    """P(demand over the lead time exceeds on-hand stock), normal approximation."""
    if sigma_demand <= 0:
        return 0.0 if current_stock >= mean_demand else 1.0
    z = (current_stock - mean_demand) / sigma_demand
    try:
        from scipy.stats import norm
        return float(1.0 - norm.cdf(z))
    except Exception:  # noqa: BLE001
        # Abramowitz & Stegun 7.1.26 approximation of the error function.
        return float(0.5 * math.erfc(z / math.sqrt(2)))


# --------------------------------------------------------------------------- #
# Demand statistics
# --------------------------------------------------------------------------- #
@dataclass
class DemandStats:
    """Per-period and per-day demand statistics for one scope."""

    scope: str
    periods: int
    mean_period: float
    std_period: float
    mean_daily: float
    std_daily: float
    basis: str = ""


def _daily_conversion(mean_native: float, std_native: float, period_days: float) -> tuple[float, float]:
    """Convert native-period statistics to daily demand.

    Mean scales linearly. For the standard deviation, independent days give
    ``sigma_period = sigma_daily * sqrt(period_days)``, so the reverse is
    ``sigma_daily = sigma_period / sqrt(period_days)``. This is an approximation
    and is stated as such wherever the numbers are shown.
    """
    mean_daily = mean_native / period_days
    std_daily = std_native / math.sqrt(period_days) if period_days > 1 else std_native
    return mean_daily, std_daily


def total_demand_stats(prepared: PreparedData) -> DemandStats:
    """Demand statistics for the whole business."""
    sales = prepared.series["sales"].astype(float)
    mean_daily, std_daily = _daily_conversion(
        float(sales.mean()), float(sales.std(ddof=1)) if len(sales) > 1 else 0.0,
        prepared.day_equivalent_period_length(),
    )
    return DemandStats(
        scope="All products",
        periods=int(len(sales)),
        mean_period=float(sales.mean()),
        std_period=float(sales.std(ddof=1)) if len(sales) > 1 else 0.0,
        mean_daily=mean_daily,
        std_daily=std_daily,
        basis="Weekly totals converted to daily equivalents (σ scaled by 1/√7)."
        if prepared.freq_label == "weekly" else "Daily totals.",
    )


def category_demand_matrix(prepared: PreparedData) -> Optional[pd.DataFrame]:
    """Category × period sales matrix on the complete period grid.

    Missing category/period combinations are zero: the department simply made no
    sale in that period, which is exactly what the demand distribution needs.
    """
    clean = prepared.clean
    if "category" not in clean.columns or clean["category"].isna().all():
        return None
    working = clean.dropna(subset=["category"]).copy()
    working["category"] = working["category"].astype(str)
    matrix = (
        working.pivot_table(index="date", columns="category", values="sales", aggfunc="sum")
        .reindex(prepared.series["date"])
        .fillna(0.0)
    )
    return matrix


def category_demand_stats(prepared: PreparedData) -> pd.DataFrame:
    """Demand statistics for every category."""
    matrix = category_demand_matrix(prepared)
    if matrix is None or matrix.empty:
        return pd.DataFrame(
            columns=["category", "periods", "mean_period", "std_period",
                     "mean_daily", "std_daily"]
        )
    period_days = prepared.day_equivalent_period_length()
    rows: List[Dict[str, object]] = []
    for category in matrix.columns:
        values = matrix[category].astype(float)
        mean_daily, std_daily = _daily_conversion(
            float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            period_days,
        )
        rows.append({
            "category": str(category),
            "periods": int(len(values)),
            "mean_period": float(values.mean()),
            "std_period": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "mean_daily": mean_daily,
            "std_daily": std_daily,
        })
    return pd.DataFrame(rows).sort_values("mean_daily", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
@dataclass
class InventoryPlan:
    """The full recommendation table plus the assumptions used to build it."""

    lines: pd.DataFrame
    lead_time_days: int
    service_level: float
    z: float
    period_days: int
    stock_provided: bool
    notes: List[str] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        if self.lines.empty or "status" not in self.lines:
            return {}
        return self.lines["status"].value_counts().to_dict()

    @property
    def reorder_lines(self) -> pd.DataFrame:
        if self.lines.empty:
            return self.lines
        return self.lines[self.lines["status"] == "Reorder now"]

    def to_csv(self) -> bytes:
        frame = self.lines.copy()
        for column in ("avg_daily_demand", "std_daily", "safety_stock",
                       "lead_time_demand", "reorder_point", "stockout_risk_pct"):
            if column in frame:
                frame[column] = frame[column].astype(float).round(2)
        return frame.to_csv(index=False).encode("utf-8")


def _status(current_stock: Optional[float], reorder_point: float,
            risk: Optional[float]) -> str:
    if current_stock is None or (isinstance(current_stock, float) and np.isnan(current_stock)):
        return "Input required"
    if current_stock <= reorder_point:
        return "Reorder now"
    if risk is not None and risk > config.STOCKOUT_RISK_THRESHOLD:
        return "Watch"
    return "Adequate"


def build_plan(
    prepared: PreparedData,
    lead_time_days: int = config.DEFAULT_LEAD_TIME_DAYS,
    service_level: float = config.DEFAULT_SERVICE_LEVEL,
    current_stock: Optional[float] = None,
    per_category_stock: Optional[Dict[str, float]] = None,
    include_categories: bool = True,
) -> InventoryPlan:
    """Compute safety stock, reorder points and stockout risk.

    *current_stock* applies to the whole business; *per_category_stock* overrides
    it for individual categories so the user can be as specific as they like.
    """
    lead = int(np.clip(lead_time_days, config.LEAD_TIME_MIN, config.LEAD_TIME_MAX))
    level = float(np.clip(service_level, 0.5, 0.999))
    z = z_factor(level)
    period_days = int(prepared.day_equivalent_period_length())
    notes: List[str] = []

    rows: List[Dict[str, object]] = []

    def add_row(scope: str, stats: DemandStats, stock: Optional[float], basis: str) -> None:
        lead_time_demand = stats.mean_daily * lead
        sigma_lead = stats.std_daily * math.sqrt(lead)
        safety = z * stats.std_daily * math.sqrt(lead)
        reorder_point = lead_time_demand + safety

        risk = None
        coverage = None
        shortfall = None
        order_qty = None
        if stock is not None and not (isinstance(stock, float) and np.isnan(stock)):
            risk = stockout_probability(stock, lead_time_demand, sigma_lead)
            coverage = stock / stats.mean_daily if stats.mean_daily > 0 else None
            shortfall = float(stock - reorder_point)
            order_qty = max(0.0, math.ceil(reorder_point - stock)) if stock < reorder_point else 0.0

        rows.append({
            "scope": scope,
            "avg_daily_demand": stats.mean_daily,
            "std_daily": stats.std_daily,
            "lead_time_days": lead,
            "lead_time_demand": lead_time_demand,
            "safety_stock": safety,
            "reorder_point": reorder_point,
            "current_stock": stock,
            "coverage_days": coverage,
            "stockout_risk_pct": risk * 100 if risk is not None else np.nan,
            "projected_gap": shortfall,
            "recommended_order": order_qty,
            "status": _status(stock, reorder_point, risk),
            "basis": basis,
        })

    total_stats = total_demand_stats(prepared)
    add_row("All products", total_stats, current_stock,
            "Demand estimate from historical sales; no stock data in the dataset.")

    if include_categories:
        cat_stats = category_demand_stats(prepared)
        for _, row in cat_stats.iterrows():
            stock = None
            if per_category_stock:
                stock = per_category_stock.get(str(row["category"]))
            elif current_stock is not None and row["mean_daily"] > 0 and total_stats.mean_daily > 0:
                # Allocate the business-level stock in proportion to demand so the
                # per-category view stays consistent; flagged as an allocation.
                stock = current_stock * (row["mean_daily"] / total_stats.mean_daily)
            basis = (
                "Stock allocated by share of demand."
                if stock is not None and not per_category_stock else
                "User-entered stock." if per_category_stock else
                "Demand estimate from historical sales; no stock data in the dataset."
            )
            stats = DemandStats(
                scope=str(row["category"]),
                periods=int(row["periods"]),
                mean_period=float(row["mean_period"]),
                std_period=float(row["std_period"]),
                mean_daily=float(row["mean_daily"]),
                std_daily=float(row["std_daily"]),
            )
            add_row(str(row["category"]), stats, stock, basis)

    lines = pd.DataFrame(rows)

    if period_days == 7:
        notes.append(
            "Weekly dataset: daily demand is estimated as weekly demand ÷ 7 and "
            f"σ_daily as σ_weekly ÷ √7 (≈ {1 / math.sqrt(7):.2f}). This is an "
            "approximation because it assumes independent daily demand inside the week."
        )
    notes.append(
        "The dataset contains no inventory balances and no supplier lead times. "
        "Safety stock and reorder points are therefore *estimates*, valid only for "
        "the lead time and service level you selected."
    )
    if current_stock is None and not per_category_stock:
        notes.append(
            "Enter current stock to turn these estimates into an actionable reorder "
            "decision; without it the risk and status columns stay blank on purpose."
        )
    if per_category_stock:
        notes.append(
            "Per-category stock values were supplied, so the 'All products' row uses "
            "the business-level input while each category uses its own balance."
        )
    notes.append(
        f"Service level {level * 100:.1f}% → Z = {z:.3f}. A higher service level "
        "raises safety stock and the reorder point."
    )

    return InventoryPlan(
        lines=lines,
        lead_time_days=lead,
        service_level=level,
        z=z,
        period_days=period_days,
        stock_provided=current_stock is not None or bool(per_category_stock),
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #
def highlight_extremes(prepared: PreparedData, top_n: int = 5) -> Dict[str, pd.DataFrame]:
    """High-demand and low-performing categories by average daily demand."""
    stats = category_demand_stats(prepared)
    if stats.empty:
        return {"high": stats, "low": stats}
    return {
        "high": stats.head(top_n).copy(),
        "low": stats.tail(top_n).sort_values("mean_daily").copy(),
    }


def risk_table(plan: InventoryPlan) -> pd.DataFrame:
    """Only the lines that need attention, most urgent first."""
    lines = plan.lines
    if lines.empty:
        return lines
    flagged = lines[lines["status"].isin(["Reorder now", "Watch"])].copy()
    if "stockout_risk_pct" in flagged:
        flagged = flagged.sort_values("stockout_risk_pct", ascending=False)
    return flagged.reset_index(drop=True)


def sensitivity_table(
    prepared: PreparedData,
    lead_times: Optional[List[int]] = None,
    service_levels: Optional[List[float]] = None,
    current_stock: Optional[float] = None,
) -> pd.DataFrame:
    """Reorder point across service levels × lead times, for the heat map."""
    lead_times = lead_times or [3, 7, 14, 30]
    service_levels = service_levels or [0.80, 0.90, 0.95, 0.99]
    stats = total_demand_stats(prepared)
    rows = []
    for lead in lead_times:
        row: Dict[str, object] = {"Lead time (days)": lead}
        for level in service_levels:
            z = z_factor(level)
            row[f"{level * 100:.0f}%"] = stats.mean_daily * lead + z * stats.std_daily * math.sqrt(lead)
        rows.append(row)
    return pd.DataFrame(rows)