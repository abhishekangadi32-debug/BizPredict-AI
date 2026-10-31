"""
Forecast error metrics.

All three headline metrics are the textbook definitions requested by the brief:

    MAE  = (1/n) * sum |y_i - y_hat_i|
    RMSE = sqrt( (1/n) * sum (y_i - y_hat_i)^2 )
    MAPE = (100/n) * sum |(y_i - y_hat_i) / y_i|

Because a period with zero sales makes MAPE undefined (division by zero), such
periods are *excluded from the MAPE denominator* and the exclusion is reported
rather than hidden. WAPE is provided as a scale-free companion that stays valid
when zeros are present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

import config


@dataclass
class ErrorMetrics:
    """Metrics for one model on one evaluation window."""

    mae: float
    rmse: float
    mape: Optional[float]          # None when no non-zero actuals exist
    n_observations: int
    n_mape_used: int
    n_mape_excluded: int
    wape: Optional[float] = None
    bias: Optional[float] = None   # mean signed error (y_hat - y)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE (%)": self.mape,
            "WAPE (%)": self.wape,
            "Bias": self.bias,
            "Observations": self.n_observations,
            "MAPE periods used": self.n_mape_used,
            "MAPE periods excluded (zero actual)": self.n_mape_excluded,
        }


def _aligned(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if a.shape != p.shape:
        raise ValueError(f"actual and predicted differ in length: {a.shape} vs {p.shape}")
    mask = ~(np.isnan(a) | np.isnan(p))
    return a[mask], p[mask]


def mae(actual, predicted) -> float:
    """Mean Absolute Error."""
    a, p = _aligned(actual, predicted)
    return float(np.mean(np.abs(a - p))) if a.size else float("nan")


def rmse(actual, predicted) -> float:
    """Root Mean Squared Error."""
    a, p = _aligned(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2))) if a.size else float("nan")


def mape(actual, predicted) -> float:
    """Mean Absolute Percentage Error in percent, ignoring zero actuals."""
    return error_report(actual, predicted).mape or float("nan")


def wape(actual, predicted) -> float:
    """Weighted Absolute Percentage Error: sum|e| / sum|y| (safe with zeros)."""
    a, p = _aligned(actual, predicted)
    denominator = np.abs(a).sum()
    if a.size == 0 or denominator <= config.MAPE_ZERO_THRESHOLD:
        return float("nan")
    return float(100.0 * np.abs(a - p).sum() / denominator)


def error_report(actual, predicted, label: str = "") -> ErrorMetrics:
    """Full metric set with explicit accounting of excluded periods."""
    a, p = _aligned(actual, predicted)
    if a.size == 0:
        return ErrorMetrics(
            mae=float("nan"),
            rmse=float("nan"),
            mape=None,
            n_observations=0,
            n_mape_used=0,
            n_mape_excluded=0,
            notes=["No overlapping non-null observations to evaluate."],
        )

    errors = a - p
    valid_mape = np.abs(a) > config.MAPE_ZERO_THRESHOLD
    excluded = int((~valid_mape).sum())
    mape_value: Optional[float] = None
    if valid_mape.any():
        mape_value = float(100.0 * np.mean(np.abs(errors[valid_mape] / a[valid_mape])))

    notes: List[str] = []
    if excluded:
        notes.append(
            f"MAPE excludes {excluded} of {a.size} periods whose actual sales were "
            "zero, where percentage error is undefined. MAE/RMSE/WAPE cover all "
            "periods, so compare models on those when zeros are frequent."
        )
    if not valid_mape.any():
        notes.append(
            "Every actual value in the evaluation window is zero, so MAPE cannot "
            "be computed at all."
        )

    return ErrorMetrics(
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        mape=mape_value,
        n_observations=int(a.size),
        n_mape_used=int(valid_mape.sum()),
        n_mape_excluded=excluded,
        wape=wape(a, p),
        bias=float(np.mean(p - a)),
        notes=notes,
    )


def comparison_frame(dates, actual, predicted, lower=None, upper=None) -> pd.DataFrame:
    """Per-period actual vs predicted table for the dashboard."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(dates)),
            "actual": np.asarray(actual, dtype=float),
            "predicted": np.asarray(predicted, dtype=float),
        }
    )
    frame["error"] = frame["predicted"] - frame["actual"]
    frame["abs_error"] = frame["error"].abs()
    frame["abs_pct_error"] = np.where(
        frame["actual"].abs() > config.MAPE_ZERO_THRESHOLD,
        100 * frame["abs_error"] / frame["actual"].abs(),
        np.nan,
    )
    if lower is not None:
        frame["lower"] = np.asarray(lower, dtype=float)
    if upper is not None:
        frame["upper"] = np.asarray(upper, dtype=float)
    return frame


def format_metrics(metrics: ErrorMetrics, currency_symbol: str = "₹") -> dict:
    """Human-readable metric strings for metric cards."""
    def money(value: Optional[float]) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        return f"{currency_symbol}{value:,.0f}"

    def pct(value: Optional[float]) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        return f"{value:,.1f}%"

    return {
        "MAE": money(metrics.mae),
        "RMSE": money(metrics.rmse),
        "MAPE": pct(metrics.mape),
        "WAPE": pct(metrics.wape),
    }