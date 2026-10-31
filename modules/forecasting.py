"""
Time-series forecasting engine.

Primary model: Prophet (trend + yearly/weekly seasonality + optional holiday
regressors). Fallbacks: Holt-Winters exponential smoothing, then a linear trend
with normally distributed residuals. The fallback chain means the dashboard
still produces a real, evaluated forecast when Prophet cannot be installed or
its optimiser fails - and the UI always states which model actually ran.

Evaluation is strictly chronological: the model is fitted on the older portion
and scored on the later portion. No test observation ever reaches training.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from modules.evaluation import ErrorMetrics, error_report
from modules.preprocessing import PreparedData

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

LOGGER = logging.getLogger(__name__)
# Prophet's Stan backend logs an INFO line per chain; keep the dashboard clean.
for _name in ("prophet", "prophet.models", "cmdstanpy"):
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.ERROR)
    _logger.propagate = False
    for _handler in _logger.handlers:
        _handler.setLevel(logging.ERROR)

MODEL_AUTO = "Auto (Prophet → Holt-Winters)"
MODEL_PROPHET = "Prophet"
MODEL_HW = "Holt-Winters"
MODEL_LINEAR = "Linear trend"

#: Normal quantile for the configured prediction interval width (0.80).
Z_INTERVAL = 1.2816


# --------------------------------------------------------------------------- #
# Capability checks
# --------------------------------------------------------------------------- #
def prophet_available() -> bool:
    try:
        import prophet  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def statsmodels_available() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def available_models() -> List[str]:
    """Models the current environment can actually run."""
    models = [MODEL_AUTO]
    if prophet_available():
        models.append(MODEL_PROPHET)
    if statsmodels_available():
        models.append(MODEL_HW)
    models.append(MODEL_LINEAR)
    return models


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class ForecastResult:
    """Everything the dashboard needs after one forecast run."""

    model_name: str
    requested_model: str
    series_label: str
    history: pd.DataFrame
    fitted: pd.DataFrame
    future: pd.DataFrame
    evaluation: pd.DataFrame
    metrics: ErrorMetrics
    components: Optional[pd.DataFrame]
    horizon_days: int
    horizon_periods: int
    period_days: int
    granularity: str
    train_periods: int
    test_periods: int
    notes: List[str] = field(default_factory=list)

    @property
    def forecast_total(self) -> float:
        return float(self.future["yhat"].sum())

    @property
    def forecast_mean(self) -> float:
        return float(self.future["yhat"].mean())

    def to_csv(self) -> bytes:
        """Downloadable CSV of the future forecast."""
        frame = self.future.copy()
        frame["period_start"] = frame["ds"].dt.date
        frame = frame.rename(
            columns={
                "ds": "date",
                "yhat": "forecast_sales",
                "yhat_lower": "lower_bound",
                "yhat_upper": "upper_bound",
            }
        )
        frame["model"] = self.model_name
        frame["horizon_days"] = self.horizon_days
        columns = ["date", "period_start", "forecast_sales", "lower_bound",
                   "upper_bound", "model", "horizon_days"]
        return frame[columns].to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- #
# History preparation
# --------------------------------------------------------------------------- #
def history_frame(prepared: PreparedData) -> pd.DataFrame:
    """Prophet-style ``ds``/``y`` frame from the aggregated series."""
    frame = prepared.series[["date", "sales"]].copy()
    frame = frame.rename(columns={"date": "ds", "sales": "y"})
    frame["ds"] = pd.to_datetime(frame["ds"])
    return frame.dropna().sort_values("ds").reset_index(drop=True)


def chronological_split(frame: pd.DataFrame, test_fraction: float = config.TEST_FRACTION):
    """Split oldest->newest; never shuffle a time series."""
    n = len(frame)
    n_test = max(1, int(round(n * test_fraction)))
    n_train = n - n_test
    if n_train < 5:
        raise ValueError(
            f"Only {n} periods available - a chronological split needs more history."
        )
    return frame.iloc[:n_train].reset_index(drop=True), frame.iloc[n_train:].reset_index(drop=True)


def _future_index(last_date: pd.Timestamp, horizon: int, freq: str) -> pd.DatetimeIndex:
    """Dates for the periods that follow *last_date* at the modelling frequency."""
    if freq == "MS":
        return pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    if freq.startswith("7"):
        return pd.date_range(last_date + pd.Timedelta(days=7), periods=horizon, freq="7D")
    return pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")


def _interval(horizon: int, sigma: float, z: float = Z_INTERVAL) -> np.ndarray:
    """Residual-based interval that widens with the square root of the step."""
    steps = np.arange(1, horizon + 1)
    return z * sigma * np.sqrt(steps)


# --------------------------------------------------------------------------- #
# Model backends
# --------------------------------------------------------------------------- #
def _silence_backend_logs() -> None:
    """Prophet's Stan backend adds its own handler when imported, so the log
    suppression has to be repeated after the import."""
    for name in ("prophet", "prophet.models", "cmdstanpy"):
        logger_ = logging.getLogger(name)
        logger_.setLevel(logging.ERROR)
        logger_.propagate = False
        for handler_ in logger_.handlers:
            handler_.setLevel(logging.ERROR)


def _predict_prophet(
    frame: pd.DataFrame,
    horizon: int,
    freq: str,
    holiday_frame: Optional[pd.DataFrame],
) -> Dict[str, object]:
    """Fit Prophet and return fitted values, future forecast and components."""
    from prophet import Prophet

    _silence_backend_logs()

    is_weekly = freq.startswith("7")
    # Prophet needs roughly two years of *observations* for a stable yearly cycle.
    # A sampled daily extract may span years while holding only a few hundred rows,
    # in which case yearly seasonality is under-identified and is switched off.
    span_days = (frame["ds"].max() - frame["ds"].min()).days
    enough_for_yearly = (
        (is_weekly and len(frame) >= 80)
        or (not is_weekly and span_days >= 730 and len(frame) >= 500)
    )

    model = Prophet(
        growth="linear",
        interval_width=config.PROPHET_INTERVAL_WIDTH,
        yearly_seasonality=(config.PROPHET_YEARLY_SEASONALITY and enough_for_yearly),
        weekly_seasonality=(config.PROPHET_WEEKLY_SEASONALITY and not is_weekly),
        daily_seasonality=False,
        changepoint_prior_scale=config.PROPHET_CHANGEPOINT_PRIOR_SCALE,
        holidays=holiday_frame if holiday_frame is not None and not holiday_frame.empty else None,
    )
    _silence_backend_logs()   # cmdstanpy attaches its handler on model creation
    model.fit(frame[["ds", "y"]].copy())

    future = model.make_future_dataframe(periods=horizon, freq=freq, include_history=True)
    forecast = model.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    history_len = len(frame)
    fitted = forecast.iloc[:history_len][["ds", "yhat", "yhat_lower", "yhat_upper"]]
    future_part = forecast.iloc[history_len:][["ds", "yhat", "yhat_lower", "yhat_upper"]]

    component_cols = [c for c in ("trend", "weekly", "yearly", "holidays")
                      if c in forecast.columns]
    components = forecast[["ds"] + component_cols].copy() if component_cols else None

    seasonal_bits = []
    if "yearly" in component_cols:
        seasonal_bits.append("yearly")
    if "weekly" in component_cols:
        seasonal_bits.append("weekly")
    note = (
        "Prophet fitted with a linear trend"
        + (f" and {' + '.join(seasonal_bits)} seasonality" if seasonal_bits else "")
        + (" plus holiday regressors" if holiday_frame is not None and not holiday_frame.empty else "")
        + f", {int(config.PROPHET_INTERVAL_WIDTH * 100)}% uncertainty interval."
    )
    return {
        "fitted": fitted.reset_index(drop=True),
        "future": future_part.reset_index(drop=True),
        "components": components,
        "note": note,
        "model_name": MODEL_PROPHET,
    }


def _predict_holt_winters(
    frame: pd.DataFrame,
    horizon: int,
    freq: str,
) -> Dict[str, object]:
    """Holt-Winters exponential smoothing with residual-based intervals."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = frame["y"].astype(float).to_numpy()
    is_weekly = freq.startswith("7")
    season_length = 52 if is_weekly else config.HW_SEASONAL_PERIODS

    use_seasonal = (
        len(y) >= max(config.HW_MIN_PERIODS_FOR_SEASONALITY, 2 * season_length)
        and season_length >= 2
    )
    if is_weekly and len(y) < 2 * season_length:
        season_length = 4 if len(y) >= 12 else None
        use_seasonal = season_length is not None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=True,
            seasonal="add" if use_seasonal else None,
            seasonal_periods=season_length if use_seasonal else None,
            initialization_method="estimated",
        ).fit(optimized=True)

    fitted_values = np.asarray(model.fittedvalues, dtype=float)
    point = np.asarray(model.forecast(horizon), dtype=float)

    residuals = y - fitted_values
    sigma = float(np.nanstd(residuals, ddof=1)) if len(residuals) > 2 else float(np.nanstd(y))
    half_width = _interval(horizon, sigma)

    future_dates = _future_index(frame["ds"].iloc[-1], horizon, freq)
    fitted = pd.DataFrame(
        {
            "ds": frame["ds"].to_numpy(),
            "yhat": np.clip(fitted_values, 0, None),
            "yhat_lower": np.clip(fitted_values - sigma * Z_INTERVAL, 0, None),
            "yhat_upper": fitted_values + sigma * Z_INTERVAL,
        }
    )
    future = pd.DataFrame(
        {
            "ds": future_dates,
            "yhat": np.clip(point, 0, None),
            "yhat_lower": np.clip(point - half_width, 0, None),
            "yhat_upper": point + half_width,
        }
    )
    note = (
        "Holt-Winters additive "
        f"{'damped trend + seasonality' if use_seasonal else 'trend (series too short for seasonality)'}"
        f" with seasonal period {season_length if use_seasonal else 'n/a'}. Intervals are "
        "approximated from residual spread growing with √h, which is wider than a "
        "simulated interval - treat them as indicative."
    )
    return {
        "fitted": fitted,
        "future": future,
        "components": None,
        "note": note,
        "model_name": MODEL_HW,
    }


def _predict_linear(frame: pd.DataFrame, horizon: int, freq: str) -> Dict[str, object]:
    """Least-squares linear trend; the last-resort baseline."""
    y = frame["y"].astype(float).to_numpy()
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    fitted_values = intercept + slope * x
    residuals = y - fitted_values
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 2 else float(np.std(y))

    future_x = np.arange(len(y), len(y) + horizon)
    point = intercept + slope * future_x
    half_width = _interval(horizon, sigma)

    future_dates = _future_index(frame["ds"].iloc[-1], horizon, freq)
    return {
        "fitted": pd.DataFrame(
            {
                "ds": frame["ds"].to_numpy(),
                "yhat": np.clip(fitted_values, 0, None),
                "yhat_lower": np.clip(fitted_values - sigma * Z_INTERVAL, 0, None),
                "yhat_upper": fitted_values + sigma * Z_INTERVAL,
            }
        ),
        "future": pd.DataFrame(
            {
                "ds": future_dates,
                "yhat": np.clip(point, 0, None),
                "yhat_lower": np.clip(point - half_width, 0, None),
                "yhat_upper": point + half_width,
            }
        ),
        "components": None,
        "note": (
            "Statistical fallback: an ordinary least-squares trend on the period "
            "index. It captures direction only, has no seasonality and its "
            "intervals are approximate."
        ),
        "model_name": MODEL_LINEAR,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _run_backend(
    model: str,
    frame: pd.DataFrame,
    horizon: int,
    freq: str,
    holiday_frame: Optional[pd.DataFrame],
) -> Dict[str, object]:
    if model == MODEL_PROPHET:
        return _predict_prophet(frame, horizon, freq, holiday_frame)
    if model == MODEL_HW:
        return _predict_holt_winters(frame, horizon, freq)
    if model == MODEL_LINEAR:
        return _predict_linear(frame, horizon, freq)
    raise ValueError(f"Unknown model: {model}")


def _fit_with_fallback(
    requested: str,
    frame: pd.DataFrame,
    horizon: int,
    freq: str,
    holiday_frame: Optional[pd.DataFrame],
) -> Tuple[Dict[str, object], List[str]]:
    """Honour the requested model, falling back only when it truly fails."""
    notes: List[str] = []
    order: List[str]
    if requested == MODEL_PROPHET:
        order = [MODEL_PROPHET, MODEL_LINEAR]
    elif requested == MODEL_HW:
        order = [MODEL_HW, MODEL_LINEAR]
    elif requested == MODEL_LINEAR:
        order = [MODEL_LINEAR]
    else:  # Auto
        order = [MODEL_PROPHET, MODEL_HW, MODEL_LINEAR]

    last_error: Optional[Exception] = None
    for candidate in order:
        if candidate == MODEL_PROPHET and not prophet_available():
            notes.append("Prophet is not installed in this environment.")
            continue
        if candidate == MODEL_HW and not statsmodels_available():
            notes.append("statsmodels is not installed, so Holt-Winters is skipped.")
            continue
        try:
            result = _run_backend(candidate, frame, horizon, freq, holiday_frame)
            if candidate != order[0]:
                notes.append(
                    f"{order[0]} could not be used here, so {candidate} produced this "
                    "forecast instead."
                )
            return result, notes
        except Exception as exc:  # noqa: BLE001 - any backend failure triggers fallback
            last_error = exc
            notes.append(f"{candidate} failed: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "No forecasting backend could be fitted. Last error: "
        f"{last_error}. Try a longer history or the linear trend model."
    )


@dataclass
class ForecastInput:
    """The exact series a forecast is fitted on, after optional re-aggregation."""

    frame: pd.DataFrame      # columns: ds, y
    freq: str                # pandas offset for future dates: "D" or "7D"
    period_days: int         # calendar days covered by one period
    granularity: str         # "Daily" | "Weekly" | "Monthly"
    holiday: Optional[pd.DataFrame]

    @property
    def label(self) -> str:
        return self.granularity.lower()


def build_forecast_input(prepared: PreparedData, granularity: str = "Native") -> ForecastInput:
    """Prepare the modelling series, optionally re-aggregated to weekly totals.

    Daily data from a *sampled* extract is extremely noisy - a single large
    order dominates the day - so the dashboard lets the user forecast weekly
    totals instead. The choice is explicit and is echoed in the results.
    """
    native = prepared.freq_label.capitalize()          # "Daily" | "Weekly"
    choice = granularity if granularity in {"Daily", "Weekly", "Monthly"} else native
    if prepared.freq_label == "weekly":
        choice = "Weekly"

    if choice == native and prepared.freq_label == "daily":
        frame = history_frame(prepared)
        return ForecastInput(frame, "D", 1, "Daily", prepared.holiday_frame)

    if choice == "Daily":
        frame = history_frame(prepared)
        return ForecastInput(frame, "D", 1, "Daily", prepared.holiday_frame)

    if choice == "Monthly":
        from modules.preprocessing import resample_series
        monthly = resample_series(prepared.series, "Monthly")
        frame = pd.DataFrame({"ds": monthly["date"], "y": monthly["sales"]}).dropna()
        return ForecastInput(frame.reset_index(drop=True), "MS", 30, "Monthly", prepared.holiday_frame)

    # Weekly aggregation (native for weekly datasets)
    if prepared.freq_label == "weekly":
        frame = history_frame(prepared)
    else:
        from modules.preprocessing import resample_series
        weekly = resample_series(prepared.series, "Weekly")
        frame = pd.DataFrame({"ds": weekly["date"], "y": weekly["sales"]}).dropna()
    return ForecastInput(frame.reset_index(drop=True), "7D", 7, "Weekly", prepared.holiday_frame)


def run_forecast(
    prepared: PreparedData,
    horizon_days: int = config.FORECAST_HORIZON_DEFAULT_DAYS,
    model: str = MODEL_AUTO,
    granularity: str = "Native",
) -> ForecastResult:
    """Evaluate on a holdout, then refit on all data and forecast forward."""
    forecast_input = build_forecast_input(prepared, granularity)
    frame = forecast_input.frame
    if len(frame) < config.MIN_HISTORY_PERIODS:
        raise ValueError(
            f"Only {len(frame)} periods are available. At least "
            f"{config.MIN_HISTORY_PERIODS} are required for a reliable forecast."
        )

    horizon_days = int(np.clip(
        horizon_days, config.FORECAST_HORIZON_MIN_DAYS, config.FORECAST_HORIZON_MAX_DAYS
    ))
    period_days = forecast_input.period_days
    horizon = max(1, int(round(horizon_days / period_days)))
    freq = forecast_input.freq
    holidays = forecast_input.holiday

    train, test = chronological_split(frame)

    # ---- 1. chronological evaluation --------------------------------------
    eval_result, eval_notes = _fit_with_fallback(model, train, len(test), freq, holidays)
    test_pred = eval_result["future"].head(len(test))
    metrics = error_report(test["y"].to_numpy(), test_pred["yhat"].to_numpy())

    evaluation = pd.DataFrame(
        {
            "date": test["ds"].to_numpy(),
            "actual": test["y"].to_numpy(),
            "predicted": test_pred["yhat"].to_numpy(),
            "lower": test_pred["yhat_lower"].to_numpy(),
            "upper": test_pred["yhat_upper"].to_numpy(),
        }
    )
    evaluation["error"] = evaluation["predicted"] - evaluation["actual"]
    evaluation["abs_pct_error"] = np.where(
        evaluation["actual"].abs() > config.MAPE_ZERO_THRESHOLD,
        100 * evaluation["error"].abs() / evaluation["actual"].abs(),
        np.nan,
    )

    # ---- 2. refit on the full history for the forward forecast -------------
    full_result, full_notes = _fit_with_fallback(model, frame, horizon, freq, holidays)

    notes = list(dict.fromkeys(eval_notes + full_notes + metrics.notes))
    notes.append(full_result["note"])
    notes.append(
        "Evaluation split: "
        f"{len(train):,} periods trained, {len(test):,} later periods scored. "
        "The forward forecast is then refitted on all history."
    )
    if period_days > 1:
        unit = "month" if period_days >= 30 else "week"
        notes.append(
            f"Modelling at {forecast_input.granularity.lower()} granularity: a "
            f"{horizon_days}-day horizon maps to {horizon} {unit} period(s) "
            f"({period_days} days each)."
        )
    notes.append("Predictions are floored at zero because negative sales are not physical.")

    return ForecastResult(
        model_name=str(full_result["model_name"]),
        requested_model=model,
        series_label="Total sales per period",
        history=frame,
        fitted=full_result["fitted"],
        future=full_result["future"],
        evaluation=evaluation,
        metrics=metrics,
        components=full_result["components"],
        horizon_days=horizon_days,
        horizon_periods=horizon,
        period_days=period_days,
        granularity=forecast_input.granularity,
        train_periods=len(train),
        test_periods=len(test),
        notes=notes,
    )


def compare_models(prepared: PreparedData, granularity: str = "Native") -> pd.DataFrame:
    """Score every available backend on the same chronological split."""
    forecast_input = build_forecast_input(prepared, granularity)
    frame = forecast_input.frame
    train, test = chronological_split(frame)
    rows: List[Dict[str, object]] = []
    for candidate in (MODEL_PROPHET, MODEL_HW, MODEL_LINEAR):
        if candidate == MODEL_PROPHET and not prophet_available():
            continue
        if candidate == MODEL_HW and not statsmodels_available():
            continue
        try:
            result = _run_backend(
                candidate, train, len(test), forecast_input.freq, forecast_input.holiday
            )
            pred = result["future"].head(len(test))
            metrics = error_report(test["y"].to_numpy(), pred["yhat"].to_numpy())
            rows.append({
                "Model": candidate,
                "MAE": metrics.mae,
                "RMSE": metrics.rmse,
                "MAPE (%)": metrics.mape,
                "WAPE (%)": metrics.wape,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "Model": candidate, "MAE": np.nan, "RMSE": np.nan,
                "MAPE (%)": np.nan, "WAPE (%)": np.nan,
                "Note": f"failed: {type(exc).__name__}: {exc}",
            })
    table = pd.DataFrame(rows)
    if "Note" not in table.columns:
        table["Note"] = ""
    return table


def trend_summary(result: ForecastResult) -> Dict[str, object]:
    """Direction and magnitude of the fitted trend, stated plainly."""
    history = result.fitted["yhat"].to_numpy()
    if len(history) < 4:
        return {"direction": "unknown", "change_pct": None}
    head = float(np.mean(history[: max(1, len(history) // 4)]))
    tail = float(np.mean(history[-max(1, len(history) // 4):]))
    change = 100 * (tail - head) / head if head > 0 else None
    direction = "rising" if (change or 0) > 2 else "falling" if (change or 0) < -2 else "flat"
    return {"direction": direction, "change_pct": change}


def seasonal_strength(result: ForecastResult) -> Dict[str, float]:
    """Share of variation explained by the seasonal components, if available."""
    if result.components is None:
        return {}
    strength: Dict[str, float] = {}
    for column in ("weekly", "yearly"):
        if column in result.components.columns:
            values = result.components[column].dropna()
            if not values.empty:
                strength[column] = float(values.std())
    return strength