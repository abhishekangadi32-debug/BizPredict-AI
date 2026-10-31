"""
BizPredict AI - interactive business intelligence dashboard.

Run with:

    streamlit run app.py

The module is deliberately thin: every calculation lives in ``modules/`` and is
unit-testable without Streamlit. This file only handles state, layout and charts.
"""

from __future__ import annotations

import html
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from modules import analytics, forecasting, inventory
from modules.data_loader import (
    DataSourceError,
    SchemaError,
    load_default_dataset,
    load_from_bytes,
    load_merged_dataset,
    profile_dataset,
)
from modules.preprocessing import PreparedData, prepare

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title=f"{config.APP_TITLE} · Retail Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

SOURCE_DEFAULT = "Default dataset (Pentaho sample)"
SOURCE_MERGED = "Merged weekly dataset (156k)"
SOURCE_UPLOAD = "Upload my own CSV"
SOURCE_KEYS = {
    SOURCE_DEFAULT: "default",
    SOURCE_MERGED: "merged",
    SOURCE_UPLOAD: "upload",
}


def initialise_state() -> None:
    defaults: Dict[str, object] = {
        "theme": "Light",
        "source": SOURCE_DEFAULT,
        "page": config.PAGES[0],
        "uploaded_bytes": None,
        "uploaded_name": None,
        "forecast_result": None,
        "forecast_key": None,
        "inventory_stock": 0.0,
        "inventory_has_stock": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


initialise_state()


# --------------------------------------------------------------------------- #
# Theming
# --------------------------------------------------------------------------- #
THEMES = {
    "Light": {
        "bg": "#F7F8FC",
        "panel": "#FFFFFF",
        "text": "#0F172A",
        "muted": "#64748B",
        "border": "#E2E8F0",
        "grid": "rgba(148,163,184,0.20)",
        "sidebar": "#FFFFFF",
        "accent_soft": "rgba(79,70,229,0.08)",
        "shadow": "0 1px 2px rgba(15,23,42,0.06), 0 8px 24px -12px rgba(15,23,42,0.18)",
    },
    "Dark": {
        "bg": "#0B1120",
        "panel": "#131C2E",
        "text": "#E2E8F0",
        "muted": "#94A3B8",
        "border": "#243049",
        "grid": "rgba(148,163,184,0.16)",
        "sidebar": "#0B1120",
        "accent_soft": "rgba(129,140,248,0.14)",
        "shadow": "0 1px 2px rgba(0,0,0,0.4), 0 12px 32px -16px rgba(0,0,0,0.7)",
    },
}


def palette() -> Dict[str, str]:
    return THEMES[st.session_state["theme"]]


def inject_css() -> None:
    """Premium SaaS styling: cards, spacing, typography and sidebar nav."""
    c = palette()
    is_dark = st.session_state["theme"] == "Dark"
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: {("'Inter', " if True else "")}-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }}
        .stApp {{ background: {c['bg']}; color: {c['text']}; }}
        section[data-testid="stSidebar"] {{
            background: {c['sidebar']};
            border-right: 1px solid {c['border']};
        }}
        section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
        .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }}
        h1, h2, h3, h4 {{ color: {c['text']}; letter-spacing: -0.018em; }}
        h1 {{ font-size: 1.65rem !important; font-weight: 700 !important; }}
        h2 {{ font-size: 1.2rem !important; font-weight: 650 !important; }}
        h3 {{ font-size: 1rem !important; font-weight: 600 !important; }}
        p, span, label, div {{ color: {c['text']}; }}

        /* ---- KPI cards ---- */
        .kpi-grid {{ display: grid; gap: 0.85rem; }}
        .kpi {{
            background: {c['panel']};
            border: 1px solid {c['border']};
            border-radius: 14px;
            padding: 0.95rem 1.05rem;
            box-shadow: {c['shadow']};
            height: 100%;
        }}
        .kpi .label {{
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em;
            text-transform: uppercase; color: {c['muted']}; margin-bottom: 0.35rem;
        }}
        .kpi .value {{
            font-size: 1.5rem; font-weight: 700; line-height: 1.15;
            letter-spacing: -0.02em; color: {c['text']};
            word-break: break-word;
        }}
        .kpi .caption {{ font-size: 0.74rem; color: {c['muted']}; margin-top: 0.3rem; }}

        /* ---- section headers ---- */
        .section-title {{
            display: flex; align-items: center; gap: 0.55rem;
            margin: 1.5rem 0 0.6rem 0;
            font-size: 0.95rem; font-weight: 650; color: {c['text']};
        }}
        .section-title::before {{
            content: ""; width: 3px; height: 16px; border-radius: 2px;
            background: linear-gradient(180deg, {config.PALETTE['primary']}, {config.PALETTE['accent']});
        }}

        /* ---- badges ---- */
        .badge {{
            display: inline-block; padding: 0.16rem 0.55rem; border-radius: 999px;
            font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em;
            background: {c['accent_soft']}; color: {config.PALETTE['primary_soft'] if is_dark else config.PALETTE['primary']};
            border: 1px solid {c['border']};
        }}
        .badge.alt {{ background: rgba(6,182,212,0.12); color: {config.PALETTE['accent']}; }}

        /* ---- app header ---- */
        .app-head {{ margin-bottom: 0.4rem; }}
        .app-head .eyebrow {{
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em;
            text-transform: uppercase; color: {c['muted']};
        }}
        .app-head .title {{ font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; }}
        .app-head .sub {{ font-size: 0.86rem; color: {c['muted']}; margin-top: 0.15rem; }}
        .rule {{ height: 1px; background: {c['border']}; margin: 0.9rem 0 1.1rem 0; }}

        /* ---- insight rows ---- */
        .insight {{
            border-radius: 10px; padding: 0.6rem 0.8rem; margin-bottom: 0.45rem;
            font-size: 0.83rem; border: 1px solid {c['border']};
            background: {c['panel']}; color: {c['text']};
        }}
        .insight.warning {{ border-left: 3px solid {config.PALETTE['warning']}; }}
        .insight.danger  {{ border-left: 3px solid {config.PALETTE['danger']}; }}
        .insight.success {{ border-left: 3px solid {config.PALETTE['success']}; }}

        /* ---- sidebar nav ---- */
        section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 0.15rem; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            padding: 0.45rem 0.6rem; border-radius: 8px; width: 100%;
            font-size: 0.88rem; transition: background 120ms ease;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background: {c['accent_soft']};
        }}

        /* ---- tables & widgets ---- */
        [data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
        div[data-testid="stMetric"] {{
            background: {c['panel']}; border: 1px solid {c['border']};
            border-radius: 12px; padding: 0.75rem 0.9rem; box-shadow: {c['shadow']};
        }}
        div[data-testid="stMetricLabel"] p {{ font-size: 0.74rem !important; color: {c['muted']} !important; }}
        .stButton button {{
            border-radius: 9px; font-weight: 600; border: 1px solid {c['border']};
        }}
        .stButton button[kind="primary"] {{ background: {config.PALETTE['primary']}; }}
        .footer {{ font-size: 0.72rem; color: {c['muted']}; margin-top: 2rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def esc(value: object) -> str:
    """Escape dataset-derived text before injecting it into HTML cards.

    ``$`` becomes an HTML entity so that a card containing several currency
    values cannot be mistaken for a LaTeX span by Streamlit's Markdown renderer.
    """
    return html.escape(str(value)).replace("$", "&#36;")


def md(text: object) -> str:
    """Escape currency symbols for Markdown.

    Streamlit renders Markdown, where a pair of ``$`` characters is treated as a
    LaTeX formula. Every generated sentence that contains money has to escape
    them or the prose turns into equations.
    """
    return str(text).replace("$", r"\$")


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{esc(title)}</div>', unsafe_allow_html=True)


def kpi_cards(cards: List[Tuple[str, str, str]], columns: int = 3) -> None:
    """Render KPI cards in a responsive grid."""
    for start in range(0, len(cards), columns):
        cols = st.columns(columns, gap="small")
        for col, (label, value, caption) in zip(cols, cards[start:start + columns]):
            with col:
                st.markdown(
                    f"""<div class="kpi">
                        <div class="label">{esc(label)}</div>
                        <div class="value">{esc(value)}</div>
                        <div class="caption">{esc(caption)}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        st.write("")


def insights(items: List[Tuple[str, str]]) -> None:
    if not items:
        return
    for severity, message in items:
        st.markdown(
            f'<div class="insight {esc(severity)}">{esc(message)}</div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Chart helpers
# --------------------------------------------------------------------------- #
def style_fig(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    c = palette()
    fig.update_layout(
        height=height,
        margin=dict(l=6, r=6, t=18, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=c["text"]),
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        colorway=config.CHART_SEQUENCE,
    )
    fig.update_xaxes(gridcolor=c["grid"], zeroline=False, linecolor=c["border"],
                     tickfont=dict(size=11, color=c["muted"]))
    fig.update_yaxes(gridcolor=c["grid"], zeroline=False, linecolor=c["border"],
                     tickfont=dict(size=11, color=c["muted"]))
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "scrollZoom": False},
    )


def chart_header(title: str, subtitle: str = "") -> None:
    c = palette()
    st.markdown(
        f'<div style="margin:0.2rem 0 0.1rem 0">'
        f'<div style="font-size:0.9rem;font-weight:600;color:{c["text"]}">{esc(title)}</div>'
        f'<div style="font-size:0.75rem;color:{c["muted"]}">{esc(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def revenue_figure(frame: pd.DataFrame, granularity: str, show_average: bool = True) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame["date"], y=frame["sales"], name="Revenue", mode="lines",
        line=dict(color=config.PALETTE["primary"], width=2.2),
        fill="tozeroy", fillcolor="rgba(79,70,229,0.12)",
    ))
    if show_average and len(frame) > 3:
        fig.add_trace(go.Scatter(
            x=frame["date"], y=frame["moving_average"], name="3-period average",
            mode="lines", line=dict(color=config.PALETTE["accent"], width=1.8, dash="dot"),
        ))
    fig.update_yaxes(title_text=f"{config.CURRENCY_SYMBOL} revenue")
    return style_fig(fig, height=330, legend=True)


def category_figure(perf: pd.DataFrame, top_n: int = 10) -> go.Figure:
    data = perf.head(top_n).sort_values("revenue")
    fig = go.Figure(go.Bar(
        x=data["revenue"], y=data["category"].astype(str), orientation="h",
        marker=dict(color=data["revenue"], colorscale=[[0, "#A5B4FC"], [1, config.PALETTE["primary"]]],
                    line=dict(width=0)),
        text=[f"{v:,.0f}" for v in data["revenue"]],
        texttemplate="%{text}", textposition="outside",
        hovertemplate="%{y}<br>Revenue: %{x:,.0f}<extra></extra>",
    ))
    fig.update_xaxes(title_text=f"{config.CURRENCY_SYMBOL} revenue")
    return style_fig(fig, height=max(260, 34 * len(data) + 70))


def donut_figure(perf: pd.DataFrame, top_n: int = 7) -> go.Figure:
    data = perf.head(top_n).copy()
    if len(perf) > top_n:
        other = perf["revenue"].iloc[top_n:].sum()
        data = pd.concat(
            [data, pd.DataFrame([{"category": "Other", "revenue": other}])],
            ignore_index=True,
        )
    fig = go.Figure(go.Pie(
        labels=data["category"].astype(str), values=data["revenue"], hole=0.58,
        textinfo="percent", textfont=dict(size=11),
        hovertemplate="%{label}<br>%{value:,.0f} (%{percent})<extra></extra>",
    ))
    fig.update_layout(showlegend=True)
    return style_fig(fig, height=330, legend=True)


# --------------------------------------------------------------------------- #
# Data access (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=6)
def load_prepared(
    source_key: str,
    payload: Optional[bytes],
    filename: Optional[str],
) -> Tuple[PreparedData, Dict[str, object], str]:
    """Load, profile and preprocess a dataset. Cached per input."""
    if source_key == "default":
        loaded = load_default_dataset()
    elif source_key == "merged":
        loaded = load_merged_dataset()
    elif payload is not None:
        loaded = load_from_bytes(payload, filename or "uploaded.csv")
    else:
        raise DataSourceError("No file has been uploaded yet.")

    profile = profile_dataset(loaded)
    prepared = prepare(loaded, profile)
    return prepared, loaded.summary(), loaded.encoding


@st.cache_data(show_spinner=False, max_entries=8)
def cached_forecast(
    source_key: str,
    payload: Optional[bytes],
    filename: Optional[str],
    horizon: int,
    model: str,
    granularity: str,
):
    prepared, _, _ = load_prepared(source_key, payload, filename)
    return forecasting.run_forecast(prepared, horizon, model, granularity)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_comparison(
    source_key: str,
    payload: Optional[bytes],
    filename: Optional[str],
    granularity: str,
):
    prepared, _, _ = load_prepared(source_key, payload, filename)
    return forecasting.compare_models(prepared, granularity)


def current_identity() -> Tuple[str, Optional[bytes], Optional[str]]:
    return (
        SOURCE_KEYS[st.session_state["source"]],
        st.session_state["uploaded_bytes"],
        st.session_state["uploaded_name"],
    )


# --------------------------------------------------------------------------- #
# Shared UI pieces
# --------------------------------------------------------------------------- #
def app_header(title: str, subtitle: str, badges: List[str]) -> None:
    badge_html = " ".join(f'<span class="badge">{esc(b)}</span>' for b in badges)
    st.markdown(
        f"""<div class="app-head">
            <div class="eyebrow">{esc(config.APP_TAGLINE)}</div>
            <div class="title">{esc(title)}</div>
            <div class="sub">{esc(subtitle)}</div>
            <div style="margin-top:0.5rem">{badge_html}</div>
        </div>
        <div class="rule"></div>""",
        unsafe_allow_html=True,
    )


def dataset_badges(prepared: PreparedData) -> List[str]:
    return [
        prepared.profile.source_label,
        f"{prepared.stats.final_rows:,} records",
        f"{prepared.freq_label} series · {prepared.stats.periods_created:,} periods",
        f"{prepared.stats.date_min} → {prepared.stats.date_max}",
    ]


def download_frame(frame: pd.DataFrame, filename: str, label: str, key: str) -> None:
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
        use_container_width=False,
    )


# --------------------------------------------------------------------------- #
# Page 1 - Executive overview
# --------------------------------------------------------------------------- #
def page_overview(prepared: PreparedData) -> None:
    kpis = analytics.executive_kpis(prepared)
    app_header(
        "Executive Overview",
        "A single view of revenue, demand and category performance for the loaded period.",
        dataset_badges(prepared),
    )

    kpi_cards(kpis.cards(), columns=3)

    left, right = st.columns([2.1, 1], gap="medium")
    with left:
        chart_header("Revenue trend", "Aggregated from the loaded records; hover for exact values.")
        trend = analytics.revenue_trend(prepared, prepared.granularities[-1])
        show(revenue_figure(trend, prepared.granularities[-1]))
    with right:
        chart_header("Business health", "Derived indicators, not external benchmarks.")
        rows = [
            ("Peak period", f"{kpis.peak_period_date}", analytics.money(kpis.peak_period_sales, True)),
            ("Periods modelled", f"{kpis.period_count:,}", prepared.freq_label),
            (
                "Growth",
                "n/a" if kpis.growth_pct is None else f"{kpis.growth_pct:+.1f}%",
                kpis.growth_window,
            ),
            ("Zero-sales periods", f"{kpis.zero_periods:,}", "kept, not dropped"),
        ]
        for label, value, caption in rows:
            st.markdown(
                f"""<div class="kpi" style="margin-bottom:0.5rem">
                    <div class="label">{esc(label)}</div>
                    <div class="value" style="font-size:1.05rem">{esc(value)}</div>
                    <div class="caption">{esc(caption)}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    section("Performance breakdown")
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        chart_header("Revenue by month", "Monthly totals across the full history.")
        monthly = analytics.revenue_trend(prepared, "Monthly")
        fig = go.Figure(go.Bar(
            x=monthly["date"], y=monthly["sales"], marker_color=config.PALETTE["primary_soft"],
            hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra></extra>",
        ))
        fig.update_yaxes(title_text=f"{config.CURRENCY_SYMBOL} revenue")
        show(style_fig(fig, height=300))
    with col_b:
        if prepared.profile.has_category:
            perf = analytics.category_performance(prepared)
            chart_header("Revenue share by category", "Top contributors to total revenue.")
            show(donut_figure(perf))
        else:
            chart_header("Revenue distribution", "Histogram of period revenue.")
            fig = go.Figure(go.Histogram(
                x=prepared.series["sales"], nbinsx=30,
                marker_color=config.PALETTE["accent"],
            ))
            show(style_fig(fig, height=300))

    section("Business performance summary")
    for sentence in analytics.business_summary(prepared, kpis):
        st.markdown(f"- {md(sentence)}")

    findings = analytics.insights(prepared)
    if findings:
        section("Signals worth attention")
        insights(findings)


# --------------------------------------------------------------------------- #
# Page 2 - Sales analytics
# --------------------------------------------------------------------------- #
def page_sales(prepared: PreparedData) -> None:
    app_header(
        "Sales Analytics",
        "Explore revenue at daily, weekly and monthly resolution with category and holiday breakdowns.",
        dataset_badges(prepared),
    )

    control_a, control_b, control_c = st.columns([1.2, 2, 1], gap="medium")
    with control_a:
        granularity = st.radio(
            "Resolution",
            prepared.granularities,
            index=0,
            horizontal=False,
            key="sales_granularity",
        )
    with control_c:
        top_n = st.slider("Categories shown", 5, 20, 10, key="sales_top_n")

    trend = analytics.revenue_trend(prepared, granularity)
    summary = analytics.distribution_summary(prepared)

    kpi_cards(
        [
            (f"{granularity} revenue", analytics.money(trend["sales"].sum(), True),
             f"{len(trend):,} periods"),
            ("Best period", analytics.money(float(trend["sales"].max()), True),
             trend.loc[trend["sales"].idxmax(), "date"].strftime("%d %b %Y")),
            ("Average per period", analytics.money(float(trend["sales"].mean()), True),
             f"median {analytics.money(summary['median'], True)}"),
            ("Volatility (CV)", f"{summary['std'] / summary['mean']:.2f}" if summary["mean"] else "n/a",
             "σ ÷ mean of period revenue"),
        ],
        columns=4,
    )

    chart_header(f"{granularity} revenue", "Bars show the underlying periods; the line is the trend.")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=trend["date"], y=trend["sales"], name=granularity,
        marker_color="rgba(129,140,248,0.65)",
        hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["moving_average"], name="3-period average",
        line=dict(color=config.PALETTE["primary"], width=2),
    ))
    show(style_fig(fig, height=330, legend=True))
    download_frame(trend, f"revenue_{granularity.lower()}.csv", "Download this table", "dl_rev")

    section("Category and store performance")
    perf = analytics.category_performance(prepared)
    col_a, col_b = st.columns([1.25, 1], gap="medium")
    with col_a:
        if not perf.empty:
            chart_header(
                "Revenue by category",
                f"Top {min(top_n, len(perf))} of {len(perf)} categories.",
            )
            show(category_figure(perf, top_n))
        else:
            st.info("This dataset has no recognisable category column.")
    with col_b:
        if prepared.profile.has_store:
            stores = analytics.store_performance(prepared)
            chart_header("Revenue by store", f"Top {min(top_n, len(stores))} of {len(stores)} stores.")
            fig = go.Figure(go.Bar(
                x=stores.head(top_n)["store"].astype(str),
                y=stores.head(top_n)["revenue"],
                marker_color=config.PALETTE["accent"],
                hovertemplate="Store %{x}<br>%{y:,.0f}<extra></extra>",
            ))
            fig.update_yaxes(title_text=f"{config.CURRENCY_SYMBOL} revenue")
            show(style_fig(fig, height=320))
        else:
            chart_header("Revenue distribution", "How period revenue is spread.")
            fig = go.Figure(go.Histogram(
                x=prepared.series["sales"], nbinsx=32,
                marker_color=config.PALETTE["accent"], opacity=0.85,
            ))
            fig.update_xaxes(title_text=f"{config.CURRENCY_SYMBOL} per period")
            show(style_fig(fig, height=320))

    section("Seasonality and patterns")
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        chart_header("Average revenue by month", "Seasonal shape across all years present.")
        month = analytics.seasonality_profile(prepared)
        fig = go.Figure(go.Scatter(
            x=month["month_name_short"], y=month["sales"], mode="lines+markers",
            line=dict(color=config.PALETTE["primary"], width=2.4, shape="spline"),
            marker=dict(size=7),
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
        ))
        show(style_fig(fig, height=300))
    with col_b:
        weekday = analytics.weekday_profile(prepared)
        if weekday is not None:
            chart_header("Average revenue by weekday", "Daily datasets only.")
            fig = go.Figure(go.Bar(
                x=weekday["day_name"], y=weekday["sales"],
                marker_color=config.PALETTE["primary_soft"],
                hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
            ))
            show(style_fig(fig, height=300))
        else:
            chart_header("Revenue heat map", "Year × month totals.")
            pivot = analytics.monthly_heatmap(prepared)
            if not pivot.empty:
                fig = go.Figure(go.Heatmap(
                    z=pivot.values, x=pivot.columns, y=pivot.index.astype(str),
                    colorscale=[[0, "#EEF2FF"], [1, config.PALETTE["primary"]]],
                    hovertemplate="%{y} %{x}<br>%{z:,.0f}<extra></extra>",
                ))
                show(style_fig(fig, height=300))

    holiday = analytics.holiday_comparison(prepared)
    if holiday:
        section("Holiday versus regular periods")
        col_a, col_b = st.columns([1, 1.4], gap="medium")
        with col_a:
            kpi_cards(
                [
                    ("Holiday avg", analytics.money(holiday["holiday_mean"], True),
                     f"{holiday['holiday_periods']:,} periods"),
                    ("Regular avg", analytics.money(holiday["regular_mean"], True),
                     f"{holiday['regular_periods']:,} periods"),
                    ("Lift", "n/a" if holiday["lift_pct"] is None else f"{holiday['lift_pct']:+.1f}%",
                     "holiday vs regular"),
                ],
                columns=1,
            )
        with col_b:
            chart_header("Average revenue: holiday vs regular", "Computed from the row-level holiday flag.")
            fig = go.Figure(go.Bar(
                x=["Holiday", "Regular"],
                y=[holiday["holiday_mean"], holiday["regular_mean"]],
                marker_color=[config.PALETTE["warning"], config.PALETTE["neutral"]],
                text=[analytics.money(holiday["holiday_mean"], True),
                      analytics.money(holiday["regular_mean"], True)],
                textposition="outside",
                hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
            ))
            show(style_fig(fig, height=300))
    elif prepared.profile.has_holiday_flag:
        st.info("Holiday flags exist but every period has the same flag, so no comparison is possible.")
    else:
        st.info(
            "This dataset has no holiday indicator, so holiday-versus-regular analysis is "
            "not shown. Nothing is inferred in its place."
        )

    if not perf.empty:
        section("Best and worst performing categories")
        best, worst = perf.head(5), perf.tail(5).sort_values("revenue")
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            st.markdown("**Strongest**")
            st.dataframe(
                best[["category", "revenue", "share_pct"]].rename(
                    columns={"category": "Category", "revenue": "Revenue", "share_pct": "Share %"}
                ),
                use_container_width=True, hide_index=True,
            )
        with col_b:
            st.markdown("**Weakest**")
            st.dataframe(
                worst[["category", "revenue", "share_pct"]].rename(
                    columns={"category": "Category", "revenue": "Revenue", "share_pct": "Share %"}
                ),
                use_container_width=True, hide_index=True,
            )


# --------------------------------------------------------------------------- #
# Page 3 - Forecasting
# --------------------------------------------------------------------------- #
def page_forecast(prepared: PreparedData) -> None:
    app_header(
        "AI Sales Forecasting",
        "Chronologically validated forecasts with uncertainty intervals and downloadable output.",
        dataset_badges(prepared),
    )

    native = prepared.freq_label.capitalize()
    granularity_options = ["Daily", "Weekly"] if native == "Daily" else ["Weekly"]
    granularity_options.append("Monthly")
    default_granularity = "Weekly" if native == "Daily" else "Weekly"

    if prepared.freq_label == "daily":
        st.markdown(
            '<div class="insight">Daily transaction data from a sampled extract is very '
            'noisy: a single large order dominates a day. Weekly aggregation usually '
            'produces a materially more accurate forecast, so it is offered here.</div>',
            unsafe_allow_html=True,
        )

    col_a, col_b, col_c = st.columns([1.3, 1, 1], gap="medium")
    with col_a:
        horizon = st.slider(
            "Forecast horizon (days)",
            config.FORECAST_HORIZON_MIN_DAYS,
            config.FORECAST_HORIZON_MAX_DAYS,
            config.FORECAST_HORIZON_DEFAULT_DAYS,
            step=1,
            help="Weekly or monthly modelling converts this into whole periods.",
        )
    with col_b:
        model = st.selectbox("Model", forecasting.available_models(), index=0)
    with col_c:
        granularity = st.selectbox(
            "Modelling granularity",
            granularity_options,
            index=granularity_options.index(default_granularity),
            help="Aggregation applied before fitting. The native frequency is always available.",
        )

    period_days = {"Daily": 1, "Weekly": 7, "Monthly": 30}[granularity]
    if prepared.freq_label == "weekly":
        period_days = 7
    periods = max(1, int(round(horizon / period_days)))
    st.caption(
        f"A {horizon}-day horizon fits **{periods}** {granularity.lower()} period(s) "
        f"at {period_days} day(s) each."
    )

    run = st.button("Generate forecast", type="primary", use_container_width=False)
    forecast_key = (
        st.session_state["source"], horizon, model,
        granularity if prepared.freq_label == "daily" else "Weekly",
        prepared.stats.final_rows,
    )

    if run:
        with st.spinner("Training the model and validating on a hold-out window…"):
            try:
                st.session_state["forecast_result"] = cached_forecast(
                    *current_identity(), horizon, model, granularity
                )
                st.session_state["forecast_key"] = forecast_key
            except (ValueError, RuntimeError) as exc:
                st.session_state["forecast_result"] = None
                st.error(f"Forecast could not be produced: {exc}")

    result = st.session_state.get("forecast_result")
    if result is None:
        st.info(
            "Choose a horizon and press **Generate forecast**. The model is trained on the "
            "older part of the history and scored on the most recent part before the "
            "forward forecast is produced."
        )
        return

    section("Model performance (chronological hold-out)")
    metrics = result.metrics
    cols = st.columns(4, gap="small")
    for col, (label, value) in zip(
        cols,
        [
            ("MAE", analytics.money(metrics.mae, True)),
            ("RMSE", analytics.money(metrics.rmse, True)),
            ("MAPE", "n/a" if metrics.mape is None else f"{metrics.mape:,.1f}%"),
            ("WAPE", "n/a" if metrics.wape is None else f"{metrics.wape:,.1f}%"),
        ],
    ):
        col.metric(label, value)

    st.caption(
        f"Trained on {result.train_periods:,} periods, scored on the following "
        f"{result.test_periods:,}. Model actually used: **{result.model_name}** · "
        f"{result.granularity.lower()} granularity."
    )
    if metrics.n_mape_excluded:
        st.warning(
            f"MAPE excludes {metrics.n_mape_excluded:,} of {metrics.n_observations:,} "
            "hold-out periods with zero actual sales, where percentage error is undefined."
        )

    section("Historical sales and forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.history["ds"], y=result.history["y"], name="Actual",
        mode="lines", line=dict(color=config.PALETTE["primary"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=result.future["ds"], y=result.future["yhat"], name="Forecast",
        mode="lines+markers",
        line=dict(color=config.PALETTE["accent"], width=2.4, dash="dash"),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([result.future["ds"], result.future["ds"][::-1]]),
        y=pd.concat([result.future["yhat_upper"], result.future["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(6,182,212,0.16)",
        line=dict(color="rgba(0,0,0,0)"), name="Interval",
        hoverinfo="skip", showlegend=True,
    ))
    # ``add_vline`` performs arithmetic on the x value, which fails on a datetime
    # axis. A shape plus a paper-anchored annotation is the reliable way here.
    split_date = pd.Timestamp(result.history["ds"].max()).isoformat()
    fig.add_shape(
        type="line", xref="x", yref="paper", x0=split_date, x1=split_date,
        y0=0, y1=1, line=dict(width=1, dash="dot", color=config.PALETTE["neutral"]),
    )
    fig.add_annotation(
        x=split_date, y=1, yref="paper", text="last observation",
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(size=10, color=palette()["muted"]),
    )
    fig.update_yaxes(title_text=f"{config.CURRENCY_SYMBOL} per period")
    show(style_fig(fig, height=380, legend=True))

    col_a, col_b = st.columns([1.4, 1], gap="medium")
    with col_a:
        chart_header(
            "Actual vs predicted on the hold-out",
            "The shaded region is the same uncertainty interval, applied to observed dates.",
        )
        ev = result.evaluation
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ev["date"], y=ev["actual"], name="Actual", mode="lines+markers",
            line=dict(color=config.PALETTE["primary"], width=2), marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=ev["date"], y=ev["predicted"], name="Predicted", mode="lines+markers",
            line=dict(color=config.PALETTE["warning"], width=2, dash="dash"),
            marker=dict(size=5),
        ))
        show(style_fig(fig, height=300, legend=True))
    with col_b:
        chart_header("Residuals", "Predicted minus actual; bars above zero are over-forecasts.")
        fig = go.Figure(go.Bar(
            x=ev["date"], y=ev["error"],
            marker_color=[config.PALETTE["danger"] if v < 0 else config.PALETTE["success"]
                          for v in ev["error"]],
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>",
        ))
        show(style_fig(fig, height=300))

    trend = forecasting.trend_summary(result)
    if trend["change_pct"] is not None:
        st.caption(
            f"Fitted trend: **{trend['direction']}** "
            f"({trend['change_pct']:+.1f}% between the first and last quarter of the "
            "fitted history)."
        )

    if result.components is not None:
        with st.expander("Seasonality components (Prophet decomposition)"):
            comps = result.components
            available = [c for c in ("trend", "weekly", "yearly") if c in comps.columns]
            if available:
                fig = make_subplots(rows=len(available), cols=1, shared_xaxes=True,
                                    subplot_titles=[c.capitalize() for c in available])
                for index, column in enumerate(available, start=1):
                    fig.add_trace(
                        go.Scatter(x=comps["ds"], y=comps[column], name=column.capitalize(),
                                   line=dict(color=config.CHART_SEQUENCE[index - 1], width=1.8)),
                        row=index, col=1,
                    )
                fig.update_layout(height=200 * len(available), showlegend=False)
                show(style_fig(fig, height=200 * len(available)))

    section("Forecast table")
    future = result.future.copy()
    future["date"] = future["ds"].dt.strftime("%Y-%m-%d")
    future = future.rename(columns={
        "yhat": "Forecast", "yhat_lower": "Lower bound", "yhat_upper": "Upper bound",
    })[["date", "Forecast", "Lower bound", "Upper bound"]]
    st.dataframe(
        future.style.format({c: "{:,.0f}" for c in ("Forecast", "Lower bound", "Upper bound")}),
        use_container_width=True, hide_index=True,
    )

    total = result.forecast_total
    st.caption(md(
        f"Expected revenue across the horizon: **{analytics.money(total)}** "
        f"(range {analytics.money(float(result.future['yhat_lower'].sum()))} – "
        f"{analytics.money(float(result.future['yhat_upper'].sum()))})."
    ))
    st.download_button(
        "Download forecast CSV",
        data=result.to_csv(),
        file_name=f"bizpredict_forecast_{result.horizon_days}d.csv",
        mime="text/csv",
    )

    with st.expander("Model comparison on the same hold-out"):
        st.caption(
            "Every available backend is fitted on the identical chronological split. "
            "Lower MAE/RMSE is better; MAPE is only comparable when no actuals are zero."
        )
        try:
            # Score at the granularity that produced the displayed forecast, not
            # whatever the selector currently says.
            comparison = cached_comparison(*current_identity(), result.granularity)
            st.dataframe(
                comparison.style.format({
                    "MAE": "{:,.0f}", "RMSE": "{:,.0f}",
                    "MAPE (%)": "{:,.1f}", "WAPE (%)": "{:,.1f}",
                }, na_rep="—"),
                use_container_width=True, hide_index=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.info(f"Comparison unavailable: {exc}")

    section("How to read this forecast")
    for note in result.notes:
        st.markdown(f"- {md(note)}")
    st.markdown(
        "- A forecast is a statistical extrapolation of past behaviour. It cannot know "
        "about a new competitor, a price change or a supply disruption."
    )


# --------------------------------------------------------------------------- #
# Page 4 - Inventory
# --------------------------------------------------------------------------- #
def page_inventory(prepared: PreparedData) -> None:
    app_header(
        "Inventory Intelligence",
        "Safety stock, reorder points and stockout risk from historical demand - with every assumption shown.",
        dataset_badges(prepared),
    )

    st.markdown(
        '<div class="insight warning"><b>Data limitation:</b> the loaded dataset contains '
        'product sales only. It has no inventory balances and no supplier lead times, so '
        'the figures below are <b>estimates</b> built from your lead time and service '
        'level. Enter your current stock to turn them into a reorder decision.</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns([1, 1, 1.1], gap="medium")
    with col_a:
        lead_time = st.slider("Lead time (days)", config.LEAD_TIME_MIN, config.LEAD_TIME_MAX,
                              config.DEFAULT_LEAD_TIME_DAYS)
    with col_b:
        service_level = st.select_slider(
            "Service level", options=[0.80, 0.85, 0.90, 0.95, 0.975, 0.99],
            value=config.DEFAULT_SERVICE_LEVEL,
            format_func=lambda v: f"{v * 100:.1f}%",
        )
    with col_c:
        has_stock = st.checkbox("I have current stock figures", value=False)
        current_stock = 0.0
        if has_stock:
            current_stock = st.number_input(
                f"Current stock ({config.CURRENCY_SYMBOL} value of goods, or units)",
                min_value=0.0, value=float(st.session_state["inventory_stock"]),
                step=1000.0,
                help="Use consistent units. Demand here is measured in revenue, so stock "
                     "in the same revenue units keeps the comparison valid.",
            )
            st.session_state["inventory_stock"] = current_stock

    per_category_stock: Dict[str, float] = {}
    category_stats = inventory.category_demand_stats(prepared)
    if has_stock and not category_stats.empty:
        with st.expander("Optional: enter stock per category"):
            st.caption(
                "Leave a value at zero to let BizPredict allocate stock by share of demand."
            )
            editable = category_stats[["category"]].head(15).copy()
            editable["current_stock"] = 0.0
            edited = st.data_editor(
                editable, hide_index=True, use_container_width=True,
                column_config={
                    "category": st.column_config.TextColumn("Category", disabled=True),
                    "current_stock": st.column_config.NumberColumn(
                        "Current stock", min_value=0.0, step=100.0, format="%.2f"
                    ),
                },
                key="stock_editor",
            )
            per_category_stock = {
                str(row["category"]): float(row["current_stock"])
                for _, row in edited.iterrows() if float(row["current_stock"]) > 0
            }

    plan = inventory.build_plan(
        prepared,
        lead_time_days=lead_time,
        service_level=service_level,
        current_stock=current_stock if has_stock else None,
        per_category_stock=per_category_stock or None,
    )
    if has_stock and per_category_stock:
        total_stock = sum(per_category_stock.values())
        plan = inventory.build_plan(
            prepared, lead_time_days=lead_time, service_level=service_level,
            current_stock=total_stock, per_category_stock=per_category_stock,
        )

    totals = plan.lines.iloc[0]
    kpi_cards(
        [
            ("Avg daily demand", analytics.money(float(totals["avg_daily_demand"]), True),
             f"σ {analytics.money(float(totals['std_daily']), True)} per day"),
            (f"Safety stock (Z={plan.z:.2f})", analytics.money(float(totals["safety_stock"]), True),
             f"{service_level * 100:.1f}% service level, {lead_time}-day lead time"),
            ("Reorder point", analytics.money(float(totals["reorder_point"]), True),
             "lead-time demand + safety stock"),
            ("Status",
             str(totals["status"]),
             "based on the stock you entered" if has_stock else "enter stock to activate"),
        ],
        columns=4,
    )

    section("Recommendations")
    table = plan.lines.copy()
    display_cols = [
        "scope", "avg_daily_demand", "safety_stock", "reorder_point",
        "current_stock", "stockout_risk_pct", "status", "recommended_order",
    ]
    display = table[display_cols].rename(columns={
        "scope": "Category", "avg_daily_demand": "Avg daily demand",
        "safety_stock": "Safety stock", "reorder_point": "Reorder point",
        "current_stock": "Current stock", "stockout_risk_pct": "Stockout risk %",
        "status": "Status", "recommended_order": "Order qty",
    })
    st.dataframe(
        display.style.format({
            "Avg daily demand": "{:,.0f}", "Safety stock": "{:,.0f}",
            "Reorder point": "{:,.0f}", "Current stock": "{:,.0f}",
            "Stockout risk %": "{:.1f}", "Order qty": "{:,.0f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True, height=380,
    )
    st.download_button(
        "Download inventory plan CSV", data=plan.to_csv(),
        file_name=f"inventory_plan_{lead_time}d.csv", mime="text/csv",
    )

    if has_stock:
        risk = inventory.risk_table(plan)
        section("Needs attention")
        if risk.empty:
            st.success("No category is below its reorder point at the entered stock level.")
        else:
            st.markdown(
                f"**{len(risk)}** line(s) are at or below the reorder point. "
                "Order quantities bring each line back up to its reorder point."
            )
            st.dataframe(
                risk[["scope", "stockout_risk_pct", "projected_gap", "recommended_order", "status"]]
                .rename(columns={
                    "scope": "Category", "stockout_risk_pct": "Stockout risk %",
                    "projected_gap": "Gap to ROP", "recommended_order": "Order qty",
                    "status": "Status",
                }),
                use_container_width=True, hide_index=True,
            )

    section("Demand intelligence")
    col_a, col_b = st.columns(2, gap="medium")
    demand = inventory.highlight_extremes(prepared)
    if not demand["high"].empty:
        with col_a:
            chart_header("Highest average daily demand", "Top categories by mean daily sales.")
            data = demand["high"].iloc[::-1]
            fig = go.Figure(go.Bar(
                x=data["mean_daily"], y=data["category"].astype(str), orientation="h",
                marker_color=config.PALETTE["success"], name="Mean daily demand",
                hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>",
            ))
            show(style_fig(fig, height=300))
        with col_b:
            chart_header("Lowest average daily demand", "Weakest categories by mean daily sales.")
            data = demand["low"]
            fig = go.Figure(go.Bar(
                x=data["mean_daily"], y=data["category"].astype(str), orientation="h",
                marker_color=config.PALETTE["danger"], name="Mean daily demand",
                hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>",
            ))
            show(style_fig(fig, height=300))

    section("Reorder point sensitivity")
    st.caption(
        "Reorder point for the whole business across lead times and service levels. "
        "Use it to see how much a higher service level actually costs in stock."
    )
    sensitivity = inventory.sensitivity_table(prepared)
    fig = go.Figure(go.Heatmap(
        z=sensitivity.drop(columns=["Lead time (days)"]).values,
        x=[c for c in sensitivity.columns if c != "Lead time (days)"],
        y=sensitivity["Lead time (days)"].astype(str),
        colorscale=[[0, "#EEF2FF"], [1, config.PALETTE["primary"]]],
        hovertemplate="Lead time %{y}d · %{x}<br>ROP %{z:,.0f}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Service level")
    fig.update_yaxes(title_text="Lead time (days)")
    show(style_fig(fig, height=280))

    section("Formulas and assumptions")
    st.latex(r"\text{Safety Stock} = Z \times \sigma_d \times \sqrt{L}")
    st.latex(r"\text{Reorder Point} = (\bar{d} \times L) + \text{Safety Stock}")
    st.caption(
        r"$Z$ = service-level factor, $\sigma_d$ = standard deviation of daily demand, "
        r"$L$ = lead time in days, $\bar{d}$ = average daily demand."
    )
    for note in plan.notes:
        st.markdown(f"- {md(note)}")


# --------------------------------------------------------------------------- #
# Page 5 - Data explorer
# --------------------------------------------------------------------------- #
def page_explorer(prepared: PreparedData, loaded_summary: Dict[str, object]) -> None:
    app_header(
        "Data Explorer",
        "Inspect the raw schema, the cleaning decisions and the resulting model-ready dataset.",
        dataset_badges(prepared),
    )

    with st.expander("Load a different dataset", expanded=False):
        st.caption(
            "Any CSV with a recognisable date column and a sales column works. "
            "Column names are matched case- and punctuation-insensitively."
        )
        upload = st.file_uploader("Upload a CSV", type=["csv"], key="explorer_upload")
        if upload is not None and upload.name != st.session_state.get("uploaded_name"):
            st.session_state["uploaded_bytes"] = upload.getvalue()
            st.session_state["uploaded_name"] = upload.name
            st.session_state["source"] = SOURCE_UPLOAD
            st.session_state["forecast_result"] = None
            st.rerun()

    section("Dataset summary")
    left, right = st.columns([1, 1.2], gap="medium")
    with left:
        summary = {
            **loaded_summary,
            "Frequency": prepared.freq_label,
            "Periods modelled": f"{prepared.stats.periods_created:,}",
            "Date range": f"{prepared.stats.date_min} → {prepared.stats.date_max}",
        }
        st.dataframe(
            pd.DataFrame({"Property": list(summary), "Value": [str(v) for v in summary.values()]}),
            use_container_width=True, hide_index=True,
        )
    with right:
        mapping = prepared.profile.schema.describe()
        st.markdown("**Column mapping** (canonical field → detected header)")
        st.dataframe(
            pd.DataFrame(
                {
                    "Canonical field": list(mapping),
                    "Detected column": list(mapping.values()),
                }
            ),
            use_container_width=True, hide_index=True,
        )
        if prepared.profile.schema.unmapped:
            st.caption(
                "Unmapped columns (kept for reference, not used in modelling): "
                + ", ".join(map(str, prepared.profile.schema.unmapped[:15]))
            )

    if prepared.profile.warnings:
        with st.expander("Schema warnings", expanded=True):
            for warning in prepared.profile.warnings:
                st.markdown(f"- {warning}")

    section("Preprocessing summary")
    stats = prepared.stats.as_dict()
    st.dataframe(
        pd.DataFrame({"Step": list(stats), "Result": [str(v) for v in stats.values()]}),
        use_container_width=True, hide_index=True,
    )
    for note in prepared.notes:
        st.markdown(f"- {md(note)}")

    section("Raw preview")
    col_a, col_b = st.columns([1, 1], gap="medium")
    with col_a:
        rows = st.slider("Rows to preview", 5, 100, 10, key="preview_rows")
        st.markdown("**Cleaned records**")
        preview_cols = [c for c in prepared.clean.columns if not c.startswith("ctx_")][:10]
        st.dataframe(prepared.clean[preview_cols].head(rows),
                     use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Aggregated time series**")
        st.dataframe(prepared.series.head(rows), use_container_width=True, hide_index=True)

    section("Data quality")
    col_a, col_b = st.columns([1.3, 1], gap="medium")
    with col_a:
        chart_header("Missing values per column", "Share of rows without a value.")
        missing = prepared.clean.isna().mean().mul(100).sort_values(ascending=False)
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values remain in the cleaned dataset.")
        else:
            fig = go.Figure(go.Bar(
                x=missing.values, y=missing.index.astype(str), orientation="h",
                marker_color=config.PALETTE["warning"],
                hovertemplate="%{y}<br>%{x:.2f}% missing<extra></extra>",
            ))
            fig.update_xaxes(title_text="% missing")
            show(style_fig(fig, height=max(220, 26 * len(missing) + 60)))
    with col_b:
        chart_header("Column types", "Inferred from the raw file.")
        dtypes = pd.DataFrame(
            {"Column": list(prepared.profile.dtypes), "Detected type": list(prepared.profile.dtypes.values())}
        )
        st.dataframe(dtypes, use_container_width=True, hide_index=True, height=300)

    section("Descriptive statistics")
    numeric = prepared.clean.select_dtypes(include=[np.number])
    if not numeric.empty:
        st.dataframe(
            numeric.describe().T.style.format("{:,.2f}"),
            use_container_width=True,
        )

    if prepared.profile.has_category:
        section("Category totals")
        perf = analytics.category_performance(prepared)
        st.dataframe(
            perf.style.format({
                "revenue": "{:,.2f}", "share_pct": "{:.2f}",
                "avg_ticket": "{:,.2f}", "units": "{:,.0f}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True,
        )

    section("Downloads")
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        download_frame(prepared.clean, "bizpredict_cleaned_records.csv",
                       "Download cleaned records (CSV)", "dl_clean")
    with col_b:
        download_frame(prepared.series, "bizpredict_series.csv",
                       "Download aggregated series (CSV)", "dl_series")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    with st.sidebar:
        if (config.ASSETS_DIR / "logo.png").exists():
            st.image(str(config.ASSETS_DIR / "logo.png"), width=180)
        else:
            st.markdown(f"### {config.APP_TITLE}")

        st.caption(config.APP_TAGLINE)
        st.divider()

        st.markdown("**Dataset**")
        source = st.radio(
            "Dataset",
            list(SOURCE_KEYS),
            index=list(SOURCE_KEYS).index(st.session_state["source"]),
            label_visibility="collapsed",
        )
        if source != st.session_state["source"]:
            st.session_state["source"] = source
            st.session_state["forecast_result"] = None

        if source == SOURCE_UPLOAD:
            upload = st.file_uploader("Upload a CSV", type=["csv"], key="sidebar_upload")
            if upload is not None and upload.getvalue() != st.session_state["uploaded_bytes"]:
                st.session_state["uploaded_bytes"] = upload.getvalue()
                st.session_state["uploaded_name"] = upload.name
                st.session_state["forecast_result"] = None
                st.rerun()
            elif st.session_state["uploaded_name"]:
                st.caption(f"Loaded: {st.session_state['uploaded_name']}")

        st.divider()
        st.markdown("**Navigation**")
        page = st.radio(
            "Page",
            list(config.PAGES),
            index=list(config.PAGES).index(st.session_state["page"]),
            label_visibility="collapsed",
        )
        st.session_state["page"] = page

        st.divider()
        st.markdown("**Appearance**")
        theme = st.radio(
            "Theme", ["Light", "Dark"],
            index=["Light", "Dark"].index(st.session_state["theme"]),
            horizontal=True, label_visibility="collapsed",
        )
        st.session_state["theme"] = theme

        st.divider()
        st.caption(f"v{config.APP_VERSION} · Streamlit {st.__version__}")
        st.caption(
            "Forecasting: "
            + ("Prophet" if forecasting.prophet_available() else "Holt-Winters (Prophet not installed)")
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    # The sidebar is rendered first so that a theme change applied by its radio
    # button is already in session state when the stylesheet is built. Calling
    # inject_css() first would delay every theme switch by one interaction.
    render_sidebar()
    inject_css()

    identity = current_identity()
    source_key, payload, filename = identity

    if source_key == "upload" and payload is None:
        app_header(
            "Load a dataset",
            "Upload a CSV to begin, or switch to one of the bundled datasets in the sidebar.",
            [],
        )
        st.info(
            "The CSV needs a date column and a sales column. Recognised names include "
            "`date`, `ORDERDATE`, `weekly_sales`, `SALES`, and many variants - matching "
            "ignores case, spaces and punctuation."
        )
        return

    try:
        with st.spinner("Loading and validating the dataset…"):
            prepared, loaded_summary, encoding = load_prepared(*identity)
    except (DataSourceError, SchemaError) as exc:
        app_header("Dataset problem", "The file could not be used as-is.", [])
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - surface anything unexpected clearly
        app_header("Dataset problem", "The file could not be processed.", [])
        st.error(f"Unexpected error while preparing the dataset: {type(exc).__name__}: {exc}")
        return

    page = st.session_state["page"]
    if page == config.PAGES[0]:
        page_overview(prepared)
    elif page == config.PAGES[1]:
        page_sales(prepared)
    elif page == config.PAGES[2]:
        page_forecast(prepared)
    elif page == config.PAGES[3]:
        page_inventory(prepared)
    else:
        page_explorer(prepared, loaded_summary)

    st.markdown(
        f'<div class="footer">{esc(config.APP_TITLE)} v{config.APP_VERSION} · '
        "Models are trained locally on the loaded dataset. Figures are analytical "
        "estimates, not financial advice.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()