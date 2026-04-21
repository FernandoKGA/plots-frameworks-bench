"""
benchmark_plots.py
==================
Generates interactive Plotly charts from the CSVs produced by the
aggregation pipeline (rounds.csv / summary_by_version.csv).

Available charts
----------------
1. box_requests(df, framework, endpoint, metric)
   Box plot by version — response times for one endpoint (rounds.csv)

2. box_energy(df, framework, metric)
   Box plot by version — energy metrics (rounds.csv)

3. line_energy(df, framework, metric)
   Line chart — mean energy consumption per version (rounds.csv)

Quick usage
-----------
    from benchmark_plots import load_rounds, box_requests, box_energy, line_energy

    df = load_rounds("rounds.csv")

    fig = box_requests(df, framework="fastapi", endpoint="api", metric="rt_p95")
    fig.show()

Usage with Dash
---------------
    import dash
    from dash import dcc, html, Input, Output
    from benchmark_plots import load_rounds, box_requests, box_energy, line_energy

    df = load_rounds("rounds.csv")
    app = dash.Dash(__name__)
    # build your callbacks using the functions above returning fig
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Constants ─────────────────────────────────────────────────────────────────

ENDPOINTS = ["api", "html", "upload"]

REQUEST_METRICS: dict[str, str] = {
    "rt_mean":        "Mean response time (s)",
    "rt_p50":         "Median / p50 (s)",
    "rt_p75":         "p75 (s)",
    "rt_p90":         "p90 (s)",
    "rt_p95":         "p95 (s)",
    "rt_p99":         "p99 (s)",
    "rt_min":         "Minimum (s)",
    "rt_max":         "Maximum (s)",
    "throughput_rps": "Throughput (req/s)",
}

ENERGY_METRICS: dict[str, str] = {
    "emission_energy_consumed": "Total energy (kWh)",
    "emission_cpu_energy":      "CPU energy (kWh)",
    "emission_ram_energy":      "RAM energy (kWh)",
    "emission_emissions":       "CO2 emissions (kg)",
    "emission_duration":        "Duration (s)",
    "emission_cpu_power":       "CPU power (W)",
    "emission_ram_power":       "RAM power (W)",
    "emission_emissions_rate":  "Emission rate (kg CO2/s)",
}

# Discrete palette — works well for ordinal versions
_VERSION_PALETTE = px.colors.qualitative.Plotly

# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_version(v: str):
    """Parses a version string using packaging.version.Version, with fallback."""
    from packaging.version import Version, InvalidVersion
    try:
        return Version(v)
    except InvalidVersion:
        return Version("0")


def load_rounds(path: str | Path = "rounds.csv") -> pd.DataFrame:
    """
    Reads rounds.csv and adds the `version_label` column (sortable string).
    Expects columns: framework, framework_version, exc_n, and the metrics.
    """
    df = pd.read_csv(path)
    df["version_label"] = df["framework_version"].astype(str)

    # Sort by framework and then by semantic version
    df["_ver_key"] = df["version_label"].map(_parse_version)
    df = df.sort_values(["framework", "_ver_key", "exc_n"]).drop(columns="_ver_key")
    df = df.reset_index(drop=True)
    return df


def _framework_versions(df: pd.DataFrame, framework: str) -> list[str]:
    """Returns the list of versions for a framework sorted by semver."""
    mask = df["framework"].str.lower() == framework.lower()
    versions = df.loc[mask, "version_label"].unique().tolist()
    return sorted(versions, key=_parse_version)


def _col(endpoint: str, metric: str) -> str:
    """Builds the column name: 'api_rt_p95', 'html_throughput_rps', etc."""
    return f"{endpoint}_{metric}"


# ── Chart 1 — Box plot: response time per version ────────────────────────────

def box_requests(
    df: pd.DataFrame,
    framework: str,
    endpoint: str = "api",
    metric: str = "rt_p95",
    title: str | None = None,
) -> go.Figure:
    """
    Box plot of a request metric (per version, over rounds).

    Parameters
    ----------
    df        : DataFrame returned by load_rounds()
    framework : framework name (case-insensitive)
    endpoint  : 'api' | 'html' | 'upload'
    metric    : key in REQUEST_METRICS (e.g. 'rt_p95', 'throughput_rps')
    title     : custom title (optional)
    """
    col = _col(endpoint, metric)
    mask = df["framework"].str.lower() == framework.lower()
    data = df[mask].copy()

    if col not in data.columns:
        raise ValueError(
            f"Column '{col}' not found. "
            f"Available columns: {[c for c in data.columns if endpoint in c]}"
        )

    versions = _framework_versions(df, framework)
    y_label  = REQUEST_METRICS.get(metric, metric)
    endpoint_label = endpoint.upper()

    fig = go.Figure()

    for i, ver in enumerate(versions):
        subset = data[data["version_label"] == ver][col].dropna()
        color  = _VERSION_PALETTE[i % len(_VERSION_PALETTE)]

        fig.add_trace(go.Box(
            y=subset,
            name=ver,
            boxpoints="all",        # shows individual points (rounds)
            jitter=0.3,
            pointpos=-1.6,
            marker=dict(size=5, opacity=0.6, color=color),
            line=dict(color=color),
            fillcolor=color,
            opacity=0.7,
            hovertemplate=(
                f"<b>{framework} %{{x}}</b><br>"
                f"Endpoint: {endpoint_label}<br>"
                f"{y_label}: %{{y:.6f}}<br>"
                "<extra></extra>"
            ),
        ))

    _title = title or (
        f"{framework.title()} — {endpoint_label}: {y_label} by version"
    )

    fig.update_layout(
        title=dict(text=_title, font=dict(size=16)),
        xaxis_title="Version",
        yaxis_title=y_label,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=13),
        margin=dict(t=60, b=60, l=70, r=30),
        hovermode="closest",
        xaxis=dict(
            categoryorder="array",
            categoryarray=versions,
            showgrid=False,
            linecolor="#ccc",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#eee",
            zeroline=False,
        ),
    )

    return fig


# ── Chart 2 — Box plot: energy per version ────────────────────────────────────

def box_energy(
    df: pd.DataFrame,
    framework: str,
    metric: str = "emission_energy_consumed",
    title: str | None = None,
) -> go.Figure:
    """
    Box plot of an energy metric (per version, over rounds).

    Parameters
    ----------
    df        : DataFrame returned by load_rounds()
    framework : framework name (case-insensitive)
    metric    : key in ENERGY_METRICS (e.g. 'energy_consumed', 'emissions')
    title     : custom title (optional)
    """
    mask = df["framework"].str.lower() == framework.lower()
    data = df[mask].copy()

    if metric not in data.columns:
        energy_cols = [c for c in data.columns if c in ENERGY_METRICS]
        raise ValueError(
            f"Column '{metric}' not found in DataFrame. "
            f"Available energy columns: {energy_cols}"
        )

    versions = _framework_versions(df, framework)
    y_label  = ENERGY_METRICS.get(metric, metric)

    fig = go.Figure()

    for i, ver in enumerate(versions):
        subset = data[data["version_label"] == ver][metric].dropna()
        color  = _VERSION_PALETTE[i % len(_VERSION_PALETTE)]

        fig.add_trace(go.Box(
            y=subset,
            name=ver,
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.6,
            marker=dict(size=5, opacity=0.6, color=color),
            line=dict(color=color),
            fillcolor=color,
            opacity=0.7,
            hovertemplate=(
                f"<b>{framework} %{{x}}</b><br>"
                f"{y_label}: %{{y:.8f}}<br>"
                "<extra></extra>"
            ),
        ))

    _title = title or f"{framework.title()} — {y_label} by version"

    fig.update_layout(
        title=dict(text=_title, font=dict(size=16)),
        xaxis_title="Version",
        yaxis_title=y_label,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=13),
        margin=dict(t=60, b=60, l=80, r=30),
        hovermode="closest",
        xaxis=dict(
            categoryorder="array",
            categoryarray=versions,
            showgrid=False,
            linecolor="#ccc",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#eee",
            zeroline=False,
        ),
    )

    return fig


# ── Chart 3 — Line: energy consumption per version ───────────────────────────

def line_energy(
    df: pd.DataFrame,
    framework: str,
    metric: str = "emission_energy_consumed",
    show_ci: bool = True,
    title: str | None = None,
) -> go.Figure:
    """
    Line chart — mean energy per version, with confidence interval
    (mean +/- std) as a shaded area when show_ci=True.

    Parameters
    ----------
    df        : DataFrame returned by load_rounds()
    framework : framework name (case-insensitive)
    metric    : key in ENERGY_METRICS
    show_ci   : display mean +/- std band
    title     : custom title (optional)
    """
    mask = df["framework"].str.lower() == framework.lower()
    data = df[mask].copy()

    versions = _framework_versions(df, framework)
    y_label  = ENERGY_METRICS.get(metric, metric)

    # Aggregate by version
    agg = (
        data.groupby("version_label")[metric]
        .agg(mean="mean", std="std", count="count")
        .reindex(versions)
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)

    color_main = "#2563EB"          # blue
    color_ci   = "rgba(37,99,235,0.15)"

    fig = go.Figure()

    # Uncertainty band (mean +/- std)
    if show_ci:
        fig.add_trace(go.Scatter(
            x=list(agg["version_label"]) + list(reversed(agg["version_label"])),
            y=list(agg["mean"] + agg["std"]) + list(reversed(agg["mean"] - agg["std"])),
            fill="toself",
            fillcolor=color_ci,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Mean +/- std deviation",
        ))

    # Main line
    fig.add_trace(go.Scatter(
        x=agg["version_label"],
        y=agg["mean"],
        mode="lines+markers",
        name=f"Mean — {y_label}",
        line=dict(color=color_main, width=2.5),
        marker=dict(size=7, color=color_main, symbol="circle"),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{y_label}: %{{y:.8f}}<br>"
            "<extra></extra>"
        ),
    ))

    _title = title or f"{framework.title()} — {y_label} by version (mean +/- sd)"

    fig.update_layout(
        title=dict(text=_title, font=dict(size=16)),
        xaxis_title="Version",
        yaxis_title=y_label,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=13),
        margin=dict(t=60, b=60, l=80, r=30),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            categoryorder="array",
            categoryarray=versions,
            tickangle=-30,
            showgrid=False,
            linecolor="#ccc",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#eee",
            zeroline=False,
        ),
    )

    return fig


# ── Full framework dashboard ──────────────────────────────────────────────────

def dashboard_framework(
    df: pd.DataFrame,
    framework: str,
    energy_metric: str = "emission_energy_consumed",
) -> go.Figure:
    """
    2x3 dashboard with all request box plots (api/html/upload at p95)
    + energy box plot + energy line chart in a single Figure with subplots.

    Useful for exporting as a standalone HTML or for quick preview.
    """
    endpoints = [e for e in ENDPOINTS if any(
        f"{e}_rt_p95" in df.columns for _ in [None]
    )]

    rows, cols = 2, 3
    subplot_titles = (
        [f"{e.upper()}: response p95" for e in ENDPOINTS]
        + [
            f"Energy ({ENERGY_METRICS[energy_metric]}) — box",
            f"Energy ({ENERGY_METRICS[energy_metric]}) — line",
            "",   # empty cell
        ]
    )

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]

    # Request box plots per endpoint
    for idx, ep in enumerate(ENDPOINTS):
        r, c = positions[idx]
        sub = box_requests(df, framework, endpoint=ep, metric="rt_p95")
        for trace in sub.data:
            trace.showlegend = False
            fig.add_trace(trace, row=r, col=c)

    # Energy box plot
    r, c = positions[3]
    sub_e = box_energy(df, framework, metric=energy_metric)
    for trace in sub_e.data:
        trace.showlegend = False
        fig.add_trace(trace, row=r, col=c)

    # Energy line chart
    r, c = positions[4]
    sub_l = line_energy(df, framework, metric=energy_metric, show_ci=True)
    for trace in sub_l.data:
        trace.showlegend = False
        fig.add_trace(trace, row=r, col=c)

    versions = _framework_versions(df, framework)
    for axis in ("xaxis", "xaxis2", "xaxis3", "xaxis4", "xaxis5"):
        fig.update_layout(**{
            axis: dict(
                categoryorder="array",
                categoryarray=versions,
                tickangle=-30,
            )
        })

    fig.update_layout(
        title=dict(
            text=f"<b>{framework.title()}</b> — Full Dashboard",
            font=dict(size=18),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11),
        height=680,
        showlegend=False,
        margin=dict(t=80, b=40, l=60, r=30),
    )
    fig.update_annotations(font_size=12)

    return fig


# ── Structured export (Option A — by framework) ──────────────────────────────

def export_all(
    df: pd.DataFrame,
    output_dir: str | Path = "plots",
    energy_metric: str = "emission_energy_consumed",
    include_plotlyjs: str = "cdn",
) -> dict[str, list[Path]]:
    """
    Exports all charts organised by framework (Option A):

        plots/
        +-- {framework}/
            +-- requests/
            |   +-- {endpoint}_{metric}.html   (request box plots)
            |   +-- ...
            +-- energy/
            |   +-- box_{metric}.html          (energy box plots)
            |   +-- line_{metric}.html         (energy line charts)
            +-- dashboard.html                 (full 2x3 dashboard)

    Parameters
    ----------
    df               : DataFrame returned by load_rounds()
    output_dir       : root output directory (created if it does not exist)
    energy_metric    : energy metric used in the dashboard (default: energy_consumed)
    include_plotlyjs : "cdn" (smaller) | "inline" (no internet) | True (local copy)

    Returns
    -------
    dict framework -> list of generated Paths
    """
    root = Path(output_dir)
    frameworks = sorted(df["framework"].unique())
    generated: dict[str, list[Path]] = {}

    req_metrics    = list(REQUEST_METRICS.keys())
    energy_metrics = list(ENERGY_METRICS.keys())

    for fw in frameworks:
        fw_dir  = root / fw
        req_dir = fw_dir / "requests"
        nrg_dir = fw_dir / "energy"
        req_dir.mkdir(parents=True, exist_ok=True)
        nrg_dir.mkdir(parents=True, exist_ok=True)

        files: list[Path] = []
        print(f"\n[{fw}]")

        # ── Request box plots ────────────────────────────────────────────────
        for ep in ENDPOINTS:
            ep_col_base = f"{ep}_rt_p50"
            if ep_col_base not in df.columns:
                print(f"  skip {ep} (no data)")
                continue

            for metric in req_metrics:
                col = _col(ep, metric)
                if col not in df.columns or df[col].isna().all():
                    continue
                try:
                    fig  = box_requests(df, framework=fw, endpoint=ep, metric=metric)
                    path = req_dir / f"{ep}_{metric}.html"
                    fig.write_html(path, include_plotlyjs=include_plotlyjs)
                    files.append(path)
                    print(f"  requests/{ep}_{metric}.html")
                except Exception as e:
                    print(f"  [WARN] {ep}_{metric}: {e}")

        # ── Energy box plots ─────────────────────────────────────────────────
        #for metric in energy_metrics:
        #    if metric not in df.columns or df[metric].isna().all():
        #        continue
        #    try:
        #        fig  = box_energy(df, framework=fw, metric=metric)
        #        path = nrg_dir / f"box_{metric}.html"
        #        fig.write_html(path, include_plotlyjs=include_plotlyjs)
        #        files.append(path)
        #        print(f"  energy/box_{metric}.html")
        #    except Exception as e:
        #        print(f"  [WARN] box_{metric}: {e}")

        # ── Energy line charts ───────────────────────────────────────────────
        for metric in energy_metrics:
            if metric not in df.columns or df[metric].isna().all():
                continue
            if metric in ["emission_ram_power", "emission_cpu_power"]:
                continue
            try:
                fig  = line_energy(df, framework=fw, metric=metric, show_ci=True)
                path = nrg_dir / f"line_{metric}.html"
                fig.write_html(path, include_plotlyjs=include_plotlyjs)
                files.append(path)
                print(f"  energy/line_{metric}.html")
            except Exception as e:
                print(f"  [WARN] line_{metric}: {e}")

        # ── Full dashboard ───────────────────────────────────────────────────
        try:
            fig  = dashboard_framework(df, framework=fw, energy_metric=energy_metric)
            path = fw_dir / "dashboard.html"
            fig.write_html(path, include_plotlyjs=include_plotlyjs)
            files.append(path)
            print(f"  dashboard.html")
        except Exception as e:
            print(f"  [WARN] dashboard: {e}")

        generated[fw] = files

    total = sum(len(v) for v in generated.values())
    print(f"\n[OK] {total} file(s) generated in: {root.resolve()}/")
    return generated


# ── CLI / usage example ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    rounds_path = sys.argv[1] if len(sys.argv) > 1 else "rounds.csv"
    out_dir     = sys.argv[2] if len(sys.argv) > 2 else "plots"

    df = load_rounds(rounds_path)
    frameworks = df["framework"].unique().tolist()
    print(f"Frameworks found: {frameworks}")

    export_all(df, output_dir=out_dir)