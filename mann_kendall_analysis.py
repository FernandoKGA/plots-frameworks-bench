"""
mann_kendall_analysis.py
========================
Mann-Kendall trend analysis for Python async framework benchmark results.

Quick usage
-----------
    from mann_kendall_analysis import run_mk_analysis, mk_summary_table, plot_mk_heatmap

    summary = pd.read_csv("summary_by_version.csv")
    results = run_mk_analysis(summary)

    print(mk_summary_table(results))
    plot_mk_heatmap(results, metric_group="energy").show()

Test variant
------------
The default variant is ``hamed_rao`` (Hamed & Rao, 1998), which corrects the
variance of the S statistic for serial autocorrelation. This is important here
because consecutive versions of the same framework tend to have similar
performance characteristics (positive autocorrelation). Using ``original_test``
artificially inflates significance (p-values lower than they should be).

For n < 10 versions, the function automatically falls back to ``original_test``
because hamed_rao is numerically unstable on very short series.
"""

from __future__ import annotations

import warnings
from typing import Literal

import pandas as pd
import pymannkendall as mk
import plotly.graph_objects as go
import plotly.express as px
from packaging.version import Version, InvalidVersion

# ── Metric mapping → RQ group and human-readable label ───────────────────────

#: Default metrics analysed, grouped by research question.
#: Key   : column name in summary_by_version.csv (suffix _mean)
#: Value : (human-readable label, group)
DEFAULT_METRICS: dict[str, tuple[str, str]] = {
    # Energy
    "emission_energy_consumed_mean": ("Total energy (kWh)",        "energy"),
    "emission_cpu_energy_mean":      ("CPU energy (kWh)",          "energy"),
    "emission_ram_energy_mean":      ("RAM energy (kWh)",          "energy"),
    # Carbon
    "emission_emissions_mean":       ("CO2 emitted (kg)",          "carbon"),
    "emission_emissions_rate_mean":  ("CO2 rate (kg CO2/s)",       "carbon"),
    # Throughput
    "api_throughput_rps_mean":       ("API throughput (req/s)",    "throughput"),
    "html_throughput_rps_mean":      ("HTML throughput (req/s)",   "throughput"),
    "upload_throughput_rps_mean":    ("Upload throughput (req/s)", "throughput"),
    # Latency
    "api_rt_mean_mean":               ("API latency Mean(s)",      "latency"),
    "html_rt_mean_mean":              ("HTML latency Mean(s)",     "latency"),
    "upload_rt_mean_mean":            ("Upload latency Mean(s)",   "latency"),
    "api_rt_p95_mean":               ("API latency p95 (s)",      "latency"),
    "html_rt_p95_mean":              ("HTML latency p95 (s)",     "latency"),
    "upload_rt_p95_mean":            ("Upload latency p95 (s)",   "latency"),
}

# Minimum number of versions required for the test to be meaningful
_MIN_VERSIONS_HAMED = 10   # below this, fall back to original_test
_MIN_VERSIONS_RUN   = 4    # below this, skip the framework / metric entirely


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_version(v: str) -> Version:
    """Parse a semantic version string, with a safe fallback for invalid values."""
    try:
        return Version(str(v))
    except InvalidVersion:
        return Version("0")


def _sort_by_version(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a DataFrame by semantic version using packaging.version.Version."""
    df = df.copy()
    df["_ver_key"] = df["framework_version"].map(_parse_version)
    df = df.sort_values("_ver_key").drop(columns="_ver_key").reset_index(drop=True)
    return df


def _run_single(
    series: list | pd.Series,
    variant: Literal["hamed_rao", "original", "yue_wang"] = "hamed_rao",
    alpha: float = 0.05,
) -> dict:
    """Run the MK test on a single series and return a result dictionary."""
    s = pd.Series(series).dropna()
    n = len(s)

    if n < _MIN_VERSIONS_RUN:
        return {"_skip": True, "note": f"n={n} < {_MIN_VERSIONS_RUN} (insufficient data)"}

    # Choose effective variant
    effective_variant = variant
    if variant == "hamed_rao" and n < _MIN_VERSIONS_HAMED:
        effective_variant = "original"
        note = f"hamed_rao->original (n={n} < {_MIN_VERSIONS_HAMED})"
    else:
        note = ""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if effective_variant == "hamed_rao":
                result = mk.hamed_rao_modification_test(s.values, alpha=alpha)
            elif effective_variant == "yue_wang":
                result = mk.yue_wang_modification_test(s.values, alpha=alpha)
            else:
                result = mk.original_test(s.values, alpha=alpha)
    except Exception as e:
        return {"_skip": True, "note": f"Test error: {e}"}

    return {
        "_skip":        False,
        "trend":        result.trend,       # 'increasing' | 'decreasing' | 'no trend'
        "h":            result.h,           # True if trend is statistically significant
        "p_value":      result.p,
        "z":            result.z,
        "tau":          result.Tau,         # Kendall's Tau: -1..+1
        "slope":        result.slope,       # Theil-Sen slope per version step
        "intercept":    result.intercept,
        "n_versions":   n,
        "variant_used": effective_variant,
        "note":         note,
    }


# ── Main analysis function ────────────────────────────────────────────────────

def run_mk_analysis(
    df_summary: pd.DataFrame,
    frameworks: list[str] | None = None,
    metrics: dict[str, tuple[str, str]] | list[str] | None = None,
    alpha: float = 0.05,
    variant: Literal["hamed_rao", "original", "yue_wang"] = "hamed_rao",
) -> pd.DataFrame:
    """
    Run the Mann-Kendall test for every (framework, metric) combination.

    Parameters
    ----------
    df_summary : DataFrame with one row per (framework, framework_version).
                 Must contain columns ``framework``, ``framework_version``,
                 and the metric columns (``_mean`` suffix).
    frameworks : List of frameworks to analyse. ``None`` = all.
    metrics    : Dict ``{column: (label, group)}`` or a plain list of column names.
                 ``None`` = DEFAULT_METRICS.
    alpha      : Significance level (default 0.05).
    variant    : MK test variant.
                 ``'hamed_rao'`` (default) -- corrects for serial autocorrelation.
                 ``'original'``            -- no autocorrelation correction.
                 ``'yue_wang'``            -- alternative autocorrelation correction.

    Returns
    -------
    DataFrame with columns:
        framework, metric, metric_label, group, n_versions,
        trend, h, p_value, z, tau, slope, intercept, variant_used, note
    """
    # Normalise the metrics argument
    if metrics is None:
        metric_map = DEFAULT_METRICS
    elif isinstance(metrics, list):
        metric_map = {m: (m, "custom") for m in metrics}
    else:
        metric_map = metrics

    # Frameworks to analyse
    all_frameworks = df_summary["framework"].unique().tolist()
    if frameworks is None:
        frameworks = all_frameworks

    rows = []

    for fw in frameworks:
        fw_df = df_summary[df_summary["framework"] == fw]
        fw_df = _sort_by_version(fw_df)
        n_total = len(fw_df)

        if n_total < _MIN_VERSIONS_RUN:
            print(f"[SKIP] {fw}: only {n_total} version(s) — insufficient for MK test.")
            continue

        for col, (label, group) in metric_map.items():
            if col not in fw_df.columns:
                continue

            series = fw_df[col].values
            res = _run_single(series, variant=variant, alpha=alpha)

            row = {
                "framework":    fw,
                "metric":       col,
                "metric_label": label,
                "group":        group,
            }

            if res.get("_skip"):
                row.update({
                    "n_versions":   n_total,
                    "trend":        "skip",
                    "h":            None,
                    "p_value":      None,
                    "z":            None,
                    "tau":          None,
                    "slope":        None,
                    "intercept":    None,
                    "variant_used": variant,
                    "note":         res.get("note", ""),
                })
            else:
                row.update(res)

            rows.append(row)

    results = pd.DataFrame(rows)
    return results


# ── Summary tables ────────────────────────────────────────────────────────────

def mk_summary_table(
    results: pd.DataFrame,
    group: str | None = None,
    significant_only: bool = False,
    sort_by: str = "framework",
) -> pd.DataFrame:
    """
    Return a formatted summary table of Mann-Kendall results.

    Parameters
    ----------
    results          : DataFrame returned by ``run_mk_analysis``.
    group            : Filter by metric group (``'energy'``, ``'carbon'``,
                       ``'throughput'``, ``'latency'``). ``None`` = all.
    significant_only : If True, keep only rows where h=True.
    sort_by          : Primary sort column.
    """
    df = results.copy()

    if group is not None:
        df = df[df["group"] == group]

    if significant_only:
        df = df[df["h"] == True]

    df = df[df["trend"] != "skip"]

    display_cols = [
        "framework", "metric_label", "group",
        "n_versions", "trend", "h",
        "p_value", "tau", "slope", "note",
    ]
    df = df[[c for c in display_cols if c in df.columns]]

    # Format numeric columns
    if "p_value" in df.columns:
        df["p_value"] = df["p_value"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
    if "tau" in df.columns:
        df["tau"] = df["tau"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "-")
    if "slope" in df.columns:
        df["slope"] = df["slope"].map(lambda x: f"{x:.3e}" if pd.notna(x) else "-")

    df = df.sort_values(sort_by).reset_index(drop=True)
    return df


def mk_pivot_table(
    results: pd.DataFrame,
    value: Literal["trend", "p_value", "tau", "h"] = "trend",
    group: str | None = None,
) -> pd.DataFrame:
    """
    Pivot table: frameworks (rows) x metrics (columns).

    Suitable for direct inclusion in a scientific paper as a summary table.

    Parameters
    ----------
    value : Value to display in each cell.
            ``'trend'``   -- 'increasing' / 'decreasing' / 'no trend'
            ``'p_value'`` -- numeric p-value
            ``'tau'``     -- Kendall's Tau
            ``'h'``       -- True / False (significant trend)
    group : Filter by metric group.
    """
    df = results.copy()
    df = df[df["trend"] != "skip"]

    if group is not None:
        df = df[df["group"] == group]

    pivot = df.pivot(index="framework", columns="metric_label", values=value)

    if value == "p_value":
        pivot = pivot.map(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
    elif value == "tau":
        pivot = pivot.map(lambda x: f"{x:+.3f}" if pd.notna(x) else "-")

    return pivot


# ── Visualisations ────────────────────────────────────────────────────────────

_TREND_COLOR = {
    "increasing": "#d62728",   # red        — worsening (energy/latency rise)
    "decreasing": "#2ca02c",   # green      — improvement
    "no trend":   "#aec7e8",   # light blue — no significant trend
    "skip":       "#eeeeee",   # grey       — insufficient data
}

_TREND_SYMBOL = {
    "increasing": "^",
    "decreasing": "v",
    "no trend":   "o",
    "skip":       "-",
}


def plot_mk_heatmap(
    results: pd.DataFrame,
    metric_group: str | list[str] | None = None,
    alpha: float = 0.05,
    title: str | None = None,
    higher_is_better: bool = False,
) -> go.Figure:
    """
    MK trend heatmap: frameworks (Y-axis) x metrics (X-axis).

    Colours (default, ``higher_is_better=False`` — e.g. energy, carbon):
        Red        -> significant increasing trend  (worsening)
        Green      -> significant decreasing trend  (improvement)
        Light blue -> no significant trend
        Grey       -> insufficient data / skipped

    When ``higher_is_better=True`` (e.g. throughput):
        Green      -> significant increasing trend  (improvement)
        Red        -> significant decreasing trend  (worsening)

    Each cell shows the trend direction symbol and the p-value.

    Parameters
    ----------
    results           : DataFrame returned by ``run_mk_analysis``.
    metric_group      : Filter by group name (``'energy'``, ``'carbon'``, etc.)
                        or a list of group names to combine into one heatmap
                        (e.g. ``['energy', 'carbon']``). ``None`` = all groups.
    alpha             : Significance level used for colouring.
    title             : Custom plot title.
    higher_is_better  : When True, increasing trend = green (good).
                        Use for throughput metrics. Default False.
    """
    df = results.copy()
    if metric_group is not None:
        groups = [metric_group] if isinstance(metric_group, str) else metric_group
        df = df[df["group"].isin(groups)]

    frameworks = sorted(df["framework"].unique())
    metrics    = df["metric_label"].unique().tolist()

    # Build matrices for z (numeric colorscale), cell text, and hover text
    # z convention: +1 = "bad" direction (red), -1 = "good" direction (green)
    # For higher_is_better metrics, increasing is good → flip sign
    def _trend_to_z(trend: str, sig: bool) -> float:
        if not sig or trend in ("no trend", "skip"):
            return 0.0 if trend != "skip" else 0.5
        base = {"increasing": 1.0, "decreasing": -1.0}.get(trend, 0.5)
        return -base if higher_is_better else base

    z_mat, text_mat, hover_mat = [], [], []

    for fw in frameworks:
        z_row, t_row, h_row = [], [], []
        fw_df = df[df["framework"] == fw]
        for m_label in metrics:
            row = fw_df[fw_df["metric_label"] == m_label]
            if row.empty:
                z_row.append(0.5)
                t_row.append("-")
                h_row.append(f"{fw} | {m_label}<br>No data")
            else:
                r     = row.iloc[0]
                trend = r.get("trend", "skip")
                p     = r.get("p_value")
                tau   = r.get("tau")
                n     = r.get("n_versions")
                sig   = (r.get("h") == True)

                z_row.append(_trend_to_z(trend, sig))
                sym   = _TREND_SYMBOL.get(trend, "-")
                p_str = f"{p:.4f}" if pd.notna(p) else "-"
                t_row.append(f"{sym}<br><sub>{p_str}</sub>" if sig else f"<sub>{p_str}</sub>")
                tau_str = f"tau = {tau:+.3f}<br>" if pd.notna(tau) else ""
                h_row.append(
                    f"<b>{fw}</b> - {m_label}<br>"
                    f"Trend: {trend}<br>"
                    f"p = {p_str}<br>"
                    f"{tau_str}"
                    f"n = {n}"
                )
        z_mat.append(z_row)
        text_mat.append(t_row)
        hover_mat.append(h_row)

    # Colorscale: green = good (-1), light blue = neutral (0),
    #             grey = skip (0.5), red = bad (+1)
    colorscale = [
        [0.0,  "#2ca02c"],   # -1   -> green (good direction)
        [0.5,  "#aec7e8"],   #  0   -> light blue (no trend)
        [0.75, "#eeeeee"],   #  0.5 -> grey (skip)
        [1.0,  "#d62728"],   #  1   -> red (bad direction)
    ]

    fig = go.Figure(go.Heatmap(
        z=z_mat,
        x=metrics,
        y=frameworks,
        text=text_mat,
        texttemplate="%{text}",
        hovertext=hover_mat,
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale=colorscale,
        zmin=-1, zmax=1,
        showscale=False,
        xgap=2,
        ygap=2,
    ))

    if metric_group is None:
        group_label = ""
    elif isinstance(metric_group, list):
        group_label = f" - {' & '.join(g.title() for g in metric_group)}"
    else:
        group_label = f" - {metric_group.title()}"
    fig.update_layout(
        title=dict(
            text=title or f"Mann-Kendall trends{group_label} (alpha={alpha})",
            font=dict(size=16),
        ),
        xaxis=dict(tickangle=-35, side="bottom"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11),
        height=max(300, 60 * len(frameworks) + 120),
        margin=dict(t=60, b=120, l=100, r=30),
    )

    # Legend labels depend on the semantic direction of the metrics
    if higher_is_better:
        legend_entries = [
            ("Increasing (sig.) — improvement", "#2ca02c", "^"),
            ("Decreasing (sig.) — worsening",   "#d62728", "v"),
            ("No significant trend",             "#aec7e8", "o"),
        ]
    else:
        legend_entries = [
            ("Increasing (sig.) — worsening",   "#d62728", "^"),
            ("Decreasing (sig.) — improvement", "#2ca02c", "v"),
            ("No significant trend",             "#aec7e8", "o"),
        ]

    for label, color, sym in legend_entries:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=color, symbol="square"),
            name=f"{sym} {label}",
            showlegend=True,
        ))

    return fig


def plot_mk_trend_line(
    df_summary: pd.DataFrame,
    framework: str,
    metric: str,
    mk_result_row: pd.Series | None = None,
    alpha: float = 0.05,
) -> go.Figure:
    """
    Line chart of a metric across versions with the Theil-Sen trend line overlaid.

    Parameters
    ----------
    df_summary    : summary_by_version.csv as a DataFrame.
    framework     : Framework name.
    metric        : Column name (e.g. ``'emission_energy_consumed_mean'``).
    mk_result_row : Row from the DataFrame returned by ``run_mk_analysis``
                    for this framework + metric (optional). When provided,
                    the Theil-Sen line and p-value annotation are added.
    alpha         : Significance level used for the title colour annotation.
    """
    fw_df = df_summary[df_summary["framework"] == framework].copy()
    fw_df = _sort_by_version(fw_df)
    fw_df = fw_df.reset_index(drop=True)

    x        = list(range(len(fw_df)))
    y        = fw_df[metric].values
    versions = fw_df["framework_version"].tolist()

    fig = go.Figure()

    # Confidence band (mean +/- std) when the std column is available
    std_col = metric.replace("_mean", "_std")
    if std_col in fw_df.columns:
        y_std = fw_df[std_col].values
        fig.add_trace(go.Scatter(
            x=versions + versions[::-1],
            y=list(y + y_std) + list((y - y_std)[::-1]),
            fill="toself",
            fillcolor="rgba(31,119,180,0.15)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            name="mean +/- std",
        ))

    # Main line
    fig.add_trace(go.Scatter(
        x=versions, y=y,
        mode="lines+markers",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=4),
        name=metric,
    ))

    # Theil-Sen trend line
    if mk_result_row is not None:
        slope     = mk_result_row.get("slope")
        intercept = mk_result_row.get("intercept")
        p_val     = mk_result_row.get("p_value")
        trend     = mk_result_row.get("trend", "")
        h         = mk_result_row.get("h", False)

        if pd.notna(slope) and pd.notna(intercept):
            y_ts     = [intercept + slope * xi for xi in x]
            color_ts = (
                "#d62728" if trend == "increasing" else
                "#2ca02c" if trend == "decreasing" else
                "#888888"
            )
            fig.add_trace(go.Scatter(
                x=versions, y=y_ts,
                mode="lines",
                line=dict(color=color_ts, width=2, dash="dash"),
                name=f"Theil-Sen (slope={slope:.3e})",
            ))

        if pd.notna(p_val):
            sig_marker = "sig." if h else "n.s."
            fig.update_layout(
                title=dict(
                    text=(
                        f"<b>{framework.title()}</b> - {metric}<br>"
                        f"<sub>MK: {trend} | p={p_val:.4f} | {sig_marker}</sub>"
                    ),
                )
            )

    fig.update_layout(
        xaxis=dict(
            title="Version",
            categoryorder="array",
            categoryarray=versions,
            tickangle=-35,
        ),
        yaxis=dict(title=metric),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11),
        height=400,
        legend=dict(orientation="h", y=-0.25),
        margin=dict(t=80, b=100, l=70, r=30),
    )

    return fig


# ── Export ────────────────────────────────────────────────────────────────────

def export_mk_results(
    results: pd.DataFrame,
    output_csv: str = "mk_results.csv",
    output_html: str | None = "mk_heatmap.html",
    alpha: float = 0.05
) -> None:
    """
    Save results to CSV and (optionally) the heatmap to a standalone HTML file.

    Parameters
    ----------
    results     : DataFrame returned by ``run_mk_analysis``.
    output_csv  : Path for the CSV output file.
    output_html : Path for the HTML heatmap. Pass ``None`` to skip.
    """
    results.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

    if output_html:
        fig = plot_mk_heatmap(results, alpha=alpha)
        fig.write_html(output_html, include_plotlyjs="cdn")
        print(f"Heatmap saved to: {output_html}")