"""
pettitt_analysis.py
====================
Pettitt change-point test for the async Python frameworks benchmark.

For each (framework, metric) pair the module:
  1. Sorts versions by semantic version (packaging.version.Version).
  2. Applies Pettitt's non-parametric homogeneity test to detect a structural
     break in the metric series across versions.
  3. Collects results including the change-point version name, mean before/after,
     and p-value.

Reference
---------
Pettitt, A. N. (1979). A non-parametric approach to the change-point problem.
Journal of the Royal Statistical Society: Series C (Applied Statistics), 28(2),
126-135. https://doi.org/10.2307/2346729
"""

from __future__ import annotations

import warnings
from collections import namedtuple
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from packaging.version import Version
import pyhomogeneity as hg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRICS = [
    # Energy / emissions
    "emission_energy_consumed_mean",
    "emission_cpu_energy_mean",
    "emission_ram_energy_mean",
    "emission_emissions_mean",
    "emission_emissions_rate_mean",
    "emission_duration_mean",
    "emission_energy_per_request_mean",
    "emission_co2_per_request_mean",
    # HTTP performance - api
    "api_throughput_rps_mean",
    "api_rt_p50_mean",
    "api_rt_p95_mean",
    "api_rt_p99_mean",
    # HTTP performance - html
    "html_throughput_rps_mean",
    "html_rt_p50_mean",
    "html_rt_p95_mean",
    "html_rt_p99_mean",
    # HTTP performance - upload
    "upload_throughput_rps_mean",
    "upload_rt_p50_mean",
    "upload_rt_p95_mean",
    "upload_rt_p99_mean",
]

ALPHA = 0.05
SIM = 20000       # Monte Carlo replications for p-value (pyhomogeneity default)
MIN_VERSIONS = 4  # Minimum distinct versions required to run the test

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary(csv_path: str | Path) -> pd.DataFrame:
    """
    Load summary_by_version.csv and normalise the version column name to
    ``'version'`` for use throughout this module.

    The CSV produced by the benchmark pipeline uses the column name
    ``framework_version``; this function renames it to ``version`` so that
    all downstream functions receive a consistent interface.

    Parameters
    ----------
    csv_path : str or Path
        Path to ``summary_by_version.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``version`` column ready for
        :func:`run_pettitt_test`.
    """
    df = pd.read_csv(csv_path)

    if "framework_version" in df.columns and "version" not in df.columns:
        df = df.rename(columns={"framework_version": "version"})

    required = {"framework", "version"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"load_summary: expected columns {required}, but {missing} are missing. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

PettittRow = namedtuple(
    "PettittRow",
    [
        "framework",
        "metric",
        "n_versions",
        "h",           # True = change point detected (nonhomogeneous)
        "cp_index",    # 0-based index within the sorted + valid sub-series
        "cp_version",  # version string at the change point
        "p_value",
        "U_stat",
        "mu1",         # mean BEFORE the change point
        "mu2",         # mean AFTER the change point
        "delta",       # mu2 - mu1
        "delta_pct",   # relative change (%)
    ],
)


def _sort_versions(versions: List[str]) -> List[str]:
    """Sort version strings using semantic versioning."""
    return sorted(versions, key=lambda v: Version(v))


def run_pettitt_test(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    alpha: float = ALPHA,
    sim: int = SIM,
    min_versions: int = MIN_VERSIONS,
) -> pd.DataFrame:
    """
    Run Pettitt change-point tests across all (framework, metric) combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated benchmark results. Must contain ``framework`` and
        ``version`` columns. Use :func:`load_summary` to load
        ``summary_by_version.csv``, which renames ``framework_version``
        to ``version`` automatically.
    metrics : list[str], optional
        Metric columns to test. Defaults to ``METRICS``.
    alpha : float
        Significance level (default 0.05).
    sim : int
        Monte Carlo replications for p-value estimation (default 20 000).
        Reduce to 1 000 for quick exploratory runs.
    min_versions : int
        Frameworks with fewer distinct valid versions are skipped.

    Returns
    -------
    pd.DataFrame
        One row per (framework, metric) with Pettitt test results.
    """
    if metrics is None:
        metrics = METRICS

    available = [m for m in metrics if m in df.columns]
    missing_metrics = set(metrics) - set(available)
    if missing_metrics:
        warnings.warn(
            f"Pettitt: the following metrics are not in the dataframe and will be "
            f"skipped: {sorted(missing_metrics)}",
            stacklevel=2,
        )

    rows: List[PettittRow] = []

    for framework, grp in df.groupby("framework"):
        sorted_versions = _sort_versions(grp["version"].dropna().unique().tolist())

        if len(sorted_versions) < min_versions:
            warnings.warn(
                f"Pettitt: '{framework}' has only {len(sorted_versions)} versions "
                f"(< {min_versions}). Skipping.",
                stacklevel=2,
            )
            continue

        version_set = set(grp["version"].values)
        ordered_versions = [v for v in sorted_versions if v in version_set]
        grp_sorted = (
            grp.set_index("version")
            .loc[ordered_versions]
            .reset_index()
        )

        for metric in available:
            series = grp_sorted[metric].values.astype(float)
            valid_mask = ~np.isnan(series)

            if valid_mask.sum() < min_versions:
                continue

            valid_indices = np.where(valid_mask)[0]
            valid_series = series[valid_mask]
            valid_versions = [ordered_versions[i] for i in valid_indices]

            try:
                result = hg.pettitt_test(valid_series, alpha=alpha, sim=sim)
            except Exception as exc:
                warnings.warn(
                    f"Pettitt: test failed for {framework}/{metric}: {exc}",
                    stacklevel=2,
                )
                continue

            h = bool(result.h)
            cp_idx = int(result.cp)
            cp_version = valid_versions[cp_idx]

            mu1 = float(result.avg.mu1)
            mu2 = float(result.avg.mu2)
            delta = mu2 - mu1
            delta_pct = (delta / mu1 * 100) if mu1 != 0 else float("nan")

            rows.append(
                PettittRow(
                    framework=framework,
                    metric=metric,
                    n_versions=len(valid_versions),
                    h=h,
                    cp_index=cp_idx,
                    cp_version=cp_version,
                    p_value=float(result.p),
                    U_stat=float(result.U),
                    mu1=mu1,
                    mu2=mu2,
                    delta=delta,
                    delta_pct=delta_pct,
                )
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_pettitt_heatmap(
    pt_results: pd.DataFrame,
    value: str = "h",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Return an interactive heatmap of Pettitt test results.

    Parameters
    ----------
    pt_results : pd.DataFrame
        Output of :func:`run_pettitt_test`.
    value : str
        Column to pivot: ``'h'``, ``'p_value'``, ``'delta_pct'``, or
        ``'cp_version'``.
    title : str, optional
        Plot title override.
    """
    valid_values = {"h", "p_value", "delta_pct", "cp_version"}
    if value not in valid_values:
        raise ValueError(f"value must be one of {valid_values}")

    pivot = pt_results.pivot_table(
        index="framework",
        columns="metric",
        values=value,
        aggfunc="first",
    )

    if value == "h":
        z = pivot.astype(float).values
        text = pivot.map(lambda x: "yes" if x else "no" if pd.notna(x) else "").values
        colorscale = [[0, "#d4edda"], [1, "#f8d7da"]]
        zmid = None
        title = title or "Pettitt Test — Change Point Detected"
        colorbar_title = "Detected"
    elif value == "p_value":
        z = pivot.astype(float).values
        text = pivot.map(lambda x: f"{x:.4f}" if pd.notna(x) else "—").values
        colorscale = "RdYlGn_r"
        zmid = ALPHA
        title = title or f"Pettitt Test — p-value (alpha = {ALPHA})"
        colorbar_title = "p-value"
    elif value == "delta_pct":
        z = pivot.astype(float).values
        text = pivot.map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—").values
        colorscale = "RdBu_r"
        zmid = 0
        title = title or "Pettitt Test — Relative Change at Break (mu2 - mu1) / mu1"
        colorbar_title = "delta %"
    else:  # cp_version
        z = np.ones(pivot.shape)
        text = pivot.map(lambda x: str(x) if pd.notna(x) else "—").values
        colorscale = [[0, "#e9ecef"], [1, "#e9ecef"]]
        zmid = None
        title = title or "Pettitt Test — Change-Point Version"
        colorbar_title = ""

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(pivot.columns),
            y=list(pivot.index),
            text=text,
            texttemplate="%{text}",
            colorscale=colorscale,
            zmid=zmid,
            showscale=(value != "cp_version"),
            colorbar=dict(title=colorbar_title),
            hoverongaps=False,
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-45),
        height=max(350, 40 * len(pivot.index) + 150),
        margin=dict(l=120, r=40, t=60, b=160),
        template="plotly_white",
    )
    return fig


def plot_pettitt_series(
    df: pd.DataFrame,
    framework: str,
    metric: str,
    pt_result_row=None,
) -> go.Figure:
    """
    Plot a metric series for a single framework with the Pettitt change point
    highlighted (if available).

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_summary` (must have a ``version`` column).
    framework : str
        Framework name.
    metric : str
        Metric column (e.g. ``'emission_energy_consumed_mean'``).
    pt_result_row : pd.Series or single-row pd.DataFrame, optional
        Corresponding row from :func:`run_pettitt_test` output.
    """
    grp = df[df["framework"] == framework].copy()
    grp["_v"] = grp["version"].apply(Version)
    grp = grp.sort_values("_v").reset_index(drop=True)

    versions = grp["version"].tolist()
    y_mean = grp[metric].values
    std_col = metric.replace("_mean", "_std")
    y_std = grp[std_col].values if std_col in grp.columns else np.zeros_like(y_mean)

    fig = go.Figure()

    # std band
    fig.add_trace(
        go.Scatter(
            x=versions + versions[::-1],
            y=list(y_mean + y_std) + list((y_mean - y_std)[::-1]),
            fill="toself",
            fillcolor="rgba(99,110,250,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # mean line
    fig.add_trace(
        go.Scatter(
            x=versions,
            y=y_mean,
            mode="lines+markers",
            name=metric,
            line=dict(color="#636EFA", width=2),
            marker=dict(size=6),
        )
    )

    # change point annotation
    if pt_result_row is not None:
        row = (
            pt_result_row.iloc[0]
            if isinstance(pt_result_row, pd.DataFrame)
            else pt_result_row
        )
        cp_v = row["cp_version"]
        mu1 = float(row["mu1"])
        mu2 = float(row["mu2"])
        h = bool(row["h"])
        p = float(row["p_value"])

        # add_vline does not support categorical (string) x-axes — use
        # add_shape + add_annotation with xref="x" instead.
        line_color = "#EF553B" if h else "#AAAAAA"
        fig.add_shape(
            type="line",
            x0=cp_v, x1=cp_v,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color=line_color, dash="dash", width=2),
        )
        fig.add_annotation(
            x=cp_v,
            y=1.0,
            xref="x", yref="paper",
            text=f"Change point: {cp_v}<br>p={p:.4f} {'(sig.)' if h else '(n.s.)'}",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=11, color=line_color),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor=line_color,
            borderwidth=1,
        )

        if cp_v in versions:
            cp_idx = versions.index(cp_v)
            # cp_idx is the FIRST index of segment 2 (pyhomogeneity convention:
            # mu1 = mean(x[:cp]), mu2 = mean(x[cp:])). So mu1 covers everything
            # BEFORE cp_v and mu2 covers FROM cp_v onwards — no overlap.
            before_v = versions[:cp_idx]
            after_v  = versions[cp_idx:]
            if before_v:
                fig.add_trace(
                    go.Scatter(
                        x=before_v, y=[mu1] * len(before_v),
                        mode="lines", name=f"mu1 = {mu1:.4g}",
                        line=dict(color="#00CC96", dash="dot", width=2),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=after_v, y=[mu2] * len(after_v),
                    mode="lines", name=f"mu2 = {mu2:.4g}",
                    line=dict(color="#AB63FA", dash="dot", width=2),
                )
            )

    fig.update_layout(
        title=f"{framework} — {metric}",
        xaxis=dict(title="Version", tickangle=-45),
        yaxis=dict(title=metric),
        template="plotly_white",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_pettitt_summary_bar(
    pt_results: pd.DataFrame,
    metric_group: str = "energy",
) -> go.Figure:
    """
    Horizontal bar chart: number of significant change points per framework.

    Parameters
    ----------
    pt_results : pd.DataFrame
        Output of :func:`run_pettitt_test`.
    metric_group : str
        ``'energy'``, ``'http'``, or ``'all'``.
    """
    df = pt_results.copy()
    if metric_group == "energy":
        df = df[df["metric"].str.startswith("emission_")]
    elif metric_group == "http":
        df = df[~df["metric"].str.startswith("emission_")]

    detected = (
        df[df["h"] == True]  # noqa: E712
        .groupby("framework").size()
        .reset_index(name="detected")
    )
    total = df.groupby("framework").size().reset_index(name="total")
    counts = total.merge(detected, on="framework", how="left").fillna(0)
    counts["detected"] = counts["detected"].astype(int)
    counts["pct"] = counts["detected"] / counts["total"] * 100
    counts = counts.sort_values("detected", ascending=True)

    group_label = {
        "energy": "energy/emissions", "http": "HTTP performance", "all": "all"
    }.get(metric_group, metric_group)

    fig = go.Figure(
        go.Bar(
            x=counts["detected"],
            y=counts["framework"],
            orientation="h",
            text=counts.apply(
                lambda r: f"{int(r['detected'])}/{int(r['total'])} ({r['pct']:.0f}%)",
                axis=1,
            ),
            textposition="outside",
            marker_color="#EF553B",
        )
    )
    fig.update_layout(
        title=f"Frameworks with Significant Change Points — {group_label} metrics",
        xaxis=dict(title="# metrics with detected change point"),
        yaxis=dict(title="Framework"),
        template="plotly_white",
        height=max(350, 40 * len(counts) + 100),
        margin=dict(r=120),
    )
    return fig


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_pettitt_results(
    pt_results: pd.DataFrame,
    output_csv: str = "pettitt_results.csv",
    output_html: str = "pettitt_heatmap.html",
) -> None:
    """
    Save results to CSV and an interactive standalone HTML heatmap.

    Parameters
    ----------
    pt_results : pd.DataFrame
        Output of :func:`run_pettitt_test`.
    output_csv : str
        Destination path for the CSV file.
    output_html : str
        Destination path for the standalone HTML heatmap.
    """
    pt_results.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    fig = plot_pettitt_heatmap(pt_results, value="h")
    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"Saved: {output_html}")


# ---------------------------------------------------------------------------
# Quick smoke test (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if csv_arg:
        print(f"Loading: {csv_arg}")
        df = load_summary(csv_arg)
        print(f"Shape: {df.shape} | Frameworks: {sorted(df['framework'].unique())}")
        results = run_pettitt_test(df, sim=500)
        print(f"\nResults: {len(results)} rows")
        print(results[["framework", "metric", "h", "cp_version", "p_value", "delta_pct"]].to_string())
    else:
        np.random.seed(42)
        n = 30
        series = np.concatenate([
            np.random.normal(100, 5, 15),
            np.random.normal(130, 5, 15),
        ])
        mock_df = pd.DataFrame({
            "framework": ["mock"] * n,
            "version": [f"0.{i}.0" for i in range(n)],
            "emission_energy_consumed_mean": series,
        })
        results = run_pettitt_test(mock_df, metrics=["emission_energy_consumed_mean"], sim=5000)
        print(results.to_string())
        assert results.iloc[0]["h"], "Expected change point to be detected"
        print("\nSmoke test passed.")