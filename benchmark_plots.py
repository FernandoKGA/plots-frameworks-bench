"""
benchmark_plots.py
==================
Gera gráficos Plotly interativos a partir dos CSVs produzidos pelo pipeline
de agregação (rounds.csv / summary_by_version.csv).

Gráficos disponíveis
--------------------
1. box_requests(df, framework, endpoint, metric)
   Box plot por versão — tempos de resposta de um endpoint (rounds.csv)

2. box_energy(df, framework, metric)
   Box plot por versão — métricas de energia (rounds.csv)

3. line_energy(df, framework, metric)
   Gráfico de linha — consumo de energia médio por versão (rounds.csv)

Uso rápido
----------
    from benchmark_plots import load_rounds, box_requests, box_energy, line_energy

    df = load_rounds("rounds.csv")

    fig = box_requests(df, framework="fastapi", endpoint="api", metric="rt_p95")
    fig.show()

Uso com Dash
------------
    import dash
    from dash import dcc, html, Input, Output
    from benchmark_plots import load_rounds, box_requests, box_energy, line_energy

    df = load_rounds("rounds.csv")
    app = dash.Dash(__name__)
    # monte seus callbacks com as funções acima retornando fig
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Constantes ────────────────────────────────────────────────────────────────

ENDPOINTS = ["api", "html", "upload"]

REQUEST_METRICS: dict[str, str] = {
    "rt_mean":   "Tempo médio (s)",
    "rt_p50":    "Mediana / p50 (s)",
    "rt_p75":    "p75 (s)",
    "rt_p90":    "p90 (s)",
    "rt_p95":    "p95 (s)",
    "rt_p99":    "p99 (s)",
    "rt_min":    "Mínimo (s)",
    "rt_max":    "Máximo (s)",
    "throughput_rps": "Throughput (req/s)",
}

ENERGY_METRICS: dict[str, str] = {
    "emission_energy_consumed": "Energia total (kWh)",
    "emission_cpu_energy":      "Energia CPU (kWh)",
    "emission_ram_energy":      "Energia RAM (kWh)",
    "emission_emissions":       "Emissões CO₂ (kg)",
    "emission_duration":        "Duração (s)",
    "emission_cpu_power":       "Potência CPU (W)",
    "emission_ram_power":       "Potência RAM (W)",
    "emission_emissions_rate":  "Taxa de emissão (kg CO₂/s)",
}

# Paleta discreta — sobra bem para versões ordinais
_VERSION_PALETTE = px.colors.qualitative.Plotly

# ── Carregamento de dados ─────────────────────────────────────────────────────

def _parse_version(v: str):
    """Parseia uma string de versão com packaging.version.Version, com fallback."""
    from packaging.version import Version, InvalidVersion
    try:
        return Version(v)
    except InvalidVersion:
        return Version("0")


def load_rounds(path: str | Path = "rounds.csv") -> pd.DataFrame:
    """
    Lê rounds.csv e adiciona a coluna `version_label` (string ordenável).
    Espera as colunas: framework, framework_version, exc_n, e as métricas.
    """
    df = pd.read_csv(path)
    df["version_label"] = df["framework_version"].astype(str)

    # Ordena por framework e depois por versão semântica
    df["_ver_key"] = df["version_label"].map(_parse_version)
    df = df.sort_values(["framework", "_ver_key", "exc_n"]).drop(columns="_ver_key")
    df = df.reset_index(drop=True)
    return df


def _framework_versions(df: pd.DataFrame, framework: str) -> list[str]:
    """Retorna lista de versões do framework ordenada por semver."""
    mask = df["framework"].str.lower() == framework.lower()
    versions = df.loc[mask, "version_label"].unique().tolist()
    return sorted(versions, key=_parse_version)


def _col(endpoint: str, metric: str) -> str:
    """Monta nome da coluna: 'api_rt_p95', 'html_throughput_rps', etc."""
    return f"{endpoint}_{metric}"


# ── Gráfico 1 — Box plot: tempo de resposta por versão ───────────────────────

def box_requests(
    df: pd.DataFrame,
    framework: str,
    endpoint: str = "api",
    metric: str = "rt_p95",
    title: str | None = None,
) -> go.Figure:
    """
    Box plot de uma métrica de requisição (por versão, over rodadas).

    Parâmetros
    ----------
    df        : DataFrame retornado por load_rounds()
    framework : nome do framework (case-insensitive)
    endpoint  : 'api' | 'html' | 'upload'
    metric    : chave em REQUEST_METRICS (ex: 'rt_p95', 'throughput_rps')
    title     : título customizado (opcional)
    """
    col = _col(endpoint, metric)
    mask = df["framework"].str.lower() == framework.lower()
    data = df[mask].copy()

    if col not in data.columns:
        raise ValueError(
            f"Coluna '{col}' não encontrada. "
            f"Colunas disponíveis: {[c for c in data.columns if endpoint in c]}"
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
            boxpoints="all",        # mostra pontos individuais (rodadas)
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
        f"{framework.title()} — {endpoint_label}: {y_label} por versão"
    )

    fig.update_layout(
        title=dict(text=_title, font=dict(size=16)),
        xaxis_title="Versão",
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


# ── Gráfico 2 — Box plot: energia por versão ──────────────────────────────────

def box_energy(
    df: pd.DataFrame,
    framework: str,
    metric: str = "emission_energy_consumed",
    title: str | None = None,
) -> go.Figure:
    """
    Box plot de uma métrica de energia (por versão, over rodadas).

    Parâmetros
    ----------
    df        : DataFrame retornado por load_rounds()
    framework : nome do framework (case-insensitive)
    metric    : chave em ENERGY_METRICS (ex: 'energy_consumed', 'emissions')
    title     : título customizado (opcional)
    """
    mask = df["framework"].str.lower() == framework.lower()
    data = df[mask].copy()

    if metric not in data.columns:
        energy_cols = [c for c in data.columns if c in ENERGY_METRICS]
        raise ValueError(
            f"Coluna '{metric}' não encontrada no DataFrame. "
            f"Colunas de energia disponíveis: {energy_cols}"
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

    _title = title or f"{framework.title()} — {y_label} por versão"

    fig.update_layout(
        title=dict(text=_title, font=dict(size=16)),
        xaxis_title="Versão",
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


# ── Gráfico 3 — Linha: consumo de energia por versão ─────────────────────────

def line_energy(
    df: pd.DataFrame,
    framework: str,
    metric: str = "emission_energy_consumed",
    show_ci: bool = True,
    title: str | None = None,
) -> go.Figure:
    """
    Gráfico de linha — média de energia por versão, com intervalo de confiança
    (mean ± std) como área sombreada quando show_ci=True.

    Parâmetros
    ----------
    df        : DataFrame retornado por load_rounds()
    framework : nome do framework (case-insensitive)
    metric    : chave em ENERGY_METRICS
    show_ci   : exibe banda mean ± std
    title     : título customizado (opcional)
    """
    mask = df["framework"].str.lower() == framework.lower()
    data = df[mask].copy()

    versions = _framework_versions(df, framework)
    y_label  = ENERGY_METRICS.get(metric, metric)

    # Agrega por versão
    agg = (
        data.groupby("version_label")[metric]
        .agg(mean="mean", std="std", count="count")
        .reindex(versions)
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)

    color_main = "#2563EB"   # azul
    color_ci   = "rgba(37,99,235,0.15)"

    fig = go.Figure()

    # Banda de incerteza (mean ± std)
    if show_ci:
        fig.add_trace(go.Scatter(
            x=list(agg["version_label"]) + list(reversed(agg["version_label"])),
            y=list(agg["mean"] + agg["std"]) + list(reversed(agg["mean"] - agg["std"])),
            fill="toself",
            fillcolor=color_ci,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Média ± desvio padrão",
        ))

    # Linha principal
    fig.add_trace(go.Scatter(
        x=agg["version_label"],
        y=agg["mean"],
        mode="lines+markers",
        name=f"Média — {y_label}",
        line=dict(color=color_main, width=2.5),
        marker=dict(size=7, color=color_main, symbol="circle"),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{y_label}: %{{y:.8f}}<br>"
            "<extra></extra>"
        ),
    ))

    _title = title or f"{framework.title()} — {y_label} por versão (média ± dp)"

    fig.update_layout(
        title=dict(text=_title, font=dict(size=16)),
        xaxis_title="Versão",
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


# ── Painel completo de um framework ──────────────────────────────────────────

def dashboard_framework(
    df: pd.DataFrame,
    framework: str,
    energy_metric: str = "emission_energy_consumed",
) -> go.Figure:
    """
    Painel 2×3 com todos os box plots de requisição (api/html/upload por p95)
    + box plot de energia + linha de energia num único Figure com subplots.

    Útil para exportar como HTML standalone ou para preview rápido.
    """
    endpoints = [e for e in ENDPOINTS if any(
        f"{e}_rt_p95" in df.columns for _ in [None]
    )]

    rows, cols = 2, 3
    subplot_titles = (
        [f"{e.upper()}: p95 de resposta" for e in ENDPOINTS]
        + [
            f"Energia ({ENERGY_METRICS[energy_metric]}) — box",
            f"Energia ({ENERGY_METRICS[energy_metric]}) — linha",
            "",   # célula vazia
        ]
    )

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]

    # Box plots de requisição por endpoint
    for idx, ep in enumerate(ENDPOINTS):
        r, c = positions[idx]
        sub = box_requests(df, framework, endpoint=ep, metric="rt_p95")
        for trace in sub.data:
            trace.showlegend = False
            fig.add_trace(trace, row=r, col=c)

    # Box plot de energia
    r, c = positions[3]
    sub_e = box_energy(df, framework, metric=energy_metric)
    for trace in sub_e.data:
        trace.showlegend = False
        fig.add_trace(trace, row=r, col=c)

    # Linha de energia
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
            text=f"<b>{framework.title()}</b> — Painel completo",
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


# ── Exportação estruturada (Opção A — por framework) ─────────────────────────

def export_all(
    df: pd.DataFrame,
    output_dir: str | Path = "plots",
    energy_metric: str = "emission_energy_consumed",
    include_plotlyjs: str = "cdn",
) -> dict[str, list[Path]]:
    """
    Exporta todos os gráficos organizados por framework (Opção A):

        plots/
        └── {framework}/
            ├── requests/
            │   ├── {endpoint}_{metric}.html   (box plots de requisição)
            │   └── ...
            ├── energy/
            │   ├── box_{metric}.html          (box plots de energia)
            │   └── line_{metric}.html         (linhas de energia)
            └── dashboard.html                 (painel completo 2×3)

    Parâmetros
    ----------
    df              : DataFrame retornado por load_rounds()
    output_dir      : pasta raiz de saída (criada se não existir)
    energy_metric   : métrica de energia usada no dashboard (padrão: energy_consumed)
    include_plotlyjs : "cdn" (menor) | "inline" (sem internet) | True (cópia local)

    Retorna
    -------
    dict framework -> lista de Paths gerados
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

        # ── Box plots de requisição ──────────────────────────────────────────
        for ep in ENDPOINTS:
            ep_col_base = f"{ep}_rt_p50"
            if ep_col_base not in df.columns:
                print(f"  skip {ep} (sem dados)")
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

        # ── Box plots de energia ─────────────────────────────────────────────
        for metric in energy_metrics:
            if metric not in df.columns or df[metric].isna().all():
                continue
            try:
                fig  = box_energy(df, framework=fw, metric=metric)
                path = nrg_dir / f"box_{metric}.html"
                fig.write_html(path, include_plotlyjs=include_plotlyjs)
                files.append(path)
                print(f"  energy/box_{metric}.html")
            except Exception as e:
                print(f"  [WARN] box_{metric}: {e}")

        # ── Linhas de energia ────────────────────────────────────────────────
        for metric in energy_metrics:
            if metric not in df.columns or df[metric].isna().all():
                continue
            try:
                fig  = line_energy(df, framework=fw, metric=metric, show_ci=True)
                path = nrg_dir / f"line_{metric}.html"
                fig.write_html(path, include_plotlyjs=include_plotlyjs)
                files.append(path)
                print(f"  energy/line_{metric}.html")
            except Exception as e:
                print(f"  [WARN] line_{metric}: {e}")

        # ── Dashboard completo ───────────────────────────────────────────────
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
    print(f"\n✓ {total} arquivo(s) gerado(s) em: {root.resolve()}/")
    return generated


# ── CLI / exemplo de uso ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    rounds_path = sys.argv[1] if len(sys.argv) > 1 else "rounds.csv"
    out_dir     = sys.argv[2] if len(sys.argv) > 2 else "plots"

    df = load_rounds(rounds_path)
    frameworks = df["framework"].unique().tolist()
    print(f"Frameworks encontrados: {frameworks}")

    export_all(df, output_dir=out_dir)
