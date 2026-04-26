#!/usr/bin/env python3
"""
generate_index.py
Generates index.html (root) and plots/{framework}/index.html for each framework.

Run AFTER all analyses have completed:
    1. aggregate_results.ipynb
    2. benchmark_plots.ipynb
    3. statistical_analysis.ipynb  (Mann-Kendall)
    4. pettitt_analysis.ipynb      (Pettitt change-point)
    5. python generate_index.py [--plots-dir plots] [--csv summary_by_version.csv]
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict
from packaging.version import Version

# ── Configuration ────────────────────────────────────────────────────────────

FRAMEWORK_LABELS = {
    "aiohttp":   "aiohttp",
    "baize":     "BáiZé",
    "django":    "Django",
    "falcon":    "Falcon",
    "fastapi":   "FastAPI",
    "muffin":    "Muffin",
    "quart":     "Quart",
    "sanic":     "Sanic",
    "starlette": "Starlette",
    "tornado":   "Tornado",
}

ENDPOINT_LABELS = {
    "api":    "JSON Endpoint",
    "html":   "HTML Endpoint",
    "upload": "Upload Endpoint",
}

METRIC_LABELS = {
    "throughput_rps":   "Throughput (req/s)",
    "rt_mean":          "Response Time – Mean",
    "rt_min":           "Response Time – Min",
    "rt_max":           "Response Time – Max",
    "rt_p50":           "Response Time – P50 (Median)",
    "rt_p75":           "Response Time – P75",
    "rt_p90":           "Response Time – P90",
    "rt_p95":           "Response Time – P95",
    "rt_p99":           "Response Time – P99",
    # energy / emissions metrics
    "emission_duration":        "Measurement Duration",
    "emission_energy_consumed": "Total Energy Consumed",
    "emission_cpu_energy":      "CPU Energy",
    "emission_ram_energy":      "RAM Energy",
    "emission_emissions":       "CO₂ Emissions",
    "emission_emissions_rate":  "CO₂ Emission Rate",
}

PLOT_TYPE_LABELS = {
    "line":    "Line Chart",
    "box":     "Box Plot",
    "bar":     "Bar Chart",
    "scatter": "Scatter Plot",
}

# ── Labels for global (cross-framework) analysis files ───────────────────────

# Keys are matched against the full file stem (case-insensitive startswith).
# More specific prefixes must come before shorter ones.
GLOBAL_ANALYSIS_LABELS = {
    "cmk_heatmap_energy_carbon":  ("Mann-Kendall",  "Energy & Carbon Heatmap (corrected)"),
    "mk_heatmap_energy_carbon":   ("Mann-Kendall",  "Energy & Carbon Heatmap"),
    "mk_heatmap_throughput":      ("Mann-Kendall",  "Throughput Heatmap"),
    "mk_heatmap_latency":         ("Mann-Kendall",  "Latency Heatmap"),
    "mk_heatmap":                 ("Mann-Kendall",  "Combined Heatmap"),
    "pettitt_heatmap_delta_pct":  ("Pettitt",       "Relative Change (Δ%) Heatmap"),
    "pettitt_heatmap_cp_version": ("Pettitt",       "Change-Point Version Heatmap"),
    "pettitt_heatmap":            ("Pettitt",       "Detection Heatmap"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def describe_plot(stem: str) -> str:
    """Convert a file stem into a human-readable description."""
    parts = stem.split("_")

    # ── Statistical per-framework plots ──────────────────────────────────────
    # mk_theil_sen_trend_{fw}
    if stem.startswith("mk_theil_sen_trend_"):
        fw = stem[len("mk_theil_sen_trend_"):]
        return f"Mann-Kendall – Theil-Sen Trend Line ({fw})"

    # pettitt_emission_emissions_{fw}
    if stem.startswith("pettitt_emission_emissions_"):
        fw = stem[len("pettitt_emission_emissions_"):]
        return f"Pettitt – CO₂ Emissions Change Point ({fw})"

    # ── Standard benchmark plots ─────────────────────────────────────────────
    plot_type = ""
    if parts[0] in PLOT_TYPE_LABELS:
        plot_type = PLOT_TYPE_LABELS[parts[0]]
        parts = parts[1:]

    endpoint = ""
    if parts and parts[0] in ENDPOINT_LABELS:
        endpoint = ENDPOINT_LABELS[parts[0]]
        parts = parts[1:]

    metric_key = "_".join(parts)
    metric = METRIC_LABELS.get(metric_key) or metric_key.replace("_", " ").title()

    pieces = [p for p in [endpoint, metric, plot_type] if p]
    return " – ".join(pieces) if pieces else stem.replace("_", " ").title()


def describe_global_plot(stem: str):
    """Return (analysis_type, description) for a global analysis HTML file."""
    for key, (analysis_type, desc) in GLOBAL_ANALYSIS_LABELS.items():
        if stem == key:
            return analysis_type, desc
    # Fallback for unrecognised stems
    return "Analysis", stem.replace("_", " ").title()


def load_version_info(csv_path: Path) -> dict:
    """Returns {framework: {"count": N, "min": v, "max": v}}."""
    data = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            data[row["framework"]].append(row["framework_version"])
    result = {}
    for fw, versions in data.items():
        unique = sorted(set(versions), key=Version)
        result[fw] = {"count": len(unique), "min": unique[0], "max": unique[-1]}
    return result


def collect_plots(plots_dir: Path) -> tuple[dict, list]:
    """
    Returns:
        structure  — {framework: {"requests": [...], "energy": [...],
                                  "statistical": [...], "other": [...]}}
        global_files — list of .html files sitting directly in plots_dir
                       (cross-framework analysis heatmaps, etc.)
    """
    # ── Per-framework plots ───────────────────────────────────────────────────
    structure = {}
    for fw_path in sorted(plots_dir.iterdir()):
        if fw_path.name.startswith('.'):
            continue
        if not fw_path.is_dir():
            continue
        fw = fw_path.name
        buckets: dict[str, list] = {
            "requests":    [],
            "energy":      [],
            "statistical": [],
            "other":       [],
        }
        for category in ["requests", "energy"]:
            cat_path = fw_path / category
            if cat_path.is_dir():
                buckets[category] = sorted(
                    [f for f in cat_path.iterdir() if f.suffix == ".html"],
                    key=lambda p: p.stem,
                )
        # Statistical per-framework files live directly in plots/{fw}/
        for f in sorted(fw_path.glob("*.html")):
            stem = f.stem
            if (stem.startswith("mk_theil_sen_trend_") or
                    stem.startswith("pettitt_")):
                buckets["statistical"].append(f)
            else:
                buckets["other"].append(f)
        structure[fw] = buckets

    # ── Global analysis files (directly in plots/) ───────────────────────────
    global_files = sorted(
        [f for f in plots_dir.glob("*.html") if f.is_file()],
        key=lambda p: p.stem,
    )

    return structure, global_files


# ── CSS / HTML shared style ───────────────────────────────────────────────────

SHARED_STYLE = """
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #0d1117;
      --surface:  #161b22;
      --border:   #30363d;
      --accent:   #58a6ff;
      --accent2:  #3fb950;
      --accent3:  #d2a8ff;
      --text:     #e6edf3;
      --muted:    #8b949e;
      --tag-bg:   #1f2937;
      --radius:   6px;
      font-size:  15px;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'JetBrains Mono', 'Fira Mono', 'Cascadia Code', monospace;
      line-height: 1.6;
      padding: 2rem 1.5rem 4rem;
      max-width: 1100px;
      margin: 0 auto;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
      padding-bottom: 1.25rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 2rem;
    }
    .topbar h1 { font-size: 1.15rem; font-weight: 600; color: var(--text); letter-spacing: -0.3px; }
    .topbar h1 span { color: var(--accent); }

    .breadcrumb {
      font-size: .8rem;
      color: var(--muted);
      margin-bottom: 1.5rem;
    }
    .breadcrumb a { color: var(--accent); text-decoration: none; }
    .breadcrumb a:hover { text-decoration: underline; }

    h2 {
      font-size: .95rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 1rem;
      padding-bottom: .4rem;
      border-bottom: 1px solid var(--border);
    }
    section { margin-bottom: 2.5rem; }

    table { width: 100%; border-collapse: collapse; font-size: .85rem; }
    thead th {
      text-align: left;
      padding: .5rem .75rem;
      font-size: .72rem;
      text-transform: uppercase;
      letter-spacing: .07em;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }
    tbody tr { border-bottom: 1px solid var(--border); transition: background .1s; }
    tbody tr:last-child { border-bottom: none; }
    tbody tr:hover { background: var(--surface); }
    tbody td { padding: .55rem .75rem; vertical-align: middle; }

    .tag {
      display: inline-block;
      padding: .15rem .5rem;
      border-radius: 999px;
      font-size: .72rem;
      font-weight: 600;
      background: var(--tag-bg);
      color: var(--accent);
      border: 1px solid var(--border);
    }
    .tag.green  { color: var(--accent2); }
    .tag.purple { color: var(--accent3); }

    .range { color: var(--muted); font-size: .8rem; }
    .range strong { color: var(--text); font-weight: 500; }

    a.open-link {
      color: var(--accent);
      text-decoration: none;
      font-size: .8rem;
      padding: .2rem .55rem;
      border: 1px solid transparent;
      border-radius: var(--radius);
      transition: border-color .15s;
    }
    a.open-link:hover { border-color: var(--accent); }

    a.fw-link { color: var(--accent); text-decoration: none; font-weight: 500; }
    a.fw-link:hover { text-decoration: underline; }

    .desc { color: var(--muted); }

    .badge-mk      { color: var(--accent);  }
    .badge-pettitt { color: var(--accent3); }

    .empty { padding: 1.5rem; text-align: center; color: var(--muted); font-size: .85rem; }

    footer {
      margin-top: 3rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
      font-size: .75rem;
      color: var(--muted);
      text-align: center;
    }
  </style>
"""

# ── Page generators ───────────────────────────────────────────────────────────

def build_root_index(
    plots_dir: Path,
    version_info: dict,
    structure: dict,
    global_files: list,
) -> str:

    # ── Framework rows ────────────────────────────────────────────────────────
    fw_rows = []
    for fw in sorted(structure.keys()):
        label     = FRAMEWORK_LABELS.get(fw, fw.title())
        vi        = version_info.get(fw, {})
        count     = vi.get("count", "–")
        v_min     = vi.get("min",   "–")
        v_max     = vi.get("max",   "–")
        link_href = f"{plots_dir.name}/{fw}/index.html"
        req_count  = len(structure[fw]["requests"])
        eng_count  = len(structure[fw]["energy"])
        stat_count = len(structure[fw]["statistical"])
        fw_rows.append(f"""
        <tr>
          <td><a class="fw-link" href="{link_href}">{label}</a></td>
          <td><span class="tag">{count} versions</span></td>
          <td class="range"><strong>{v_min}</strong> &rarr; <strong>{v_max}</strong></td>
          <td><span class="tag green">{req_count}</span></td>
          <td><span class="tag green">{eng_count}</span></td>
          <td><span class="tag purple">{stat_count}</span></td>
          <td><a class="open-link" href="{link_href}">View &rarr;</a></td>
        </tr>""")

    fw_rows_html = "\n".join(fw_rows) if fw_rows else \
        '<tr><td colspan="7" class="empty">No frameworks found.</td></tr>'

    # ── Global analysis rows ──────────────────────────────────────────────────
    global_rows = []
    for f in global_files:
        analysis_type, desc = describe_global_plot(f.stem)
        badge_class = "badge-mk" if "Mann-Kendall" in analysis_type else "badge-pettitt"
        global_rows.append(f"""
        <tr>
          <td><code>{f.stem}</code></td>
          <td class="{badge_class}">{analysis_type}</td>
          <td class="desc">{desc}</td>
          <td><a class="open-link" href="{plots_dir.name}/{f.name}" target="_blank">Open &rarr;</a></td>
        </tr>""")

    if global_rows:
        global_section = f"""
  <section>
    <h2>Statistical Analysis — Cross-Framework</h2>
    <table>
      <thead>
        <tr>
          <th>File</th>
          <th>Test</th>
          <th>Description</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {"".join(global_rows)}
      </tbody>
    </table>
  </section>"""
    else:
        global_section = ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Python Web Framework Benchmark – Plot Index</title>
{SHARED_STYLE}
</head>
<body>

  <div class="topbar">
    <h1>Python Web Framework Benchmark &mdash; <span>Plot Index</span></h1>
  </div>

  <section>
    <h2>Frameworks</h2>
    <table>
      <thead>
        <tr>
          <th>Library</th>
          <th>Versions tested</th>
          <th>Version range</th>
          <th>Request plots</th>
          <th>Energy plots</th>
          <th>Statistical plots</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {fw_rows_html}
      </tbody>
    </table>
  </section>

  {global_section}

  <footer>
    Generated by <code>generate_index.py</code>
  </footer>

</body>
</html>"""


def build_framework_index(
    fw: str,
    fw_dir: Path,
    version_info: dict,
    buckets: dict,
) -> str:
    label = FRAMEWORK_LABELS.get(fw, fw.title())
    vi    = version_info.get(fw, {})
    count = vi.get("count", "–")
    v_min = vi.get("min",   "–")
    v_max = vi.get("max",   "–")

    def plot_rows(files: list, subdir: str | None = None) -> str:
        if not files:
            return '<tr><td colspan="3" class="empty">No plots found.</td></tr>'
        rows = []
        for f in files:
            desc = describe_plot(f.stem)
            rel  = f"{subdir}/{f.name}" if subdir else f.name
            rows.append(f"""
        <tr>
          <td><code>{f.stem}</code></td>
          <td class="desc">{desc}</td>
          <td><a class="open-link" href="{rel}" target="_blank">Open &rarr;</a></td>
        </tr>""")
        return "\n".join(rows)

    def stat_rows(files: list) -> str:
        """Statistical rows include the test type as a coloured badge."""
        if not files:
            return '<tr><td colspan="4" class="empty">No plots found.</td></tr>'
        rows = []
        for f in files:
            desc = describe_plot(f.stem)
            if f.stem.startswith("mk_"):
                test_label = "Mann-Kendall"
                badge_class = "badge-mk"
            else:
                test_label = "Pettitt"
                badge_class = "badge-pettitt"
            rows.append(f"""
        <tr>
          <td><code>{f.stem}</code></td>
          <td class="{badge_class}">{test_label}</td>
          <td class="desc">{desc}</td>
          <td><a class="open-link" href="{f.name}" target="_blank">Open &rarr;</a></td>
        </tr>""")
        return "\n".join(rows)

    req_rows  = plot_rows(buckets["requests"],  "requests")
    eng_rows  = plot_rows(buckets["energy"],    "energy")
    stat_html = stat_rows(buckets["statistical"])

    # Optional dashboard / other links
    dashboard_html = ""
    if buckets["other"]:
        links = " &nbsp;|&nbsp; ".join(
            f'<a class="open-link" href="{f.name}" target="_blank">{f.stem}</a>'
            for f in buckets["other"]
        )
        dashboard_html = (
            f'<section><h2>Dashboard</h2>'
            f'<p style="padding:.5rem 0">{links}</p></section>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{label} – Benchmark Plots</title>
{SHARED_STYLE}
</head>
<body>

  <div class="topbar">
    <h1>Python Web Framework Benchmark &mdash; <span>{label}</span></h1>
  </div>

  <div class="breadcrumb">
    <a href="../../index.html">&larr; All frameworks</a>
    &nbsp;/&nbsp; {label}
    &nbsp;&mdash;&nbsp; {count} versions tested &nbsp;
    (<strong style="color:var(--text)">{v_min}</strong>
    &rarr; <strong style="color:var(--text)">{v_max}</strong>)
  </div>

  <section>
    <h2>Request Performance Plots</h2>
    <table>
      <thead><tr><th>File</th><th>Description</th><th></th></tr></thead>
      <tbody>{req_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Energy Consumption Plots</h2>
    <table>
      <thead><tr><th>File</th><th>Description</th><th></th></tr></thead>
      <tbody>{eng_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Statistical Analysis Plots</h2>
    <table>
      <thead><tr><th>File</th><th>Test</th><th>Description</th><th></th></tr></thead>
      <tbody>{stat_html}</tbody>
    </table>
  </section>

  {dashboard_html}

  <footer>
    Generated by <code>generate_index.py</code>
  </footer>

</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate HTML plot indexes.\n"
            "Run AFTER: aggregate_results.ipynb → benchmark_plots.ipynb → "
            "statistical_analysis.ipynb → pettitt_analysis.ipynb"
        )
    )
    parser.add_argument("--plots-dir", default="plots",
                        help="Path to the plots/ directory")
    parser.add_argument("--csv", default="summary_by_version.csv",
                        help="Path to summary_by_version.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print paths without writing files")
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    csv_path  = Path(args.csv)

    if not plots_dir.is_dir():
        print(f"[error] plots directory not found: {plots_dir}")
        return

    version_info = load_version_info(csv_path) if csv_path.exists() else {}
    if not version_info:
        print(f"[warn] CSV not found or empty: {csv_path} — version info will be missing")

    structure, global_files = collect_plots(plots_dir)

    print(f"[info] Found {len(structure)} framework(s): {', '.join(sorted(structure))}")
    print(f"[info] Found {len(global_files)} global analysis file(s)")

    # ── Root index ────────────────────────────────────────────────────────────
    root_html = build_root_index(plots_dir, version_info, structure, global_files)
    root_path = Path("index.html")
    if args.dry_run:
        print(f"[dry-run] Would write: {root_path}")
    else:
        root_path.write_text(root_html, encoding="utf-8")
        print(f"[ok] {root_path}")

    # ── Per-framework indexes ─────────────────────────────────────────────────
    for fw, buckets in structure.items():
        fw_dir  = plots_dir / fw
        fw_html = build_framework_index(fw, fw_dir, version_info, buckets)
        fw_path = fw_dir / "index.html"
        if args.dry_run:
            print(f"[dry-run] Would write: {fw_path}")
        else:
            fw_path.write_text(fw_html, encoding="utf-8")
            print(f"[ok] {fw_path}")

    print("\nDone. Open index.html to start browsing.")


if __name__ == "__main__":
    main()