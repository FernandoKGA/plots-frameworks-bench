#!/usr/bin/env python3
"""
generate_index.py
Generates index.html (root) and plots/{framework}/index.html for each framework.
Run from the root of the repository:
    python generate_index.py [--plots-dir plots] [--csv summary_by_version.csv]
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict
from packaging.version import Version

# ── Configuration ────────────────────────────────────────────────────────────

GITHUB_URL = "https://github.com/FernandoKGA/plots-frameworks-bench"

FRAMEWORK_LABELS = {
    "aiohttp":   "aiohttp",
    "baize":     "Baize",
    "django":    "Django",
    "falcon":    "Falcon",
    "fastapi":   "FastAPI",
    "muffin":    "Muffin",
    "quart":     "Quart",
    "sanic":     "Sanic",
    "starlette": "Starlette",
    "tornado":   "Tornado",
}

# Tokens used to build human-readable plot descriptions
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
    # energy metrics
    "emission_duration":        "Measurement Duration",
    "emission_energy_consumed": "Total Energy Consumed",
    "emission_cpu_energy":      "CPU Energy",
    "emission_ram_energy":      "RAM Energy",
    "emission_cpu_power":       "CPU Power",
    "emission_ram_power":       "RAM Power",
    "emission_emissions":       "CO₂ Emissions",
    "emission_emissions_rate":  "CO₂ Emission Rate",
}

PLOT_TYPE_LABELS = {
    "line": "Line Chart",
    "box":  "Box Plot",
    "bar":  "Bar Chart",
    "scatter": "Scatter Plot",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def describe_plot(stem: str) -> str:
    """Convert a file stem like 'api_rt_p95' or 'box_emission_cpu_power' into a readable description."""
    parts = stem.split("_")

    # Check for plot-type prefix (line_, box_, etc.)
    plot_type = ""
    if parts[0] in PLOT_TYPE_LABELS:
        plot_type = PLOT_TYPE_LABELS[parts[0]]
        parts = parts[1:]

    # Check for endpoint prefix (api_, html_, upload_)
    endpoint = ""
    if parts and parts[0] in ENDPOINT_LABELS:
        endpoint = ENDPOINT_LABELS[parts[0]]
        parts = parts[1:]

    # Try to match the remaining tokens against known metric labels
    metric_key = "_".join(parts)
    metric = METRIC_LABELS.get(metric_key)

    # Fallback: humanise unknown metric keys
    if not metric:
        metric = metric_key.replace("_", " ").title()

    pieces = [p for p in [endpoint, metric, plot_type] if p]
    return " – ".join(pieces) if pieces else stem.replace("_", " ").title()


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


def collect_plots(plots_dir: Path) -> dict:
    """Returns {framework: {"requests": [...], "energy": [...], "other": [...]}}."""
    structure = {}
    for fw_path in sorted(plots_dir.iterdir()):
        if not fw_path.is_dir():
            continue
        fw = fw_path.name
        buckets = {"requests": [], "energy": [], "other": []}
        for category in ["requests", "energy"]:
            cat_path = fw_path / category
            if cat_path.is_dir():
                files = sorted(
                    [f for f in cat_path.iterdir() if f.suffix == ".html"],
                    key=lambda p: p.stem,
                )
                buckets[category] = files
        # dashboard or anything else at the framework root
        for f in sorted(fw_path.glob("*.html")):
            buckets["other"].append(f)
        structure[fw] = buckets
    return structure


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

    /* ── top bar ── */
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

    .gh-link {
      display: inline-flex;
      align-items: center;
      gap: .45rem;
      padding: .35rem .8rem;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      color: var(--muted);
      text-decoration: none;
      font-size: .8rem;
      transition: color .15s, border-color .15s;
    }
    .gh-link:hover { color: var(--text); border-color: var(--accent); }
    .gh-link svg { fill: currentColor; }

    /* ── breadcrumb ── */
    .breadcrumb {
      font-size: .8rem;
      color: var(--muted);
      margin-bottom: 1.5rem;
    }
    .breadcrumb a { color: var(--accent); text-decoration: none; }
    .breadcrumb a:hover { text-decoration: underline; }

    /* ── section heading ── */
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

    /* ── tables ── */
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: .85rem;
    }
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
    .tag.green { color: var(--accent2); }

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

    a.fw-link {
      color: var(--accent);
      text-decoration: none;
      font-weight: 500;
    }
    a.fw-link:hover { text-decoration: underline; }

    .desc { color: var(--muted); }

    /* ── empty state ── */
    .empty { padding: 1.5rem; text-align: center; color: var(--muted); font-size: .85rem; }

    /* ── footer ── */
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

GH_ICON = """<svg width="14" height="14" viewBox="0 0 16 16">
  <path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 005.47 7.59c.4.07.55-.17.55-.38
  0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
  -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87
  2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
  0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21
  2.2.82a7.65 7.65 0 012-.27c.68 0 1.36.09 2 .27 1.53-1.04
  2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82
  2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01
  1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8z"/>
</svg>"""


# ── Page generators ───────────────────────────────────────────────────────────

def build_root_index(plots_dir: Path, version_info: dict, structure: dict) -> str:
    rows = []
    for fw in sorted(structure.keys()):
        label = FRAMEWORK_LABELS.get(fw, fw.title())
        vi = version_info.get(fw, {})
        count = vi.get("count", "–")
        v_min = vi.get("min", "–")
        v_max = vi.get("max", "–")
        link_href = f"{plots_dir.name}/{fw}/index.html"
        req_count = len(structure[fw]["requests"])
        eng_count = len(structure[fw]["energy"])
        rows.append(f"""
        <tr>
          <td><a class="fw-link" href="{link_href}">{label}</a></td>
          <td><span class="tag">{count} versions</span></td>
          <td class="range"><strong>{v_min}</strong> &rarr; <strong>{v_max}</strong></td>
          <td><span class="tag green">{req_count}</span></td>
          <td><span class="tag green">{eng_count}</span></td>
          <td><a class="open-link" href="{link_href}">View plots &rarr;</a></td>
        </tr>""")

    rows_html = "\n".join(rows) if rows else '<tr><td colspan="6" class="empty">No frameworks found.</td></tr>'

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
    <a class="gh-link" href="{GITHUB_URL}" target="_blank" rel="noopener">
      {GH_ICON} FernandoKGA/plots-frameworks-bench
    </a>
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
          <th></th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </section>

  <footer>
    Generated by <code>generate_index.py</code> &mdash;
    <a href="{GITHUB_URL}" target="_blank" rel="noopener" style="color:var(--accent)">GitHub repository</a>
  </footer>

</body>
</html>"""


def build_framework_index(fw: str, fw_dir: Path, version_info: dict, buckets: dict) -> str:
    label = FRAMEWORK_LABELS.get(fw, fw.title())
    vi = version_info.get(fw, {})
    count = vi.get("count", "–")
    v_min = vi.get("min", "–")
    v_max = vi.get("max", "–")

    def plot_rows(files: list, subdir: str) -> str:
        if not files:
            return '<tr><td colspan="3" class="empty">No plots found.</td></tr>'
        rows = []
        for f in files:
            desc = describe_plot(f.stem)
            rel = f"{subdir}/{f.name}"
            rows.append(f"""
        <tr>
          <td><code>{f.stem}</code></td>
          <td class="desc">{desc}</td>
          <td><a class="open-link" href="{rel}" target="_blank">Open &rarr;</a></td>
        </tr>""")
        return "\n".join(rows)

    req_rows = plot_rows(buckets["requests"], "requests")
    eng_rows = plot_rows(buckets["energy"], "energy")

    # optional dashboard link
    dashboard_html = ""
    if buckets["other"]:
        links = " &nbsp;|&nbsp; ".join(
            f'<a class="open-link" href="{f.name}" target="_blank">{f.stem}</a>'
            for f in buckets["other"]
        )
        dashboard_html = f'<section><h2>Dashboard</h2><p style="padding:.5rem 0">{links}</p></section>'

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
    <a class="gh-link" href="{GITHUB_URL}" target="_blank" rel="noopener">
      {GH_ICON} FernandoKGA/plots-frameworks-bench
    </a>
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
      <thead>
        <tr>
          <th>File</th>
          <th>Description</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {req_rows}
      </tbody>
    </table>
  </section>

  <section>
    <h2>Energy Consumption Plots</h2>
    <table>
      <thead>
        <tr>
          <th>File</th>
          <th>Description</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {eng_rows}
      </tbody>
    </table>
  </section>

  {dashboard_html}

  <footer>
    Generated by <code>generate_index.py</code> &mdash;
    <a href="{GITHUB_URL}" target="_blank" rel="noopener" style="color:var(--accent)">GitHub repository</a>
  </footer>

</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate HTML plot indexes.")
    parser.add_argument("--plots-dir", default="plots", help="Path to the plots/ directory")
    parser.add_argument("--csv", default="summary_by_version.csv", help="Path to summary_by_version.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print paths without writing files")
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    csv_path  = Path(args.csv)

    if not plots_dir.is_dir():
        print(f"[error] plots directory not found: {plots_dir}")
        return

    version_info = load_version_info(csv_path) if csv_path.exists() else {}
    if not version_info:
        print(f"[warn] CSV not found or empty: {csv_path} — version info will be missing")

    structure = collect_plots(plots_dir)
    print(f"[info] Found {len(structure)} framework(s): {', '.join(sorted(structure))}")

    # Root index
    root_html = build_root_index(plots_dir, version_info, structure)
    root_path = Path("index.html")
    if args.dry_run:
        print(f"[dry-run] Would write: {root_path}")
    else:
        root_path.write_text(root_html, encoding="utf-8")
        print(f"[ok] {root_path}")

    # Per-framework indexes
    for fw, buckets in structure.items():
        fw_dir   = plots_dir / fw
        fw_html  = build_framework_index(fw, fw_dir, version_info, buckets)
        fw_path  = fw_dir / "index.html"
        if args.dry_run:
            print(f"[dry-run] Would write: {fw_path}")
        else:
            fw_path.write_text(fw_html, encoding="utf-8")
            print(f"[ok] {fw_path}")

    print("\nDone. Open index.html to start browsing.")


if __name__ == "__main__":
    main()