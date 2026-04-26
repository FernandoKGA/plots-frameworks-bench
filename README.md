# plots-frameworks-bench

Aggregation and visualization of benchmark results produced by the [py-frameworks-bench]() repository runs.

This project processes raw execution results from multiple benchmark rounds, aggregates the data across framework versions, generates plots for each metric, runs statistical analyses, and produces navigable HTML files for easy exploration.

---

## Requirements

- Python 3.x
- Docker (for running JupyterLab)

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd plots-frameworks-bench
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install packaging
```

> The `packaging` library is used for correct semantic version ordering across framework versions.

### 4. Pull the JupyterLab Docker image

```bash
docker pull quay.io/jupyter/datascience-notebook
```

---

## Populating the Data

After running the benchmark in [py-frameworks-bench](), copy the resulting `results_exc_all` folder into the root of this repository:

```
plots-frameworks-bench/
└── results_exc_all/
    ├── results_exc_1/
    │   ├── results_fastapi_0_110_0/
    │   └── ...
    ├── results_exc_2/
    └── ...
```

> This step is done **manually**. No sync script is provided — bring the folder over however is most convenient (e.g. `rsync`, `scp`, or direct copy).

---

## Running the Notebooks

Start JupyterLab using the Docker image:

```bash
docker run -it --network host --rm -p 10000:8888 \
  -v "${PWD}":/home/jovyan/ \
  quay.io/jupyter/datascience-notebook
```

Then open the URL shown in the terminal (e.g. `http://127.0.0.1:10000/lab?token=...`).

### Step 1 — Aggregate the results

Open and run **`aggregate_results.ipynb`**.

Reads from `results_exc_all`, processes all benchmark rounds, and produces the aggregated data files for each framework version, including per-metric summaries used in subsequent steps.

### Step 2 — Generate the plots

Open and run **`benchmark_plots.ipynb`**.

Reads the aggregated data and generates interactive Plotly plots for each metric (e.g. response time, throughput, energy consumption, CO₂ emissions). Plots are saved as **HTML files** under `plots/{framework}/requests/` and `plots/{framework}/energy/`.

### Step 3 — Mann-Kendall trend analysis

Open and run **`statistical_analysis.ipynb`**.

Applies the **Mann-Kendall test** (with Hamed-Rao variance correction) to detect monotonic trends across framework versions for all collected metrics. Produces:

- `mk_results.csv` — full results table
- `plots/mk_heatmap.html` — combined trend heatmap
- `plots/mk_heatmap_energy_carbon.html` — energy & carbon heatmap
- `plots/mk_heatmap_throughput.html` — throughput heatmap
- `plots/mk_heatmap_latency.html` — latency heatmap
- `plots/{framework}/mk_theil_sen_trend_{framework}.html` — per-framework Theil-Sen trend line

### Step 4 — Pettitt change-point analysis

Open and run **`pettitt_analysis.ipynb`**.

Applies the **Pettitt homogeneity test** to detect structural breaks in metrics across ordered framework versions. Unlike Mann-Kendall (which asks *"is there a monotonic trend?"*), Pettitt asks *"is there a single version where the distribution shifted?"*. Produces:

- `pettitt_results.csv` — full results table
- `plots/pettitt_heatmap.html` — detection heatmap
- `plots/pettitt_heatmap_delta_pct.html` — relative change (Δ%) at each break point
- `plots/pettitt_heatmap_cp_version.html` — change-point version heatmap
- `plots/{framework}/pettitt_emission_emissions_{framework}.html` — per-framework CO₂ emissions series with change-point annotation

---

## Generating the Navigation Index

After **all notebooks have been run**, generate the navigable index pages:

```bash
python generate_index.py
```

This produces `index.html` at the project root and one `index.html` inside each `plots/{framework}/` directory, making it easy to browse all plots — including statistical analysis results — without opening files one by one.

---

## Workflow Summary

```
py-frameworks-bench          plots-frameworks-bench
──────────────────           ──────────────────────────────────────────────────────
results_exc_all/   ──copy──► aggregate_results.ipynb    (aggregate data)
                                        │
                                        ▼
                             benchmark_plots.ipynb       (generate HTML plots)
                                        │
                                        ▼
                             statistical_analysis.ipynb  (Mann-Kendall trend tests)
                                        │
                                        ▼
                             pettitt_analysis.ipynb      (Pettitt change-point tests)
                                        │
                                        ▼
                             generate_index.py           (generate index files)
```