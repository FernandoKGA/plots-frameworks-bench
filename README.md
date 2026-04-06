# plots-frameworks-bench

Aggregation and visualization of benchmark results produced by the [py-frameworks-bench]() repository runs.

This project processes raw execution results from multiple benchmark rounds, aggregates the data across framework versions, generates plots for each metric, and produces navigable HTML files for easy exploration.

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

This notebook reads from `results_exc_all`, processes all benchmark rounds, and produces the aggregated data files for each framework version, including per-metric summaries used in the next step.

### Step 2 — Generate the plots

Open and run **`benchmark_plots.ipynb`**.

This notebook reads the aggregated data and generates plots for each metric (e.g. carbon emissions, total requests, emissions rate). The plots are saved as **HTML files** in the project directory.

---

## Generating the Navigation Index

After the HTML files have been generated, run the index generator script to create navigable index pages:

```bash
python generate_index.py
```

This produces index files that make it easy to browse through all generated plot files without having to open them one by one.

---

## Workflow Summary

```
py-frameworks-bench          plots-frameworks-bench
──────────────────           ──────────────────────────────────────────────
results_exc_all/   ──copy──► aggregate_results.ipynb  (aggregate data)
                                       │
                                       ▼
                             benchmark_plots.ipynb     (generate HTML plots)
                                       │
                                       ▼
                             generate_index.py         (generate index files)
```