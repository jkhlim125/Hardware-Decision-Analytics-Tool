## Overview

This project is a **hardware-aware design trade-off analysis and decision-support tool** built in Streamlit.
It helps engineers choose among design candidates (experiment runs, system-level configurations, and RTL-derived variants)
under explicit constraints and priorities.

The emphasis is on **engineering logic**:
- Filter by **hard feasibility constraints** first (accuracy/latency/resource budgets)
- Identify **Pareto-optimal** candidates (multi-objective)
- Recommend designs under clear, explainable rules
- Integrate **RTL simulation outputs** as comparable design candidates (even when other metrics are missing)

## Problem statement

Hardware deployment decisions are rarely “maximize one metric”.
Real constraints (latency budget, pin/slice budgets, minimum accuracy) define what is feasible, and only then do
engineering priorities determine which feasible point is best.

## Why constraint-aware filtering matters

A single weighted score can hide infeasible designs.
This tool separates the workflow:
- **Feasibility**: “Can we ship this design under the current constraints?”
- **Preference**: “Among feasible candidates, what should we choose and why?”

## Key features

- **Unified input model**: JSON experiment logs, generic CSV trade-off logs, and RTL latency comparison CSVs
- **Hard constraint filtering**: returns feasible + rejected sets with explicit rejection reasons
- **Pareto frontier**: generic Pareto-optimal marking for selectable objectives/directions
- **Recommendation engine** (feasible set only):
  - Best accuracy
  - Best efficiency
  - Best latency-aware
  - Best balanced (simple normalized composite)
- **Sensitivity / weight analysis**: adjust preference weights after constraints and see how the top design changes
- **Robustness**: missing columns/NaNs/malformed uploads degrade gracefully without crashing the app

## Input formats

### 1) JSON experiment logs

The parser supports the project’s existing JSON structure and maps it into a canonical design-candidate table.
If a baseline is detected, `accuracy_drop` is computed vs baseline accuracy.

Sample: `sample_data/sample_experiments.json`

### 2) Generic CSV trade-off logs

Any CSV with some subset of fields like `accuracy`, `latency_cycles`, `pins`, `slices`, `pin_reduction`, etc.
Columns are mapped conservatively into the canonical schema.

Sample: `sample_data/sample_tradeoff.csv`

### 3) RTL latency comparison CSV

Expected columns:
- `run_id`
- `latency_lut`
- `latency_mac`

The parser converts this into two candidates:
- `config_id=rtl_lut`, `design_type=LUT`
- `config_id=rtl_mac`, `design_type=MAC`

Each candidate contains average `latency_cycles`, a `run_count`, and `source=rtl_simulation`.
Other metrics remain `NaN` (and the tool still supports latency-oriented comparisons).

Sample: `sample_data/sample_rtl_results.csv`

## Decision flow (what the UI does)

1. **Load & normalize** candidates into a canonical dataframe (`parsers.load_and_normalize_data`)
2. **Apply hard constraints** to produce:
   - feasible candidates
   - rejected candidates + rejection reasons
3. **Compute Pareto frontier** over selected objectives (`analysis_engine.compute_pareto_frontier`)
4. **Generate recommendations** from the feasible set (`recommendation.py`)
5. **Sensitivity analysis**: adjust weights and sweep a parameter to observe recommendation changes

## Pareto frontier (definition)

A feasible candidate is **Pareto-optimal** if **no other feasible candidate** is:
- at least as good in all selected objectives, and
- strictly better in at least one objective,
given the selected maximize/minimize directions.

## Recommendation logic (high level)

All recommendation modes operate on the **feasible set only**:
- **Best accuracy**: highest `accuracy`
- **Best efficiency**: strongest hardware efficiency proxy using available reduction/efficiency metrics
- **Best latency-aware**: lowest `latency_cycles` (tie-break using accuracy/resources when available)
- **Best balanced**: min-max normalized weighted score using available metrics only

## RTL integration

RTL-derived CSV outputs are treated as first-class candidates via `parsers.parse_rtl_latency_csv`.
This supports a workflow where detailed simulation results inform system-level design selection,
even when not all metrics are available in the RTL output yet.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```text
project_root/
├── app.py                     # Streamlit UI only
├── analysis_engine.py         # constraints, Pareto, scoring, sensitivity helpers
├── parsers.py                 # JSON/CSV loaders + schema normalization (incl. RTL CSV)
├── recommendation.py          # recommendation modes (feasible-set only)
├── sample_data/
│   ├── sample_experiments.json
│   ├── sample_tradeoff.csv
│   └── sample_rtl_results.csv
├── results/                   # optional outputs
├── README.md
└── requirements.txt
```

