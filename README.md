## Overview

This project is a Streamlit-based **hardware design decision tool** for comparing candidate configurations under explicit engineering constraints.

The app is intentionally **constraint-first**:

1. Load and normalize candidate data
2. Apply hard constraints to split **feasible** and **rejected** candidates
3. Compute a Pareto frontier on the feasible set
4. Recommend a configuration under a chosen decision mode
5. Run weight-based sensitivity analysis on feasible candidates only

The goal is not to behave like a generic dashboard. It is meant to support engineering decisions with explicit filtering, traceable rejection reasons, and graceful handling of incomplete data.

## Supported inputs

### `sample_tradeoff.csv`

Structured trade-off table with metrics such as:

- `accuracy`
- `latency_cycles`
- `pins`
- `slices`
- `pin_reduction`
- `slice_reduction`

### `sample_experiments.json`

Experiment-log style input. The parser maps experiment summaries into the app's canonical candidate schema.

### `sample_rtl_results.csv`

RTL latency comparison input with columns:

- `run_id`
- `latency_lut`
- `latency_mac`

The parser converts this file into two candidates:

- `design_type = LUT`
- `design_type = MAC`

Both are tagged with `source = rtl_simulation`. The app keeps missing metrics such as `accuracy`, `pins`, and `slices` as `NaN` and skips unsupported comparisons automatically.

## Key behavior

### Hard constraints

The default UI starts with these active constraints:

- `min_accuracy = 90`
- `max_latency_cycles = 1500`

Additional resource constraints can be enabled as needed. The user can also choose how missing metrics are treated:

- treat missing metrics as infeasible
- ignore missing metrics for active constraints

Constraint evaluation returns:

- `feasible_df`
- `rejected_df`

Rejected candidates include:

- `rejection_reasons`
- `rejection_reason_str`

### Pareto frontier

`analysis_engine.compute_pareto_frontier()` now:

- returns all rows
- always adds `is_pareto`
- never raises for missing objectives or partial data

A candidate is marked Pareto-optimal if no other candidate dominates it across the selected objectives and directions.

### Recommendation modes

Recommendations run on the **feasible set only**:

- Best Accuracy
- Best Latency
- Best Efficiency
- Balanced

Each mode returns:

- the selected candidate row
- a short reason string

### Weighted scoring

Balanced scoring uses min-max normalization and combines available terms only:

`score = w_acc * acc_norm - w_lat * lat_norm + w_eff * eff_norm`

Missing terms are ignored safely so partial candidate data does not crash the tool.

### Sensitivity analysis

The app can sweep one weight at a time and show:

- top score across the sweep
- top configuration selected at each sweep value
- whether the selected configuration changes

## UI flow

The app is organized as:

- `A. Data Input`
- `B. Constraints`
- `C. Feasible Candidates`
- `D. Pareto Frontier`
- `E. Recommended Configuration`
- `F. Sensitivity`

Each section is designed to degrade gracefully:

- empty dataframes show messages instead of crashing
- missing columns skip unsupported tables and plots
- no feasible candidates skips Pareto and recommendation safely
- plot failures are caught and shown as warnings

## Project structure

```text
stream/
├── app.py
├── analysis_engine.py
├── parsers.py
├── recommendation.py
├── sample_data/
│   ├── sample_tradeoff.csv
│   ├── sample_experiments.json
│   └── sample_rtl_results.csv
├── README.md
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The app is structured to avoid Python tracebacks in normal user interaction by guarding empty, partial, and unsupported cases.
- RTL-derived candidates are intentionally allowed to remain partial so latency-only evidence can still participate in the workflow where appropriate.
