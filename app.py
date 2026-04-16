from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis_engine import (
    Constraints,
    apply_constraints,
    compute_pareto_frontier,
    compute_weighted_scores,
    summarize_rejections,
    sweep_single_weight,
)
from parsers import load_and_normalize_data
from recommendation import (
    recommend_best_accuracy,
    recommend_best_balanced,
    recommend_best_efficiency,
    recommend_best_latency,
)


APP_TITLE = "Constraint-First Hardware Design Decision Tool"
SAMPLE_DIR = Path("sample_data")
DEFAULT_MIN_ACCURACY = 90.0
DEFAULT_MAX_LATENCY = 1500.0


def _metric_str(v: object, unit: str = "", digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    try:
        x = float(v)
        return f"{x:.{digits}f}{unit}"
    except Exception:
        text = str(v)
        return text if not unit else f"{text}{unit}"


def _existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _ensure_pareto_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "is_pareto" not in out.columns:
        out["is_pareto"] = False
    out["is_pareto"] = out["is_pareto"].fillna(False).astype(bool)
    return out


def _safe_sort(df: pd.DataFrame, column: str, ascending: bool = False) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    return df.sort_values(column, ascending=ascending, na_position="last")


def _maybe_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    pareto_only: bool,
) -> Tuple[bool, Optional[object]]:
    if df.empty or x not in df.columns or y not in df.columns:
        return False, None

    d = _ensure_pareto_flag(df)
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y])
    if pareto_only:
        d = d[d["is_pareto"]]
    if d.empty:
        return False, None

    hover = _existing_columns(
        d,
        ["config_id", "design_type", "source", "accuracy", "latency_cycles", "pins", "slices"],
    )

    try:
        fig = px.scatter(
            d,
            x=x,
            y=y,
            color="is_pareto" if "is_pareto" in d.columns else None,
            symbol="source" if "source" in d.columns else None,
            hover_data=hover,
            title=title,
            color_discrete_map={True: "#0f766e", False: "#9ca3af"},
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="black")))
        fig.update_layout(
            height=440,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        return True, fig
    except Exception:
        return False, None


def _render_recommendation(rec: Optional[dict], reason: str) -> None:
    with st.container(border=True):
        st.caption(reason)
        if rec is None:
            st.write("No recommendation available for the current feasible set.")
            return

        summary = pd.DataFrame(
            [
                {"metric": "config_id", "value": rec.get("config_id", "N/A")},
                {"metric": "design_type", "value": rec.get("design_type", "N/A")},
                {"metric": "source", "value": rec.get("source", "N/A")},
                {"metric": "accuracy", "value": _metric_str(rec.get("accuracy"))},
                {"metric": "latency_cycles", "value": _metric_str(rec.get("latency_cycles"))},
                {"metric": "pins", "value": _metric_str(rec.get("pins"))},
                {"metric": "slices", "value": _metric_str(rec.get("slices"))},
                {"metric": "pin_reduction", "value": _metric_str(rec.get("pin_reduction"), unit="%")},
                {"metric": "slice_reduction", "value": _metric_str(rec.get("slice_reduction"), unit="%")},
                {"metric": "weighted_score", "value": _metric_str(rec.get("weighted_score"))},
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)


st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title(APP_TITLE)
st.caption("Load candidate designs, enforce hard constraints, then compare only the feasible design space.")

st.subheader("A. Data Input")
left, right = st.columns([1.1, 1.9], vertical_alignment="top")
with left:
    mode = st.radio("Input mode", ["Use sample dataset", "Upload file"])
    sample_choice = None
    uploaded = None

    if mode == "Use sample dataset":
        sample_choice = st.selectbox(
            "Sample dataset",
            [
                "sample_tradeoff.csv",
                "sample_experiments.json",
                "sample_rtl_results.csv",
            ],
        )
    else:
        uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])

df = pd.DataFrame()
detected_label = "N/A"
warnings: list[str] = []

with right:
    try:
        if mode == "Use sample dataset":
            sample_path = SAMPLE_DIR / str(sample_choice)
            df, detected, warnings = load_and_normalize_data(sample_path)
        else:
            if uploaded is None:
                st.info("Upload a CSV or JSON file to evaluate candidates.")
                st.stop()
            df, detected, warnings = load_and_normalize_data(uploaded)

        detected_label = f"{detected.kind}: {detected.detail}"
    except Exception as exc:
        st.error(f"Failed to load input: {exc}")
        st.stop()

    k1, k2, k3 = st.columns(3)
    k1.metric("Candidates", int(len(df)))
    k2.metric("Detected input", detected.kind if "detected" in locals() else "N/A")
    k3.metric("Schema columns", int(len(df.columns)))
    st.caption(detected_label)

    if warnings:
        with st.expander("Input warnings", expanded=False):
            for warning in warnings:
                st.warning(warning)

    with st.expander("Normalized candidates", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

if df.empty:
    st.warning("No candidate rows were loaded. Downstream sections will remain empty until valid input is provided.")

st.subheader("B. Constraints")
constraint_cols = st.columns(3)
with constraint_cols[0]:
    use_min_acc = st.checkbox("Minimum accuracy", value=True)
    min_acc = st.number_input(
        "min_accuracy",
        min_value=0.0,
        max_value=100.0,
        value=DEFAULT_MIN_ACCURACY,
        step=0.1,
        disabled=not use_min_acc,
    )
    use_min_pin_red = st.checkbox("Minimum pin_reduction", value=False)
    min_pin_red = st.number_input(
        "min_pin_reduction (%)",
        value=0.0,
        step=0.5,
        disabled=not use_min_pin_red,
    )

with constraint_cols[1]:
    use_max_lat = st.checkbox("Maximum latency_cycles", value=True)
    max_lat = st.number_input(
        "max_latency_cycles",
        min_value=0.0,
        value=DEFAULT_MAX_LATENCY,
        step=10.0,
        disabled=not use_max_lat,
    )
    use_min_slice_red = st.checkbox("Minimum slice_reduction", value=False)
    min_slice_red = st.number_input(
        "min_slice_reduction (%)",
        value=0.0,
        step=0.5,
        disabled=not use_min_slice_red,
    )

with constraint_cols[2]:
    use_max_pins = st.checkbox("Maximum pins", value=False)
    max_pins = st.number_input(
        "max_pins",
        min_value=0.0,
        value=320.0,
        step=1.0,
        disabled=not use_max_pins,
    )
    use_max_slices = st.checkbox("Maximum slices", value=False)
    max_slices = st.number_input(
        "max_slices",
        min_value=0.0,
        value=1800.0,
        step=10.0,
        disabled=not use_max_slices,
    )

missing_policy = st.selectbox(
    "Missing metric policy",
    [
        "Treat missing metrics as infeasible",
        "Ignore missing metrics for active constraints",
    ],
    index=0,
)

constraints = Constraints(
    min_accuracy=min_acc if use_min_acc else None,
    max_latency_cycles=max_lat if use_max_lat else None,
    max_pins=max_pins if use_max_pins else None,
    max_slices=max_slices if use_max_slices else None,
    min_pin_reduction=min_pin_red if use_min_pin_red else None,
    min_slice_reduction=min_slice_red if use_min_slice_red else None,
    treat_nan_as_violation=missing_policy == "Treat missing metrics as infeasible",
)

try:
    feasible_df, rejected_df = apply_constraints(df, constraints)
    rejection_summary = summarize_rejections(rejected_df)
except Exception as exc:
    st.warning(f"Constraint evaluation fell back to empty results: {exc}")
    feasible_df = pd.DataFrame(columns=df.columns)
    rejected_df = pd.DataFrame(columns=list(df.columns) + ["rejection_reasons", "rejection_reason_str", "is_feasible"])
    rejection_summary = pd.DataFrame({"reason": [], "count": []})

st.subheader("C. Feasible Candidates")
summary_cols = st.columns(3)
summary_cols[0].metric("Total candidates", int(len(df)))
summary_cols[1].metric("Feasible", int(len(feasible_df)))
summary_cols[2].metric("Rejected", int(len(rejected_df)))

feasible_preview_cols = _existing_columns(
    feasible_df,
    [
        "config_id",
        "design_type",
        "accuracy",
        "latency_cycles",
        "pins",
        "slices",
        "pin_reduction",
        "slice_reduction",
        "source",
    ],
)

if feasible_df.empty:
    st.warning("No candidates satisfy the active constraints. Pareto analysis and recommendations are skipped until a feasible set exists.")
else:
    st.dataframe(feasible_df[feasible_preview_cols], use_container_width=True, hide_index=True)

detail_cols = st.columns([1.1, 1.4], vertical_alignment="top")
with detail_cols[0]:
    st.markdown("**Rejection Summary**")
    if rejection_summary.empty:
        st.caption("No rejection reasons recorded.")
    else:
        st.dataframe(rejection_summary, use_container_width=True, hide_index=True)

with detail_cols[1]:
    st.markdown("**Rejected Candidates**")
    rejected_preview_cols = _existing_columns(
        rejected_df,
        ["config_id", "design_type", "source", "rejection_reason_str"],
    )
    if rejected_df.empty:
        st.caption("No rejected candidates.")
    else:
        st.dataframe(rejected_df[rejected_preview_cols], use_container_width=True, hide_index=True)

st.subheader("D. Pareto Frontier")
pareto_df = _ensure_pareto_flag(feasible_df)

if feasible_df.empty:
    st.info("Pareto frontier skipped because there are no feasible candidates.")
else:
    objective_group = st.selectbox(
        "Objective group",
        [
            "maximize accuracy, minimize latency_cycles, minimize slices",
            "maximize accuracy, minimize pins",
            "maximize accuracy, minimize resource_usage",
        ],
    )
    show_pareto_only = st.checkbox("Show Pareto only", value=False)

    if objective_group == "maximize accuracy, minimize latency_cycles, minimize slices":
        objectives = ["accuracy", "latency_cycles", "slices"]
        directions = ["maximize", "minimize", "minimize"]
    elif objective_group == "maximize accuracy, minimize pins":
        objectives = ["accuracy", "pins"]
        directions = ["maximize", "minimize"]
    else:
        pareto_df = pareto_df.copy()
        pins = pd.to_numeric(pareto_df["pins"], errors="coerce") if "pins" in pareto_df.columns else pd.Series(np.nan, index=pareto_df.index)
        slices = pd.to_numeric(pareto_df["slices"], errors="coerce") if "slices" in pareto_df.columns else pd.Series(np.nan, index=pareto_df.index)
        pareto_df["resource_usage"] = pd.concat([pins, slices], axis=1).sum(axis=1, min_count=1)
        objectives = ["accuracy", "resource_usage"]
        directions = ["maximize", "minimize"]

    try:
        pareto_df = compute_pareto_frontier(pareto_df, objectives=objectives, directions=directions, flag_column="is_pareto")
    except Exception as exc:
        pareto_df = _ensure_pareto_flag(feasible_df)
        st.warning(f"Pareto computation was skipped safely: {exc}")

    pareto_df = _ensure_pareto_flag(pareto_df)
    pareto_count = int(pareto_df["is_pareto"].sum()) if "is_pareto" in pareto_df.columns else 0
    st.caption(f"Pareto-optimal feasible candidates: {pareto_count}")

    pareto_table = pareto_df[pareto_df["is_pareto"]] if show_pareto_only else pareto_df
    pareto_table = _ensure_pareto_flag(pareto_table)
    if "is_pareto" in pareto_table.columns:
        pareto_table = _safe_sort(pareto_table, "is_pareto", ascending=False)

    pareto_cols = _existing_columns(
        pareto_table,
        [
            "is_pareto",
            "config_id",
            "design_type",
            "accuracy",
            "latency_cycles",
            "pins",
            "slices",
            "pin_reduction",
            "slice_reduction",
            "source",
        ],
    )

    if pareto_table.empty:
        st.info("No rows to display for the current Pareto filter.")
    else:
        st.dataframe(pareto_table[pareto_cols], use_container_width=True, hide_index=True)

    plot_cols = st.columns(2)
    with plot_cols[0]:
        ok, fig = _maybe_scatter(pareto_df, "accuracy", "latency_cycles", "Accuracy vs Latency", show_pareto_only)
        if ok and fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Skipped: `accuracy` and `latency_cycles` are required for this plot.")

    with plot_cols[1]:
        ok, fig = _maybe_scatter(pareto_df, "accuracy", "slices", "Accuracy vs Slices", show_pareto_only)
        if ok and fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Skipped: `accuracy` and `slices` are required for this plot.")

    ok, fig = _maybe_scatter(pareto_df, "pin_reduction", "accuracy", "Pin Reduction vs Accuracy", show_pareto_only)
    if ok and fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Skipped: `pin_reduction` and `accuracy` are required for this plot.")

st.subheader("E. Recommended Configuration")
if feasible_df.empty:
    st.info("Recommendation skipped because the current constraints leave no feasible candidates.")
else:
    recommendation_mode = st.selectbox(
        "Recommendation mode",
        ["Best Accuracy", "Best Latency", "Best Efficiency", "Balanced"],
    )

    if recommendation_mode == "Best Accuracy":
        recommendation, reason = recommend_best_accuracy(feasible_df)
    elif recommendation_mode == "Best Latency":
        recommendation, reason = recommend_best_latency(feasible_df)
    elif recommendation_mode == "Best Efficiency":
        recommendation, reason = recommend_best_efficiency(feasible_df)
    else:
        recommendation, reason = recommend_best_balanced(feasible_df, 1.0, 1.0, 1.0)

    _render_recommendation(recommendation, reason)

st.subheader("F. Sensitivity")
if feasible_df.empty:
    st.info("Sensitivity analysis skipped because the feasible set is empty.")
else:
    weight_cols = st.columns(3)
    with weight_cols[0]:
        weight_accuracy = st.slider("Accuracy weight", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    with weight_cols[1]:
        weight_latency = st.slider("Latency weight", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    with weight_cols[2]:
        weight_efficiency = st.slider("Efficiency weight", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    try:
        scored_df, score_meta = compute_weighted_scores(
            feasible_df,
            weight_accuracy,
            weight_latency,
            weight_efficiency,
            score_column="weighted_score",
        )
    except Exception as exc:
        st.warning(f"Weighted scoring was skipped safely: {exc}")
        scored_df = feasible_df.copy()
        scored_df["weighted_score"] = np.nan
        score_meta = {"note": "No weighted score available."}

    if score_meta.get("note"):
        st.caption(score_meta["note"])
    if score_meta.get("weights_used"):
        st.caption(f"Weights used: {score_meta['weights_used']}")

    ranked_cols = _existing_columns(
        scored_df,
        [
            "config_id",
            "design_type",
            "weighted_score",
            "accuracy",
            "latency_cycles",
            "pin_reduction",
            "slice_reduction",
            "source",
        ],
    )

    if "weighted_score" in scored_df.columns and scored_df["weighted_score"].notna().any():
        ranked_df = scored_df.dropna(subset=["weighted_score"]).sort_values("weighted_score", ascending=False)
        st.dataframe(ranked_df[ranked_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No feasible candidates could be scored with the currently available metrics.")

    sweep_metric = st.selectbox("Sweep weight", ["accuracy", "latency", "efficiency"])
    try:
        sweep_df = sweep_single_weight(
            feasible_df,
            sweep=sweep_metric,
            values=np.linspace(0.0, 5.0, 11),
            base_weights={
                "accuracy": weight_accuracy,
                "latency": weight_latency,
                "efficiency": weight_efficiency,
            },
        )
    except Exception as exc:
        st.warning(f"Sensitivity sweep was skipped safely: {exc}")
        sweep_df = pd.DataFrame({"sweep": [], "top_config_id": [], "top_score": [], "selection_changed": []})

    if not sweep_df.empty and sweep_df["top_score"].notna().any():
        try:
            sweep_fig = px.line(
                sweep_df,
                x="sweep",
                y="top_score",
                markers=True,
                hover_data=["top_config_id", "selection_changed"],
                title="Top score across weight sweep",
            )
            sweep_fig.update_layout(height=420, margin=dict(l=40, r=40, t=60, b=40))
            st.plotly_chart(sweep_fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"Sensitivity plot was skipped safely: {exc}")

        st.dataframe(sweep_df, use_container_width=True, hide_index=True)
    else:
        st.info("Sweep produced no scorable candidate changes for the current feasible set.")

with st.expander("Debug Tables", expanded=False):
    st.markdown("**Feasible candidates**")
    st.dataframe(feasible_df, use_container_width=True, hide_index=True)
    st.markdown("**Pareto working set**")
    st.dataframe(pareto_df, use_container_width=True, hide_index=True)
    st.markdown("**Rejected candidates**")
    st.dataframe(rejected_df, use_container_width=True, hide_index=True)
