from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from analysis_engine import compute_efficiency_proxy, compute_weighted_scores, ensure_numeric_columns


def _as_row_dict(row: pd.Series) -> dict:
    d = row.to_dict()
    for k, v in list(d.items()):
        if isinstance(v, (np.floating, np.integer)):
            d[k] = v.item()
    return d


def _empty(msg: str) -> dict[str, Any]:
    return {"row": None, "explanation": msg}


def _metric_value(row: pd.Series | dict | None, metric: str) -> float:
    if row is None:
        return np.nan
    try:
        value = row.get(metric) if hasattr(row, "get") else np.nan
        return float(value)
    except Exception:
        return np.nan


def _config_id(row: pd.Series | dict | None) -> str:
    if row is None:
        return "N/A"
    value = row.get("config_id") if hasattr(row, "get") else None
    return str(value) if value is not None else "N/A"


def _format_delta(
    value: float,
    unit: str = "",
    digits: int = 2,
    signed: bool = True,
) -> str:
    if not np.isfinite(value):
        return "N/A"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:.{digits}f}{unit}"


def _next_best_gap(
    d: pd.DataFrame,
    metric: str,
    ascending: bool,
    selected_idx: object,
) -> tuple[Optional[pd.Series], float]:
    ranked = d.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
    ranked = ranked[ranked.index != selected_idx]
    if ranked.empty:
        return None, np.nan
    next_best = ranked.iloc[0]
    return next_best, _metric_value(d.loc[selected_idx], metric) - _metric_value(next_best, metric)


def _best_by_metric(d: pd.DataFrame, metric: str, ascending: bool) -> Optional[pd.Series]:
    if metric not in d.columns:
        return None
    ranked = d.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
    return ranked.iloc[0] if not ranked.empty else None


def _best_efficiency_row(d: pd.DataFrame) -> Optional[pd.Series]:
    eff = compute_efficiency_proxy(d)
    temp = d.copy()
    temp["_eff_proxy"] = eff
    ranked = temp.dropna(subset=["_eff_proxy"]).sort_values(["_eff_proxy"], ascending=[False])
    return ranked.iloc[0] if not ranked.empty else None


def _explain_accuracy_choice(d: pd.DataFrame, best: pd.Series) -> str:
    lines = [f"Selected `{_config_id(best)}`:", f"- Highest feasible accuracy: {_metric_value(best, 'accuracy'):.2f}%"]

    next_best, gap = _next_best_gap(d, "accuracy", ascending=False, selected_idx=best.name)
    if next_best is not None:
        lines.append(f"- Beats `{_config_id(next_best)}` by {_format_delta(gap, unit='%', digits=2)} on accuracy")

    best_latency = _best_by_metric(d, "latency_cycles", ascending=True)
    if best_latency is not None:
        latency_gap = _metric_value(best, "latency_cycles") - _metric_value(best_latency, "latency_cycles")
        if np.isfinite(latency_gap) and latency_gap != 0:
            lines.append(f"- Sacrifice: {_format_delta(latency_gap, unit=' cycles', digits=0)} versus best latency `{_config_id(best_latency)}`")

    lines.append("-> Accuracy-prioritized trade-off")
    return "\n".join(lines)


def _explain_latency_choice(d: pd.DataFrame, best: pd.Series) -> str:
    lines = [f"Selected `{_config_id(best)}`:", f"- Lowest feasible latency: {_metric_value(best, 'latency_cycles'):.0f} cycles"]

    next_best, gap = _next_best_gap(d, "latency_cycles", ascending=True, selected_idx=best.name)
    if next_best is not None and np.isfinite(gap):
        lines.append(f"- Beats `{_config_id(next_best)}` by {_format_delta(-gap, unit=' cycles', digits=0)} on latency")

    best_accuracy = _best_by_metric(d, "accuracy", ascending=False)
    if best_accuracy is not None:
        accuracy_gap = _metric_value(best_accuracy, "accuracy") - _metric_value(best, "accuracy")
        if np.isfinite(accuracy_gap) and accuracy_gap != 0:
            lines.append(f"- Sacrifice: {_format_delta(-accuracy_gap, unit='%', digits=2)} versus best accuracy `{_config_id(best_accuracy)}`")

    lines.append("-> Latency-prioritized trade-off")
    return "\n".join(lines)


def _explain_efficiency_choice(d: pd.DataFrame, best: pd.Series) -> str:
    temp = d.copy()
    temp["_eff_proxy"] = compute_efficiency_proxy(temp)
    best_eff = _metric_value(best, "_eff_proxy")
    lines = [f"Selected `{_config_id(best)}`:", f"- Highest feasible efficiency proxy: {best_eff:.2f}"]

    next_best, gap = _next_best_gap(temp, "_eff_proxy", ascending=False, selected_idx=best.name)
    if next_best is not None and np.isfinite(gap):
        lines.append(f"- Beats `{_config_id(next_best)}` by {_format_delta(gap, digits=2)} on combined pin/slice reduction")

    best_accuracy = _best_by_metric(d, "accuracy", ascending=False)
    if best_accuracy is not None:
        accuracy_gap = _metric_value(best, "accuracy") - _metric_value(best_accuracy, "accuracy")
        if np.isfinite(accuracy_gap) and accuracy_gap != 0:
            lines.append(f"- Sacrifice: {_format_delta(accuracy_gap, unit='%', digits=2)} versus best accuracy `{_config_id(best_accuracy)}`")

    lines.append("-> Hardware-efficient trade-off")
    return "\n".join(lines)


def _explain_balanced_choice(
    scored: pd.DataFrame,
    best: pd.Series,
    meta: dict[str, str],
) -> str:
    lines = [f"Selected `{_config_id(best)}`:", f"- Highest balanced score: {_metric_value(best, 'weighted_score'):.3f}"]

    next_best, gap = _next_best_gap(scored, "weighted_score", ascending=False, selected_idx=best.name)
    if next_best is not None and np.isfinite(gap):
        lines.append(f"- Beats `{_config_id(next_best)}` by {_format_delta(gap, digits=3)} on weighted score")

    best_accuracy = _best_by_metric(scored, "accuracy", ascending=False)
    if best_accuracy is not None:
        accuracy_gap = _metric_value(best, "accuracy") - _metric_value(best_accuracy, "accuracy")
        if np.isfinite(accuracy_gap) and accuracy_gap != 0:
            lines.append(f"- Accuracy delta versus best-accuracy option: {_format_delta(accuracy_gap, unit='%', digits=2)}")

    best_latency = _best_by_metric(scored, "latency_cycles", ascending=True)
    if best_latency is not None:
        latency_gap = _metric_value(best, "latency_cycles") - _metric_value(best_latency, "latency_cycles")
        if np.isfinite(latency_gap) and latency_gap != 0:
            lines.append(f"- Latency delta versus best-latency option: {_format_delta(latency_gap, unit=' cycles', digits=0)}")

    lines.append(f"-> Balanced trade-off using {meta.get('weights_used', 'available metrics only')}")
    return "\n".join(lines)


def recommend_best_accuracy(df_feasible: pd.DataFrame) -> dict[str, Any]:
    d = ensure_numeric_columns(df_feasible).copy()
    if d.empty or "accuracy" not in d.columns or d["accuracy"].dropna().empty:
        return _empty("No feasible candidate with available `accuracy`.")

    best = d.dropna(subset=["accuracy"]).sort_values(["accuracy"], ascending=[False]).iloc[0]
    return {"row": _as_row_dict(best), "explanation": _explain_accuracy_choice(d, best)}


def recommend_best_efficiency(df_feasible: pd.DataFrame) -> dict[str, Any]:
    d = ensure_numeric_columns(df_feasible).copy()
    if d.empty:
        return _empty("No feasible candidates.")

    eff = compute_efficiency_proxy(d)
    if eff.dropna().empty:
        return _empty("No feasible candidate with available `pin_reduction + slice_reduction`.")

    d = d.copy()
    d["_eff_proxy"] = eff
    sort_cols = ["_eff_proxy"]
    ascending = [False]
    if "accuracy" in d.columns:
        sort_cols.append("accuracy")
        ascending.append(False)
    best = d.dropna(subset=["_eff_proxy"]).sort_values(sort_cols, ascending=ascending).iloc[0]
    return {
        "row": _as_row_dict(best.drop(labels=["_eff_proxy"])),
        "explanation": _explain_efficiency_choice(d, best),
    }


def recommend_best_latency(df_feasible: pd.DataFrame) -> dict[str, Any]:
    d = ensure_numeric_columns(df_feasible).copy()
    if d.empty or "latency_cycles" not in d.columns or d["latency_cycles"].dropna().empty:
        return _empty("No feasible candidate with available `latency_cycles`.")

    sort_cols = ["latency_cycles"]
    ascending = [True]
    if "accuracy" in d.columns:
        sort_cols.append("accuracy")
        ascending.append(False)
    if "slices" in d.columns:
        sort_cols.append("slices")
        ascending.append(True)

    best = d.dropna(subset=["latency_cycles"]).sort_values(sort_cols, ascending=ascending).iloc[0]
    return {"row": _as_row_dict(best), "explanation": _explain_latency_choice(d, best)}


def recommend_best_balanced(
    df_feasible: pd.DataFrame,
    w_accuracy: float = 1.0,
    w_latency: float = 1.0,
    w_efficiency: float = 1.0,
) -> dict[str, Any]:
    d = ensure_numeric_columns(df_feasible).copy()
    if d.empty:
        return _empty("No feasible candidates.")

    scored, meta = compute_weighted_scores(d, w_accuracy, w_latency, w_efficiency, score_column="weighted_score")
    if "weighted_score" not in scored.columns or scored["weighted_score"].dropna().empty:
        return _empty("No feasible candidate can be scored with the currently available metrics.")

    best = scored.dropna(subset=["weighted_score"]).sort_values(["weighted_score"], ascending=[False]).iloc[0]
    return {"row": _as_row_dict(best), "explanation": _explain_balanced_choice(scored, best, meta)}


def get_reference_candidates(df_feasible: pd.DataFrame) -> dict[str, Optional[dict]]:
    d = ensure_numeric_columns(df_feasible).copy()
    if d.empty:
        return {"accuracy": None, "latency": None, "efficiency": None}

    best_accuracy = _best_by_metric(d, "accuracy", ascending=False)
    best_latency = _best_by_metric(d, "latency_cycles", ascending=True)
    best_efficiency = _best_efficiency_row(d)

    return {
        "accuracy": _as_row_dict(best_accuracy) if best_accuracy is not None else None,
        "latency": _as_row_dict(best_latency) if best_latency is not None else None,
        "efficiency": _as_row_dict(best_efficiency.drop(labels=["_eff_proxy"])) if best_efficiency is not None else None,
    }
