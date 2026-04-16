from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


Direction = Literal["maximize", "minimize"]


CANONICAL_NUMERIC_COLS = [
    "accuracy",
    "accuracy_drop",
    "latency_cycles",
    "pins",
    "slices",
    "pin_reduction",
    "slice_reduction",
    "packing_efficiency",
    "lut_usage",
]


@dataclass(frozen=True)
class Constraints:
    min_accuracy: Optional[float] = None
    max_latency_cycles: Optional[float] = None
    max_pins: Optional[float] = None
    max_slices: Optional[float] = None
    min_pin_reduction: Optional[float] = None
    min_slice_reduction: Optional[float] = None

    treat_nan_as_violation: bool = True


def ensure_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CANONICAL_NUMERIC_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def apply_constraints(
    df: pd.DataFrame,
    constraints: Constraints,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply hard feasibility constraints and produce explicit rejection reasons.

    Returns (feasible_df, rejected_df). Both include:
      - `rejection_reasons`: list[str]
      - `rejection_reason_str`: joined reasons for easy display
      - `is_feasible`: boolean
    """
    d = ensure_numeric_columns(df)
    out = d.copy()
    out["rejection_reasons"] = [[] for _ in range(len(out))]

    def add_reason(mask: pd.Series, reason: str) -> None:
        if not mask.any():
            return
        idxs = out.index[mask].tolist()
        for i in idxs:
            out.at[i, "rejection_reasons"] = list(out.at[i, "rejection_reasons"]) + [reason]

    def violates_min(col: str, thr: float, label: str) -> None:
        if col not in out.columns:
            add_reason(pd.Series([True] * len(out), index=out.index), f"missing metric: {label}")
            return
        s = pd.to_numeric(out[col], errors="coerce")
        if constraints.treat_nan_as_violation:
            add_reason(s.isna(), f"missing metric: {label}")
        add_reason(s.notna() & (s < thr), f"rejected: {label} < {thr}")

    def violates_max(col: str, thr: float, label: str) -> None:
        if col not in out.columns:
            add_reason(pd.Series([True] * len(out), index=out.index), f"missing metric: {label}")
            return
        s = pd.to_numeric(out[col], errors="coerce")
        if constraints.treat_nan_as_violation:
            add_reason(s.isna(), f"missing metric: {label}")
        add_reason(s.notna() & (s > thr), f"rejected: {label} > {thr}")

    if constraints.min_accuracy is not None:
        violates_min("accuracy", float(constraints.min_accuracy), "accuracy")
    if constraints.max_latency_cycles is not None:
        violates_max("latency_cycles", float(constraints.max_latency_cycles), "latency_cycles")
    if constraints.max_pins is not None:
        violates_max("pins", float(constraints.max_pins), "pins")
    if constraints.max_slices is not None:
        violates_max("slices", float(constraints.max_slices), "slices")
    if constraints.min_pin_reduction is not None:
        violates_min("pin_reduction", float(constraints.min_pin_reduction), "pin_reduction")
    if constraints.min_slice_reduction is not None:
        violates_min("slice_reduction", float(constraints.min_slice_reduction), "slice_reduction")

    out["is_feasible"] = out["rejection_reasons"].apply(lambda r: len(r) == 0)
    out["rejection_reason_str"] = out["rejection_reasons"].apply(lambda r: "; ".join(r))

    feasible = out[out["is_feasible"]].copy().reset_index(drop=True)
    rejected = out[~out["is_feasible"]].copy().reset_index(drop=True)
    return feasible, rejected


def summarize_rejections(rejected_df: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    """
    Return a dataframe with columns: reason, count
    """
    if rejected_df is None or rejected_df.empty or "rejection_reasons" not in rejected_df.columns:
        return pd.DataFrame({"reason": [], "count": []})

    counts: Dict[str, int] = {}
    for reasons in rejected_df["rejection_reasons"].tolist():
        if not isinstance(reasons, list):
            continue
        for r in reasons:
            counts[r] = counts.get(r, 0) + 1

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return pd.DataFrame({"reason": [k for k, _ in items], "count": [v for _, v in items]})


def compute_pareto_frontier(
    df: pd.DataFrame,
    objectives: Sequence[str],
    directions: Sequence[Direction],
    flag_column: str = "is_pareto",
) -> pd.DataFrame:
    """
    Mark Pareto-optimal candidates across selected objectives.

    Pareto-optimal means: no other candidate is strictly better in all objectives
    (with at least one strict improvement) under the given directions.
    """
    if len(objectives) != len(directions):
        raise ValueError("`objectives` and `directions` must have the same length.")

    out = ensure_numeric_columns(df).copy()
    out[flag_column] = False

    if out.empty:
        return out

    needed = [c for c in objectives if c not in out.columns]
    if needed:
        # Cannot compute Pareto meaningfully without objective columns.
        return out

    pts = out[list(objectives)].apply(pd.to_numeric, errors="coerce")
    valid_mask = pts.notna().all(axis=1)
    if valid_mask.sum() == 0:
        return out

    pts = pts[valid_mask]

    # Convert to a "maximize everything" space by negating minimize objectives.
    pts_adj = pts.copy()
    for c, d in zip(objectives, directions):
        if d == "minimize":
            pts_adj[c] = -pts_adj[c]

    idxs = pts_adj.index.tolist()
    values = pts_adj.to_numpy()
    pareto = np.ones(len(values), dtype=bool)

    # Readable O(n^2) dominance check (good enough for typical dashboard sizes).
    for i in range(len(values)):
        if not pareto[i]:
            continue
        vi = values[i]
        for j in range(len(values)):
            if i == j:
                continue
            vj = values[j]
            # vj dominates vi if vj >= vi in all dims and > in at least one.
            if np.all(vj >= vi) and np.any(vj > vi):
                pareto[i] = False
                break

    pareto_ids = [idxs[i] for i, keep in enumerate(pareto) if keep]
    out.loc[pareto_ids, flag_column] = True
    return out


def _minmax_norm(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo = x.min(skipna=True)
    hi = x.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.nan, index=s.index)
    return (x - lo) / (hi - lo)


def compute_weighted_scores(
    df_feasible: pd.DataFrame,
    w_accuracy: float,
    w_latency: float,
    w_efficiency: float,
    score_column: str = "weighted_score",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Compute a simple, explainable weighted score among feasible candidates only.

    - accuracy term: higher is better (normalized)
    - latency term: lower is better (normalized and inverted)
    - efficiency term: average of available {pin_reduction, slice_reduction, packing_efficiency} (normalized)
    """
    d = ensure_numeric_columns(df_feasible).copy()
    if d.empty:
        d[score_column] = np.nan
        return d, {"note": "No feasible candidates to score."}

    weights = np.array([w_accuracy, w_latency, w_efficiency], dtype=float)
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")
    if weights.sum() == 0:
        weights = np.array([1.0, 1.0, 1.0])
    weights = weights / weights.sum()

    acc_norm = _minmax_norm(d["accuracy"]) if "accuracy" in d.columns else pd.Series(np.nan, index=d.index)
    lat_norm = _minmax_norm(d["latency_cycles"]) if "latency_cycles" in d.columns else pd.Series(np.nan, index=d.index)
    lat_term = 1.0 - lat_norm if lat_norm.notna().any() else pd.Series(np.nan, index=d.index)

    eff_parts: List[pd.Series] = []
    for c in ["pin_reduction", "slice_reduction", "packing_efficiency"]:
        if c in d.columns and pd.to_numeric(d[c], errors="coerce").notna().any():
            eff_parts.append(_minmax_norm(d[c]))
    eff_term = pd.concat(eff_parts, axis=1).mean(axis=1) if eff_parts else pd.Series(np.nan, index=d.index)

    d["score_accuracy_norm"] = acc_norm
    d["score_latency_norm"] = lat_term
    d["score_efficiency_norm"] = eff_term

    # Use available metrics only: if a term is all-NaN, drop its weight and renormalize.
    term_matrix = pd.DataFrame(
        {
            "accuracy": acc_norm,
            "latency": lat_term,
            "efficiency": eff_term,
        }
    )
    term_avail = term_matrix.notna().any(axis=0).to_numpy(dtype=bool)
    w = weights.copy()
    w[~term_avail] = 0.0
    if w.sum() == 0:
        d[score_column] = np.nan
        return d, {"note": "No scorable metrics available among feasible candidates."}
    w = w / w.sum()

    d[score_column] = (
        w[0] * term_matrix["accuracy"].fillna(0.0)
        + w[1] * term_matrix["latency"].fillna(0.0)
        + w[2] * term_matrix["efficiency"].fillna(0.0)
    )

    meta = {
        "weights_used": f"accuracy={w[0]:.2f}, latency={w[1]:.2f}, efficiency={w[2]:.2f}",
        "note": "Score is computed only across feasible candidates using min-max normalization.",
    }
    return d, meta


def sweep_single_weight(
    df_feasible: pd.DataFrame,
    sweep: Literal["accuracy", "latency", "efficiency"],
    values: Iterable[float],
    base_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Simple sensitivity helper: sweep one weight while keeping the other two fixed.
    Returns a table of (sweep_value, top_config_id, top_score).
    """
    if base_weights is None:
        base_weights = {"accuracy": 1.0, "latency": 1.0, "efficiency": 1.0}

    rows = []
    for v in values:
        w_acc = float(base_weights.get("accuracy", 1.0))
        w_lat = float(base_weights.get("latency", 1.0))
        w_eff = float(base_weights.get("efficiency", 1.0))

        if sweep == "accuracy":
            w_acc = float(v)
        elif sweep == "latency":
            w_lat = float(v)
        else:
            w_eff = float(v)

        scored, _ = compute_weighted_scores(df_feasible, w_acc, w_lat, w_eff)
        scored = scored.dropna(subset=["weighted_score"]) if "weighted_score" in scored.columns else scored
        if scored.empty:
            rows.append({"sweep": v, "top_config_id": None, "top_score": np.nan})
            continue
        best = scored.sort_values("weighted_score", ascending=False).iloc[0]
        rows.append({"sweep": v, "top_config_id": best.get("config_id"), "top_score": best.get("weighted_score")})

    return pd.DataFrame(rows)
