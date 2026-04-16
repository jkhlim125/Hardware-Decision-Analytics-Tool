from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Internal canonical schema (not every input provides all fields).
CANONICAL_COLUMNS = [
    "config_id",
    "design_type",
    "model_name",
    "accuracy",
    "accuracy_drop",
    "latency_cycles",
    "pins",
    "slices",
    "pin_reduction",
    "slice_reduction",
    "packing_efficiency",
    "lut_usage",
    "notes",
    # Extra but useful for provenance/debug.
    "source",
    "run_count",
]


@dataclass(frozen=True)
class DetectedInput:
    kind: str  # "json_experiment" | "generic_csv" | "rtl_latency_csv"
    detail: str


def _to_path_str(path: Union[str, Path]) -> str:
    return str(path) if not isinstance(path, Path) else str(path)


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CANONICAL_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    # Keep column order stable (extras at the end).
    ordered = CANONICAL_COLUMNS + [c for c in out.columns if c not in CANONICAL_COLUMNS]
    out = out[ordered]

    # Coerce likely numeric columns.
    numeric_cols = [
        "accuracy",
        "accuracy_drop",
        "latency_cycles",
        "pins",
        "slices",
        "pin_reduction",
        "slice_reduction",
        "packing_efficiency",
        "lut_usage",
        "run_count",
    ]
    out = _coerce_numeric(out, numeric_cols)

    # Normalize ids/types to strings where present.
    if "config_id" in out.columns:
        out["config_id"] = out["config_id"].astype(str)
    if "design_type" in out.columns:
        out["design_type"] = out["design_type"].astype(str)
    if "model_name" in out.columns:
        out["model_name"] = out["model_name"].astype(str)
    if "source" in out.columns:
        out["source"] = out["source"].astype(str)

    return out


def _make_unique_config_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Ensure `config_id` is unique and non-empty.
    Returns (df_with_unique_ids, warnings).
    """
    out = df.copy()
    warnings: list[str] = []

    if "config_id" not in out.columns:
        out["config_id"] = [f"cand_{i}" for i in range(len(out))]
        warnings.append("Missing `config_id` column; generated sequential ids.")
        return out, warnings

    out["config_id"] = out["config_id"].replace({"nan": np.nan, "None": np.nan})
    missing = out["config_id"].isna() | (out["config_id"].astype(str).str.strip() == "")
    if missing.any():
        out.loc[missing, "config_id"] = [f"cand_{i}" for i in out.index[missing]]
        warnings.append("Some candidates had empty `config_id`; generated fallback ids.")

    # De-duplicate by suffixing.
    dup = out["config_id"].duplicated(keep=False)
    if dup.any():
        warnings.append("Duplicate `config_id` detected; suffixes were added to make ids unique.")
        counts: Dict[str, int] = {}
        new_ids = []
        for cid in out["config_id"].astype(str).tolist():
            counts[cid] = counts.get(cid, 0) + 1
            suffix = counts[cid]
            new_ids.append(f"{cid}__{suffix}" if suffix > 1 else cid)
        out["config_id"] = new_ids

    return out, warnings


def detect_input_kind_from_dataframe(df: pd.DataFrame) -> DetectedInput:
    cols = {c.lower() for c in df.columns}
    if {"run_id", "latency_lut", "latency_mac"}.issubset(cols):
        return DetectedInput(kind="rtl_latency_csv", detail="Detected columns: run_id, latency_lut, latency_mac")
    if "experiments" in cols:
        return DetectedInput(kind="json_experiment", detail="Detected top-level `experiments` field")
    return DetectedInput(kind="generic_csv", detail="Defaulted to generic CSV schema")


def parse_rtl_latency_csv(path_or_bytes: Union[str, Path, bytes]) -> Tuple[pd.DataFrame, list[str]]:
    """
    Parse RTL comparison CSV:
      run_id, latency_lut, latency_mac

    Returns two design candidates:
      - LUT implementation
      - MAC implementation
    """
    warnings: list[str] = []

    if isinstance(path_or_bytes, (str, Path)):
        df = pd.read_csv(_to_path_str(path_or_bytes))
        source_label = _to_path_str(path_or_bytes)
    else:
        df = pd.read_csv(io.BytesIO(path_or_bytes))
        source_label = "uploaded_rtl_latency_csv"

    df.columns = [c.strip() for c in df.columns]
    required = {"run_id", "latency_lut", "latency_mac"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"RTL latency CSV is missing required columns: {missing}")

    df = _coerce_numeric(df, ["latency_lut", "latency_mac"])
    run_count = int(len(df))

    lut_mean = float(df["latency_lut"].mean(skipna=True)) if run_count else np.nan
    mac_mean = float(df["latency_mac"].mean(skipna=True)) if run_count else np.nan

    rows = [
        {
            "config_id": "rtl_lut",
            "design_type": "LUT",
            "model_name": "N/A",
            "accuracy": np.nan,
            "accuracy_drop": np.nan,
            "latency_cycles": lut_mean,
            "pins": np.nan,
            "slices": np.nan,
            "pin_reduction": np.nan,
            "slice_reduction": np.nan,
            "packing_efficiency": np.nan,
            "lut_usage": np.nan,
            "notes": "Derived from event-based RTL testbench latency log (LUT path).",
            "source": "rtl_simulation",
            "run_count": run_count,
            "source_file": source_label,
        },
        {
            "config_id": "rtl_mac",
            "design_type": "MAC",
            "model_name": "N/A",
            "accuracy": np.nan,
            "accuracy_drop": np.nan,
            "latency_cycles": mac_mean,
            "pins": np.nan,
            "slices": np.nan,
            "pin_reduction": np.nan,
            "slice_reduction": np.nan,
            "packing_efficiency": np.nan,
            "lut_usage": np.nan,
            "notes": "Derived from event-based RTL testbench latency log (MAC path).",
            "source": "rtl_simulation",
            "run_count": run_count,
            "source_file": source_label,
        },
    ]

    out = pd.DataFrame(rows)
    out = _ensure_canonical_columns(out)
    out, w = _make_unique_config_ids(out)
    warnings.extend(w)
    return out, warnings


def parse_generic_tradeoff_csv(path_or_bytes: Union[str, Path, bytes]) -> Tuple[pd.DataFrame, list[str]]:
    """
    Parse a generic trade-off CSV and map columns into the canonical schema when possible.
    Heuristics are conservative and favor explicitness over guessing.
    """
    warnings: list[str] = []

    if isinstance(path_or_bytes, (str, Path)):
        df = pd.read_csv(_to_path_str(path_or_bytes))
        source_label = _to_path_str(path_or_bytes)
    else:
        df = pd.read_csv(io.BytesIO(path_or_bytes))
        source_label = "uploaded_generic_csv"

    original_cols = list(df.columns)
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in lower_map:
                return lower_map[n.lower()]
        return None

    col_config = pick("config_id", "run_id", "id", "name", "experiment_id")
    col_design = pick("design_type", "design", "impl", "implementation")
    col_model = pick("model_name", "model")
    col_acc = pick("accuracy", "acc", "max_acc", "max_accuracy")
    col_acc_drop = pick("accuracy_drop", "acc_drop", "acc_drop_positive")
    col_lat = pick("latency_cycles", "latency", "latency_cycle", "cycles")
    col_pins = pick("pins", "pin_count")
    col_slices = pick("slices", "slice", "slice_count")
    col_pin_red = pick("pin_reduction", "pin_reduction_rate", "pin_reduction_rate_percent")
    col_slice_red = pick("slice_reduction", "slice_reduction_percent")
    col_pack = pick("packing_efficiency", "pack_efficiency", "packing", "pack_ratio")
    col_lut = pick("lut_usage", "lut_utilization", "lut")
    col_notes = pick("notes", "note", "comment")

    out = pd.DataFrame()
    out["config_id"] = df[col_config].astype(str) if col_config else [f"cand_{i}" for i in range(len(df))]
    out["design_type"] = df[col_design].astype(str) if col_design else "candidate"
    out["model_name"] = df[col_model].astype(str) if col_model else "N/A"
    out["accuracy"] = df[col_acc] if col_acc else np.nan
    out["accuracy_drop"] = df[col_acc_drop] if col_acc_drop else np.nan
    out["latency_cycles"] = df[col_lat] if col_lat else np.nan
    out["pins"] = df[col_pins] if col_pins else np.nan
    out["slices"] = df[col_slices] if col_slices else np.nan
    out["pin_reduction"] = df[col_pin_red] if col_pin_red else np.nan
    out["slice_reduction"] = df[col_slice_red] if col_slice_red else np.nan
    out["packing_efficiency"] = df[col_pack] if col_pack else np.nan
    out["lut_usage"] = df[col_lut] if col_lut else np.nan
    out["notes"] = df[col_notes].astype(str) if col_notes else ""
    out["source"] = "generic_csv"
    out["run_count"] = np.nan
    out["source_file"] = source_label
    out["input_columns"] = ", ".join(original_cols)

    out = _ensure_canonical_columns(out)
    out, w = _make_unique_config_ids(out)
    warnings.extend(w)
    return out, warnings


def parse_json_experiment_log(path_or_bytes: Union[str, Path, bytes]) -> Tuple[pd.DataFrame, list[str]]:
    """
    Parse the existing experiment JSON format used by the current project.
    The goal is to preserve your current metrics while mapping into the new canonical schema.
    """
    warnings: list[str] = []

    if isinstance(path_or_bytes, (str, Path)):
        raw = Path(path_or_bytes).read_text(encoding="utf-8")
        source_label = _to_path_str(path_or_bytes)
    else:
        raw = path_or_bytes.decode("utf-8", errors="replace")
        source_label = "uploaded_experiment_json"

    obj = json.loads(raw)
    experiments = obj.get("experiments", []) or []
    configs = obj.get("configurations_by_experiment", []) or []
    max_accs = obj.get("max_test_accuracy_by_experiment", []) or []
    summaries = obj.get("summaries_by_experiment", []) or []

    def safe_float(x: Any) -> float:
        try:
            if x is None:
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    rows: list[Dict[str, Any]] = []
    for i, exp in enumerate(experiments):
        exp = exp or {}
        cfg = exp.get("configuration") or (configs[i] if i < len(configs) else {}) or {}
        summ = exp.get("final_summary_physics_aware") or (summaries[i] if i < len(summaries) else {}) or {}
        max_info = exp.get("max_test_accuracy") or (max_accs[i] if i < len(max_accs) else {}) or {}

        pack_ratio = safe_float(cfg.get("pack_ratio"))
        global_sparsity = safe_float(cfg.get("global_sparsity"))
        source_log = str(exp.get("source_log", "") or "")

        is_baseline = ("baseline" in source_log.lower()) or (np.isfinite(pack_ratio) and pack_ratio == 0.0)

        # Accuracy in existing dashboard is stored as percent.
        max_acc = safe_float(max_info.get("max_test_accuracy_percent") if isinstance(max_info, dict) else np.nan)

        successful_packs = safe_float(summ.get("successful_packs_pairs"))
        failed_packs = safe_float(summ.get("failed_packs_pairs"))
        total_pairs = successful_packs + failed_packs
        packing_eff = successful_packs / total_pairs if np.isfinite(total_pairs) and total_pairs > 0 else np.nan

        if np.isfinite(pack_ratio) and np.isfinite(global_sparsity):
            config_id = f"pr_{pack_ratio:.2f}__gs_{global_sparsity:.2f}"
        else:
            config_id = f"exp_{i}"

        rows.append(
            {
                "config_id": config_id if not is_baseline else "baseline",
                "design_type": "experiment_baseline" if is_baseline else "experiment_candidate",
                "model_name": "N/A",
                "accuracy": max_acc,
                "accuracy_drop": np.nan,  # computed after baseline is known
                "latency_cycles": np.nan,
                "pins": np.nan,
                "slices": np.nan,
                "pin_reduction": safe_float(summ.get("pin_reduction_rate_percent")),
                "slice_reduction": safe_float(summ.get("slice_reduction_percent")),
                "packing_efficiency": packing_eff,
                "lut_usage": np.nan,
                "notes": f"pack_ratio={pack_ratio}, global_sparsity={global_sparsity}, source_log={source_log}",
                "source": "experiment_log",
                "run_count": 1,
                "is_baseline": bool(is_baseline),
                "pack_ratio": pack_ratio,
                "global_sparsity": global_sparsity,
                "source_file": source_label,
            }
        )

    out = pd.DataFrame(rows)
    out = _ensure_canonical_columns(out)
    out, w = _make_unique_config_ids(out)
    warnings.extend(w)

    # Compute accuracy_drop vs baseline if possible.
    if "is_baseline" in out.columns and out["is_baseline"].astype(bool).any():
        base_acc = pd.to_numeric(out.loc[out["is_baseline"].astype(bool), "accuracy"], errors="coerce").max()
        if np.isfinite(base_acc):
            out["accuracy_drop"] = (base_acc - pd.to_numeric(out["accuracy"], errors="coerce")).clip(lower=0.0)
        else:
            warnings.append("Baseline exists but baseline accuracy is missing; `accuracy_drop` left as NaN.")
    else:
        warnings.append("No baseline detected in experiment JSON; `accuracy_drop` left as NaN.")

    return out, warnings


def detect_input_kind_from_bytes(filename: str, data: bytes) -> DetectedInput:
    name = (filename or "").lower()
    if name.endswith(".json"):
        return DetectedInput(kind="json_experiment", detail="Filename suggests JSON experiment log")
    if name.endswith(".csv"):
        # Heuristic sniff: read header only.
        try:
            preview = pd.read_csv(io.BytesIO(data), nrows=5)
            return detect_input_kind_from_dataframe(preview)
        except Exception:
            return DetectedInput(kind="generic_csv", detail="CSV could not be sniffed; defaulted to generic CSV")
    return DetectedInput(kind="generic_csv", detail="Unknown extension; defaulted to generic CSV")


def load_and_normalize_data(
    file_or_path: Union[str, Path, Any],
) -> Tuple[pd.DataFrame, DetectedInput, list[str]]:
    """
    Main entrypoint for the app.

    Supports:
    - Path to a local file (CSV/JSON)
    - Streamlit UploadedFile-like object (has `.read()` and `.name`)
    """
    warnings: list[str] = []

    filename = ""
    raw_bytes: Optional[bytes] = None
    path: Optional[Union[str, Path]] = None

    if isinstance(file_or_path, (str, Path)):
        path = file_or_path
        filename = Path(file_or_path).name
    else:
        # Streamlit UploadedFile interface
        filename = getattr(file_or_path, "name", "") or ""
        raw_bytes = file_or_path.read()

    if path is not None:
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            df, w = parse_json_experiment_log(path)
            warnings.extend(w)
            detected = DetectedInput(kind="json_experiment", detail="Parsed JSON experiment log")
        elif suffix == ".csv":
            df_head = pd.read_csv(_to_path_str(path), nrows=5)
            detected = detect_input_kind_from_dataframe(df_head)
            if detected.kind == "rtl_latency_csv":
                df, w = parse_rtl_latency_csv(path)
            else:
                df, w = parse_generic_tradeoff_csv(path)
            warnings.extend(w)
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")
    else:
        if raw_bytes is None:
            raise ValueError("No file bytes provided.")
        detected = detect_input_kind_from_bytes(filename, raw_bytes)
        if detected.kind == "json_experiment":
            df, w = parse_json_experiment_log(raw_bytes)
        elif detected.kind == "rtl_latency_csv":
            df, w = parse_rtl_latency_csv(raw_bytes)
        else:
            df, w = parse_generic_tradeoff_csv(raw_bytes)
        warnings.extend(w)

    df = _ensure_canonical_columns(df)
    df, w = _make_unique_config_ids(df)
    warnings.extend(w)

    # Best-effort: clean up empty strings in notes/source.
    for c in ["notes", "source"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    return df.reset_index(drop=True), detected, warnings

