import json
from pathlib import Path

import numpy as np
import pandas as pd


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def normalize_curve(obj):
    if isinstance(obj, list):
        return obj
    return []


def parse_json_to_df(json_path, lambda_score=2.0):
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    experiments = obj.get("experiments", [])
    configs = obj.get("configurations_by_experiment", [])
    max_accs = obj.get("max_test_accuracy_by_experiment", [])
    summaries = obj.get("summaries_by_experiment", [])

    rows = []

    for i, exp in enumerate(experiments):
        exp = exp or {}

        cfg = exp.get("configuration") or (configs[i] if i < len(configs) else {}) or {}
        summ = exp.get("final_summary_physics_aware") or (summaries[i] if i < len(summaries) else {}) or {}
        max_info = exp.get("max_test_accuracy") or (max_accs[i] if i < len(max_accs) else {}) or {}

        pack_ratio = safe_float(cfg.get("pack_ratio"))
        global_sparsity = safe_float(cfg.get("global_sparsity"))

        source_log = str(exp.get("source_log", "") or "")
        is_baseline = ("baseline" in source_log.lower()) or (pack_ratio == 0.0)

        test_acc_curve = exp.get("test_accuracies_percent") or []
        test_loss_curve = exp.get("test_losses") or []
        train_loss_curve = exp.get("train_losses") or []
        epochs = exp.get("epochs") or []

        # 어떤 로그는 epoch 정보가 dict list 형태일 수도 있음
        if isinstance(epochs, list) and len(epochs) > 0 and isinstance(epochs[0], dict):
            ep_list = []
            acc_list = []
            tloss_list = []
            trloss_list = []

            for item in epochs:
                item = item or {}
                ep_list.append(item.get("epoch", len(ep_list) + 1))
                acc_list.append(safe_float(item.get("test_acc")))
                tloss_list.append(safe_float(item.get("test_loss")))
                trloss_list.append(safe_float(item.get("train_loss")))

            epochs = ep_list

            if len(test_acc_curve) == 0:
                test_acc_curve = acc_list
            if len(test_loss_curve) == 0:
                test_loss_curve = tloss_list
            if len(train_loss_curve) == 0:
                train_loss_curve = trloss_list

        max_acc = safe_float(
            max_info.get("max_test_accuracy_percent") if isinstance(max_info, dict) else np.nan
        )

        final_acc = safe_float(test_acc_curve[-1] if len(test_acc_curve) > 0 else np.nan)
        final_test_loss = safe_float(test_loss_curve[-1] if len(test_loss_curve) > 0 else np.nan)

        if len(test_acc_curve) >= 5:
            last5_mean_acc = safe_float(np.mean(test_acc_curve[-5:]))
        elif len(test_acc_curve) > 0:
            last5_mean_acc = safe_float(np.mean(test_acc_curve))
        else:
            last5_mean_acc = np.nan

        successful_packs = safe_float(summ.get("successful_packs_pairs"))
        failed_packs = safe_float(summ.get("failed_packs_pairs"))

        total_pairs = successful_packs + failed_packs
        if np.isfinite(total_pairs) and total_pairs > 0:
            pack_efficiency = successful_packs / total_pairs
        else:
            pack_efficiency = np.nan

        rows.append(
            {
                "pack_ratio": pack_ratio,
                "global_sparsity": global_sparsity,
                "is_baseline": is_baseline,
                "source_log": source_log,
                "max_acc": max_acc,
                "final_acc": final_acc,
                "last5_mean_acc": last5_mean_acc,
                "final_test_loss": final_test_loss,
                "slice_reduction": safe_float(summ.get("slice_reduction_percent")),
                "pin_reduction": safe_float(summ.get("pin_reduction_rate_percent")),
                "total_dead": safe_float(summ.get("total_dead")),
                "successful_packs": successful_packs,
                "failed_packs": failed_packs,
                "pack_efficiency": pack_efficiency,
                "epochs": epochs if isinstance(epochs, list) else [],
                "test_acc_curve": normalize_curve(test_acc_curve),
                "test_loss_curve": normalize_curve(test_loss_curve),
                "train_loss_curve": normalize_curve(train_loss_curve),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    baseline_df = df[df["is_baseline"]].copy()

    if not baseline_df.empty:
        baseline_acc = baseline_df["max_acc"].dropna().max()
        if baseline_df["final_test_loss"].notna().any():
            baseline_loss = baseline_df["final_test_loss"].dropna().min()
        else:
            baseline_loss = np.nan
    else:
        baseline_acc = np.nan
        baseline_loss = np.nan

    df["baseline_acc_global"] = baseline_acc
    df["baseline_loss_global"] = baseline_loss

    raw_drop = baseline_acc - df["max_acc"]
    df["acc_drop_positive"] = np.maximum(0.0, raw_drop)

    df["loss_delta_vs_global_baseline"] = df["final_test_loss"] - baseline_loss

    # custom score: hardware gain - penalty from accuracy drop
    df["tradeoff_score"] = df["pin_reduction"] - lambda_score * df["acc_drop_positive"]

    return df.reset_index(drop=True)


def pareto_frontier_maximize_x_y(df, xcol, ycol):
    pts = df[[xcol, ycol]].dropna()
    if pts.empty:
        return df.iloc[[]].copy()

    keep_idx = []

    for i, ri in pts.iterrows():
        xi, yi = ri[xcol], ri[ycol]
        dominated = False

        for j, rj in pts.iterrows():
            if i == j:
                continue

            xj, yj = rj[xcol], rj[ycol]
            if (xj >= xi and yj >= yi) and (xj > xi or yj > yi):
                dominated = True
                break

        if not dominated:
            keep_idx.append(i)

    return df.loc[keep_idx].copy().sort_values(xcol)


def pareto_frontier_maximize_x_minimize_y(df, xcol, ycol):
    pts = df[[xcol, ycol]].dropna()
    if pts.empty:
        return df.iloc[[]].copy()

    keep_idx = []

    for i, ri in pts.iterrows():
        xi, yi = ri[xcol], ri[ycol]
        dominated = False

        for j, rj in pts.iterrows():
            if i == j:
                continue

            xj, yj = rj[xcol], rj[ycol]
            if (xj >= xi and yj <= yi) and (xj > xi or yj < yi):
                dominated = True
                break

        if not dominated:
            keep_idx.append(i)

    return df.loc[keep_idx].copy().sort_values(xcol)


def get_top_configs(df, top_k=5):
    d = df[~df["is_baseline"]].copy()
    d = d.dropna(subset=["tradeoff_score"])
    if d.empty:
        return d
    return d.sort_values("tradeoff_score", ascending=False).head(top_k)


def build_summary_text(df, lambda_score):
    if df.empty:
        return "No experiments found."

    total_runs = len(df)
    non_baseline = df[~df["is_baseline"]].copy()

    if non_baseline.empty:
        return "Only baseline runs found."

    best_score_row = non_baseline.sort_values("tradeoff_score", ascending=False).iloc[0]

    lines = []
    lines.append(f"Total runs: {total_runs}")
    lines.append(f"Non-baseline runs: {len(non_baseline)}")
    lines.append(f"Current trade-off weight (lambda): {lambda_score:.2f}")

    if np.isfinite(best_score_row["pack_ratio"]) and np.isfinite(best_score_row["global_sparsity"]):
        lines.append(
            "Best trade-off config: "
            f"pack_ratio={best_score_row['pack_ratio']:.2f}, "
            f"global_sparsity={best_score_row['global_sparsity']:.2f}"
        )

    if np.isfinite(best_score_row["pin_reduction"]):
        lines.append(f"Pin reduction: {best_score_row['pin_reduction']:.2f}%")

    if np.isfinite(best_score_row["acc_drop_positive"]):
        lines.append(f"Accuracy drop vs baseline: {best_score_row['acc_drop_positive']:.2f}%")

    return "\n".join(lines)
