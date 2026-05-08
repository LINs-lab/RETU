"""Data loading and metric helpers for cached scaling-curve fitting."""

import numpy as np

from config.model import qwen_7b_config
from config.train import (
    easy102K_train_config,
    hard102K_train_config,
    s1K_train_config,
    sft_train_config,
    uniform102K_train_config,
)
from utils import get_combined_flops2val_performance_zoo, load_pickle_by_name
from utils.flops_est import step2flops_sft


DEFAULT_BENCHMARK_DATA_NUM = {
    "gsm8k": 1317,
    "olympiad_bench": 291,
    "math": 237,
    "minerva": 262,
    "aime": 25,
    "aime25": 25,
}

SFT_TRAIN_CONFIGS = {
    "general": sft_train_config,
    "easy102K": easy102K_train_config,
    "hard102K": hard102K_train_config,
    "s1K": s1K_train_config,
    "uniform102K": uniform102K_train_config,
}

MODEL_CONFIGS = {
    "qwen_2_5_7b": qwen_7b_config,
}


def build_overall_benchmark_config(
    fit_points_num,
    val_most,
    use_robust_reg,
    lts_alpha,
    outlier_threshold,
    fixed_c0=0,
    train_range=None,
    val_range=None,
    exclude_nonpositive_val=False,
):
    """Return the per-benchmark config for the selected overall curve."""
    if train_range is None:
        train_range = (0, fit_points_num)
    if val_range is None:
        val_range = (train_range[1], val_most)
    return {
        "overall": {
            "split_mode": "index",
            "train_range": tuple(train_range),
            "val_range": tuple(val_range),
            "use_robust_regression": bool(use_robust_reg),
            "lts_alpha": float(lts_alpha),
            "model_types": ["logistic"],
            "detect_outliers": True,
            "outlier_threshold": float(outlier_threshold),
            "metric": "auto",
            "fixed_C0": fixed_c0,
            "exclude_nonpositive_val": bool(exclude_nonpositive_val),
        }
    }


def prepare_curve_fit_data(args):
    """Load cached SFT and RL points and build the branch curve for one fit."""
    sft_config = SFT_TRAIN_CONFIGS[args.sft_scene]
    model_config = MODEL_CONFIGS[args.model_name]

    step2flops_per_ckpt = step2flops_sft(model_config, sft_config, args.sft_steps)
    flops2val_performance_sft = load_pickle_by_name(args.sft_dict_file_name)
    step2flops2val_performance_rl = load_pickle_by_name(args.rl_dict_file_name)

    combined, branch_flops = get_combined_flops2val_performance_zoo(
        step2flops_per_ckpt,
        flops2val_performance_sft,
        step2flops2val_performance_rl,
        args.rl_start_points,
        DEFAULT_BENCHMARK_DATA_NUM,
    )

    return {
        "step2flops_per_ckpt": step2flops_per_ckpt,
        "combined_flops2val_performance": combined,
        "branch_flops": branch_flops,
        "test_ckpt": args.rl_start_points[0],
    }


def summarize_fit_results(all_results, gap_weight=0.35):
    """Summarize fit quality across all plotted benchmarks."""
    if not all_results:
        return {
            "score": float("-inf"),
            "mean_r2_val": None,
            "mean_r2_train": None,
            "mean_overfit_gap": None,
            "per_benchmark": {},
        }

    r2_val = [result["r2_val"] for result in all_results.values()]
    r2_train = [result["r2_train"] for result in all_results.values()]
    mean_r2_val = float(np.mean(r2_val))
    mean_r2_train = float(np.mean(r2_train))
    overfit_gap = max(0.0, mean_r2_train - mean_r2_val)
    return {
        "score": mean_r2_val - gap_weight * overfit_gap,
        "mean_r2_val": mean_r2_val,
        "mean_r2_train": mean_r2_train,
        "mean_overfit_gap": float(mean_r2_train - mean_r2_val),
        "per_benchmark": {
            name: {
                "r2_val": result["r2_val"],
                "r2_train": result["r2_train"],
                "rmse_val": result["rmse_val"],
                "equation": result["equation"],
            }
            for name, result in all_results.items()
        },
    }

