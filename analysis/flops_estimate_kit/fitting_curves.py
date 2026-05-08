import argparse
import json
import os

from curve_fit_prepare import (
    build_overall_benchmark_config,
    prepare_curve_fit_data,
    summarize_fit_results,
)
from visualize import plot_advanced_scaling_v2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit one cached SFT->RL scaling curve with selected hyperparameters."
    )
    parser.add_argument("--sft_parquets_dir", type=str, default="", help="Unused in cache-only mode.")
    parser.add_argument("--rl_parquets_dir", type=str, default="", help="Unused in cache-only mode.")
    parser.add_argument("--rl_progress_dir", type=str, default="", help="Unused in cache-only mode.")
    parser.add_argument("--sft_dict_file_name", type=str, required=True)
    parser.add_argument("--rl_dict_file_name", type=str, required=True)
    parser.add_argument("--figure_save_path", type=str, required=True)
    parser.add_argument("--metrics_json_out", type=str, default="")

    parser.add_argument("--model_name", type=str, default="qwen_2_5_7b")
    parser.add_argument("--sft_scene", type=str, required=True)
    parser.add_argument("--rl_method", type=str, default="dapo")
    parser.add_argument("--max_step", type=str, default="auto")
    parser.add_argument("--sft_steps", type=int, nargs="+", required=True)
    parser.add_argument("--rl_start_points", type=int, nargs="+", required=True)

    parser.add_argument("--fit_points_num", type=int, required=True)
    parser.add_argument("--use_robust_reg", action="store_true")
    parser.add_argument("--lts_alpha", type=float, default=0.75)
    parser.add_argument("--outlier_threshold", type=float, default=2.5)
    parser.add_argument("--val_most", type=int, default=100)
    parser.add_argument("--train_split", type=int, nargs=2, metavar=("START", "END"), default=None)
    parser.add_argument("--val_split", type=int, nargs=2, metavar=("START", "END"), default=None)
    parser.add_argument("--gap_weight", type=float, default=0.35)
    parser.add_argument("--free_c0", action="store_true")
    parser.add_argument("--exclude_nonpositive_val", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.rl_start_points) != 1:
        raise ValueError("Each public fit script should pass exactly one --rl_start_points value.")

    print("Selected hyperparameters")
    print(f"  fit_points_num={args.fit_points_num}")
    print(f"  train_split={args.train_split}")
    print(f"  val_split={args.val_split}")
    print(f"  use_robust_reg={args.use_robust_reg}")
    print(f"  lts_alpha={args.lts_alpha}")
    print(f"  outlier_threshold={args.outlier_threshold}")
    print(f"  val_most={args.val_most}")
    print(f"  free_c0={args.free_c0}")

    data_pack = prepare_curve_fit_data(args)
    print(f"step2flops_per_ckpt: {data_pack['step2flops_per_ckpt']}")
    print(f"branch_flops: {data_pack['branch_flops']}")

    fixed_c0 = None if args.free_c0 else 0
    train_range = tuple(args.train_split) if args.train_split is not None else None
    val_range = tuple(args.val_split) if args.val_split is not None else None
    benchmark_configs = build_overall_benchmark_config(
        args.fit_points_num,
        args.val_most,
        args.use_robust_reg,
        args.lts_alpha,
        args.outlier_threshold,
        fixed_c0=fixed_c0,
        train_range=train_range,
        val_range=val_range,
        exclude_nonpositive_val=args.exclude_nonpositive_val,
    )

    default_config = {
        "split_mode": "index",
        "train_range": (0, 60),
        "val_range": (60, 100),
        "detect_outliers": True,
        "outlier_threshold": 2.5,
        "use_robust_regression": False,
        "lts_alpha": 0.75,
        "model_types": "auto",
        "metric": "val_rmse",
        "fixed_C0": fixed_c0,
        "exclude_nonpositive_val": args.exclude_nonpositive_val,
    }

    results, predictions, _ = plot_advanced_scaling_v2(
        data_pack["test_ckpt"],
        data_pack["combined_flops2val_performance"],
        data_pack["branch_flops"],
        default_config=default_config,
        benchmark_configs=benchmark_configs,
        benchmark_filter=None,
        predict_flops_list=[30000, 40000, 50000, 60000],
        save_path=args.figure_save_path,
        dpi=300,
        save_format="png",
    )

    if args.metrics_json_out:
        summary = summarize_fit_results(results, gap_weight=args.gap_weight)
        overall_cfg = benchmark_configs["overall"]
        payload = {
            "metrics": summary,
            "predictions": predictions,
            "hparams": {
                "fit_points_num": args.fit_points_num,
                "use_robust_reg": args.use_robust_reg,
                "lts_alpha": args.lts_alpha,
                "outlier_threshold": args.outlier_threshold,
                "val_most": args.val_most,
                "gap_weight": args.gap_weight,
                "train_range": list(overall_cfg["train_range"]),
                "val_range": list(overall_cfg["val_range"]),
                "free_c0": args.free_c0,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_json_out)), exist_ok=True)
        with open(args.metrics_json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote metrics to {args.metrics_json_out}")


if __name__ == "__main__":
    main()

