import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit
from sklearn.metrics import r2_score

from .scaling_law import get_rl_scatters


warnings.filterwarnings("ignore")


BENCHMARK_ORDER = [
    "gsm8k",
    "math",
    "olympiad_bench",
    "minerva",
    "aime",
    "aime25",
    "easy_overall",
    "med_overall",
    "hard_overall",
    "overall",
]


def _safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _split_by_index(x, y, config):
    train_start, train_end = config.get("train_range", (0, 60))
    val_start, val_end = config.get("val_range", (60, 100))
    n = len(x)

    if train_end >= n and val_start >= n:
        train_start = 0
        train_end = max(2, int(n * 0.7))
        val_start = train_end
        val_end = n

    train_start = max(0, min(train_start, n))
    train_end = max(train_start + 1, min(train_end, n))
    val_start = max(train_end, min(val_start, n))
    val_end = min(max(val_start + 1, val_end), n)
    return (
        x[train_start:train_end],
        y[train_start:train_end],
        x[val_start:val_end],
        y[val_start:val_end],
    )


def _linear(x, a, b):
    return a * x + b


def _log_curve(x, a, b, c):
    return a * np.log(np.maximum(x + c, 1e-8)) + b


def _power_curve(x, a, b, c):
    return a * np.power(np.maximum(x + c, 1e-8), b)


def _exponential_curve(x, a, b, c):
    return a * (1 - np.exp(-b * x)) + c


def _logistic_fixed_c0(x, b, cmid, upper, fixed_c0=0.0):
    x = np.maximum(np.asarray(x, dtype=float), 1e-8)
    cmid = max(float(cmid), 1e-8)
    return fixed_c0 + (upper - fixed_c0) * expit(-b * np.log(cmid / x))


def _logistic_free_c0(x, b, c0, cmid, upper):
    x = np.maximum(np.asarray(x, dtype=float), 1e-8)
    cmid = max(float(cmid), 1e-8)
    return c0 + (upper - c0) * expit(-b * np.log(cmid / x))


def _fit_one_model(model_name, x_train, y_train, fixed_c0=0):
    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    y_span = max(y_max - y_min, 1.0)
    x_mid = float(np.median(x_train[x_train > 0])) if np.any(x_train > 0) else 1.0

    if model_name == "linear":
        func = _linear
        p0 = [0.0, float(y_train[0])]
        bounds = (-np.inf, np.inf)
    elif model_name == "log":
        func = _log_curve
        p0 = [1.0, float(y_train[0]), 1.0]
        bounds = ([-np.inf, -np.inf, 1e-8], [np.inf, np.inf, np.inf])
    elif model_name == "power":
        func = _power_curve
        p0 = [max(y_span, 1e-3), 0.5, 1.0]
        bounds = ([-np.inf, -5.0, 1e-8], [np.inf, 5.0, np.inf])
    elif model_name == "exponential":
        func = _exponential_curve
        p0 = [y_span, 0.01, y_min]
        bounds = ([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])
    elif model_name == "logistic":
        if fixed_c0 is None:
            func = _logistic_free_c0
            p0 = [1.0, y_min, x_mid, y_max]
            bounds = ([1e-4, -100.0, 1e-8, -100.0], [10.0, 100.0, np.inf, 100.0])
        else:
            fixed = float(fixed_c0)

            def func(x, b, cmid, upper):
                return _logistic_fixed_c0(x, b, cmid, upper, fixed_c0=fixed)

            p0 = [1.0, x_mid, max(y_max, fixed + 1e-3)]
            bounds = ([1e-4, 1e-8, -100.0], [10.0, np.inf, 100.0])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    params, _ = curve_fit(func, x_train, y_train, p0=p0, bounds=bounds, maxfev=20000)
    return func, params


def _equation(model_name, params, fixed_c0=0):
    values = [float(x) for x in params]
    if model_name == "linear":
        return f"y = {values[0]:.3g}x + {values[1]:.3g}"
    if model_name == "log":
        return f"y = {values[0]:.3g} ln(x + {values[2]:.3g}) + {values[1]:.3g}"
    if model_name == "power":
        return f"y = {values[0]:.3g} (x + {values[2]:.3g})^{values[1]:.3g}"
    if model_name == "exponential":
        return f"y = {values[0]:.3g}(1 - exp(-{values[1]:.3g}x)) + {values[2]:.3g}"
    if fixed_c0 is None:
        return (
            f"y = {values[1]:.3g} + ({values[3]:.3g}-{values[1]:.3g})/"
            f"(1 + ({values[2]:.3g}/x)^{values[0]:.3g})"
        )
    return f"y = {fixed_c0:.3g} + ({values[2]:.3g}-{fixed_c0:.3g})/(1 + ({values[1]:.3g}/x)^{values[0]:.3g})"


def _fit_benchmark(data, benchmark, config):
    x = np.asarray(sorted(data.keys()), dtype=float)
    y = np.asarray([data[flops][benchmark] for flops in sorted(data.keys())], dtype=float)
    x_train, y_train, x_val, y_val = _split_by_index(x, y, config)

    if config.get("exclude_nonpositive_val"):
        mask = y_val >= 0
        x_val, y_val = x_val[mask], y_val[mask]

    requested_models = config.get("model_types", "auto")
    if requested_models == "auto":
        requested_models = ["linear", "log", "power", "exponential", "logistic"]

    best = None
    for model_name in requested_models:
        try:
            func, params = _fit_one_model(model_name, x_train, y_train, config.get("fixed_C0", 0))
            train_pred = func(x_train, *params)
            val_pred = func(x_val, *params) if len(x_val) else np.asarray([])
            rmse_val = float(np.sqrt(np.mean((y_val - val_pred) ** 2))) if len(x_val) else float("nan")
            result = {
                "benchmark": benchmark,
                "best_model": model_name,
                "params": [float(v) for v in params],
                "r2_train": _safe_r2(y_train, train_pred),
                "r2_val": _safe_r2(y_val, val_pred) if len(x_val) else float("nan"),
                "rmse_val": rmse_val,
                "func": func,
                "config": config,
                "equation": _equation(model_name, params, config.get("fixed_C0", 0)),
            }
            if best is None or result["rmse_val"] < best["rmse_val"]:
                best = result
        except Exception as exc:
            print(f"Fit skipped for benchmark={benchmark} model={model_name}: {exc}")

    if best is None:
        raise RuntimeError(f"No model fit succeeded for benchmark={benchmark}")
    return best, x, y


def _plot_results(sft_ckpt, data, results, raw_xy, psft_by_benchmark, save_path, dpi, save_format):
    benchmarks = [b for b in BENCHMARK_ORDER if b in results]
    cols = 2
    rows = int(np.ceil(len(benchmarks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, max(4, rows * 3.2)))
    axes = np.asarray(axes).reshape(-1)

    for ax, benchmark in zip(axes, benchmarks):
        result = results[benchmark]
        x, y = raw_xy[benchmark]
        x_dense = np.linspace(max(1e-8, float(np.min(x))), float(np.max(x)), 240)
        y_dense = result["func"](x_dense, *np.asarray(result["params"], dtype=float))
        ax.scatter(x, y, s=18, color="#2f5bea", alpha=0.8, label="observed")
        ax.plot(x_dense, y_dense, color="#d34a32", linewidth=2, label=result["best_model"])
        ax.axhline(0, color="#999999", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{benchmark} (Psft={psft_by_benchmark.get(benchmark, float('nan')):.2f})")
        ax.set_xlabel("Relative RL EFLOPs")
        ax.set_ylabel("Performance gain")
        ax.grid(alpha=0.25)
        ax.text(
            0.02,
            0.98,
            f"train R2={result['r2_train']:.3f}\nval RMSE={result['rmse_val']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#dddddd"},
        )

    for ax in axes[len(benchmarks):]:
        ax.axis("off")

    fig.suptitle(f"SFT checkpoint {sft_ckpt}: RL scaling curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if not os.path.splitext(save_path)[1]:
            save_path = f"{save_path}.{save_format}"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Figure saved to: {save_path}")
    plt.close(fig)


def plot_advanced_scaling_v2(
    sft_ckpt,
    combined_flops2val_performance,
    branch_flops,
    default_config=None,
    benchmark_configs=None,
    benchmark_filter=None,
    predict_flops_list=None,
    save_path=None,
    dpi=300,
    save_format="png",
):
    """Fit and plot cached SFT->RL scaling curves for one branch."""
    data = get_rl_scatters(sft_ckpt, combined_flops2val_performance, branch_flops)
    start_flops = branch_flops[sft_ckpt]
    psft_by_benchmark = combined_flops2val_performance[sft_ckpt][start_flops]

    default_config = default_config or {
        "split_mode": "index",
        "train_range": (0, 60),
        "val_range": (60, 100),
        "model_types": "auto",
        "fixed_C0": 0,
        "exclude_nonpositive_val": False,
    }
    benchmark_configs = benchmark_configs or {}

    all_benchmarks = list(next(iter(data.values())).keys())
    if benchmark_filter is None:
        benchmarks = [b for b in BENCHMARK_ORDER if b in all_benchmarks]
    elif isinstance(benchmark_filter, list):
        benchmarks = [b for b in benchmark_filter if b in all_benchmarks]
    else:
        benchmarks = [b for b in all_benchmarks if str(benchmark_filter) in b]

    print(f"Fitting SFT checkpoint {sft_ckpt}")
    print(f"Default config: {default_config}")
    print(f"Benchmark-specific configs: {len(benchmark_configs)}")

    results = {}
    raw_xy = {}
    for benchmark in benchmarks:
        config = dict(default_config)
        config.update(benchmark_configs.get(benchmark, {}))
        result, x, y = _fit_benchmark(data, benchmark, config)
        results[benchmark] = result
        raw_xy[benchmark] = (x, y)

    _plot_results(sft_ckpt, data, results, raw_xy, psft_by_benchmark, save_path, dpi, save_format)

    print("\nFit summary")
    print("benchmark | model | train_r2 | val_r2 | val_rmse | equation")
    for benchmark, result in results.items():
        print(
            f"{benchmark} | {result['best_model']} | {result['r2_train']:.4f} | "
            f"{result['r2_val']:.4f} | {result['rmse_val']:.4f} | {result['equation']}"
        )

    predictions = {}
    if predict_flops_list:
        for benchmark, result in results.items():
            predictions[benchmark] = {
                float(flops): float(result["func"](np.asarray([flops], dtype=float), *result["params"])[0])
                for flops in predict_flops_list
            }
    return results, predictions, None
