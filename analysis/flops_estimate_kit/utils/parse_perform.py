from .flops_est import sft_then_rl_flops2val_performance


def get_overall_performance(performance, data_num_dict, overall_key):
    """Compute a weighted aggregate performance value."""
    included = set(data_num_dict)
    if overall_key == "easy_overall":
        included = {"gsm8k", "math"}
    elif overall_key == "med_overall":
        included = {"minerva", "olympiad_bench"}
    elif overall_key == "hard_overall":
        included = {"aime", "aime25"}

    weighted_sum = 0.0
    total_count = 0
    for benchmark_name, data_count in data_num_dict.items():
        if benchmark_name not in included or benchmark_name not in performance:
            continue
        weighted_sum += performance[benchmark_name] * data_count
        total_count += data_count

    if total_count == 0:
        raise ValueError(f"No benchmark overlap for aggregate metric: {overall_key}")
    return round(weighted_sum / total_count, 2)


def add_aggregate_metrics(curves, data_num_dict):
    """Add overall/easy/medium/hard aggregate metrics in-place."""
    for _, flops2performance in curves.items():
        for _, performance in flops2performance.items():
            for key in ("overall", "easy_overall", "med_overall", "hard_overall"):
                try:
                    performance[key] = get_overall_performance(performance, data_num_dict, key)
                except ValueError:
                    pass
    return curves


def get_combined_flops2val_performance_zoo(
    step2flops_per_ckpt,
    flops2val_performance_sft,
    step2flops2val_performance_rl,
    sft_ckpt_num_list,
    data_num_dict,
):
    """Create all SFT->RL branch curves requested by a fitting script."""
    combined = {}
    branch_flops = {}
    for branch_step in sft_ckpt_num_list:
        combined[branch_step], branch_flops[branch_step] = sft_then_rl_flops2val_performance(
            step2flops_per_ckpt,
            flops2val_performance_sft,
            step2flops2val_performance_rl,
            branch_step,
        )
    return add_aggregate_metrics(combined, data_num_dict), branch_flops

