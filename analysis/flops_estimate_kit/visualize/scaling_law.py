def get_rl_scatters(sft_ckpt, combined_flops2val_performance, branch_flops):
    """Convert cumulative branch points into relative RL FLOPs and gains."""
    start_flops = branch_flops[sft_ckpt]
    start_performance = combined_flops2val_performance[sft_ckpt][start_flops]

    relative_curve = {}
    for accum_flops, performance in combined_flops2val_performance[sft_ckpt].items():
        if accum_flops < start_flops:
            continue
        relative_curve[round(accum_flops - start_flops, 2)] = {
            benchmark: performance[benchmark] - start_performance[benchmark]
            for benchmark in performance
            if benchmark in start_performance
        }
    return relative_curve

