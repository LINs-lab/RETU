def calculate_flops_per_token(model_config, seq_len):
    """Estimate forward-pass FLOPs per token for a decoder-only GQA model."""
    num_layers = model_config["num_layers"]
    hidden_size = model_config["hidden_size"]
    vocab_size = model_config["vocab_size"]
    ffn_dim = model_config["ffn_dim"]
    num_kv_heads = model_config["num_kv_heads"]
    num_attention_heads = model_config["num_attention_heads"]

    head_dim = hidden_size // num_attention_heads
    q_size = num_attention_heads * head_dim
    k_size = num_kv_heads * head_dim
    v_size = num_kv_heads * head_dim

    mlp_params = hidden_size * ffn_dim * 3
    attn_linear_params = hidden_size * (
        q_size + k_size + v_size + num_attention_heads * head_dim
    )
    embedding_and_lm_head_params = vocab_size * hidden_size * 2
    dense_params = (mlp_params + attn_linear_params) * num_layers + embedding_and_lm_head_params

    dense_flops = 2 * dense_params
    attention_flops = 4 * seq_len * head_dim * num_attention_heads * num_layers
    return dense_flops + attention_flops


def calculate_sft_flops(batch_size, seq_len, model_config, batch_num=1):
    """Estimate SFT FLOPs for a number of optimizer steps.

    The estimator uses forward+backward = 3x forward FLOPs.  The returned
    ``total_flops_eflops`` is rounded to two decimals because the historical
    fitting caches are keyed in EFLOPs.
    """
    tokens_per_batch = batch_size * seq_len
    forward_flops_per_token = calculate_flops_per_token(model_config, seq_len)
    total_flops_per_token = forward_flops_per_token * 3
    flops_per_batch = tokens_per_batch * total_flops_per_token
    total_flops = flops_per_batch * batch_num
    return {
        "batch_num": batch_num,
        "tokens_per_batch": tokens_per_batch,
        "flops_per_batch": flops_per_batch,
        "total_flops_eflops": round(total_flops / 1e18, 2),
    }


def step2flops_sft(model_config, sft_train_config, sft_steps):
    """Map each SFT checkpoint step to cumulative SFT EFLOPs."""
    step2flops = {}
    accum_flops = 0
    for i, step in enumerate(sft_steps):
        if step == 0:
            step2flops[step] = 0
            continue

        prev_step = sft_steps[i - 1] if i > 0 else 0
        batch_num = step - prev_step
        segment = calculate_sft_flops(
            batch_size=sft_train_config["batch_size"],
            seq_len=sft_train_config["max_length"],
            model_config=model_config,
            batch_num=batch_num,
        )
        accum_flops = round(accum_flops + segment["total_flops_eflops"], 2)
        print(f"SFT step={step} segment_steps={batch_num} cumulative_eflops={accum_flops}")
        step2flops[step] = accum_flops
    return step2flops


def _nearest_key(target, mapping, tolerance=0.2):
    """Match historical cache keys that mix one- and two-decimal EFLOPs."""
    if target in mapping:
        return target
    nearest = min(mapping.keys(), key=lambda key: abs(float(key) - float(target)))
    if abs(float(nearest) - float(target)) <= tolerance:
        return nearest
    return target


def sft_then_rl_flops2val_performance(
    step2flops_per_ckpt,
    flops2val_performance_sft,
    step2flops2val_performance_rl,
    branch_sft_step,
):
    """Build the cumulative FLOPs/performance curve for one SFT->RL branch."""
    combined = {}
    for sft_step, sft_flops in step2flops_per_ckpt.items():
        if sft_step > branch_sft_step:
            break
        sft_key = _nearest_key(sft_flops, flops2val_performance_sft)
        if sft_key in flops2val_performance_sft:
            combined[sft_key] = flops2val_performance_sft[sft_key]

    branch_flops = _nearest_key(step2flops_per_ckpt[branch_sft_step], flops2val_performance_sft)
    rl_flops2val_performance = step2flops2val_performance_rl[branch_sft_step]

    for rl_flops, rl_performance in rl_flops2val_performance.items():
        combined[round(branch_flops + rl_flops, 2)] = rl_performance
    return combined, branch_flops

