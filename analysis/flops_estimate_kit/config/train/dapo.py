# Minimal DAPO configuration retained for metadata compatibility.
#
# The public curve-fitting scripts load precomputed RL FLOPs/performance caches,
# so these values are not used to regenerate RL FLOPs by default.

train_config = {
    "gen_prompt_bsz": 128,
    "train_prompt_bsz": 64,
    "n_resp_per_prompt": 8,
    "max_prompt_length": 1024,
    "max_response_length": 8192,
    "dynamic_sampling_iterations": 2,
}

