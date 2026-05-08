# SFT training configurations used by the FLOPs estimator.
#
# The sequence lengths are empirical mean prompt+response lengths for each SFT
# dataset.  They intentionally differ across datasets because trajectory length
# is a large part of the compute budget.

sft_train_config = {
    "max_length": 3771,
    "batch_size": 512,
    "per_ckpt_interval": 360,
}

hard102K_train_config = {
    "max_length": 8532 + 101,
    "batch_size": 512,
    "per_ckpt_interval": 360,
}

easy102K_train_config = {
    "max_length": 2153 + 64,
    "batch_size": 512,
    "per_ckpt_interval": 360,
}

uniform102K_train_config = {
    "max_length": 3673 + 74,
    "batch_size": 512,
    "per_ckpt_interval": 360,
}

s1K_train_config = {
    "max_length": 9884 + 127,
    "batch_size": 16,
    "per_ckpt_interval": 62,
}

