# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# SePiCo
uda = dict(
    type='ATDOC_SePiCo',
    source_only=False,
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    enable_self_training=False,
    enable_strong_aug=True,
    start_distribution_iter=0,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    mask_mode = None,
    mask_alpha = 'same',
    mask_lambda=0,
    mask_generator=None,
    print_grad_magnitude=False,
    debug_img_interval=1000,
)
use_ddp_wrapper = True
