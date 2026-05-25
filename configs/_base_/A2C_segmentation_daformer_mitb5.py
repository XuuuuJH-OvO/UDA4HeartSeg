# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes High-Resolution Data Loading
    '../_base_/datasets/seg_A2C.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
norm_cfg = dict(type ='BN')
# Random Seed
seed = 0
# Optimizer Hyperparameters3
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=3000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=3000, max_keep_ckpts=1)
evaluation = dict(interval=3000, metric='mDice')
# Meta Information for Result Analysis
name = 'A2C_segmentation_daformer_mitb5'
exp = 'basic'
name_dataset = 'A4C'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
