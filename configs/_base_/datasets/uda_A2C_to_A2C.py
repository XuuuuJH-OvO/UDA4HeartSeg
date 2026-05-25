# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'A4CDataset'
data_root = '/home/server4090/lyx/data/CAMUS_A2C/train/'  #'/home/server4090/lyx/data/shenzhen_heart/model_data/A2C/train'         #'/home/server4090/lyx/data/CAMUS_A2C/train/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)
A2C_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(256, 256)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0),
    # dict(type='PhotoMetricDistortion'),  # is applied later in sepico.py
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
A4C_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(256, 256)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.),  # is applied later in sepico.py
    dict(type='RandomFlip', prob=0),
    # dict(type='PhotoMetricDistortion'),  # is applied later in sepico.py
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),  # is applied later in sepico.py
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            #dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type='UDADataset',
        source=dict(
            type='A2CDataset',
            data_root='/home/server4090/lyx/data/CAMUS_A2C/train/',  #'/home/server4090/lyx/data/shenzhen_heart/model_data/A3C/train/',   # '/home/server4090/lyx/data/CAMUS_A2C/train/'
            img_dir='images',   #'images',
            ann_dir='labels',
            pipeline=A2C_train_pipeline),
        target=dict(
            type='A2CDataset',
            data_root='/home/server4090/lyx/data/shenzhen_heart/model_data/A2C/train/',  #'/home/server4090/lyx/data/shenzhen_heart/model_data/A2C/train/', #'/home/server4090/lyx/data/CAMUS_A4C/train/'
            img_dir='images',         #'images',
            ann_dir='labels',
            pipeline=A2C_train_pipeline)),
    val=dict(
        type='A2CDataset',
        data_root='/home/server4090/lyx/data/shenzhen_heart/model_data/A2C/test', #'/home/server4090/lyx/data/shenzhen_heart/model_data/A2C/test/' #'/home/server4090/lyx/data/CAMUS_A2C/val/'
        img_dir='images',
        ann_dir='labels',
        pipeline=test_pipeline),
    test=dict(
        type='A2CDataset',
        data_root='/home/server4090/lyx/data/shenzhen_heart/model_data/A2C/test', #'/home/server4090/lyx/data/shenzhen_heart/model_data/A2C/test/'
        img_dir='images',
        ann_dir='labels',
        pipeline=test_pipeline))
