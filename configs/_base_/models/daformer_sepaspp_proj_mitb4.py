# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
# Adapted from: https://github.com/lhoyer/DAFormer

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoderProjector',
    pretrained='pretrained/mit_b4.pth',
    backbone=dict(type='mit_b4', style='pytorch'),
    # pretrained=None,
    # backbone=dict(
    #     type='UNet',
    #     in_channels=2,
    #     base_channels=256,
    #     num_stages=5,
    #     strides=(1, 1, 1, 1, 1),
    #     enc_num_convs=(2, 2, 2, 2, 2),
    #     dec_num_convs=(2, 2, 2, 2),
    #     downsamples=(True, True, True, True),
    #     enc_dilations=(1, 1, 1, 1, 1),
    #     dec_dilations=(1, 1, 1, 1),
    #     with_cp=False,
    #     conv_cfg=None,
    #     norm_cfg=norm_cfg,
    #     act_cfg=dict(type='ReLU'),
    #     upsample_cfg=dict(type='InterpConv'),
    #     norm_eval=False),
    # decode_head=dict(
    #     type='FCNHead',
    #     in_channels=64,
    #     in_index=4,
    #     channels=64,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=128,
    #     in_index=3,
    #     channels=64,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),


    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.5, #0.1
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,  #false
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg)),
        loss_decode=(#dict(type = 'DiceLoss', use_sigmoid = True, loss_weight = 1.0),
                     # dict(type = 'FocalLoss', use_sigmoid = True, loss_weight = 1.0)
                     dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                    )),
            # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='ProjHead',
        in_channels=[64, 128, 320, 512], #[64, 128, 320, 512]
        in_index=[0, 1, 2, 3],  # int or list, depending on value of input_transform
        input_transform='resize_concat',  # optional(None, 'resize_concat', 'multiple_select')
        channels=256,
        num_convs=2,
        dropout_ratio=0.7, #0.1
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='ContrastiveLoss', use_dist=False, use_bank=False, use_reg=False,
            use_avg_pool=True, scale_min_ratio=1, num_classes=2,
            contrast_temp=100., loss_weight=0.5, reg_relative_weight=0.01)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
