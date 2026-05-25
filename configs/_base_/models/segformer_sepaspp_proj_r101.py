# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications:
# - BN instead of SyncBN
# - Replace MiT with ResNet backbone
# This work is licensed under the NVIDIA Source Code License
# A copy of the license is available at resources/license_segformer

_base_ = ['../../_base_/models/segformer.py']

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoderProjector',
    pretrained='pretrained/resnet101_v1c-e67eebb6.pth',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768, conv_kernel_size=1),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='ProjHead',
        in_channels=2048,
        in_index=3,   # int or list, depending on value of input_transform
        input_transform=None,  # optional(None, 'resize_concat', 'multiple_select')
        channels=128,
        num_convs=2,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='ContrastiveLoss', use_dist=False, use_bank=False, use_reg=False,
            use_avg_pool=True, scale_min_ratio=1, num_classes=2,
            contrast_temp=100., loss_weight=1.0, reg_relative_weight=0.01, )),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
