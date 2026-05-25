# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# This implementation is based on:
# https://github.com/lhoyer/DAFormer
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. Licensed under the Apache License, Version 2.0
# A copy of the license is available at resources/license_daformer

import itertools
import logging
import math
import torch


def get_model_base(architecture, backbone):
    if 'daformer_' in architecture and 'proj' in architecture and 'mitb0' in backbone:
        return f'_base_/models/{architecture}_mitb0.py'
    if 'daformer_' in architecture and 'proj' in architecture and 'mitb5' in backbone:
        return f'_base_/models/{architecture}_mitb5.py'
    if 'daformer_' in architecture and 'mitb5' in backbone:
        return f'_base_/models/{architecture}_mitb5.py'
    assert 'mit' not in backbone or '-del' in backbone
    return {
        'dlv2_proj': '_base_/models/deeplabv2_proj_r50-d8.py', #
    }[architecture]


def get_pretraining_file(backbone):
    if 'mitb5' in backbone:
        return 'pretrained/mit_b5.pth'
    if 'mitb4' in backbone:
        return 'pretrained/mit_b4.pth'
    if 'mitb0' in backbone:
        return 'pretrained/mit_b5.pth'
    if 'r101v1c' in backbone:
        return 'pretrained/resnet50_v1c-2cccc1ad.pth'
    return {
        'r50v1c': 'pretrained/resnet50_v1c-2cccc1ad.pth',
    }[backbone]


def get_backbone_cfg(backbone):
    for i in [0, 1, 2, 3, 4, 5]:
        if backbone == f'mitb{i}':
            return dict(type=f'mit_b{i}')
        if backbone == f'mitb{i}-del':
            return dict(_delete_=True, type=f'mit_b{i}')
    return {
        'r50v1c': {
            'depth': 50
        },
        'r101v1c': {
            'depth': 101
        },
    }[backbone]


def update_decoder_in_channels(cfg, architecture, backbone):
    cfg.setdefault('model', {}).setdefault('decode_head', {})
    return cfg


def setup_rcs(cfg, temperature):
    cfg.setdefault('data', {}).setdefault('train', {})
    cfg['data']['train']['rare_class_sampling'] = dict(
        min_pixels=300, class_temp=temperature, min_crop_ratio=0.75)
    return cfg


def generate_experiment_cfgs(id):
    def config_from_vars():
        cfg = {'_base_': ['_base_/default_runtime.py'], 'n_gpus': n_gpus}
        if seed is not None:
            cfg['seed'] = seed

        # Setup model config
        architecture_mod = architecture
        model_base = get_model_base(architecture_mod, backbone)
        cfg['_base_'].append(model_base)
        cfg['model'] = {
            'pretrained': get_pretraining_file(backbone),
            'backbone': get_backbone_cfg(backbone),
        }
        cfg = update_decoder_in_channels(cfg, architecture_mod, backbone)

        # Setup UDA config
        if pseudo_random_crop:  # crop deleted
            cfg['_base_'].append(f'_base_/datasets/uda_{source}_to_{target}.py')
        elif fix_crop:  # crop fixed https://github.com/lhoyer/DAFormer/issues/6
            cfg['_base_'].append(f'_base_/datasets/uda_{source}_to_{target}.py')
        else:
            raise FileNotFoundError()
        cfg['_base_'].append(f'_base_/uda/{uda}.py')

        cfg.setdefault('uda', {})
        if method_name in uda and plcrop:
            cfg['uda']['pseudo_weight_ignore_top'] = 0
            cfg['uda']['pseudo_weight_ignore_bottom'] = 0
        cfg['data'] = dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=workers_per_gpu,
            train={})
        if method_name in uda and rcs_T is not None:
            cfg = setup_rcs(cfg, rcs_T)


        if mask_mode is not None:
            cfg.setdefault('uda', {})
            cfg['uda']['mask_mode'] = mask_mode
            cfg['uda']['mask_alpha'] = mask_alpha
            cfg['uda']['mask_pseudo_threshold'] = mask_pseudo_threshold
            cfg['uda']['mask_lambda'] = mask_lambda
            cfg['uda']['mask_generator'] = dict(
                type='block',
                mask_ratio=mask_ratio,
                mask_block_size=mask_block_size,
                _delete_=True)


        if method_name in uda:
            cfg['uda']['start_distribution_iter'] = start_distribution_iter
            if use_bank:
                cfg['uda']['memory_length'] = memory_length
            if pseudo_random_crop:
                cfg['uda']['pseudo_random_crop'] = pseudo_random_crop
                cfg['uda']['cat_max_ratio'] = cat_max_ratio
                crop_size = crop.split('x')
                crop_size = (int(crop_size[0]), int(crop_size[1]))
                cfg['uda']['crop_size'] = crop_size
                cfg['uda']['regen_pseudo'] = regen_pseudo
            cfg['model'].setdefault('auxiliary_head', {})
            cfg['model']['auxiliary_head']['in_channels'] = in_channels
            cfg['model']['auxiliary_head']['in_index'] = contrast_indexes
            cfg['model']['auxiliary_head']['input_transform'] = contrast_mode
            cfg['model']['auxiliary_head']['channels'] = channels
            cfg['model']['auxiliary_head']['num_convs'] = num_convs


            # cfg['model'].setdefault('classifier_head', {})
            # cfg['model']['classifier_head']['in_channels'] = in_channels
            # cfg['model']['classifier_head']['channels'] = channels
            # cfg['model']['classifier_head']['in_index'] = contrast_indexes
            # cfg['model']['classifier_head']['input_transform'] = contrast_mode




            if num_convs == 0:
                if contrast_mode == 'resize_concat':
                    cfg['model']['auxiliary_head']['channels'] = sum(in_channels)
                else:
                    cfg['model']['auxiliary_head']['channels'] = in_channels
            cfg['model']['auxiliary_head'].setdefault('loss_decode', {})
            cfg['model']['auxiliary_head']['loss_decode']['use_dist'] = use_dist
            cfg['model']['auxiliary_head']['loss_decode']['use_bank'] = use_bank
            cfg['model']['auxiliary_head']['loss_decode']['use_reg'] = use_reg
            cfg['model']['auxiliary_head']['loss_decode']['use_avg_pool'] = use_avg_pool
            cfg['model']['auxiliary_head']['loss_decode']['use_avg_pool'] = use_avg_pool
            cfg['model']['auxiliary_head']['loss_decode']['scale_min_ratio'] = scale_min_ratio
            cfg['model']['auxiliary_head']['loss_decode']['contrast_temp'] = contrastive_temperature
            cfg['model']['auxiliary_head']['loss_decode']['loss_weight'] = contrastive_weight
            cfg['model']['auxiliary_head']['loss_decode']['reg_relative_weight'] = reg_relative_weight




        if method_name in uda and enable_self_training:
            cfg['uda']['enable_self_training'] = enable_self_training

        # Setup optimizer and schedule
        if method_name in uda:
            cfg['optimizer_config'] = None  # Don't use outer optimizer

        cfg['_base_'].extend(
            [f'_base_/schedules/{opt}.py', f'_base_/schedules/{schedule}.py'])
        cfg['optimizer'] = {'lr': lr}
        cfg['optimizer'].setdefault('paramwise_cfg', {})
        cfg['optimizer']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optimizer']['paramwise_cfg']['custom_keys']
        if pmult:
            opt_param_cfg['head'] = dict(lr_mult=10.)
        if 'mit' in backbone:
            opt_param_cfg['pos_block'] = dict(decay_mult=0.)
            opt_param_cfg['norm'] = dict(decay_mult=0) #0

        # Setup runner
        cfg['runner'] = dict(type='IterBasedRunner', max_iters=iters)
        cfg['checkpoint_config'] = dict(
            by_epoch=False, interval=iters, max_keep_ckpts=1)
        cfg['evaluation'] = dict(interval=3000, metric='mDice')

        # Construct uda name
        uda_mod = uda

        if method_name in uda:
            if use_dist:
                uda_mod += '_DistCL'
            elif use_bank:
                uda_mod += '_BankCL'
            else:
                uda_mod += '_ProtoCL'
            if use_reg:
                uda_mod += f'-reg-w{reg_relative_weight * contrastive_weight}'
            uda_mod += f'-start-iter{start_distribution_iter}'
            uda_mod += f'-tau{contrastive_temperature}'
            if contrast_mode == 'multiple_select':
                for lyr in contrast_indexes:
                    uda_mod += f'-l{lyr}-w{contrastive_weight}'
            else:
                uda_mod += f'-l{contrast_indexes}-w{contrastive_weight}'
        if mask_mode is not None:
            uda_mod += f'_m{mask_block_size}-' \
                       f'{mask_ratio}-'
            if mask_alpha != 'same':
                uda_mod += f'a{mask_alpha}-'
            if mask_pseudo_threshold != 'same':
                uda_mod += f'p{mask_pseudo_threshold}-'
            uda_mod += {
                'separate': 'sep',
                'separateaug': 'spa',
                'separatesrc': 'sps',
                'separatesrcaug': 'spsa',
                'separatetrg': 'spt',
                'separatetrgaug': 'spta',
            }[mask_mode]
            if mask_lambda != 1:
                uda_mod += f'-w{mask_lambda}'

        if method_name in uda and rcs_T is not None:
            uda_mod += f'_rcs{rcs_T}'
        if method_name in uda and plcrop:
            uda_mod += '_cpl'
        if method_name in uda and enable_self_training:
            uda_mod += '_self'

        # Construct config name
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}'
        cfg['name_architecture'] = f'{architecture_mod}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture_mod
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' \
                          f'_{n_gpus}x{batch_size}_{iters}k'
        cfg['name'] = f"{cfg['name_architecture']}_{cfg['name_dataset']}"
        if seed is not None:
            cfg['name'] += f'_seed{seed}'
        cfg['name'] = cfg['name'].replace('.', '.').replace('True', 'T') \
            .replace('False', 'F').replace('cityscapes', 'cs') \
            .replace('synthia', 'syn').replace('dark_zurich', 'dz')
        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    cfgs = []
    method_name = 'sepico'
    n_gpus = 1
    batch_size = 4
    iters = 90000
    opt, lr, schedule, pmult = 'adamw', 0.00006, 'poly10warm', True
    crop = '256x256'
    datasets = [
        ('A2C', 'A4C'),
    ]
    architecture = None
    workers_per_gpu = 1
    samples_per_gpu = 4
    rcs_T = True
    plcrop = False
    fix_crop = True  # whether to fix the RandomCrop bug in DAForm
    start_distribution_iter = 4500  # 5000  # 10000
    enable_self_training = False
    pseudo_random_crop = False
    regen_pseudo = False
    cat_max_ratio = 1.0  # used for CBC

    # auxiliary head parameters
    in_channels = 1024  # in_channels = [256, 512, 1024, 2048]
    channels = 512      # default out_dim
    num_convs = 1
    contrast_indexes = 3  # int or list, depending on value of contrast_mode
    contrast_mode = None  # optional(None, 'resize_concat', 'multiple_select')
    use_dist = False
    use_bank = False
    memory_length = 200
    use_reg = True
    use_avg_pool = True
    scale_min_ratio = 1  # used for down-sampling
    contrastive_temperature = 6.5      # 6.5 is the best
    contrastive_weight = 1.0
    reg_relative_weight = 1.0  # reg_weight = reg_relative_weight * loss_weight in auxiliary head


    #MIC
    mask_mode = None
    mask_alpha = 0.999
    mask_pseudo_threshold = 0.968
    mask_lambda = 1
    mask_block_size = None
    mask_ratio = 0

    seeds = [76]  # random seeds


    # -------------------------------------------------------------------------
    # GTA -> Cityscapes [DistCL] (ResNet-101)
    # -------------------------------------------------------------------------
    if id == 1:
        # task
        model = ('daformer_sepaspp_proj', 'mitb5')
        architecture, backbone = model
        datasets = [
            ('A2C', 'A4C'),
        ]
        # general
        uda = 'sepico'
        pseudo_random_crop = True
        regen_pseudo = True
        # aux
        num_convs = 1
        in_channels = 1024
        contrast_indexes = 3  # int or list, depending on value of contrast_mode
        contrast_mode = None  # optional(None, 'resize_concat', 'multiple_select')
        # reg
        use_reg = True
        reg_relative_weight = 1.0
        # contrastive variants
        methods = [
            # use_dist, use_bank
            (True, False),  # DistCL
        ]


        # results
        for seed, (use_dist, use_bank), (source, target) in itertools.product(seeds, methods, datasets):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # GTA -> Cityscapes [BankCL] (ResNet-101)
    # -------------------------------------------------------------------------
    elif id == 2:
        # task
        model = ('dlv2_proj', 'r101v1c')
        architecture, backbone = model
        datasets = [
            ('gta', 'cityscapes'),
        ]
        # general
        uda = 'sepico'
        pseudo_random_crop = True
        regen_pseudo = True
        # aux
        num_convs = 2
        in_channels = 2048
        contrast_indexes = 3  # int or list, depending on value of contrast_mode
        contrast_mode = None  # optional(None, 'resize_concat', 'multiple_select')
        # reg
        use_reg = True
        reg_relative_weight = 1.0
        # contrastive variants
        methods = [
            # use_dist, use_bank
            (False, True),  # BankCL
        ]
        # results
        for seed, (use_dist, use_bank), (source, target) in itertools.product(seeds, methods, datasets):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # GTA -> Cityscapes [ProtoCL] (ResNet-101)
    # -------------------------------------------------------------------------
    elif id == 3:
        # task
        model = ('dlv2_proj', 'r101v1c')
        architecture, backbone = model
        datasets = [
            ('gta', 'cityscapes'),
        ]
        # general
        uda = 'sepico'
        pseudo_random_crop = True
        regen_pseudo = True
        # aux
        num_convs = 2
        in_channels = 2048
        contrast_indexes = 3  # int or list, depending on value of contrast_mode
        contrast_mode = None  # optional(None, 'resize_concat', 'multiple_select')
        # reg
        use_reg = True
        reg_relative_weight = 1.0
        # contrastive variants
        methods = [
            # use_dist, use_bank
            (False, False),  # ProtoCL
        ]
        # results
        for seed, (use_dist, use_bank), (source, target) in itertools.product(seeds, methods, datasets):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # GTA -> Cityscapes [DistCL] (MiT-B5)
    # -------------------------------------------------------------------------
    elif id == 4:
        # task
        model = ('daformer_sepaspp_proj', 'mitb0')
        architecture, backbone = model
        datasets = [
            ('A2C', 'A4C')
        ]
        # general
        uda = 'ATDOC_sepico'
        pseudo_random_crop = True
        regen_pseudo = True
        # aux
        num_convs = 2
        modes = [
            # in_channels, contrast_indexes, contrast_mode
            ([64, 128, 320, 512], [0, 1, 2, 3], 'resize_concat'),  # fusion
        ]

        # reg
        use_reg = True
        reg_relative_weight = 1
        # contrastive variants
        methods = [
            # use_dist, use_bank
            (True, False),  # DistCL
        ]

        #MIC
        mask_mode, mask_block_size, mask_ratio ='separatetrgaug', 32, 0.75


        # results
        for seed, mode, (use_dist, use_bank), (source, target) in itertools.product(seeds, modes, methods, datasets):
            in_channels, contrast_indexes, contrast_mode = mode
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # GTA -> Cityscapes [BankCL] (MiT-B5)
    # -------------------------------------------------------------------------
    elif id == 5:
        # task: CAMUS
        model = ('daformer_sepaspp_proj', 'mitb5')
        architecture, backbone = model
        datasets = [
            ('A2C', 'A4C')
        ]
        # general
        uda = 'MAE_sepico'
        pseudo_random_crop = False
        regen_pseudo = False
        # aux
        num_convs = 1
        modes = [
            # in_channels, contrast_indexes, contrast_mode
            ([64, 128, 320, 512], [0, 1, 2, 3], 'resize_concat'),  # fusion
        ]
        # reg
        use_reg = True
        reg_relative_weight = 1
        # contrastive variants
        methods = [
            # use_dist, use_bank
            (True, False),  # BankCL
        ]

        mask_mode, mask_block_size, mask_ratio = None, 32, 0.75  #'separatetrgaug'

        # results
        for seed, mode, (use_dist, use_bank), (source, target) in itertools.product(seeds, modes, methods, datasets):
            in_channels, contrast_indexes, contrast_mode = mode
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # GTA -> Cityscapes [ProtoCL] (MiT-B5)
    # -------------------------------------------------------------------------
    elif id == 6:
        # task: shenzhen_heart
        model = ('daformer_sepaspp_proj', 'mitb5')
        architecture, backbone = model
        datasets = [
            ('A2C', 'A4C')
        ]
        # general
        uda = 'MAE_sepico'
        pseudo_random_crop = False
        regen_pseudo = False
        # aux
        num_convs = 1
        modes = [
            # in_channels, contrast_indexes, contrast_mode
            ([64, 128, 320, 512], [0, 1, 2, 3], 'resize_concat'),  # fusion
        ]
        # reg
        use_reg = False
        reg_relative_weight = 1
        # contrastive variants
        methods = [
            # use_dist, use_bank
            (True, False),  # BankCL
        ]

        mask_mode, mask_block_size, mask_ratio = 'separatetrgaug', 32, 0.75  #'separatetrgaug'  'separatetrgaug'

        # results
        for seed, mode, (use_dist, use_bank), (source, target) in itertools.product(seeds, modes, methods, datasets):
            in_channels, contrast_indexes, contrast_mode = mode
            cfg = config_from_vars()
            cfgs.append(cfg)
        # task
        # model = ('daformer_sepaspp_proj', 'mitb5')
        # architecture, backbone = model
        # datasets = [
        #     ('A2C', 'A4C')
        # ]
        # # general
        # uda = 'sepico'
        # pseudo_random_crop = True
        # regen_pseudo = True
        # # aux
        # num_convs = 2
        # modes = [
        #     # in_channels, contrast_indexes, contrast_mode
        #     ([64, 128, 320, 512], [0, 1, 2, 3], 'resize_concat'),  # fusion
        # ]
        # # reg
        # use_reg = True
        # reg_relative_weight = 1.0
        # # contrastive variants
        # methods = [
        #     # use_dist, use_bank
        #     (False, False),  # ProtoCL
        # ]
        # # results
        # for seed, mode, (use_dist, use_bank), (source, target) in itertools.product(seeds, modes, methods, datasets):
        #     in_channels, contrast_indexes, contrast_mode = mode
        #     cfg = config_from_vars()
        #     cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Cityscapes -> Dark Zurich [DistCL] (ResNet-101)
    # -------------------------------------------------------------------------

    elif id == 100:
        # task: shenzhen_heart
        model = ('daformer_sepaspp_proj', 'mitb5')
        architecture, backbone = model
        datasets = [
            ('A2C', 'A2C')
        ]
        # general
        uda = 'MAE_sepico'
        pseudo_random_crop = False
        regen_pseudo = False
        # aux
        num_convs = 1
        modes = [
            # in_channels, contrast_indexes, contrast_mode
            ([64, 128, 320, 512], [0, 1, 2, 3], 'resize_concat'),  # fusion
        ]
        # reg
        use_reg = False
        reg_relative_weight = 1
        # contrastive variants
        methods = [
            # use_dist, use_bank
            (True, False),  # BankCL
        ]

        mask_mode, mask_block_size, mask_ratio = 'separatetrgaug', 32, 0.75  #'separatetrgaug'  'separatetrgaug'

        # results
        for seed, mode, (use_dist, use_bank), (source, target) in itertools.product(seeds, modes, methods, datasets):
            in_channels, contrast_indexes, contrast_mode = mode
            cfg = config_from_vars()
            cfgs.append(cfg)

    elif id == '10':
        # task: shenzhen_heart
        model = ('daformer_sepaspp_proj', 'mitb5')
        architecture, backbone = model
        datasets = [
            ('A2C', 'A4C')
        ]
        # general
        uda = 'MAE_sepico'
        pseudo_random_crop = False
        regen_pseudo = False
        # aux
        num_convs = 1
        modes = [
            # in_channels, contrast_indexes, contrast_mode
            ([64, 128, 320, 512], [0, 1, 2, 3], 'resize_concat'),  # fusion
        ]
        # reg
        use_reg = False
        reg_relative_weight = 1
        # contrastive variants
        methods = [
            # use_dist, use_bank
            (False, False),  # BankCL
        ]

        mask_mode, mask_block_size, mask_ratio = None, 32, 0.75  # 'separatetrgaug'

        # results
        for seed, mode, (use_dist, use_bank), (source, target) in itertools.product(seeds, modes, methods, datasets):
            in_channels, contrast_indexes, contrast_mode = mode
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Cityscapes -> Dark Zurich [DistCL] (MiT-B5)
    # -------------------------------------------------------------------------
    elif id == 8:
        seeds = [42]
        # task
        model = ('daformer_sepaspp_proj', 'mitb5')
        architecture, backbone = model
        datasets = [
            ('cityscapes', 'dark_zurich'),
        ]
        # general
        uda = 'sepico_dark'
        plcrop = False  # not needed for Dark Zurich
        iters = 60000
        pseudo_random_crop = True
        regen_pseudo = True
        # aux
        num_convs = 2
        modes = [
            # in_channels, contrast_indexes, contrast_mode
            ([64, 128, 320, 512], [0, 1, 2, 3], 'resize_concat'),  # fusion
        ]
        # reg
        use_reg = True
        reg_relative_weight = 1.0
        # contrastive variants
        methods = [
            # use_dist, use_bank
            (True, False),  # DistCL
            # (False, False),  # ProtoCL
            # (False, True),  # BankCL
        ]
        # dark exclusive
        corresp_root = 'data/dark_zurich/corresp/train/night/'
        weights = torch.log(torch.FloatTensor(
            [0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341, 0.00207795, 0.0055127, 0.15928651,
             0.01157818, 0.04018982, 0.01218957, 0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
             0.00413907]))
        std = 0.05  # 0.16 for test
        class_weight_seg = (torch.mean(weights) - weights) / torch.std(weights) * std + 1.0
        class_weight_seg = class_weight_seg.numpy().tolist()
        # results
        for seed, mode, (use_dist, use_bank), (source, target) in itertools.product(seeds, modes, methods, datasets):
            in_channels, contrast_indexes, contrast_mode = mode
            cfg = config_from_vars()
            cfgs.append(cfg)
    else:
        raise NotImplementedError('Unknown id {}'.format(id))

    return cfgs
