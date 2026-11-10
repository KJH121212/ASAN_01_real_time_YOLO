auto_scale_lr = dict(base_batch_size=1024)
backend_args = dict(backend='local')
base_lr = 0.004
codec = dict(
    input_size=(
        192,
        256,
    ),
    normalize=False,
    sigma=(
        4.9,
        5.66,
    ),
    simcc_split_ratio=2.0,
    type='mmpose.codecs.SimCCLabel',
    use_dark=False)
custom_hooks = [
    dict(
        ema_type='mmpose.engine.hooks.ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='mmengine.hooks.EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=180,
        switch_pipeline=[
            dict(
                backend_args=dict(backend='local'),
                type='mmpose.datasets.LoadImage'),
            dict(type='mmpose.datasets.GetBBoxCenterScale'),
            dict(direction='horizontal', type='mmpose.datasets.RandomFlip'),
            dict(type='mmpose.datasets.RandomHalfBody'),
            dict(
                rotate_factor=60,
                scale_factor=[
                    0.75,
                    1.25,
                ],
                shift_factor=0.0,
                type=
                'mmpose.datasets.transforms.common_transforms.RandomBBoxTransform'
            ),
            dict(
                input_size=(
                    192,
                    256,
                ), type='mmpose.datasets.TopdownAffine'),
            dict(type='mmdet.datasets.transforms.YOLOXHSVRandomAug'),
            dict(
                transforms=[
                    dict(p=0.1, type='albumentations.augmentations.Blur'),
                    dict(
                        p=0.1, type='albumentations.augmentations.MedianBlur'),
                    dict(
                        max_height=0.4,
                        max_holes=1,
                        max_width=0.4,
                        min_height=0.2,
                        min_holes=1,
                        min_width=0.2,
                        p=0.5,
                        type='albumentations.augmentations.CoarseDropout'),
                ],
                type=
                'mmpose.datasets.transforms.common_transforms.Albumentation'),
            dict(
                encoder=dict(
                    input_size=(
                        192,
                        256,
                    ),
                    normalize=False,
                    sigma=(
                        4.9,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='mmpose.codecs.SimCCLabel',
                    use_dark=False),
                type='mmpose.datasets.GenerateTarget'),
            dict(type='mmpose.datasets.PackPoseInputs'),
        ],
        type='mmdet.engine.hooks.PipelineSwitchHook'),
]
data_mode = 'topdown'
data_root = '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5'
dataset_type = 'mmpose.datasets.CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=-1,
        rule=None,
        save_best=None,
        type='CheckpointHook'),
    logger=dict(interval=50, type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'),
    visualization=dict(
        enable=False, type='mmpose.engine.hooks.PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/rtmpose/v2.0_with_public_step_5_fixed/epoch_390.pth'
log_level = 'INFO'
log_processor = dict(
    by_epoch=True,
    num_digits=6,
    type='mmengine.runner.LogProcessor',
    window_size=50)
max_epochs = 210
model = dict(
    backbone=dict(
        act_cfg=dict(type='torch.nn.SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.67,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth',
            prefix='backbone.',
            type='mmengine.model.PretrainedInit'),
        norm_cfg=dict(type='torch.nn.SyncBatchNorm'),
        out_indices=(4, ),
        type='mmdet.models.CSPNeXt',
        widen_factor=0.75),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmpose.models.PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='mmpose.codecs.SimCCLabel',
            use_dark=False),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn='SiLU',
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False),
        in_channels=768,
        in_featuremap_size=(
            6,
            8,
        ),
        input_size=(
            192,
            256,
        ),
        loss=dict(
            beta=10.0,
            label_softmax=True,
            type='mmpose.models.KLDiscretLoss',
            use_target_weight=True),
        out_channels=17,
        simcc_split_ratio=2.0,
        type='mmpose.models.RTMCCHead'),
    test_cfg=dict(flip_test=True),
    type='mmpose.models.TopdownPoseEstimator')
optim_wrapper = dict(
    optimizer=dict(lr=0.004, type='torch.optim.AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.0, weight_decay_mult=0.0))),
    type='mmengine.optim.OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=1000,
        start_factor=1e-05,
        type='mmengine.optim.LinearLR'),
    dict(
        T_max=105,
        begin=105,
        by_epoch=True,
        convert_to_iter_based=True,
        end=210,
        eta_min=0.0002,
        type='mmengine.optim.CosineAnnealingLR'),
]
randomness = dict(seed=21)
resume = True
stage2_num_epochs = 30
test_cfg = dict()
test_dataloader = dict(
    batch_size=256,
    dataset=dict(
        ann_file='annotations/person_keypoints_val.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root=
        '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5',
        metainfo=dict(
            dataset_name='coco',
            from_file='configs/_base_/datasets/coco.py',
            keypoint_weights=[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]),
        pipeline=[
            dict(
                backend_args=dict(backend='local'),
                type='mmpose.datasets.LoadImage'),
            dict(type='mmpose.datasets.GetBBoxCenterScale'),
            dict(
                input_size=(
                    192,
                    256,
                ), type='mmpose.datasets.TopdownAffine'),
            dict(type='mmpose.datasets.PackPoseInputs'),
        ],
        test_mode=True,
        type='mmpose.datasets.CocoDataset'),
    drop_last=False,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(
        round_up=False, shuffle=False, type='mmengine.dataset.DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5/annotations/person_keypoints_val.json',
    type='mmpose.evaluation.CocoMetric')
train_cfg = dict(by_epoch=True, max_epochs=420, val_interval=10)
train_dataloader = dict(
    batch_size=256,
    dataset=dict(
        ann_file='annotations/person_keypoints_train.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root=
        '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5',
        metainfo=dict(
            dataset_name='coco',
            from_file='configs/_base_/datasets/coco.py',
            keypoint_weights=[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]),
        pipeline=[
            dict(
                backend_args=dict(backend='local'),
                type='mmpose.datasets.LoadImage'),
            dict(type='mmpose.datasets.GetBBoxCenterScale'),
            dict(direction='horizontal', type='mmpose.datasets.RandomFlip'),
            dict(type='mmpose.datasets.RandomHalfBody'),
            dict(
                rotate_factor=80,
                scale_factor=[
                    0.6,
                    1.4,
                ],
                type=
                'mmpose.datasets.transforms.common_transforms.RandomBBoxTransform'
            ),
            dict(
                input_size=(
                    192,
                    256,
                ), type='mmpose.datasets.TopdownAffine'),
            dict(type='mmdet.datasets.transforms.YOLOXHSVRandomAug'),
            dict(
                transforms=[
                    dict(p=0.1, type='albumentations.augmentations.Blur'),
                    dict(
                        p=0.1, type='albumentations.augmentations.MedianBlur'),
                    dict(
                        max_height=0.4,
                        max_holes=1,
                        max_width=0.4,
                        min_height=0.2,
                        min_holes=1,
                        min_width=0.2,
                        p=1.0,
                        type='albumentations.augmentations.CoarseDropout'),
                ],
                type=
                'mmpose.datasets.transforms.common_transforms.Albumentation'),
            dict(
                encoder=dict(
                    input_size=(
                        192,
                        256,
                    ),
                    normalize=False,
                    sigma=(
                        4.9,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='mmpose.codecs.SimCCLabel',
                    use_dark=False),
                type='mmpose.datasets.GenerateTarget'),
            dict(type='mmpose.datasets.PackPoseInputs'),
        ],
        type='mmpose.datasets.CocoDataset'),
    drop_last=True,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='mmengine.dataset.DefaultSampler'))
train_pipeline = [
    dict(backend_args=dict(backend='local'), type='mmpose.datasets.LoadImage'),
    dict(type='mmpose.datasets.GetBBoxCenterScale'),
    dict(direction='horizontal', type='mmpose.datasets.RandomFlip'),
    dict(type='mmpose.datasets.RandomHalfBody'),
    dict(
        rotate_factor=80,
        scale_factor=[
            0.6,
            1.4,
        ],
        type='mmpose.datasets.transforms.common_transforms.RandomBBoxTransform'
    ),
    dict(input_size=(
        192,
        256,
    ), type='mmpose.datasets.TopdownAffine'),
    dict(type='mmdet.datasets.transforms.YOLOXHSVRandomAug'),
    dict(
        transforms=[
            dict(p=0.1, type='albumentations.augmentations.Blur'),
            dict(p=0.1, type='albumentations.augmentations.MedianBlur'),
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=1.0,
                type='albumentations.augmentations.CoarseDropout'),
        ],
        type='mmpose.datasets.transforms.common_transforms.Albumentation'),
    dict(
        encoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='mmpose.codecs.SimCCLabel',
            use_dark=False),
        type='mmpose.datasets.GenerateTarget'),
    dict(type='mmpose.datasets.PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=dict(backend='local'), type='mmpose.datasets.LoadImage'),
    dict(type='mmpose.datasets.GetBBoxCenterScale'),
    dict(direction='horizontal', type='mmpose.datasets.RandomFlip'),
    dict(type='mmpose.datasets.RandomHalfBody'),
    dict(
        rotate_factor=60,
        scale_factor=[
            0.75,
            1.25,
        ],
        shift_factor=0.0,
        type='mmpose.datasets.transforms.common_transforms.RandomBBoxTransform'
    ),
    dict(input_size=(
        192,
        256,
    ), type='mmpose.datasets.TopdownAffine'),
    dict(type='mmdet.datasets.transforms.YOLOXHSVRandomAug'),
    dict(
        transforms=[
            dict(p=0.1, type='albumentations.augmentations.Blur'),
            dict(p=0.1, type='albumentations.augmentations.MedianBlur'),
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=0.5,
                type='albumentations.augmentations.CoarseDropout'),
        ],
        type='mmpose.datasets.transforms.common_transforms.Albumentation'),
    dict(
        encoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='mmpose.codecs.SimCCLabel',
            use_dark=False),
        type='mmpose.datasets.GenerateTarget'),
    dict(type='mmpose.datasets.PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=256,
    dataset=dict(
        ann_file='annotations/person_keypoints_val.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root=
        '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5',
        metainfo=dict(
            dataset_name='coco',
            from_file='configs/_base_/datasets/coco.py',
            keypoint_weights=[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]),
        pipeline=[
            dict(
                backend_args=dict(backend='local'),
                type='mmpose.datasets.LoadImage'),
            dict(type='mmpose.datasets.GetBBoxCenterScale'),
            dict(
                input_size=(
                    192,
                    256,
                ), type='mmpose.datasets.TopdownAffine'),
            dict(type='mmpose.datasets.PackPoseInputs'),
        ],
        test_mode=True,
        type='mmpose.datasets.CocoDataset'),
    drop_last=False,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(
        round_up=False, shuffle=False, type='mmengine.dataset.DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5/annotations/person_keypoints_val.json',
    type='mmpose.evaluation.CocoMetric')
val_pipeline = [
    dict(backend_args=dict(backend='local'), type='mmpose.datasets.LoadImage'),
    dict(type='mmpose.datasets.GetBBoxCenterScale'),
    dict(input_size=(
        192,
        256,
    ), type='mmpose.datasets.TopdownAffine'),
    dict(type='mmpose.datasets.PackPoseInputs'),
]
vis_backends = [
    dict(type='mmengine.visualization.LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmpose.visualization.PoseLocalVisualizer',
    vis_backends=[
        dict(type='mmengine.visualization.LocalVisBackend'),
    ])
work_dir = '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/rtmpose/v2.0_with_public_step_5_fixed'
