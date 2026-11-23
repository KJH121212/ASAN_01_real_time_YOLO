"""RTMPose-M (COCO-23: Body 17 + Foot 6, 256x192) 추론 설정 파일."""

default_scope = 'mmpose'

coco23_metainfo = dict(
    dataset_name='coco23',
    paper_info=dict(
        author='Unified COCO-WholeBody + Rehab Sapiens',
        title='COCO-23 (Body 17 + Foot 6)',
        year='2026',
    ),
    keypoint_info={
        0: dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1: dict(name='left_eye', id=1, color=[51, 153, 255], type='upper', swap='right_eye'),
        2: dict(name='right_eye', id=2, color=[51, 153, 255], type='upper', swap='left_eye'),
        3: dict(name='left_ear', id=3, color=[51, 153, 255], type='upper', swap='right_ear'),
        4: dict(name='right_ear', id=4, color=[51, 153, 255], type='upper', swap='left_ear'),
        5: dict(name='left_shoulder', id=5, color=[0, 255, 0], type='upper', swap='right_shoulder'),
        6: dict(name='right_shoulder', id=6, color=[255, 128, 0], type='upper', swap='left_shoulder'),
        7: dict(name='left_elbow', id=7, color=[0, 255, 0], type='upper', swap='right_elbow'),
        8: dict(name='right_elbow', id=8, color=[255, 128, 0], type='upper', swap='left_elbow'),
        9: dict(name='left_wrist', id=9, color=[0, 255, 0], type='upper', swap='right_wrist'),
        10: dict(name='right_wrist', id=10, color=[255, 128, 0], type='upper', swap='left_wrist'),
        11: dict(name='left_hip', id=11, color=[0, 255, 0], type='lower', swap='right_hip'),
        12: dict(name='right_hip', id=12, color=[255, 128, 0], type='lower', swap='left_hip'),
        13: dict(name='left_knee', id=13, color=[0, 255, 0], type='lower', swap='right_knee'),
        14: dict(name='right_knee', id=14, color=[255, 128, 0], type='lower', swap='left_knee'),
        15: dict(name='left_ankle', id=15, color=[0, 255, 0], type='lower', swap='right_ankle'),
        16: dict(name='right_ankle', id=16, color=[255, 128, 0], type='lower', swap='left_ankle'),
        17: dict(name='left_big_toe', id=17, color=[255, 0, 255], type='lower', swap='right_big_toe'),
        18: dict(name='left_small_toe', id=18, color=[255, 0, 255], type='lower', swap='right_small_toe'),
        19: dict(name='left_heel', id=19, color=[255, 0, 255], type='lower', swap='right_heel'),
        20: dict(name='right_big_toe', id=20, color=[255, 128, 0], type='lower', swap='left_big_toe'),
        21: dict(name='right_small_toe', id=21, color=[255, 128, 0], type='lower', swap='left_small_toe'),
        22: dict(name='right_heel', id=22, color=[255, 128, 0], type='lower', swap='left_heel'),
    },
    skeleton_info={
        0: dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1: dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2: dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3: dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4: dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5: dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6: dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7: dict(link=('left_shoulder', 'right_shoulder'), id=7, color=[51, 153, 255]),
        8: dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9: dict(link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10: dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11: dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12: dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13: dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14: dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15: dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16: dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17: dict(link=('left_ankle', 'left_big_toe'), id=17, color=[255, 0, 255]),
        18: dict(link=('left_ankle', 'left_small_toe'), id=18, color=[255, 0, 255]),
        19: dict(link=('left_ankle', 'left_heel'), id=19, color=[255, 0, 255]),
        20: dict(link=('right_ankle', 'right_big_toe'), id=20, color=[255, 128, 0]),
        21: dict(link=('right_ankle', 'right_small_toe'), id=21, color=[255, 128, 0]),
        22: dict(link=('right_ankle', 'right_heel'), id=22, color=[255, 128, 0]),
    },
    joint_weights=[
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.2, 1.2, 1.5, 1.5,
        1.0, 1.0, 1.2, 1.2, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035,
        0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
        0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
        0.068, 0.066, 0.066, 0.092, 0.094, 0.094,
    ],
)

codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
    ),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=23,
        input_size=(192, 256),
        in_featuremap_size=(6, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.0,
            label_softmax=True,
        ),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=False),
)

val_pipeline = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='PackPoseInputs'),
]

val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_mode='topdown',
        test_mode=True,
        metainfo=coco23_metainfo,
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader
