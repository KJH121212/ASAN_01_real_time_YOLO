# 한국어 코드 해설: RTMPose 모델 구조와 학습·검증 파이프라인 설정입니다. 주석은 모델 설정값을 바꾸지 않습니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# `auto_scale_lr` 값에 `dict(base_batch_size=1024)` 호출 결과를 저장합니다. 대상: `auto_scale_lr`.
auto_scale_lr = dict(base_batch_size=1024)
# 데이터 파일 접근 방식 설정에 `dict(backend='local')` 호출 결과를 저장합니다. 대상: `backend_args`.
backend_args = dict(backend='local')
# 학습 기본 학습률에 0.004을 저장합니다. 대상: `base_lr`.
base_lr = 0.004
# 관절 좌표와 모델 출력 사이의 인코딩·디코딩 규칙에 `dict(input_size=(192, 256), normalize=False, sigma=(4.9, 5.66), simcc_split_ratio=2.0, type='mmpose.codecs.SimCCLabel', use_dark=False)` 호출 결과를 저장합니다. 대상: `codec`.
codec = dict(
    # 구성 항목: 모델에 입력할 영상 크기(가로·세로)(`input_size`)에 순서·구성을 지정한 `(192, 256)`을 지정합니다.
    input_size=(
        # 앞줄에서 시작한 식·인수·자료 구조를 `192,` 내용으로 이어서 구성합니다.
        192,
        # 앞줄에서 시작한 식·인수·자료 구조를 `256,` 내용으로 이어서 구성합니다.
        256,
    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
    ),
    # 구성 항목: `normalize` 값(`normalize`)에 False을 지정합니다.
    normalize=False,
    # 구성 항목: `sigma` 값(`sigma`)에 순서·구성을 지정한 `(4.9, 5.66)`을 지정합니다.
    sigma=(
        # 앞줄에서 시작한 식·인수·자료 구조를 `4.9,` 내용으로 이어서 구성합니다.
        4.9,
        # 앞줄에서 시작한 식·인수·자료 구조를 `5.66,` 내용으로 이어서 구성합니다.
        5.66,
    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
    ),
    # 구성 항목: simcc split 비율(`simcc_split_ratio`)에 2.0을 지정합니다.
    simcc_split_ratio=2.0,
    # 구성 항목: `type` 값(`type`)에 'mmpose.codecs.SimCCLabel'을 지정합니다.
    type='mmpose.codecs.SimCCLabel',
    # 구성 항목: `use_dark` 값(`use_dark`)에 False을 지정합니다.
    use_dark=False)
# `custom_hooks` 값에 순서·구성을 지정한 `[dict(ema_type='mmpose.engine.hooks.ExpMomentumEMA', momentum=0.0002, priority=49, type='mmengine.hooks.EMAHook', update_buffers=True), dict(switch_epoch=180, switch_pip…`을 저장합니다. 대상: `custom_hooks`.
custom_hooks = [
    # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
    dict(
        # 구성 항목: `ema_type` 값(`ema_type`)에 'mmpose.engine.hooks.ExpMomentumEMA'을 지정합니다.
        ema_type='mmpose.engine.hooks.ExpMomentumEMA',
        # 구성 항목: `momentum` 값(`momentum`)에 0.0002을 지정합니다.
        momentum=0.0002,
        # 구성 항목: `priority` 값(`priority`)에 49을 지정합니다.
        priority=49,
        # 구성 항목: `type` 값(`type`)에 'mmengine.hooks.EMAHook'을 지정합니다.
        type='mmengine.hooks.EMAHook',
        # 구성 항목: 갱신 buffers(`update_buffers`)에 True을 지정합니다.
        update_buffers=True),
    # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
    dict(
        # 구성 항목: `switch_epoch` 값(`switch_epoch`)에 180을 지정합니다.
        switch_epoch=180,
        # 구성 항목: `switch_pipeline` 값(`switch_pipeline`)에 순서·구성을 지정한 `[dict(backend_args=dict(backend='local'), type='mmpose.datasets.LoadImage'), dict(type='mmpose.datasets.GetBBoxCenterScale'), dict(direction='horizontal', type='mmpose.d…`을 지정합니다.
        switch_pipeline=[
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: 데이터 파일 접근 방식 설정(`backend_args`)에 `dict(backend='local')` 호출 결과; `backend` 값(`backend`)에 'local'을 지정합니다.
                backend_args=dict(backend='local'),
                # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.LoadImage'을 지정합니다.
                type='mmpose.datasets.LoadImage'),
            # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.GetBBoxCenterScale'을 지정합니다.
            dict(type='mmpose.datasets.GetBBoxCenterScale'),
            # 구성 항목: 운동 방향(`direction`)에 'horizontal'; `type` 값(`type`)에 'mmpose.datasets.RandomFlip'을 지정합니다.
            dict(direction='horizontal', type='mmpose.datasets.RandomFlip'),
            # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.RandomHalfBody'을 지정합니다.
            dict(type='mmpose.datasets.RandomHalfBody'),
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: `rotate_factor` 값(`rotate_factor`)에 60을 지정합니다.
                rotate_factor=60,
                # 구성 항목: `scale_factor` 값(`scale_factor`)에 순서·구성을 지정한 `[0.75, 1.25]`을 지정합니다.
                scale_factor=[
                    # 앞줄에서 시작한 식·인수·자료 구조를 `0.75,` 내용으로 이어서 구성합니다.
                    0.75,
                    # 앞줄에서 시작한 식·인수·자료 구조를 `1.25,` 내용으로 이어서 구성합니다.
                    1.25,
                # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                ],
                # 구성 항목: `shift_factor` 값(`shift_factor`)에 0.0을 지정합니다.
                shift_factor=0.0,
                # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.transforms.common_transforms.RandomBBoxTransform'을 지정합니다.
                type=
                # 앞줄에서 시작한 식·인수·자료 구조를 `'mmpose.datasets.transforms.common_transforms.RandomBBoxTransform'` 내용으로 이어서 구성합니다.
                'mmpose.datasets.transforms.common_transforms.RandomBBoxTransform'
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            ),
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: 모델에 입력할 영상 크기(가로·세로)(`input_size`)에 순서·구성을 지정한 `(192, 256)`을 지정합니다.
                input_size=(
                    # 앞줄에서 시작한 식·인수·자료 구조를 `192,` 내용으로 이어서 구성합니다.
                    192,
                    # 앞줄에서 시작한 식·인수·자료 구조를 `256,` 내용으로 이어서 구성합니다.
                    256,
                # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.TopdownAffine'을 지정합니다.
                ), type='mmpose.datasets.TopdownAffine'),
            # 구성 항목: `type` 값(`type`)에 'mmdet.datasets.transforms.YOLOXHSVRandomAug'을 지정합니다.
            dict(type='mmdet.datasets.transforms.YOLOXHSVRandomAug'),
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: `transforms` 값(`transforms`)에 순서·구성을 지정한 `[dict(p=0.1, type='albumentations.augmentations.Blur'), dict(p=0.1, type='albumentations.augmentations.MedianBlur'), dict(max_height=0.4, max_holes=1, max_width=0.4, min…`을 지정합니다.
                transforms=[
                    # 구성 항목: `p` 값(`p`)에 0.1; `type` 값(`type`)에 'albumentations.augmentations.Blur'을 지정합니다.
                    dict(p=0.1, type='albumentations.augmentations.Blur'),
                    # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
                    dict(
                        # 구성 항목: `p` 값(`p`)에 0.1; `type` 값(`type`)에 'albumentations.augmentations.MedianBlur'을 지정합니다.
                        p=0.1, type='albumentations.augmentations.MedianBlur'),
                    # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
                    dict(
                        # 구성 항목: `max_height` 값(`max_height`)에 0.4을 지정합니다.
                        max_height=0.4,
                        # 구성 항목: `max_holes` 값(`max_holes`)에 1을 지정합니다.
                        max_holes=1,
                        # 구성 항목: `max_width` 값(`max_width`)에 0.4을 지정합니다.
                        max_width=0.4,
                        # 구성 항목: `min_height` 값(`min_height`)에 0.2을 지정합니다.
                        min_height=0.2,
                        # 구성 항목: `min_holes` 값(`min_holes`)에 1을 지정합니다.
                        min_holes=1,
                        # 구성 항목: `min_width` 값(`min_width`)에 0.2을 지정합니다.
                        min_width=0.2,
                        # 구성 항목: `p` 값(`p`)에 0.5을 지정합니다.
                        p=0.5,
                        # 구성 항목: `type` 값(`type`)에 'albumentations.augmentations.CoarseDropout'을 지정합니다.
                        type='albumentations.augmentations.CoarseDropout'),
                # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                ],
                # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.transforms.common_transforms.Albumentation'을 지정합니다.
                type=
                # 앞줄에서 시작한 식·인수·자료 구조를 `'mmpose.datasets.transforms.common_transforms.Albumentation'),` 내용으로 이어서 구성합니다.
                'mmpose.datasets.transforms.common_transforms.Albumentation'),
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: `encoder` 값(`encoder`)에 `dict(input_size=(192, 256), normalize=False, sigma=(4.9, 5.66), simcc_split_ratio=2.0, type='mmpose.codecs.SimCCLabel', use_dark=False)` 호출 결과를 지정합니다.
                encoder=dict(
                    # 구성 항목: 모델에 입력할 영상 크기(가로·세로)(`input_size`)에 순서·구성을 지정한 `(192, 256)`을 지정합니다.
                    input_size=(
                        # 앞줄에서 시작한 식·인수·자료 구조를 `192,` 내용으로 이어서 구성합니다.
                        192,
                        # 앞줄에서 시작한 식·인수·자료 구조를 `256,` 내용으로 이어서 구성합니다.
                        256,
                    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                    ),
                    # 구성 항목: `normalize` 값(`normalize`)에 False을 지정합니다.
                    normalize=False,
                    # 구성 항목: `sigma` 값(`sigma`)에 순서·구성을 지정한 `(4.9, 5.66)`을 지정합니다.
                    sigma=(
                        # 앞줄에서 시작한 식·인수·자료 구조를 `4.9,` 내용으로 이어서 구성합니다.
                        4.9,
                        # 앞줄에서 시작한 식·인수·자료 구조를 `5.66,` 내용으로 이어서 구성합니다.
                        5.66,
                    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                    ),
                    # 구성 항목: simcc split 비율(`simcc_split_ratio`)에 2.0을 지정합니다.
                    simcc_split_ratio=2.0,
                    # 구성 항목: `type` 값(`type`)에 'mmpose.codecs.SimCCLabel'을 지정합니다.
                    type='mmpose.codecs.SimCCLabel',
                    # 구성 항목: `use_dark` 값(`use_dark`)에 False을 지정합니다.
                    use_dark=False),
                # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.GenerateTarget'을 지정합니다.
                type='mmpose.datasets.GenerateTarget'),
            # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.PackPoseInputs'을 지정합니다.
            dict(type='mmpose.datasets.PackPoseInputs'),
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        ],
        # 구성 항목: `type` 값(`type`)에 'mmdet.engine.hooks.PipelineSwitchHook'을 지정합니다.
        type='mmdet.engine.hooks.PipelineSwitchHook'),
# 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
]
# 데이터 mode에 'topdown'을 저장합니다. 대상: `data_mode`.
data_mode = 'topdown'
# 데이터 최상위 경로에 '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5'을 저장합니다. 대상: `data_root`.
data_root = '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5'
# `dataset_type` 값에 'mmpose.datasets.CocoDataset'을 저장합니다. 대상: `dataset_type`.
dataset_type = 'mmpose.datasets.CocoDataset'
# 학습 진행 중 기본 콜백 설정에 `dict(checkpoint=dict(interval=10, max_keep_ckpts=-1, rule=None, save_best=None, type='CheckpointHook'), logger=dict(interval=50, type='mmengine.hooks.LoggerHook'), param…` 호출 결과를 저장합니다. 대상: `default_hooks`.
default_hooks = dict(
    # 구성 항목: `checkpoint` 값(`checkpoint`)에 `dict(interval=10, max_keep_ckpts=-1, rule=None, save_best=None, type='CheckpointHook')` 호출 결과를 지정합니다.
    checkpoint=dict(
        # 구성 항목: `interval` 값(`interval`)에 10을 지정합니다.
        interval=10,
        # 구성 항목: `max_keep_ckpts` 값(`max_keep_ckpts`)에 계산식 `-1`의 결과를 지정합니다.
        max_keep_ckpts=-1,
        # 구성 항목: `rule` 값(`rule`)에 None을 지정합니다.
        rule=None,
        # 구성 항목: 저장 best(`save_best`)에 None을 지정합니다.
        save_best=None,
        # 구성 항목: `type` 값(`type`)에 'CheckpointHook'을 지정합니다.
        type='CheckpointHook'),
    # 구성 항목: `logger` 값(`logger`)에 `dict(interval=50, type='mmengine.hooks.LoggerHook')` 호출 결과; `interval` 값(`interval`)에 50; `type` 값(`type`)에 'mmengine.hooks.LoggerHook'을 지정합니다.
    logger=dict(interval=50, type='mmengine.hooks.LoggerHook'),
    # 구성 항목: 학습 단계별 학습률 변경 규칙(`param_scheduler`)에 `dict(type='mmengine.hooks.ParamSchedulerHook')` 호출 결과; `type` 값(`type`)에 'mmengine.hooks.ParamSchedulerHook'을 지정합니다.
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    # 구성 항목: `sampler_seed` 값(`sampler_seed`)에 `dict(type='mmengine.hooks.DistSamplerSeedHook')` 호출 결과; `type` 값(`type`)에 'mmengine.hooks.DistSamplerSeedHook'을 지정합니다.
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    # 구성 항목: `timer` 값(`timer`)에 `dict(type='mmengine.hooks.IterTimerHook')` 호출 결과; `type` 값(`type`)에 'mmengine.hooks.IterTimerHook'을 지정합니다.
    timer=dict(type='mmengine.hooks.IterTimerHook'),
    # 구성 항목: `visualization` 값(`visualization`)에 `dict(enable=False, type='mmpose.engine.hooks.PoseVisualizationHook')` 호출 결과를 지정합니다.
    visualization=dict(
        # 구성 항목: `enable` 값(`enable`)에 False; `type` 값(`type`)에 'mmpose.engine.hooks.PoseVisualizationHook'을 지정합니다.
        enable=False, type='mmpose.engine.hooks.PoseVisualizationHook'))
# 기본 scope에 'mmpose'을 저장합니다. 대상: `default_scope`.
default_scope = 'mmpose'
# `env_cfg` 값에 `dict(cudnn_benchmark=False, dist_cfg=dict(backend='nccl'), mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))` 호출 결과를 저장합니다. 대상: `env_cfg`.
env_cfg = dict(
    # 구성 항목: `cudnn_benchmark` 값(`cudnn_benchmark`)에 False을 지정합니다.
    cudnn_benchmark=False,
    # 구성 항목: `dist_cfg` 값(`dist_cfg`)에 `dict(backend='nccl')` 호출 결과; `backend` 값(`backend`)에 'nccl'을 지정합니다.
    dist_cfg=dict(backend='nccl'),
    # 구성 항목: `mp_cfg` 값(`mp_cfg`)에 `dict(mp_start_method='fork', opencv_num_threads=0)` 호출 결과; mp 시작 method(`mp_start_method`)에 'fork'; `opencv_num_threads` 값(`opencv_num_threads`)에 0을 지정합니다.
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
# 불러오기 from에 '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/rtmpose/v2.0_with_public_step_5_fixed/epoch_390.pth'을 저장합니다. 대상: `load_from`.
load_from = '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/rtmpose/v2.0_with_public_step_5_fixed/epoch_390.pth'
# 로그 level에 'INFO'을 저장합니다. 대상: `log_level`.
log_level = 'INFO'
# 로그 processor에 `dict(by_epoch=True, num_digits=6, type='mmengine.runner.LogProcessor', window_size=50)` 호출 결과를 저장합니다. 대상: `log_processor`.
log_processor = dict(
    # 구성 항목: `by_epoch` 값(`by_epoch`)에 True을 지정합니다.
    by_epoch=True,
    # 구성 항목: `num_digits` 값(`num_digits`)에 6을 지정합니다.
    num_digits=6,
    # 구성 항목: `type` 값(`type`)에 'mmengine.runner.LogProcessor'을 지정합니다.
    type='mmengine.runner.LogProcessor',
    # 구성 항목: window 크기(`window_size`)에 50을 지정합니다.
    window_size=50)
# `max_epochs` 값에 210을 저장합니다. 대상: `max_epochs`.
max_epochs = 210
# RTMPose 신경망 구조 설정에 `dict(backbone=dict(act_cfg=dict(type='torch.nn.SiLU'), arch='P5', channel_attention=True, deepen_factor=0.67, expand_ratio=0.5, init_cfg=dict(checkpoint='https://downloa…` 호출 결과를 저장합니다. 대상: `model`.
model = dict(
    # 구성 항목: `backbone` 값(`backbone`)에 `dict(act_cfg=dict(type='torch.nn.SiLU'), arch='P5', channel_attention=True, deepen_factor=0.67, expand_ratio=0.5, init_cfg=dict(checkpoint='https://download.openmmlab.co…` 호출 결과를 지정합니다.
    backbone=dict(
        # 구성 항목: `act_cfg` 값(`act_cfg`)에 `dict(type='torch.nn.SiLU')` 호출 결과; `type` 값(`type`)에 'torch.nn.SiLU'을 지정합니다.
        act_cfg=dict(type='torch.nn.SiLU'),
        # 구성 항목: `arch` 값(`arch`)에 'P5'을 지정합니다.
        arch='P5',
        # 구성 항목: `channel_attention` 값(`channel_attention`)에 True을 지정합니다.
        channel_attention=True,
        # 구성 항목: `deepen_factor` 값(`deepen_factor`)에 0.67을 지정합니다.
        deepen_factor=0.67,
        # 구성 항목: expand 비율(`expand_ratio`)에 0.5을 지정합니다.
        expand_ratio=0.5,
        # 구성 항목: `init_cfg` 값(`init_cfg`)에 `dict(checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth', prefix='backbone.', type='mmengine…` 호출 결과를 지정합니다.
        init_cfg=dict(
            # 구성 항목: `checkpoint` 값(`checkpoint`)에 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'을 지정합니다.
            checkpoint=
            # 앞줄에서 시작한 식·인수·자료 구조를 `'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth',` 내용으로 이어서 구성합니다.
            'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth',
            # 구성 항목: `prefix` 값(`prefix`)에 'backbone.'을 지정합니다.
            prefix='backbone.',
            # 구성 항목: `type` 값(`type`)에 'mmengine.model.PretrainedInit'을 지정합니다.
            type='mmengine.model.PretrainedInit'),
        # 구성 항목: `norm_cfg` 값(`norm_cfg`)에 `dict(type='torch.nn.SyncBatchNorm')` 호출 결과; `type` 값(`type`)에 'torch.nn.SyncBatchNorm'을 지정합니다.
        norm_cfg=dict(type='torch.nn.SyncBatchNorm'),
        # 구성 항목: `out_indices` 값(`out_indices`)에 순서·구성을 지정한 `(4,)`을 지정합니다.
        out_indices=(4, ),
        # 구성 항목: `type` 값(`type`)에 'mmdet.models.CSPNeXt'을 지정합니다.
        type='mmdet.models.CSPNeXt',
        # 구성 항목: `widen_factor` 값(`widen_factor`)에 0.75을 지정합니다.
        widen_factor=0.75),
    # 구성 항목: 데이터 preprocessor(`data_preprocessor`)에 `dict(bgr_to_rgb=True, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], type='mmpose.models.PoseDataPreprocessor')` 호출 결과를 지정합니다.
    data_preprocessor=dict(
        # 구성 항목: `bgr_to_rgb` 값(`bgr_to_rgb`)에 True을 지정합니다.
        bgr_to_rgb=True,
        # 구성 항목: `mean` 값(`mean`)에 순서·구성을 지정한 `[123.675, 116.28, 103.53]`을 지정합니다.
        mean=[
            # 앞줄에서 시작한 식·인수·자료 구조를 `123.675,` 내용으로 이어서 구성합니다.
            123.675,
            # 앞줄에서 시작한 식·인수·자료 구조를 `116.28,` 내용으로 이어서 구성합니다.
            116.28,
            # 앞줄에서 시작한 식·인수·자료 구조를 `103.53,` 내용으로 이어서 구성합니다.
            103.53,
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        ],
        # 구성 항목: `std` 값(`std`)에 순서·구성을 지정한 `[58.395, 57.12, 57.375]`을 지정합니다.
        std=[
            # 앞줄에서 시작한 식·인수·자료 구조를 `58.395,` 내용으로 이어서 구성합니다.
            58.395,
            # 앞줄에서 시작한 식·인수·자료 구조를 `57.12,` 내용으로 이어서 구성합니다.
            57.12,
            # 앞줄에서 시작한 식·인수·자료 구조를 `57.375,` 내용으로 이어서 구성합니다.
            57.375,
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        ],
        # 구성 항목: `type` 값(`type`)에 'mmpose.models.PoseDataPreprocessor'을 지정합니다.
        type='mmpose.models.PoseDataPreprocessor'),
    # 구성 항목: `head` 값(`head`)에 `dict(decoder=dict(input_size=(192, 256), normalize=False, sigma=(4.9, 5.66), simcc_split_ratio=2.0, type='mmpose.codecs.SimCCLabel', use_dark=False), final_layer_kernel_…` 호출 결과를 지정합니다.
    head=dict(
        # 구성 항목: `decoder` 값(`decoder`)에 `dict(input_size=(192, 256), normalize=False, sigma=(4.9, 5.66), simcc_split_ratio=2.0, type='mmpose.codecs.SimCCLabel', use_dark=False)` 호출 결과를 지정합니다.
        decoder=dict(
            # 구성 항목: 모델에 입력할 영상 크기(가로·세로)(`input_size`)에 순서·구성을 지정한 `(192, 256)`을 지정합니다.
            input_size=(
                # 앞줄에서 시작한 식·인수·자료 구조를 `192,` 내용으로 이어서 구성합니다.
                192,
                # 앞줄에서 시작한 식·인수·자료 구조를 `256,` 내용으로 이어서 구성합니다.
                256,
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            ),
            # 구성 항목: `normalize` 값(`normalize`)에 False을 지정합니다.
            normalize=False,
            # 구성 항목: `sigma` 값(`sigma`)에 순서·구성을 지정한 `(4.9, 5.66)`을 지정합니다.
            sigma=(
                # 앞줄에서 시작한 식·인수·자료 구조를 `4.9,` 내용으로 이어서 구성합니다.
                4.9,
                # 앞줄에서 시작한 식·인수·자료 구조를 `5.66,` 내용으로 이어서 구성합니다.
                5.66,
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            ),
            # 구성 항목: simcc split 비율(`simcc_split_ratio`)에 2.0을 지정합니다.
            simcc_split_ratio=2.0,
            # 구성 항목: `type` 값(`type`)에 'mmpose.codecs.SimCCLabel'을 지정합니다.
            type='mmpose.codecs.SimCCLabel',
            # 구성 항목: `use_dark` 값(`use_dark`)에 False을 지정합니다.
            use_dark=False),
        # 구성 항목: final layer kernel 크기(`final_layer_kernel_size`)에 7을 지정합니다.
        final_layer_kernel_size=7,
        # 구성 항목: `gau_cfg` 값(`gau_cfg`)에 `dict(act_fn='SiLU', drop_path=0.0, dropout_rate=0.0, expansion_factor=2, hidden_dims=256, pos_enc=False, s=128, use_rel_bias=False)` 호출 결과를 지정합니다.
        gau_cfg=dict(
            # 구성 항목: `act_fn` 값(`act_fn`)에 'SiLU'을 지정합니다.
            act_fn='SiLU',
            # 구성 항목: drop 경로(`drop_path`)에 0.0을 지정합니다.
            drop_path=0.0,
            # 구성 항목: `dropout_rate` 값(`dropout_rate`)에 0.0을 지정합니다.
            dropout_rate=0.0,
            # 구성 항목: `expansion_factor` 값(`expansion_factor`)에 2을 지정합니다.
            expansion_factor=2,
            # 구성 항목: `hidden_dims` 값(`hidden_dims`)에 256을 지정합니다.
            hidden_dims=256,
            # 구성 항목: `pos_enc` 값(`pos_enc`)에 False을 지정합니다.
            pos_enc=False,
            # 구성 항목: `s` 값(`s`)에 128을 지정합니다.
            s=128,
            # 구성 항목: `use_rel_bias` 값(`use_rel_bias`)에 False을 지정합니다.
            use_rel_bias=False),
        # 구성 항목: `in_channels` 값(`in_channels`)에 768을 지정합니다.
        in_channels=768,
        # 구성 항목: in featuremap 크기(`in_featuremap_size`)에 순서·구성을 지정한 `(6, 8)`을 지정합니다.
        in_featuremap_size=(
            # 앞줄에서 시작한 식·인수·자료 구조를 `6,` 내용으로 이어서 구성합니다.
            6,
            # 앞줄에서 시작한 식·인수·자료 구조를 `8,` 내용으로 이어서 구성합니다.
            8,
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        ),
        # 구성 항목: 모델에 입력할 영상 크기(가로·세로)(`input_size`)에 순서·구성을 지정한 `(192, 256)`을 지정합니다.
        input_size=(
            # 앞줄에서 시작한 식·인수·자료 구조를 `192,` 내용으로 이어서 구성합니다.
            192,
            # 앞줄에서 시작한 식·인수·자료 구조를 `256,` 내용으로 이어서 구성합니다.
            256,
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        ),
        # 구성 항목: `loss` 값(`loss`)에 `dict(beta=10.0, label_softmax=True, type='mmpose.models.KLDiscretLoss', use_target_weight=True)` 호출 결과를 지정합니다.
        loss=dict(
            # 구성 항목: 움직임 속도에 따른 차단 주파수 증가 계수(`beta`)에 10.0을 지정합니다.
            beta=10.0,
            # 구성 항목: `label_softmax` 값(`label_softmax`)에 True을 지정합니다.
            label_softmax=True,
            # 구성 항목: `type` 값(`type`)에 'mmpose.models.KLDiscretLoss'을 지정합니다.
            type='mmpose.models.KLDiscretLoss',
            # 구성 항목: use 목표 weight(`use_target_weight`)에 True을 지정합니다.
            use_target_weight=True),
        # 구성 항목: `out_channels` 값(`out_channels`)에 17을 지정합니다.
        out_channels=17,
        # 구성 항목: simcc split 비율(`simcc_split_ratio`)에 2.0을 지정합니다.
        simcc_split_ratio=2.0,
        # 구성 항목: `type` 값(`type`)에 'mmpose.models.RTMCCHead'을 지정합니다.
        type='mmpose.models.RTMCCHead'),
    # 구성 항목: 회귀 검증 시나리오 test_cfg(`test_cfg`)에 `dict(flip_test=True)` 호출 결과; flip 검증(`flip_test`)에 True을 지정합니다.
    test_cfg=dict(flip_test=True),
    # 구성 항목: `type` 값(`type`)에 'mmpose.models.TopdownPoseEstimator'을 지정합니다.
    type='mmpose.models.TopdownPoseEstimator')
# 최적화 알고리즘과 파라미터 설정에 `dict(optimizer=dict(lr=0.004, type='torch.optim.AdamW', weight_decay=0.05), paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.0, weight_decay_mult=0.0))), type…` 호출 결과를 저장합니다. 대상: `optim_wrapper`.
optim_wrapper = dict(
    # 구성 항목: `optimizer` 값(`optimizer`)에 `dict(lr=0.004, type='torch.optim.AdamW', weight_decay=0.05)` 호출 결과; `lr` 값(`lr`)에 0.004; `type` 값(`type`)에 'torch.optim.AdamW'; `weight_decay` 값(`weight_decay`)에 0.05을 지정합니다.
    optimizer=dict(lr=0.004, type='torch.optim.AdamW', weight_decay=0.05),
    # 구성 항목: `paramwise_cfg` 값(`paramwise_cfg`)에 `dict(custom_keys=dict(backbone=dict(lr_mult=0.0, weight_decay_mult=0.0)))` 호출 결과를 지정합니다.
    paramwise_cfg=dict(
        # 구성 항목: `custom_keys` 값(`custom_keys`)에 `dict(backbone=dict(lr_mult=0.0, weight_decay_mult=0.0))` 호출 결과; `backbone` 값(`backbone`)에 `dict(lr_mult=0.0, weight_decay_mult=0.0)` 호출 결과; `lr_mult` 값(`lr_mult`)에 0.0; `weight_decay_mult` 값(`weight_decay_mult`)에 0.0을 지정합니다.
        custom_keys=dict(backbone=dict(lr_mult=0.0, weight_decay_mult=0.0))),
    # 구성 항목: `type` 값(`type`)에 'mmengine.optim.OptimWrapper'을 지정합니다.
    type='mmengine.optim.OptimWrapper')
# 학습 단계별 학습률 변경 규칙에 순서·구성을 지정한 `[dict(begin=0, by_epoch=False, end=1000, start_factor=1e-05, type='mmengine.optim.LinearLR'), dict(T_max=105, begin=105, by_epoch=True, convert_to_iter_based=True, end=2…`을 저장합니다. 대상: `param_scheduler`.
param_scheduler = [
    # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
    dict(
        # 구성 항목: `begin` 값(`begin`)에 0을 지정합니다.
        begin=0,
        # 구성 항목: `by_epoch` 값(`by_epoch`)에 False을 지정합니다.
        by_epoch=False,
        # 구성 항목: `end` 값(`end`)에 1000을 지정합니다.
        end=1000,
        # 구성 항목: 시작 factor(`start_factor`)에 1e-05을 지정합니다.
        start_factor=1e-05,
        # 구성 항목: `type` 값(`type`)에 'mmengine.optim.LinearLR'을 지정합니다.
        type='mmengine.optim.LinearLR'),
    # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
    dict(
        # 구성 항목: `T_max` 값(`T_max`)에 105을 지정합니다.
        T_max=105,
        # 구성 항목: `begin` 값(`begin`)에 105을 지정합니다.
        begin=105,
        # 구성 항목: `by_epoch` 값(`by_epoch`)에 True을 지정합니다.
        by_epoch=True,
        # 구성 항목: `convert_to_iter_based` 값(`convert_to_iter_based`)에 True을 지정합니다.
        convert_to_iter_based=True,
        # 구성 항목: `end` 값(`end`)에 210을 지정합니다.
        end=210,
        # 구성 항목: `eta_min` 값(`eta_min`)에 0.0002을 지정합니다.
        eta_min=0.0002,
        # 구성 항목: `type` 값(`type`)에 'mmengine.optim.CosineAnnealingLR'을 지정합니다.
        type='mmengine.optim.CosineAnnealingLR'),
# 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
]
# `randomness` 값에 `dict(seed=21)` 호출 결과를 저장합니다. 대상: `randomness`.
randomness = dict(seed=21)
# `resume` 값에 True을 저장합니다. 대상: `resume`.
resume = True
# `stage2_num_epochs` 값에 30을 저장합니다. 대상: `stage2_num_epochs`.
stage2_num_epochs = 30
# 회귀 검증 시나리오 test_cfg에 `dict()` 호출 결과를 저장합니다. 대상: `test_cfg`.
test_cfg = dict()
# 테스트 데이터 읽기 설정에 `dict(batch_size=256, dataset=dict(ann_file='annotations/person_keypoints_val.json', data_mode='topdown', data_prefix=dict(img='images/'), data_root='/workspace/nas203/ds…` 호출 결과를 저장합니다. 대상: `test_dataloader`.
test_dataloader = dict(
    # 구성 항목: 한 번에 추론 또는 학습할 표본 수(`batch_size`)에 256을 지정합니다.
    batch_size=256,
    # 구성 항목: `dataset` 값(`dataset`)에 `dict(ann_file='annotations/person_keypoints_val.json', data_mode='topdown', data_prefix=dict(img='images/'), data_root='/workspace/nas203/ds_RehabilitationMedicineData/I…` 호출 결과를 지정합니다.
    dataset=dict(
        # 구성 항목: ann 파일(`ann_file`)에 'annotations/person_keypoints_val.json'을 지정합니다.
        ann_file='annotations/person_keypoints_val.json',
        # 구성 항목: 데이터 mode(`data_mode`)에 'topdown'을 지정합니다.
        data_mode='topdown',
        # 구성 항목: 데이터 prefix(`data_prefix`)에 `dict(img='images/')` 호출 결과; `img` 값(`img`)에 'images/'을 지정합니다.
        data_prefix=dict(img='images/'),
        # 구성 항목: 데이터 최상위 경로(`data_root`)에 '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5'을 지정합니다.
        data_root=
        # 앞줄에서 시작한 식·인수·자료 구조를 `'/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5',` 내용으로 이어서 구성합니다.
        '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5',
        # 구성 항목: `metainfo` 값(`metainfo`)에 `dict(dataset_name='coco', from_file='configs/_base_/datasets/coco.py', keypoint_weights=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, …` 호출 결과를 지정합니다.
        metainfo=dict(
            # 구성 항목: `dataset_name` 값(`dataset_name`)에 'coco'을 지정합니다.
            dataset_name='coco',
            # 구성 항목: from 파일(`from_file`)에 'configs/_base_/datasets/coco.py'을 지정합니다.
            from_file='configs/_base_/datasets/coco.py',
            # 구성 항목: `keypoint_weights` 값(`keypoint_weights`)에 순서·구성을 지정한 `[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`을 지정합니다.
            keypoint_weights=[
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            ]),
        # 구성 항목: `pipeline` 값(`pipeline`)에 순서·구성을 지정한 `[dict(backend_args=dict(backend='local'), type='mmpose.datasets.LoadImage'), dict(type='mmpose.datasets.GetBBoxCenterScale'), dict(input_size=(192, 256), type='mmpose.da…`을 지정합니다.
        pipeline=[
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: 데이터 파일 접근 방식 설정(`backend_args`)에 `dict(backend='local')` 호출 결과; `backend` 값(`backend`)에 'local'을 지정합니다.
                backend_args=dict(backend='local'),
                # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.LoadImage'을 지정합니다.
                type='mmpose.datasets.LoadImage'),
            # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.GetBBoxCenterScale'을 지정합니다.
            dict(type='mmpose.datasets.GetBBoxCenterScale'),
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: 모델에 입력할 영상 크기(가로·세로)(`input_size`)에 순서·구성을 지정한 `(192, 256)`을 지정합니다.
                input_size=(
                    # 앞줄에서 시작한 식·인수·자료 구조를 `192,` 내용으로 이어서 구성합니다.
                    192,
                    # 앞줄에서 시작한 식·인수·자료 구조를 `256,` 내용으로 이어서 구성합니다.
                    256,
                # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.TopdownAffine'을 지정합니다.
                ), type='mmpose.datasets.TopdownAffine'),
            # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.PackPoseInputs'을 지정합니다.
            dict(type='mmpose.datasets.PackPoseInputs'),
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        ],
        # 구성 항목: 회귀 검증 시나리오 test_mode(`test_mode`)에 True을 지정합니다.
        test_mode=True,
        # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.CocoDataset'을 지정합니다.
        type='mmpose.datasets.CocoDataset'),
    # 구성 항목: `drop_last` 값(`drop_last`)에 False을 지정합니다.
    drop_last=False,
    # 구성 항목: `num_workers` 값(`num_workers`)에 16을 지정합니다.
    num_workers=16,
    # 구성 항목: `persistent_workers` 값(`persistent_workers`)에 True을 지정합니다.
    persistent_workers=True,
    # 구성 항목: `sampler` 값(`sampler`)에 `dict(round_up=False, shuffle=False, type='mmengine.dataset.DefaultSampler')` 호출 결과를 지정합니다.
    sampler=dict(
        # 구성 항목: `round_up` 값(`round_up`)에 False; `shuffle` 값(`shuffle`)에 False; `type` 값(`type`)에 'mmengine.dataset.DefaultSampler'을 지정합니다.
        round_up=False, shuffle=False, type='mmengine.dataset.DefaultSampler'))
# 회귀 검증 시나리오 test_evaluator에 `dict(ann_file='/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5/annotations/person_keypoints_val.json', type='mmpo…` 호출 결과를 저장합니다. 대상: `test_evaluator`.
test_evaluator = dict(
    # 구성 항목: ann 파일(`ann_file`)에 '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5/annotations/person_keypoints_val.json'을 지정합니다.
    ann_file=
    # 앞줄에서 시작한 식·인수·자료 구조를 `'/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5/annotations/person_keypoints_val.json',` 내용으로 이어서 구성합니다.
    '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5/annotations/person_keypoints_val.json',
    # 구성 항목: `type` 값(`type`)에 'mmpose.evaluation.CocoMetric'을 지정합니다.
    type='mmpose.evaluation.CocoMetric')
# `train_cfg` 값에 `dict(by_epoch=True, max_epochs=420, val_interval=10)` 호출 결과를 저장합니다. 대상: `train_cfg`.
train_cfg = dict(by_epoch=True, max_epochs=420, val_interval=10)
# 학습 데이터 묶음 읽기 설정에 `dict(batch_size=256, dataset=dict(ann_file='annotations/person_keypoints_train.json', data_mode='topdown', data_prefix=dict(img='images/'), data_root='/workspace/nas203/…` 호출 결과를 저장합니다. 대상: `train_dataloader`.
train_dataloader = dict(
    # 구성 항목: 한 번에 추론 또는 학습할 표본 수(`batch_size`)에 256을 지정합니다.
    batch_size=256,
    # 구성 항목: `dataset` 값(`dataset`)에 `dict(ann_file='annotations/person_keypoints_train.json', data_mode='topdown', data_prefix=dict(img='images/'), data_root='/workspace/nas203/ds_RehabilitationMedicineData…` 호출 결과를 지정합니다.
    dataset=dict(
        # 구성 항목: ann 파일(`ann_file`)에 'annotations/person_keypoints_train.json'을 지정합니다.
        ann_file='annotations/person_keypoints_train.json',
        # 구성 항목: 데이터 mode(`data_mode`)에 'topdown'을 지정합니다.
        data_mode='topdown',
        # 구성 항목: 데이터 prefix(`data_prefix`)에 `dict(img='images/')` 호출 결과; `img` 값(`img`)에 'images/'을 지정합니다.
        data_prefix=dict(img='images/'),
        # 구성 항목: 데이터 최상위 경로(`data_root`)에 '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5'을 지정합니다.
        data_root=
        # 앞줄에서 시작한 식·인수·자료 구조를 `'/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5',` 내용으로 이어서 구성합니다.
        '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5',
        # 구성 항목: `metainfo` 값(`metainfo`)에 `dict(dataset_name='coco', from_file='configs/_base_/datasets/coco.py', keypoint_weights=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, …` 호출 결과를 지정합니다.
        metainfo=dict(
            # 구성 항목: `dataset_name` 값(`dataset_name`)에 'coco'을 지정합니다.
            dataset_name='coco',
            # 구성 항목: from 파일(`from_file`)에 'configs/_base_/datasets/coco.py'을 지정합니다.
            from_file='configs/_base_/datasets/coco.py',
            # 구성 항목: `keypoint_weights` 값(`keypoint_weights`)에 순서·구성을 지정한 `[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`을 지정합니다.
            keypoint_weights=[
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            ]),
        # 구성 항목: `pipeline` 값(`pipeline`)에 순서·구성을 지정한 `[dict(backend_args=dict(backend='local'), type='mmpose.datasets.LoadImage'), dict(type='mmpose.datasets.GetBBoxCenterScale'), dict(direction='horizontal', type='mmpose.d…`을 지정합니다.
        pipeline=[
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: 데이터 파일 접근 방식 설정(`backend_args`)에 `dict(backend='local')` 호출 결과; `backend` 값(`backend`)에 'local'을 지정합니다.
                backend_args=dict(backend='local'),
                # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.LoadImage'을 지정합니다.
                type='mmpose.datasets.LoadImage'),
            # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.GetBBoxCenterScale'을 지정합니다.
            dict(type='mmpose.datasets.GetBBoxCenterScale'),
            # 구성 항목: 운동 방향(`direction`)에 'horizontal'; `type` 값(`type`)에 'mmpose.datasets.RandomFlip'을 지정합니다.
            dict(direction='horizontal', type='mmpose.datasets.RandomFlip'),
            # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.RandomHalfBody'을 지정합니다.
            dict(type='mmpose.datasets.RandomHalfBody'),
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: `rotate_factor` 값(`rotate_factor`)에 80을 지정합니다.
                rotate_factor=80,
                # 구성 항목: `scale_factor` 값(`scale_factor`)에 순서·구성을 지정한 `[0.6, 1.4]`을 지정합니다.
                scale_factor=[
                    # 앞줄에서 시작한 식·인수·자료 구조를 `0.6,` 내용으로 이어서 구성합니다.
                    0.6,
                    # 앞줄에서 시작한 식·인수·자료 구조를 `1.4,` 내용으로 이어서 구성합니다.
                    1.4,
                # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                ],
                # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.transforms.common_transforms.RandomBBoxTransform'을 지정합니다.
                type=
                # 앞줄에서 시작한 식·인수·자료 구조를 `'mmpose.datasets.transforms.common_transforms.RandomBBoxTransform'` 내용으로 이어서 구성합니다.
                'mmpose.datasets.transforms.common_transforms.RandomBBoxTransform'
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            ),
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: 모델에 입력할 영상 크기(가로·세로)(`input_size`)에 순서·구성을 지정한 `(192, 256)`을 지정합니다.
                input_size=(
                    # 앞줄에서 시작한 식·인수·자료 구조를 `192,` 내용으로 이어서 구성합니다.
                    192,
                    # 앞줄에서 시작한 식·인수·자료 구조를 `256,` 내용으로 이어서 구성합니다.
                    256,
                # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.TopdownAffine'을 지정합니다.
                ), type='mmpose.datasets.TopdownAffine'),
            # 구성 항목: `type` 값(`type`)에 'mmdet.datasets.transforms.YOLOXHSVRandomAug'을 지정합니다.
            dict(type='mmdet.datasets.transforms.YOLOXHSVRandomAug'),
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: `transforms` 값(`transforms`)에 순서·구성을 지정한 `[dict(p=0.1, type='albumentations.augmentations.Blur'), dict(p=0.1, type='albumentations.augmentations.MedianBlur'), dict(max_height=0.4, max_holes=1, max_width=0.4, min…`을 지정합니다.
                transforms=[
                    # 구성 항목: `p` 값(`p`)에 0.1; `type` 값(`type`)에 'albumentations.augmentations.Blur'을 지정합니다.
                    dict(p=0.1, type='albumentations.augmentations.Blur'),
                    # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
                    dict(
                        # 구성 항목: `p` 값(`p`)에 0.1; `type` 값(`type`)에 'albumentations.augmentations.MedianBlur'을 지정합니다.
                        p=0.1, type='albumentations.augmentations.MedianBlur'),
                    # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
                    dict(
                        # 구성 항목: `max_height` 값(`max_height`)에 0.4을 지정합니다.
                        max_height=0.4,
                        # 구성 항목: `max_holes` 값(`max_holes`)에 1을 지정합니다.
                        max_holes=1,
                        # 구성 항목: `max_width` 값(`max_width`)에 0.4을 지정합니다.
                        max_width=0.4,
                        # 구성 항목: `min_height` 값(`min_height`)에 0.2을 지정합니다.
                        min_height=0.2,
                        # 구성 항목: `min_holes` 값(`min_holes`)에 1을 지정합니다.
                        min_holes=1,
                        # 구성 항목: `min_width` 값(`min_width`)에 0.2을 지정합니다.
                        min_width=0.2,
                        # 구성 항목: `p` 값(`p`)에 1.0을 지정합니다.
                        p=1.0,
                        # 구성 항목: `type` 값(`type`)에 'albumentations.augmentations.CoarseDropout'을 지정합니다.
                        type='albumentations.augmentations.CoarseDropout'),
                # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                ],
                # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.transforms.common_transforms.Albumentation'을 지정합니다.
                type=
                # 앞줄에서 시작한 식·인수·자료 구조를 `'mmpose.datasets.transforms.common_transforms.Albumentation'),` 내용으로 이어서 구성합니다.
                'mmpose.datasets.transforms.common_transforms.Albumentation'),
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: `encoder` 값(`encoder`)에 `dict(input_size=(192, 256), normalize=False, sigma=(4.9, 5.66), simcc_split_ratio=2.0, type='mmpose.codecs.SimCCLabel', use_dark=False)` 호출 결과를 지정합니다.
                encoder=dict(
                    # 구성 항목: 모델에 입력할 영상 크기(가로·세로)(`input_size`)에 순서·구성을 지정한 `(192, 256)`을 지정합니다.
                    input_size=(
                        # 앞줄에서 시작한 식·인수·자료 구조를 `192,` 내용으로 이어서 구성합니다.
                        192,
                        # 앞줄에서 시작한 식·인수·자료 구조를 `256,` 내용으로 이어서 구성합니다.
                        256,
                    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                    ),
                    # 구성 항목: `normalize` 값(`normalize`)에 False을 지정합니다.
                    normalize=False,
                    # 구성 항목: `sigma` 값(`sigma`)에 순서·구성을 지정한 `(4.9, 5.66)`을 지정합니다.
                    sigma=(
                        # 앞줄에서 시작한 식·인수·자료 구조를 `4.9,` 내용으로 이어서 구성합니다.
                        4.9,
                        # 앞줄에서 시작한 식·인수·자료 구조를 `5.66,` 내용으로 이어서 구성합니다.
                        5.66,
                    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                    ),
                    # 구성 항목: simcc split 비율(`simcc_split_ratio`)에 2.0을 지정합니다.
                    simcc_split_ratio=2.0,
                    # 구성 항목: `type` 값(`type`)에 'mmpose.codecs.SimCCLabel'을 지정합니다.
                    type='mmpose.codecs.SimCCLabel',
                    # 구성 항목: `use_dark` 값(`use_dark`)에 False을 지정합니다.
                    use_dark=False),
                # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.GenerateTarget'을 지정합니다.
                type='mmpose.datasets.GenerateTarget'),
            # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.PackPoseInputs'을 지정합니다.
            dict(type='mmpose.datasets.PackPoseInputs'),
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        ],
        # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.CocoDataset'을 지정합니다.
        type='mmpose.datasets.CocoDataset'),
    # 구성 항목: `drop_last` 값(`drop_last`)에 True을 지정합니다.
    drop_last=True,
    # 구성 항목: `num_workers` 값(`num_workers`)에 16을 지정합니다.
    num_workers=16,
    # 구성 항목: `persistent_workers` 값(`persistent_workers`)에 True을 지정합니다.
    persistent_workers=True,
    # 구성 항목: `sampler` 값(`sampler`)에 `dict(shuffle=True, type='mmengine.dataset.DefaultSampler')` 호출 결과; `shuffle` 값(`shuffle`)에 True; `type` 값(`type`)에 'mmengine.dataset.DefaultSampler'을 지정합니다.
    sampler=dict(shuffle=True, type='mmengine.dataset.DefaultSampler'))
# 학습 영상 전처리 순서에 순서·구성을 지정한 `[dict(backend_args=dict(backend='local'), type='mmpose.datasets.LoadImage'), dict(type='mmpose.datasets.GetBBoxCenterScale'), dict(direction='horizontal', type='mmpose.d…`을 저장합니다. 대상: `train_pipeline`.
train_pipeline = [
    # 구성 항목: 데이터 파일 접근 방식 설정(`backend_args`)에 `dict(backend='local')` 호출 결과; `type` 값(`type`)에 'mmpose.datasets.LoadImage'; `backend` 값(`backend`)에 'local'을 지정합니다.
    dict(backend_args=dict(backend='local'), type='mmpose.datasets.LoadImage'),
    # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.GetBBoxCenterScale'을 지정합니다.
    dict(type='mmpose.datasets.GetBBoxCenterScale'),
    # 구성 항목: 운동 방향(`direction`)에 'horizontal'; `type` 값(`type`)에 'mmpose.datasets.RandomFlip'을 지정합니다.
    dict(direction='horizontal', type='mmpose.datasets.RandomFlip'),
    # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.RandomHalfBody'을 지정합니다.
    dict(type='mmpose.datasets.RandomHalfBody'),
    # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
    dict(
        # 구성 항목: `rotate_factor` 값(`rotate_factor`)에 80을 지정합니다.
        rotate_factor=80,
        # 구성 항목: `scale_factor` 값(`scale_factor`)에 순서·구성을 지정한 `[0.6, 1.4]`을 지정합니다.
        scale_factor=[
            # 앞줄에서 시작한 식·인수·자료 구조를 `0.6,` 내용으로 이어서 구성합니다.
            0.6,
            # 앞줄에서 시작한 식·인수·자료 구조를 `1.4,` 내용으로 이어서 구성합니다.
            1.4,
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        ],
        # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.transforms.common_transforms.RandomBBoxTransform'을 지정합니다.
        type='mmpose.datasets.transforms.common_transforms.RandomBBoxTransform'
    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
    ),
    # 구성 항목: 모델에 입력할 영상 크기(가로·세로)(`input_size`)에 순서·구성을 지정한 `(192, 256)`을 지정합니다.
    dict(input_size=(
        # 앞줄에서 시작한 식·인수·자료 구조를 `192,` 내용으로 이어서 구성합니다.
        192,
        # 앞줄에서 시작한 식·인수·자료 구조를 `256,` 내용으로 이어서 구성합니다.
        256,
    # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.TopdownAffine'을 지정합니다.
    ), type='mmpose.datasets.TopdownAffine'),
    # 구성 항목: `type` 값(`type`)에 'mmdet.datasets.transforms.YOLOXHSVRandomAug'을 지정합니다.
    dict(type='mmdet.datasets.transforms.YOLOXHSVRandomAug'),
    # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
    dict(
        # 구성 항목: `transforms` 값(`transforms`)에 순서·구성을 지정한 `[dict(p=0.1, type='albumentations.augmentations.Blur'), dict(p=0.1, type='albumentations.augmentations.MedianBlur'), dict(max_height=0.4, max_holes=1, max_width=0.4, min…`을 지정합니다.
        transforms=[
            # 구성 항목: `p` 값(`p`)에 0.1; `type` 값(`type`)에 'albumentations.augmentations.Blur'을 지정합니다.
            dict(p=0.1, type='albumentations.augmentations.Blur'),
            # 구성 항목: `p` 값(`p`)에 0.1; `type` 값(`type`)에 'albumentations.augmentations.MedianBlur'을 지정합니다.
            dict(p=0.1, type='albumentations.augmentations.MedianBlur'),
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: `max_height` 값(`max_height`)에 0.4을 지정합니다.
                max_height=0.4,
                # 구성 항목: `max_holes` 값(`max_holes`)에 1을 지정합니다.
                max_holes=1,
                # 구성 항목: `max_width` 값(`max_width`)에 0.4을 지정합니다.
                max_width=0.4,
                # 구성 항목: `min_height` 값(`min_height`)에 0.2을 지정합니다.
                min_height=0.2,
                # 구성 항목: `min_holes` 값(`min_holes`)에 1을 지정합니다.
                min_holes=1,
                # 구성 항목: `min_width` 값(`min_width`)에 0.2을 지정합니다.
                min_width=0.2,
                # 구성 항목: `p` 값(`p`)에 1.0을 지정합니다.
                p=1.0,
                # 구성 항목: `type` 값(`type`)에 'albumentations.augmentations.CoarseDropout'을 지정합니다.
                type='albumentations.augmentations.CoarseDropout'),
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        ],
        # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.transforms.common_transforms.Albumentation'을 지정합니다.
        type='mmpose.datasets.transforms.common_transforms.Albumentation'),
    # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
    dict(
        # 구성 항목: `encoder` 값(`encoder`)에 `dict(input_size=(192, 256), normalize=False, sigma=(4.9, 5.66), simcc_split_ratio=2.0, type='mmpose.codecs.SimCCLabel', use_dark=False)` 호출 결과를 지정합니다.
        encoder=dict(
            # 구성 항목: 모델에 입력할 영상 크기(가로·세로)(`input_size`)에 순서·구성을 지정한 `(192, 256)`을 지정합니다.
            input_size=(
                # 앞줄에서 시작한 식·인수·자료 구조를 `192,` 내용으로 이어서 구성합니다.
                192,
                # 앞줄에서 시작한 식·인수·자료 구조를 `256,` 내용으로 이어서 구성합니다.
                256,
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            ),
            # 구성 항목: `normalize` 값(`normalize`)에 False을 지정합니다.
            normalize=False,
            # 구성 항목: `sigma` 값(`sigma`)에 순서·구성을 지정한 `(4.9, 5.66)`을 지정합니다.
            sigma=(
                # 앞줄에서 시작한 식·인수·자료 구조를 `4.9,` 내용으로 이어서 구성합니다.
                4.9,
                # 앞줄에서 시작한 식·인수·자료 구조를 `5.66,` 내용으로 이어서 구성합니다.
                5.66,
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            ),
            # 구성 항목: simcc split 비율(`simcc_split_ratio`)에 2.0을 지정합니다.
            simcc_split_ratio=2.0,
            # 구성 항목: `type` 값(`type`)에 'mmpose.codecs.SimCCLabel'을 지정합니다.
            type='mmpose.codecs.SimCCLabel',
            # 구성 항목: `use_dark` 값(`use_dark`)에 False을 지정합니다.
            use_dark=False),
        # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.GenerateTarget'을 지정합니다.
        type='mmpose.datasets.GenerateTarget'),
    # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.PackPoseInputs'을 지정합니다.
    dict(type='mmpose.datasets.PackPoseInputs'),
# 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
]
# `train_pipeline_stage2` 값에 순서·구성을 지정한 `[dict(backend_args=dict(backend='local'), type='mmpose.datasets.LoadImage'), dict(type='mmpose.datasets.GetBBoxCenterScale'), dict(direction='horizontal', type='mmpose.d…`을 저장합니다. 대상: `train_pipeline_stage2`.
train_pipeline_stage2 = [
    # 구성 항목: 데이터 파일 접근 방식 설정(`backend_args`)에 `dict(backend='local')` 호출 결과; `type` 값(`type`)에 'mmpose.datasets.LoadImage'; `backend` 값(`backend`)에 'local'을 지정합니다.
    dict(backend_args=dict(backend='local'), type='mmpose.datasets.LoadImage'),
    # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.GetBBoxCenterScale'을 지정합니다.
    dict(type='mmpose.datasets.GetBBoxCenterScale'),
    # 구성 항목: 운동 방향(`direction`)에 'horizontal'; `type` 값(`type`)에 'mmpose.datasets.RandomFlip'을 지정합니다.
    dict(direction='horizontal', type='mmpose.datasets.RandomFlip'),
    # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.RandomHalfBody'을 지정합니다.
    dict(type='mmpose.datasets.RandomHalfBody'),
    # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
    dict(
        # 구성 항목: `rotate_factor` 값(`rotate_factor`)에 60을 지정합니다.
        rotate_factor=60,
        # 구성 항목: `scale_factor` 값(`scale_factor`)에 순서·구성을 지정한 `[0.75, 1.25]`을 지정합니다.
        scale_factor=[
            # 앞줄에서 시작한 식·인수·자료 구조를 `0.75,` 내용으로 이어서 구성합니다.
            0.75,
            # 앞줄에서 시작한 식·인수·자료 구조를 `1.25,` 내용으로 이어서 구성합니다.
            1.25,
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        ],
        # 구성 항목: `shift_factor` 값(`shift_factor`)에 0.0을 지정합니다.
        shift_factor=0.0,
        # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.transforms.common_transforms.RandomBBoxTransform'을 지정합니다.
        type='mmpose.datasets.transforms.common_transforms.RandomBBoxTransform'
    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
    ),
    # 구성 항목: 모델에 입력할 영상 크기(가로·세로)(`input_size`)에 순서·구성을 지정한 `(192, 256)`을 지정합니다.
    dict(input_size=(
        # 앞줄에서 시작한 식·인수·자료 구조를 `192,` 내용으로 이어서 구성합니다.
        192,
        # 앞줄에서 시작한 식·인수·자료 구조를 `256,` 내용으로 이어서 구성합니다.
        256,
    # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.TopdownAffine'을 지정합니다.
    ), type='mmpose.datasets.TopdownAffine'),
    # 구성 항목: `type` 값(`type`)에 'mmdet.datasets.transforms.YOLOXHSVRandomAug'을 지정합니다.
    dict(type='mmdet.datasets.transforms.YOLOXHSVRandomAug'),
    # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
    dict(
        # 구성 항목: `transforms` 값(`transforms`)에 순서·구성을 지정한 `[dict(p=0.1, type='albumentations.augmentations.Blur'), dict(p=0.1, type='albumentations.augmentations.MedianBlur'), dict(max_height=0.4, max_holes=1, max_width=0.4, min…`을 지정합니다.
        transforms=[
            # 구성 항목: `p` 값(`p`)에 0.1; `type` 값(`type`)에 'albumentations.augmentations.Blur'을 지정합니다.
            dict(p=0.1, type='albumentations.augmentations.Blur'),
            # 구성 항목: `p` 값(`p`)에 0.1; `type` 값(`type`)에 'albumentations.augmentations.MedianBlur'을 지정합니다.
            dict(p=0.1, type='albumentations.augmentations.MedianBlur'),
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: `max_height` 값(`max_height`)에 0.4을 지정합니다.
                max_height=0.4,
                # 구성 항목: `max_holes` 값(`max_holes`)에 1을 지정합니다.
                max_holes=1,
                # 구성 항목: `max_width` 값(`max_width`)에 0.4을 지정합니다.
                max_width=0.4,
                # 구성 항목: `min_height` 값(`min_height`)에 0.2을 지정합니다.
                min_height=0.2,
                # 구성 항목: `min_holes` 값(`min_holes`)에 1을 지정합니다.
                min_holes=1,
                # 구성 항목: `min_width` 값(`min_width`)에 0.2을 지정합니다.
                min_width=0.2,
                # 구성 항목: `p` 값(`p`)에 0.5을 지정합니다.
                p=0.5,
                # 구성 항목: `type` 값(`type`)에 'albumentations.augmentations.CoarseDropout'을 지정합니다.
                type='albumentations.augmentations.CoarseDropout'),
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        ],
        # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.transforms.common_transforms.Albumentation'을 지정합니다.
        type='mmpose.datasets.transforms.common_transforms.Albumentation'),
    # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
    dict(
        # 구성 항목: `encoder` 값(`encoder`)에 `dict(input_size=(192, 256), normalize=False, sigma=(4.9, 5.66), simcc_split_ratio=2.0, type='mmpose.codecs.SimCCLabel', use_dark=False)` 호출 결과를 지정합니다.
        encoder=dict(
            # 구성 항목: 모델에 입력할 영상 크기(가로·세로)(`input_size`)에 순서·구성을 지정한 `(192, 256)`을 지정합니다.
            input_size=(
                # 앞줄에서 시작한 식·인수·자료 구조를 `192,` 내용으로 이어서 구성합니다.
                192,
                # 앞줄에서 시작한 식·인수·자료 구조를 `256,` 내용으로 이어서 구성합니다.
                256,
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            ),
            # 구성 항목: `normalize` 값(`normalize`)에 False을 지정합니다.
            normalize=False,
            # 구성 항목: `sigma` 값(`sigma`)에 순서·구성을 지정한 `(4.9, 5.66)`을 지정합니다.
            sigma=(
                # 앞줄에서 시작한 식·인수·자료 구조를 `4.9,` 내용으로 이어서 구성합니다.
                4.9,
                # 앞줄에서 시작한 식·인수·자료 구조를 `5.66,` 내용으로 이어서 구성합니다.
                5.66,
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            ),
            # 구성 항목: simcc split 비율(`simcc_split_ratio`)에 2.0을 지정합니다.
            simcc_split_ratio=2.0,
            # 구성 항목: `type` 값(`type`)에 'mmpose.codecs.SimCCLabel'을 지정합니다.
            type='mmpose.codecs.SimCCLabel',
            # 구성 항목: `use_dark` 값(`use_dark`)에 False을 지정합니다.
            use_dark=False),
        # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.GenerateTarget'을 지정합니다.
        type='mmpose.datasets.GenerateTarget'),
    # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.PackPoseInputs'을 지정합니다.
    dict(type='mmpose.datasets.PackPoseInputs'),
# 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
]
# `val_cfg` 값에 `dict()` 호출 결과를 저장합니다. 대상: `val_cfg`.
val_cfg = dict()
# 검증 데이터 읽기 설정에 `dict(batch_size=256, dataset=dict(ann_file='annotations/person_keypoints_val.json', data_mode='topdown', data_prefix=dict(img='images/'), data_root='/workspace/nas203/ds…` 호출 결과를 저장합니다. 대상: `val_dataloader`.
val_dataloader = dict(
    # 구성 항목: 한 번에 추론 또는 학습할 표본 수(`batch_size`)에 256을 지정합니다.
    batch_size=256,
    # 구성 항목: `dataset` 값(`dataset`)에 `dict(ann_file='annotations/person_keypoints_val.json', data_mode='topdown', data_prefix=dict(img='images/'), data_root='/workspace/nas203/ds_RehabilitationMedicineData/I…` 호출 결과를 지정합니다.
    dataset=dict(
        # 구성 항목: ann 파일(`ann_file`)에 'annotations/person_keypoints_val.json'을 지정합니다.
        ann_file='annotations/person_keypoints_val.json',
        # 구성 항목: 데이터 mode(`data_mode`)에 'topdown'을 지정합니다.
        data_mode='topdown',
        # 구성 항목: 데이터 prefix(`data_prefix`)에 `dict(img='images/')` 호출 결과; `img` 값(`img`)에 'images/'을 지정합니다.
        data_prefix=dict(img='images/'),
        # 구성 항목: 데이터 최상위 경로(`data_root`)에 '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5'을 지정합니다.
        data_root=
        # 앞줄에서 시작한 식·인수·자료 구조를 `'/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5',` 내용으로 이어서 구성합니다.
        '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5',
        # 구성 항목: `metainfo` 값(`metainfo`)에 `dict(dataset_name='coco', from_file='configs/_base_/datasets/coco.py', keypoint_weights=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, …` 호출 결과를 지정합니다.
        metainfo=dict(
            # 구성 항목: `dataset_name` 값(`dataset_name`)에 'coco'을 지정합니다.
            dataset_name='coco',
            # 구성 항목: from 파일(`from_file`)에 'configs/_base_/datasets/coco.py'을 지정합니다.
            from_file='configs/_base_/datasets/coco.py',
            # 구성 항목: `keypoint_weights` 값(`keypoint_weights`)에 순서·구성을 지정한 `[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`을 지정합니다.
            keypoint_weights=[
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `0.0,` 내용으로 이어서 구성합니다.
                0.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
                # 앞줄에서 시작한 식·인수·자료 구조를 `1.0,` 내용으로 이어서 구성합니다.
                1.0,
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            ]),
        # 구성 항목: `pipeline` 값(`pipeline`)에 순서·구성을 지정한 `[dict(backend_args=dict(backend='local'), type='mmpose.datasets.LoadImage'), dict(type='mmpose.datasets.GetBBoxCenterScale'), dict(input_size=(192, 256), type='mmpose.da…`을 지정합니다.
        pipeline=[
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: 데이터 파일 접근 방식 설정(`backend_args`)에 `dict(backend='local')` 호출 결과; `backend` 값(`backend`)에 'local'을 지정합니다.
                backend_args=dict(backend='local'),
                # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.LoadImage'을 지정합니다.
                type='mmpose.datasets.LoadImage'),
            # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.GetBBoxCenterScale'을 지정합니다.
            dict(type='mmpose.datasets.GetBBoxCenterScale'),
            # 앞줄에서 시작한 식·인수·자료 구조를 `dict(` 내용으로 이어서 구성합니다.
            dict(
                # 구성 항목: 모델에 입력할 영상 크기(가로·세로)(`input_size`)에 순서·구성을 지정한 `(192, 256)`을 지정합니다.
                input_size=(
                    # 앞줄에서 시작한 식·인수·자료 구조를 `192,` 내용으로 이어서 구성합니다.
                    192,
                    # 앞줄에서 시작한 식·인수·자료 구조를 `256,` 내용으로 이어서 구성합니다.
                    256,
                # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.TopdownAffine'을 지정합니다.
                ), type='mmpose.datasets.TopdownAffine'),
            # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.PackPoseInputs'을 지정합니다.
            dict(type='mmpose.datasets.PackPoseInputs'),
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        ],
        # 구성 항목: 회귀 검증 시나리오 test_mode(`test_mode`)에 True을 지정합니다.
        test_mode=True,
        # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.CocoDataset'을 지정합니다.
        type='mmpose.datasets.CocoDataset'),
    # 구성 항목: `drop_last` 값(`drop_last`)에 False을 지정합니다.
    drop_last=False,
    # 구성 항목: `num_workers` 값(`num_workers`)에 16을 지정합니다.
    num_workers=16,
    # 구성 항목: `persistent_workers` 값(`persistent_workers`)에 True을 지정합니다.
    persistent_workers=True,
    # 구성 항목: `sampler` 값(`sampler`)에 `dict(round_up=False, shuffle=False, type='mmengine.dataset.DefaultSampler')` 호출 결과를 지정합니다.
    sampler=dict(
        # 구성 항목: `round_up` 값(`round_up`)에 False; `shuffle` 값(`shuffle`)에 False; `type` 값(`type`)에 'mmengine.dataset.DefaultSampler'을 지정합니다.
        round_up=False, shuffle=False, type='mmengine.dataset.DefaultSampler'))
# `val_evaluator` 값에 `dict(ann_file='/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5/annotations/person_keypoints_val.json', type='mmpo…` 호출 결과를 저장합니다. 대상: `val_evaluator`.
val_evaluator = dict(
    # 구성 항목: ann 파일(`ann_file`)에 '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5/annotations/person_keypoints_val.json'을 지정합니다.
    ann_file=
    # 앞줄에서 시작한 식·인수·자료 구조를 `'/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5/annotations/person_keypoints_val.json',` 내용으로 이어서 구성합니다.
    '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/12_RTM_DATASET/v2.0_with_public_step_5/annotations/person_keypoints_val.json',
    # 구성 항목: `type` 값(`type`)에 'mmpose.evaluation.CocoMetric'을 지정합니다.
    type='mmpose.evaluation.CocoMetric')
# 검증 영상 전처리 순서에 순서·구성을 지정한 `[dict(backend_args=dict(backend='local'), type='mmpose.datasets.LoadImage'), dict(type='mmpose.datasets.GetBBoxCenterScale'), dict(input_size=(192, 256), type='mmpose.da…`을 저장합니다. 대상: `val_pipeline`.
val_pipeline = [
    # 구성 항목: 데이터 파일 접근 방식 설정(`backend_args`)에 `dict(backend='local')` 호출 결과; `type` 값(`type`)에 'mmpose.datasets.LoadImage'; `backend` 값(`backend`)에 'local'을 지정합니다.
    dict(backend_args=dict(backend='local'), type='mmpose.datasets.LoadImage'),
    # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.GetBBoxCenterScale'을 지정합니다.
    dict(type='mmpose.datasets.GetBBoxCenterScale'),
    # 구성 항목: 모델에 입력할 영상 크기(가로·세로)(`input_size`)에 순서·구성을 지정한 `(192, 256)`을 지정합니다.
    dict(input_size=(
        # 앞줄에서 시작한 식·인수·자료 구조를 `192,` 내용으로 이어서 구성합니다.
        192,
        # 앞줄에서 시작한 식·인수·자료 구조를 `256,` 내용으로 이어서 구성합니다.
        256,
    # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.TopdownAffine'을 지정합니다.
    ), type='mmpose.datasets.TopdownAffine'),
    # 구성 항목: `type` 값(`type`)에 'mmpose.datasets.PackPoseInputs'을 지정합니다.
    dict(type='mmpose.datasets.PackPoseInputs'),
# 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
]
# `vis_backends` 값에 순서·구성을 지정한 `[dict(type='mmengine.visualization.LocalVisBackend')]`을 저장합니다. 대상: `vis_backends`.
vis_backends = [
    # 구성 항목: `type` 값(`type`)에 'mmengine.visualization.LocalVisBackend'을 지정합니다.
    dict(type='mmengine.visualization.LocalVisBackend'),
# 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
]
# `visualizer` 값에 `dict(name='visualizer', type='mmpose.visualization.PoseLocalVisualizer', vis_backends=[dict(type='mmengine.visualization.LocalVisBackend')])` 호출 결과를 저장합니다. 대상: `visualizer`.
visualizer = dict(
    # 구성 항목: `name` 값(`name`)에 'visualizer'을 지정합니다.
    name='visualizer',
    # 구성 항목: `type` 값(`type`)에 'mmpose.visualization.PoseLocalVisualizer'을 지정합니다.
    type='mmpose.visualization.PoseLocalVisualizer',
    # 구성 항목: `vis_backends` 값(`vis_backends`)에 순서·구성을 지정한 `[dict(type='mmengine.visualization.LocalVisBackend')]`을 지정합니다.
    vis_backends=[
        # 구성 항목: `type` 값(`type`)에 'mmengine.visualization.LocalVisBackend'을 지정합니다.
        dict(type='mmengine.visualization.LocalVisBackend'),
    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
    ])
# `work_dir` 값에 '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/rtmpose/v2.0_with_public_step_5_fixed'을 저장합니다. 대상: `work_dir`.
work_dir = '/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/rtmpose/v2.0_with_public_step_5_fixed'
