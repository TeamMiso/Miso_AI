# Config 설정하고 Pretrained 모델 다운로드
config_file = ''
checkpoint_file = ''

from mmcv import Config
from mmdet.apis import set_random_seed

cfg = Config.fromfile(config_file)

''' 파일 절대 경로 : D:/Separate_Collection/Miso_AI/'''

# dataset에 대한 환경 파라미터 수정.
cfg.dataset_type = 'AihubDataset'
cfg.data_root = 'D:/Separate_Collection/Miso_AI/생활 폐기물 이미지/'

# train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정.
cfg.data.train.type = 'AihubDataset'
cfg.data.train.data_root = 'D:/Separate_Collection/Miso_AI/생활 폐기물 이미지/'
cfg.data.train.ann_file = 'Training_라벨링데이터'
cfg.data.train.img_prefix = 'Training'

cfg.data.val.type = 'AihubDataset'
cfg.data.val.data_root = 'D:/Separate_Collection/Miso_AI/생활 폐기물 이미지/'
cfg.data.val.ann_file = '[V라벨링]라벨링데이터'
cfg.data.val.img_prefix = 'Validation'

cfg.data.test.type = 'AihubDataset'
cfg.data.test.data_root = 'D:/Separate_Collection/Miso_AI/생활 폐기물 이미지/'
cfg.data.test.ann_file = '[V라벨링]라벨링데이터'
cfg.data.test.img_prefix = 'Validation'

# class의 갯수 수정.
cfg.model.roi_head.bbox_head.num_classes = 15
# pretrained 모델
cfg.load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정.
cfg.work_dir = './tutorial_exps'

# 학습율 변경 환경 파라미터 설정.
cfg.optimizer.lr = 0.02 / 8

cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# config 수행 시마다 policy값이 없어지는 bug로 인하여 설정.
cfg.lr_config.policy = 'step'

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# ConfigDict' object has no attribute 'device 오류 발생시 반드시 설정 필요. https://github.com/open-mmlab/mmdetection/issues/7901
cfg.device='cuda'