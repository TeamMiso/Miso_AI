from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import set_random_seed, train_detector

import mmcv
import os.path as osp

# Load the configuration from the file
config_file = 'mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoint/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

from mmcv import Config
cfg = Config.fromfile(config_file)

# Update specific configurations
cfg.dataset_type = 'AihubDataset'
cfg.data_root = 'D:/생활 폐기물 이미지/'

cfg.data.train.type = 'AihubDataset'
cfg.data.train.data_root = 'D:/생활 폐기물 이미지/'
cfg.data.train.ann_file = 'Training_라벨링데이터'
cfg.data.train.img_prefix = 'Training'

cfg.data.val.type = 'AihubDataset'
cfg.data.val.data_root = 'D:/생활 폐기물 이미지/'
cfg.data.val.ann_file = '[V라벨링]라벨링데이터'
cfg.data.val.img_prefix = 'Validation'

cfg.data.test.type = 'AihubDataset'
cfg.data.test.data_root = 'D:/생활 폐기물 이미지/'
cfg.data.test.ann_file = '[V라벨링]라벨링데이터'
cfg.data.test.img_prefix = 'Validation'

cfg.model.roi_head.bbox_head.num_classes = 15
cfg.load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

cfg.work_dir = './tutorial_exps'

cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10
cfg.lr_config.policy = 'step'

cfg.evaluation.metric = 'mAP'
cfg.evaluation.interval = 12
cfg.checkpoint_config.interval = 12

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda:1'

def train():
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':
    train()