from mmdet.datasets import DATASETS, PIPELINES
# from mmdet.models import build_model
# from mmdet.apis import set_random_seed, train_detector
# from mmdet.datasets import CustomDataset
# from mmdet.models import build_detector
# from mmdet.datasets.builder import build_dataset

from mmdet.datasets.coco import CocoDataset
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import set_random_seed, train_detector     

import os
import shutil
import json
import copy
import os.path as osp
import cv2
import mmcv
import numpy as np
import pycocotools._mask as _mask

# Load the configuration from the file
config_file = 'mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'C:/cv_project/Recycling_trash/Separate_Collection/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

from mmcv import Config
cfg = Config.fromfile(config_file)

# Update specific configurations
cfg.dataset_type = 'TrashDataset'
cfg.data_root = 'customdata/'

cfg.data.train.type = 'TrashDataset'
cfg.data.train.data_root = 'customdata/'
cfg.data.train.ann_file = 'train.json'
cfg.data.train.img_prefix = ''

cfg.data.val.type = 'TrashDataset'
cfg.data.val.data_root = 'customdata/'
cfg.data.val.ann_file = 'valid_0.json'
cfg.data.val.img_prefix = ''

cfg.data.test.type = 'AihubDataset'
cfg.data.test.data_root = 'customdata/'
cfg.data.test.ann_file = 'valid_0.json'
cfg.data.test.img_prefix = ''

cfg.model.roi_head.bbox_head.num_classes = 10
cfg.load_from = 'C:/cv_project/Recycling_trash/Separate_Collection/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

cfg.work_dir = './tutorial_exps'

cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10
cfg.lr_config.policy = 'step'

cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 12
cfg.checkpoint_config.interval = 12

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'

cfg.dump('faster_rcnn_config.py')

@DATASETS.register_module(force=True)
class TrashDataset(CocoDataset):
    CLASSES = ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic','Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    
def train():
    # config에서 train 데이터셋 정보 가져오기
    # train_dataset = copy.deepcopy(cfg.data.train)
    # train_dataset.pipeline = cfg.train_pipeline

    # 데이터셋 빌드를 위한 설정 추가
    # dataset_info = dict(
    #     type=cfg.dataset_type,
    #     data_root=cfg.data_root,
    #     ann_file=cfg.data.train.ann_file,  # ann_file 추가
    #     img_prefix=train_dataset.img_prefix,
    #     classes=cfg.model.roi_head.bbox_head.num_classes,
    #     pipeline=train_dataset.pipeline
    # )

    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=False)

if __name__ == '__main__':
    train()