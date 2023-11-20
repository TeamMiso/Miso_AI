from mmdet.datasets import DATASETS, PIPELINES

from mmdet.datasets.coco import CocoDataset
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import set_random_seed, train_detector,single_gpu_test

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
cfg.data_root = 'C:/cv_project/Recycling_trash/Separate_Collection/customdata/'

cfg.data.train.type = 'TrashDataset'
cfg.data.train.data_root = 'C:/cv_project/Recycling_trash/Separate_Collection/customdata/'
cfg.data.train.ann_file = 'train.json'
cfg.data.train.img_prefix = 'new_train2'

cfg.data.val.type = 'TrashDataset'
cfg.data.val.data_root = 'C:/cv_project/Recycling_trash/Separate_Collection/customdata/'
cfg.data.val.ann_file = 'test.json'
cfg.data.val.img_prefix = 'new_train2'

cfg.data.test.type = 'AihubDataset'
cfg.data.test.data_root = 'C:/cv_project/Recycling_trash/Separate_Collection/customdata/'
cfg.data.test.ann_file = 'test.json'
cfg.data.test.img_prefix = 'new_train2'

cfg.model.roi_head.bbox_head.num_classes = 11
cfg.load_from = 'C:/cv_project/Recycling_trash/Separate_Collection/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

cfg.work_dir = './tutorial_exps'

cfg.optimizer.lr = 0.001
# cfg.lr_config.warmup = "linear"
# cfg.lr_config.warmup_iters = 500
cfg.log_config.interval = 10
cfg.lr_config.policy = 'step'

cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 12
cfg.checkpoint_config.interval = 12

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'

cfg.data.samples_per_gpu = 1  # 한 GPU 당 처리되는 샘플 수
cfg.data.workers_per_gpu = 1 

cfg.dump('faster_rcnn_config.py')

@DATASETS.register_module(force=True)
class TrashDataset(CocoDataset):
    CLASSES = ['UNKNOWN','General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic','Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

def train():

    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=False)

if __name__ == '__main__':
    train()