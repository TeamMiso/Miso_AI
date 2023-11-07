from mmdet.datasets.builder import DATASETS, PIPELINES
# from mmdet.models import build_model
# from mmdet.apis import set_random_seed, train_detector
# from mmdet.datasets import CustomDataset
# from mmdet.models import build_detector
# from mmdet.datasets.builder import build_dataset

from mmdet.datasets.custom import CustomDataset
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

import mmcv
import os.path as osp
import pycocotools._mask as _mask

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

@DATASETS.register_module(force=True)
class AihubDataset(CustomDataset):
    # CLASSES = ('Paper', 'Plastic', 'Glass', 'Can', 'Metal', 'Clothes', 'Electronic Product', 'Styrofoam', 'Pottery',
    #            'Vinyl', 'Furniture', 'Bicycle', 'Fluorescent lamp', 'Plastic bottle', 'Tree')
    CLASSES = ('가구류', '고철류', '나무', '도기류', '비닐', '스티로폼', '유리병', '의류', '자전거', '전자제품', '종이류',
               '캔류', '페트병', '플라스틱류', '형광등')
    
    def __init__(self, ann_file, data_root, img_prefix, pipeline, classes):
        self.CLASSES = classes  # classes를 CLASSES 변수에 할당

        super(AihubDataset, self).__init__(ann_file, data_root, img_prefix, pipeline)
    def load_annotations(self, ann_file):
        print('##### self.data_root:', self.data_root, 'self.ann_file:', self.ann_file, 'self.img_prefix:', self.img_prefix)
        print('#### ann_file:', ann_file)
        cat2label = {Class_Value: Class_Key for Class_Key, Class_Value in enumerate(self.CLASSES)}

        data_infos = []

        base_dir = os.path.join(self.data_root, self.img_prefix)

        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.json'):
                    json_file = os.path.join(root, file)
                    image_file = file.replace('.json', '.jpg')
                    image_path = os.path.join(root, image_file)

                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Annotation 정보를 확인하고 필요한 정보를 추출하여 리스트에 추가합니다.
                        gt_labels = cat2label[data['Bounding']['CLASS']]
                        gt_bboxes = [
                            data['Bounding']['x1'],
                            data['Bounding']['y1'],
                            data['Bounding']['x2'],
                            data['Bounding']['y2']
                        ]

                        image = cv2.imread(image_path)
                        height, width = image.shape[:2]

                        data_info = {
                            'filename': image_file,
                            'width': width,
                            'height': height,
                            'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                            'labels': np.array(gt_labels, dtype=np.long),
                            'bboxes_ignore': [],
                            'labels_ignore': []
                        }
                        data_infos.append(data_info)

        return data_infos
    
def train():
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':
    train()