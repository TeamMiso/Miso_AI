from mmdet.datasets.builder import DATASETS, PIPELINES
from mmdet.models import build_model
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import CustomDataset
from mmdet.models import build_detector
from mmdet.datasets.builder import build_dataset

# from mmdet.datasets import build_dataset
# from mmdet.models import build_detector
# from mmdet.apis import set_random_seed, train_detector

import os
import shutil
import json
import logging
import copy
import os.path as osp
import cv2
import mmcv
import numpy as np

import mmcv
import os.path as osp
import pycocotools._mask as _mask

# logger 세팅
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 파일 핸들러, 포매터 세팅
formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
file_handler = logging.FileHandler()
logger.addHandler(file_handler)
file_handler.setFormatter(formatter)


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
    def load_annotations(self, ann_file):
        logging.debug('##### self.data_root:', self.data_root, 'self.ann_file:', self.ann_file, 'self.img_prefix:', self.img_prefix)
        logging.debug('#### ann_file:',ann_file)
        cat2label = {Class_Value:Class_Key for Class_Key, Class_Value in enumerate(self.CLASSES)}
        
        data_list = []
        # D:/Separate_Collection/Miso_AI/생활 폐기물 이미지/Training 안의 폴더의 이름을 list로 가져온다.
        
        base_dir = '{0:}/{1:}'.format(self.data_root,self.img_prefix)
        
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.json'):
                    json_file = os.path.join(root, file)
                    # 여기에서 JSON 파일을 읽고 파싱하여 annotation 정보를 반환합니다.
                    # 입력 예시의 형식에 맞게 JSON 파일을 읽어서 반환하는 코드를 작성하세요.
                    annotation = self.parse_json(json_file)
                    
                    # 이미지 파일 경로 구성 예시: '11_X001_C012_1215_11_x001_C012_1215_0~4.jpg'
                    image_file = os.path.splitext(file)[0] + '.jpg'
                    image_path = os.path.join(root, image_file)

                    # 이미지 절대경로, annotation json파일 정보, 
                    data_list.append({'image_path': image_path, 'annotation': annotation})
        
        for image_ in data_list:
            image = cv2.imread(image_)
            height, width = image.shape[:2]

            # JSON 파일 읽기
            with open(data_list['annotation'], 'r') as json_file:
                data = json.load(json_file)
            
            # labels = CLASSES의 id
            gt_labels = cat2label[data['Bounding']['CLASS']]

            # bboxes = x1, y1, x2, y2
            gt_bboxes = [data['Bounding']['x1'], data['Bounding']['y1'], data['Bounding']['x2'], data['Bounding']['y2']]

            data_infos = {
                'filename' : data['FILE NAME'],
                'width' : width,
                'height' : height,
                'bboxes' : np.array(gt_bboxes, dtype=np.float32).reshape(-1,4),
                'labels' : np.array(gt_labels, dtype=np.long),
                'bboxes_ignore' : [],
                'labels_ignore' : []
            }

            return data_infos
def train():
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':
    train()