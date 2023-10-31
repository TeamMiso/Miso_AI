import os
import shutil
import json
import logging
import copy
import os.path as osp
import cv2

import mmcv
import numpy as np

from mmdet.registry import DATASETS
from mmengine.dataset import BaseDataset

# logger 세팅
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 파일 핸들러, 포매터 세팅
formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
file_handler = logging.FileHandler()
logger.addHandler(file_handler)
file_handler.setFormatter(formatter)

# # 원본 annotation 경로
# annoDir = ""
# # image가 있는 경로
# imageDir = ""

# # anno_dir 내 annotation 파일 이름을 리스트로 변경
# annoList = os.listdir(annoDir)
# logging.debug(len(annoList))

@DATASETS.register_module(force=True)
class AihubDataset(BaseDataset):
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