import os
import shutil
import json
import logging
import copy
import os.path as osp
import cv2

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

# logger 세팅
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 파일 핸들러, 포매터 세팅
formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
file_handler = logging.FileHandler()
logger.addHandler(file_handler)
file_handler.setFormatter(formatter)

# 원본 annotation 경로
annoDir = ""
# image가 있는 경로
imageDir = ""

# anno_dir 내 annotation 파일 이름을 리스트로 변경
annoList = os.listdir(annoDir)
logging.debug(len(annoList))

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
        
        # D:/Separate_Collection/Miso_AI/생활 폐기물 이미지/Training 안의 폴더의 이름을 list로 가져온다.
        Training_folder_list = []
        base_dir = '{0:}/{1:}'.format(self.data_root,self.img_prefix)
        
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                Training_folder_list.append(item)
        
        
        data_infos = []
        image_num = 0
        
        for image_id in Training_folder_list:
            filename = '{0}_{1}.jpg'.format()
            # 원본 이미지의 너비, 높이를 image를 직접 로드하여 구함.
            image = cv2.imread(filename)
            height, width = image.shape[:2]
            # 개별 image의 annotation 정보 저장용 Dict 생성. key값 filename 에는 image의 파일명만 들어감(디렉토리는 제외)
            data_info = {'filename': str(image_id) + '.jpeg',
                        'width': width, 'height': height}
            # 개별 annotation이 있는 서브 디렉토리의 prefix 변환.
            label_prefix = self.img_prefix.replace('image_2', 'label_2')
            # 개별 annotation 파일을 1개 line 씩 읽어서 list 로드
            lines = mmcv.list_from_file(osp.join(label_prefix, str(image_id)+'.txt'))

            # 전체 lines를 개별 line별 공백 레벨로 parsing 하여 다시 list로 저장. content는 list의 list형태임.
            # ann 정보는 numpy array로 저장되나 텍스트 처리나 데이터 가공이 list 가 편하므로 일차적으로 list로 변환 수행.
            content = [line.strip().split(' ') for line in lines]
            # 오브젝트의 클래스명은 bbox_names로 저장.
            bbox_names = [x[0] for x in content]
            # bbox 좌표를 저장
            bboxes = [ [float(info) for info in x[4:8]] for x in content]

            # 클래스명이 해당 사항이 없는 대상 Filtering out, 'DontCare'sms ignore로 별도 저장.
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            for bbox_name, bbox in zip(bbox_names, bboxes):
                # 만약 bbox_name이 클래스명에 해당 되면, gt_bboxes와 gt_labels에 추가, 그렇지 않으면 gt_bboxes_ignore, gt_labels_ignore에 추가
                if bbox_name in cat2label:
                    gt_bboxes.append(bbox)
                    # gt_labels에는 class id를 입력
                    gt_labels.append(cat2label[bbox_name])

                else:
                    gt_bboxes_ignore.append(bbox)
                    gt_labels_ignore.append(-1)
            # 개별 image별 annotation 정보를 가지는 Dict 생성. 해당 Dict의 value값은 모두 np.array임.
            data_anno = {
                'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                'labels': np.array(gt_labels, dtype=np.long),
                'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
                'labels_ignore': np.array(gt_labels_ignore, dtype=np.long)
            }
            # image에 대한 메타 정보를 가지는 data_info Dict에 'ann' key값으로 data_anno를 value로 저장.
            data_info.update(ann=data_anno)
            # 전체 annotation 파일들에 대한 정보를 가지는 data_infos에 data_info Dict를 추가
            data_infos.append(data_info)

        return data_infos