import firebase_admin
from firebase_admin import credentials, initialize_app, storage, db

import time
from io import BytesIO
from PIL import Image
import requests
import numpy as np

from mmdet.apis import inference_detector, init_detector

model = init_detector(config='faster_rcnn_config.py', checkpoint='mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')

# Firebase Admin SDK 초기화
cred = credentials.Certificate('C:/cv_project/Recycling_trash/Separate_Collection/mykey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://miso-f8a77-default-rtdb.firebaseio.com'
})

bucket = storage.bucket("gs://miso-f8a77.appspot.com")

def process_data(img):

    result = inference_detector(model, img)
    
    max_score = -1
    best_class = None

    for bbox in result:
        if bbox.shape[0] != 0:
            class_score = bbox[0, 4]
            if class_score > max_score:
                max_score = class_score
                best_class = int(bbox[0, 0])
        
    return best_class

def send_result(result,str_data):
    ref = db.reference('ai') #경로가 없으면 생성한다.
    ref.update({f'response{str_data[-1]}': result })


if __name__ == '__main__':
    # 이전 데이터 저장 변수
    previous_data = None
    epoch = 1
    while True:
        try:
            ref = db.reference('user')
            
            # 마지막 key와 value 저장
            data = ref.order_by_key().limit_to_last(1).get()

            #data의 key값을 string으로 변환
            str_key = str(data.keys())[0]
            img_filename = list(data.values())[0]

            print(img_filename)
            img_path = f"miso-f8a77.appspot.com/{img_filename}"

            # key값으로 이미지 이름을 로컬에 저장함.
            # 스토리지에 있는 이미지 local로 다운로드
            blob = bucket.blob(img_path)
            local_image_path = f'C:/cv_project/Recycling_trash/Separate_Collection/image_jpeg/{str_key}.jpeg'
            blob.download_to_filename(local_image_path)
            
            img = Image.open(local_image_path)
            # 이전 데이터와 현재 데이터가 다르면 동작 수행
            if str_key != previous_data:
                print("Data updated:", str_key)

                result = process_data(img)
                print("Best_class :", result)

                send_result(result, str_key)
                
                # 현재 데이터를 이전 데이터로 업데이트
                previous_data = str_key
            
            print('Epoch : {d}'.format(epoch))
            # 1초마다 반복
            time.sleep(1)

        except KeyboardInterrupt:
            print("Exiting the loop.")
            break
