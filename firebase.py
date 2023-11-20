import firebase_admin
from firebase_admin import credentials, initialize_app, storage, db

import time
from io import BytesIO
from PIL import Image
import requests
import numpy as np

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = init_detector(config='C:/cv_project/Recycling_trash/Separate_Collection/faster_rcnn_config.py', checkpoint='C:/cv_project/Recycling_trash/Separate_Collection/tutorial_exps/latest.pth')

# Firebase Admin SDK 초기화
cred = credentials.Certificate('C:/cv_project/Recycling_trash/Separate_Collection/mykey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://miso-android-app-2-default-rtdb.firebaseio.com'
})

bucket = storage.bucket("miso-android-app-2.appspot.com")
def process_data(img):

    result = inference_detector(model, img)

    max_confidence = 0
    max_class_id = None

    # 모든 결과를 반복
    for class_id, bbox_result in enumerate(result):
        # bbox 결과가 비어있지 않은지 확인
        if bbox_result.shape[0] > 0:
            confidence = bbox_result[:, 4].max()
            if confidence > max_confidence:
                max_confidence = confidence
                max_class_id = class_id

    # 클래스 이름 정의
    class_names = model.CLASSES
    # 클래스 ID를 클래스 이름에 매핑
    max_class_name = class_names[max_class_id] if max_class_id is not None else None

    return max_class_name


def send_result(result,str_data):
    ref = db.reference('ai') #경로가 없으면 생성한다.
    ref.update({f'response{str_data[-1]}': result })


def main():
    # 이전 데이터 저장 변수
    previous_data = None
    while True:
        try:
            ref = db.reference('user')
            
            # 마지막 key와 value 저장
            data = ref.order_by_key().limit_to_last(1).get()

            #data의 key값을 string으로 변환
            str_key = str(list(data.keys())[0])
            img_filename = list(data.values())[0]

            img_path = f"{str(img_filename)}.jpeg"

            # key값으로 이미지 이름을 로컬에 저장함.
            # 스토리지에 있는 이미지 local로 다운로드
            print(str_key)
            print(img_path)
            blob = bucket.blob(img_path)

            if blob.exists():
                print("Blob exists!")
            else:
                print("Blob does not exist.")
            
            local_image_path = f'C:/cv_project/Recycling_trash/Separate_Collection/image_jpeg/{str_key}.jpeg'
            blob.download_to_filename(local_image_path)
            
            img = Image.open(local_image_path)
            # 이전 데이터와 현재 데이터가 다르면 동작 수행
            if str_key != previous_data:
                print("Data updated: {}".format(str_key))

                result = process_data(local_image_path)
                if result is not None:
                    print("Best_class :", result)
                    send_result(result, str_key)
                
                # 현재 데이터를 이전 데이터로 업데이트
                previous_data = str_key
            
            # 1초마다 반복
            time.sleep(1)

        except KeyboardInterrupt:
            print("Exiting the loop.")
            break

if __name__ == '__main__':
    main()