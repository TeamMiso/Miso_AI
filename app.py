from flask import Flask
from flask import Blueprint
from flask import request
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
import os.path as osp
from io import BytesIO
from PIL import Image
import base64
import time

from config import cfg
from train import model
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
app = Flask(__name__)

# AI 모델 판단 후 결과 응답
@app.route("/model",methods=['GET', 'POST'])
def decision():
    if(request.method == 'POST'):
        # json 요청에서 id키에 해당하는 값
        params = request.get_json()['id']
        
        # base64로 인코딩된 이미지
        img = Image.open(BytesIO(base64.b64decode(params)))
        result = img_to_result(img)
        return result
    
    elif(request.method =='GET'):
        return 'Backend-server Connect'

def img_to_result(image):
    image = np.asarray(image)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    
    model.cfg = cfg

    result = inference_detector(model, image)
    
    # 예측 클래스와 확률(또는 점수)을 저장할 변수
    max_score = -1  # 가장 높은 점수
    best_class = None  # 가장 높은 점수를 갖는 클래스

    # 결과를 반복하며 가장 높은 점수를 갖는 클래스 찾기
    for bbox in result:
        if bbox.shape[0] != 0:
            # 클래스 정보는 0번째 열에, 해당 클래스의 점수는 1번째 열에 있다고 가정합니다.
            class_score = bbox[0, 4]  # 예측된 객체의 클래스 점수

            # 현재 클래스의 점수가 최고 점수보다 높으면 갱신
            if class_score > max_score:
                max_score = class_score
                best_class = int(bbox[0, 0])  # 해당 클래스

    return best_class

if __name__ == '__main__':
    app.run(host='0.0.0.0' ,port = 8001, debug=True)