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

import config
import train
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
    
    
    return

if __name__ == '__main__':
    app.run(host='0.0.0.0' ,port = 8001, debug=True)