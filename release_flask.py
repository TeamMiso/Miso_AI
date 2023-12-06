from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import requests
import numpy as np
import logging
import urllib.request
from mmdet.apis import inference_detector, init_detector
import urllib.request
import base64

app = Flask(__name__)
logger = logging.getLogger("gunicorn.error")

# Load the model and configurations
model = init_detector(config='/home/torch/recycle_trash/Miso_AI/faster_rcnn_config.py', checkpoint='/home/torch/recycle_trash/Miso_AI/tutorial_exps/latest.pth')

def url_to_best_class(image_url):

    sorted_results = []
    image_bytes = base64.b64decode(image_url)
        
    # BytesIO를 사용하여 PIL 이미지로 변환
    image = Image.open(BytesIO(image_bytes))
        
    # PIL 이미지를 numpy 배열로 변환
    image = np.array(image)
    
    result = inference_detector(model, image)
    
    for class_id, bbox_result in enumerate(result):
        if bbox_result.shape[0] > 0:
            confidence = bbox_result[:, 4].max()
            sorted_results.append((class_id, confidence))

    sorted_results = sorted(
        sorted_results,
        key=lambda x: x[1],  # confidence score를 기준으로 정렬
        reverse=True
    )

    class_names = [model.CLASSES[class_id] for class_id, _ in sorted_results]

    return class_names

# Route to accept image URL and process it
@app.route("/process_image", methods=['POST'])
def process_image():
    if request.method == 'POST':
        try:
            # JSON 데이터를 가져옴
            json_data = request.get_json()
            image_url = json_data["image_url"]
            
            class_list = url_to_best_class(image_url)
            return jsonify(class_list)
        
        except Exception as e:
            # 예외가 발생하면 오류 응답을 반환
            return jsonify(str(e), 400)
    else:
        return "Method Not Allowed", 405