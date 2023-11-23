from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import requests
import numpy as np
import logging
import urllib.request
from mmdet.apis import inference_detector, init_detector

app = Flask(__name__)
logger = logging.getLogger("gunicorn.error")

# Load the model and configurations
model = init_detector(config='faster_rcnn_config.py', checkpoint='/home/torch/recycle_trash/Miso_AI/tutorial_exps/latest.pth',device='cpu')

# Function to process image from URL and get the best class
def url_to_best_class(image_url):
    
    image = urllib.request.urlopen(image_url).read()
    result = inference_detector(model, image)

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

# Route to accept image URL and process it
@app.route("/process_image", methods=['POST'])
def process_image():
    if request.method == 'POST':
        try:
            # JSON 데이터를 가져옴
            json_data = request.get_json()
            image_url = json_data["body"]["image_url"]
            
            # 이미지 URL을 통해 최적의 클래스 찾기
            best_class = url_to_best_class(image_url)

            return jsonify({
                "header": {},
                "body": {
                    "best_class": best_class,
                }
            })
        except Exception as e:
            # 예외가 발생하면 오류 응답을 반환
            return jsonify({
                "header": {},
                "body": {
                    "error": str(e),
                }
            }), 400
    else:
        return "Method Not Allowed", 405