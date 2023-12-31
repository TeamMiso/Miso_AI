from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import requests
import numpy as np

# from train import Config as cfg
# from mmdet.apis import inference_detector, init_detector

app = Flask(__name__)

# Load the model and configurations
# model = init_detector(config='mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', checkpoint='mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', device='cuda:1')

# Function to process image from URL and get the best class
# def url_to_best_class(image_url):
#     response = requests.get(image_url)
#     image = Image.open(BytesIO(response.content))
#     image = np.array(image)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
#     result = inference_detector(model, image)
    
#     max_score = -1
#     best_class = None

#     for bbox in result:
#         if bbox.shape[0] != 0:
#             class_score = bbox[0, 4]
#             if class_score > max_score:
#                 max_score = class_score
#                 best_class = int(bbox[0, 0])

#     return best_class

# Route to accept image URL and process it
@app.route("/process_image", methods=['POST'])
def process_image():
    if request.method == 'POST':
        # Get the image URL from the JSON request with key 'image_url'
        image_url = request.get_json()['image_url']
        
        # best_class = url_to_best_class(image_url)
        return jsonify({'best_class': 0}) #bjsonify({'best_class': best_class})
    else:
        return 'Backend-server Connect'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
