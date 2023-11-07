from flask import Flask, request
from io import BytesIO
from PIL import Image
import base64
import cv2
import numpy as np

from train import Config as cfg
from mmdet.apis import inference_detector, init_detector

app = Flask(__name__)

# Load the model and configurations
model = init_detector(config='path_to_config_file', checkpoint='path_to_checkpoint_file', device='cuda:0')

# AI 모델 판단 후 결과 응답
@app.route("/model", methods=['POST'])
def decision():
    if request.method == 'POST':
        # Get the base64-encoded image from the JSON request with key 'id'
        params = request.get_json()['id']
        
        # Convert the base64 encoded image to PIL Image
        img = Image.open(BytesIO(base64.b64decode(params)))
        result = img_to_result(img)
        return str(result)
    else:
        return 'Backend-server Connect'

def img_to_result(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    result = inference_detector(model, image)
    
    # Variables to store predicted class and probability (or score)
    max_score = -1  # Highest score
    best_class = None  # Class with the highest score

    # Iterate through the results to find the class with the highest score
    for bbox in result:
        if bbox.shape[0] != 0:
            # Assuming class information is in the 0th column and the score of that class is in the 1st column
            class_score = bbox[0, 4]  # Score of the predicted object class

            # If the score for the current class is higher than the highest score, update it
            if class_score > max_score:
                max_score = class_score
                best_class = int(bbox[0, 0])  # The corresponding class

    return best_class

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
