import os
import json

input_folder = 'naverconnect-trash-data_dataset'
output_file = 'merged_data.json'

# 빈 리스트를 생성합니다.
merged_data = {"images": [], "annotations": []}

# 이미지 및 어노테이션의 ID를 추적하기 위한 변수 초기화
image_id_counter = 0
annotation_id_counter = 0

for batch_folder in os.listdir(input_folder):
    batch_folder_path = os.path.join(input_folder, batch_folder)
    data_json_path = os.path.join(batch_folder_path, 'data.json')

    with open(data_json_path, 'r') as f:
        data = json.load(f)

    # images와 annotations에 대한 ID를 갱신
    for img_info in data['images']:
        img_info['id'] = image_id_counter
        img_info['file_name'] = f"{batch_folder}/{img_info['file_name']}"
        merged_data['images'].append(img_info)
        image_id_counter += 1

    for ann_info in data['annotations']:
        ann_info['id'] = annotation_id_counter
        merged_data['annotations'].append(ann_info)
        annotation_id_counter += 1

# 하나의 JSON 파일로 데이터를 저장
with open(output_file, 'w') as f:
    json.dump(merged_data, f)