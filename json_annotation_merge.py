import os
import json
from tqdm import tqdm

input_folder = 'C:/cv_project/Recycling_trash/Separate_Collection/naverconnect-trash-data_dataset'
output_train_file = 'C:/cv_project/Recycling_trash/Separate_Collection/train_3.json'
output_test_file = 'C:/cv_project/Recycling_trash/Separate_Collection/test_3.json'

# 빈 리스트를 생성합니다.
merged_data = {
    "info": {
        "year": 2021,
        "version": "1.0",
        "description": "Recycle Trash",
        "contributor": "Upstage",
        "url": None,
        "date_created": "2021-02-02 01:10:00"
    },
    "licenses": [
        {
            "id": 0,
            "name": "CC BY 4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/deed.ast"
        }
    ],
    "images": [],
    "annotations": [],
    "categories": [ {"id": 0, "name": "UNKNOWN", "supercategory": "UNKNOWN"},
                    {"id": 1, "name": "General trash", "supercategory": "General trash"},
                    {"id": 2, "name": "Paper", "supercategory": "Paper"},
                    {"id": 3, "name": "Paper pack", "supercategory": "Paper pack"},
                    {"id": 4, "name": "Metal", "supercategory": "Metal"},
                    {"id": 5, "name": "Glass", "supercategory": "Glass"},
                    {"id": 6, "name": "Plastic", "supercategory": "Plastic"},
                    {"id": 7, "name": "Styrofoam", "supercategory": "Styrofoam"},
                    {"id": 8, "name": "Plastic bag", "supercategory": "Plastic bag"},
                    {"id": 9, "name": "Battery", "supercategory": "Battery"},
                    {"id": 10, "name": "Clothing", "supercategory": "Clothing"}
                ]
}

# 이미지 및 어노테이션의 ID를 추적하기 위한 변수 초기화
image_id_counter = 0
annotation_id_counter = 0

for batch_folder in tqdm(os.listdir(input_folder), desc="Processing batches"):
    batch_folder_path = os.path.join(input_folder, batch_folder)
    data_json_path = os.path.join(batch_folder_path, 'data.json')

    with open(data_json_path, 'r') as f:
        data = json.load(f)

    # images와 annotations에 대한 ID를 갱신
    for img_info in data['images']:
        img_info['id'] = image_id_counter

        file_num = f"{image_id_counter:04d}" if image_id_counter < 1000 else str(image_id_counter)
        img_info['file_name'] = f"{file_num}.jpg"

        merged_data['images'].append(img_info)
        image_id_counter += 1

    for ann_info in data['annotations']:
        ann_info['id'] = annotation_id_counter
        merged_data['annotations'].append(ann_info)
        annotation_id_counter += 1

# 이미지의 총 개수를 확인
total_images = len(merged_data['images'])

# 나눌 위치 설정 (여기서는 80%를 train에 할당)
split_index = int(0.8 * total_images)

# Train 데이터 생성
train_data = {
    "info": merged_data["info"],
    "licenses": merged_data["licenses"],
    "images": merged_data["images"][:split_index],
    "annotations": merged_data["annotations"]
}

# Test 데이터 생성
test_data = {
    "info": merged_data["info"],
    "licenses": merged_data["licenses"],
    "images": merged_data["images"][split_index:],
    "annotations": merged_data["annotations"]
}

# Train JSON 파일로 저장
with open(output_train_file, 'w') as f_train:
    json.dump(train_data, f_train)

# Test JSON 파일로 저장
with open(output_test_file, 'w') as f_test:
    json.dump(test_data, f_test)