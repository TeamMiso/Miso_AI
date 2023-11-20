import os
import shutil
from tqdm import tqdm

def split_data():
    # 원본 데이터셋 폴더와 저장할 폴더 지정
    original_dataset_folder = 'C:/cv_project/Recycling_trash/Separate_Collection/naverconnect-trash-data_dataset'
    train_folder = 'C:/cv_project/Recycling_trash/Separate_Collection/new_train2'

    # train 폴더가 없으면 생성
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    # 각 batch 폴더를 돌면서 jpg 파일을 train 폴더에 복사
    start_index = 0
    for batch_folder in tqdm(os.listdir(original_dataset_folder), desc="Processing batches"):
        batch_folder_path = os.path.join(original_dataset_folder, batch_folder)

        # batch 폴더가 존재하는지 확인
        if os.path.isdir(batch_folder_path):
            # batch 폴더 안에 있는 파일들을 정렬
            file_list = sorted(os.listdir(batch_folder_path))

            # train 폴더에 jpg 파일을 복사하면서 이름을 변경
            for i, file_name in enumerate(file_list):
                # 확장자가 jpg인지 확인
                if file_name.lower().endswith(".jpg"):
                    # 숫자 부분을 4자리로 맞추고 0을 추가
                    index = start_index + i
                    padded_number = f"{index:04d}" if index < 1000 else str(index)
                    new_file_name = f"{padded_number}.jpg"

                    source_path = os.path.join(batch_folder_path, file_name)
                    destination_path = os.path.join(train_folder, new_file_name)
                    shutil.copyfile(source_path, destination_path)

            # 다음 batch 폴더에서 시작할 인덱스 업데이트
            start_index += len(file_list) -1

if __name__ == '__main__':
    split_data()
