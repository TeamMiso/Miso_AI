import os
import shutil

def split_data():
    # 원본 데이터셋 폴더와 저장할 폴더 지정
    original_dataset_folder = 'C:/cv_project/Recycling_trash/Separate_Collection/trash_dataset'
    train_folder = 'C:/cv_project/Recycling_trash/Separate_Collection/train'

    # train 폴더가 없으면 생성
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    # 각 batch 폴더를 돌면서 파일을 읽어와 train 폴더에 저장
    start_index = 0
    for batch_folder in range(1, 45):
        batch_folder_name = f'batch_{batch_folder}'
        batch_folder_path = os.path.join(original_dataset_folder, batch_folder_name)

        # batch 폴더가 존재하는지 확인
        if os.path.exists(batch_folder_path):
            # batch 폴더 안에 있는 파일들을 정렬
            file_list = sorted(os.listdir(batch_folder_path))

            # train 폴더에 파일을 복사하면서 이름을 변경
            for i, file_name in enumerate(file_list):
                # 숫자 부분을 4자리로 맞추고 0을 추가
                base_name, extension = os.path.splitext(file_name)
                padded_number = f"{int(base_name):04d}"
                new_file_name = f"{padded_number}{extension}"

                source_path = os.path.join(batch_folder_path, file_name)
                destination_path = os.path.join(train_folder, f'{start_index + i}_{new_file_name}')
                shutil.copyfile(source_path, destination_path)

            # 다음 batch 폴더에서 시작할 인덱스 업데이트
            start_index += len(file_list)

    print("파일 복사가 완료되었습니다.")

if __name__ == '__main__':
    split_data()
