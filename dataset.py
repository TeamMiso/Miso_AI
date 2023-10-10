import os
import shutil
import json
import logging

# logger 세팅
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 파일 핸들러, 포매터 세팅
formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
file_handler = logging.FileHandler()
logger.addHandler(file_handler)
file_handler.setFormatter(formatter)

# 원본 annotation 경로
annoDir = ""
# image가 있는 경로
imageDir = ""

# anno_dir 내 annotation 파일 이름을 리스트로 변경
annoList = os.listdir(annoDir)
logging.info(len(annoList))