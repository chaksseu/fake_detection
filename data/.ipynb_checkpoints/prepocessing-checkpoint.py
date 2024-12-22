from ultralytics import YOLO
import os
import pandas as pd
from PIL import Image, ImageOps
from natsort import natsorted  # 자연 정렬용 라이브러리

# YOLO 모델 로드
model = YOLO("./data/YOLO_model/best.pt")

# 입력 및 출력 디렉토리 설정
# base_dir = "./data/raw"
base_dir = "./data/split"
output_dir = "./data/processed"
os.makedirs(output_dir, exist_ok=True)

# 결과 저장을 위한 CSV 리스트 초기화
csv_data = []

# 배치 단위 이미지 전처리 함수
def process_images_in_batch(image_paths, folder_name, output_dir, csv_data):
    # 배치 처리 (YOLO는 GPU를 자동으로 사용함)
    results = model(image_paths)

    for image_path, result in zip(image_paths, results):
        image_file = os.path.basename(image_path)
        img = None

        try:
            # 이미지 로드
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"이미지 로드 실패: {image_path}, 오류: {e}")
            continue

        # YOLO 결과 처리
        if len(result.boxes) > 0:
            # 가장 높은 confidence를 가진 box 선택
            best_box = max(result.boxes, key=lambda box: box.conf[0])

            # Bounding Box 정보 가져오기
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()  # .cpu()를 사용하여 텐서를 numpy로 변환
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)  # 좌표 범위 확인
            confidence = best_box.conf[0].cpu().numpy()  # .cpu()로 변환
            class_id = int(best_box.cls[0].cpu().numpy())  # .cpu()로 변환

            # Bounding Box와 관련된 데이터 기록
            csv_data.append({
                "folder": folder_name,
                "file": image_file,
                "class_id": class_id,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

            # 잘린 부분 이미지 저장
            cropped = img.crop((x1, y1, x2, y2))
            cropped_save_path = os.path.join(output_dir, folder_name, "cropped")
            os.makedirs(cropped_save_path, exist_ok=True)
            cropped.save(os.path.join(cropped_save_path, f"cropped_{os.path.splitext(image_file)[0]}.png"))

            # 잘린 부분 제외 영역 이미지 저장
            excluded = img.copy()
            mask = Image.new("L", img.size, 0)  # 단일 채널 마스크 생성
            mask.paste(255, (int(x1), int(y1), int(x2), int(y2)))  # Bounding Box를 마스크로
            excluded_final = Image.composite(Image.new("RGB", img.size, (0, 0, 0)), excluded, mask)
            excluded_save_path = os.path.join(output_dir, folder_name, "excluded")
            os.makedirs(excluded_save_path, exist_ok=True)
            excluded_final.save(os.path.join(excluded_save_path, f"excluded_{os.path.splitext(image_file)[0]}.png"))
        else:
            # YOLO 결과가 없는 경우 기본 정보 기록
            csv_data.append({
                "folder": folder_name,
                "file": image_file,
                "class_id": None,
                "confidence": None,
                "x1": None,
                "y1": None,
                "x2": None,
                "y2": None
            })

# raw 폴더의 모든 서브 디렉토리 처리
batch_size = 64  # 원하는 배치 크기 설정
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        # 해당 디렉토리의 이미지 파일 필터링 및 자연 정렬
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        image_paths = [os.path.join(folder_path, image_file) for image_file in natsorted(image_files)]  # 자연 정렬 적용

        # 배치 단위로 이미지 처리
        for i in range(0, len(image_paths), batch_size):
            batch_image_paths = image_paths[i:i + batch_size]
            process_images_in_batch(batch_image_paths, folder_name, output_dir, csv_data)

# CSV 데이터를 이름 순으로 정렬
df = pd.DataFrame(csv_data)
df = df.sort_values(by=["folder", "file"])  # 폴더와 파일 이름 기준 정렬

# 결과를 CSV 파일로 저장
csv_output_path = os.path.join(output_dir, "results.csv")
df.to_csv(csv_output_path, index=False)

print(f"모든 이미지 처리 완료! 결과는 {csv_output_path}에 저장되었습니다.")
