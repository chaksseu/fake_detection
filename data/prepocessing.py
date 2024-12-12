from ultralytics import YOLO
import os

# YOLO 모델 로드
model = YOLO("./best.pt")

# 입력 및 출력 디렉토리 설정
image_dir = "../diseasecode_sorting/fraud/"
output_dir = "./f_results"
os.makedirs(output_dir, exist_ok=True)

# 이미지 파일 필터링
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

# 각 이미지에 대해 추론 수행
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    
    # YOLO 모델 추론
    results = model(image_path)
    
    # Bounding Box 정보 출력
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 좌표 (좌상단 x, y / 우하단 x, y)
        confidence = box.conf[0].cpu().numpy()  # Confidence Score
        class_id = int(box.cls[0].cpu().numpy())  # 클래스 ID
        
        print(f"Image: {image_file}, Class: {class_id}, Confidence: {confidence:.2f}, Box: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")
    
    # 결과 시각화 및 저장
    save_path = os.path.join(output_dir, f"result_{image_file}")
    results[0].plot(show=False)
    results[0].save(save_path)

print("모든 이미지에 대한 추론 및 Bounding Box 정보 출력이 완료되었습니다!")