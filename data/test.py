import os
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
from tqdm import tqdm
import random
import argparse

def load_csv(data_path, data_type, mode):
    file_path = os.path.join(data_path, data_type, f"{mode}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def random_color_jitter(image):
    """
    이미지에 색조, 밝기, 대비, 채도, 선명도를 조정하여 변화를 적용
    """
    # 색조 조정
    color_enhancer = ImageEnhance.Color(image)
    color_factor = random.uniform(0.4, 1.6)  # 색조 변환 정도 (0.4 ~ 1.6)
    image = color_enhancer.enhance(color_factor)

    # 밝기 조정
    brightness_enhancer = ImageEnhance.Brightness(image)
    brightness_factor = random.uniform(0.6, 1.4)  # 밝기 변환 정도 (0.6 ~ 1.4)
    image = brightness_enhancer.enhance(brightness_factor)

    # 대비 조정
    contrast_enhancer = ImageEnhance.Contrast(image)
    contrast_factor = random.uniform(0.7, 1.5)  # 대비 변환 정도 (0.7 ~ 1.5)
    image = contrast_enhancer.enhance(contrast_factor)

    # 선명도 조정
    sharpness_enhancer = ImageEnhance.Sharpness(image)
    sharpness_factor = random.uniform(0.7, 1.5)  # 선명도 변환 정도 (0.7 ~ 1.5)
    image = sharpness_enhancer.enhance(sharpness_factor)

    return image

def resize_to_1000(img_path):
    # 이미지를 다시 로드하고 1000x1000으로 리사이즈 후 덮어쓰기
    img = Image.open(img_path)
    resized_img = img.resize((1000, 1000), Image.BILINEAR)
    resized_img.save(img_path)

def validate_excluded_images(data_path, data_type, mode, output_path, num_pseudo=5, num_color_jitter=5, generate_negatives=True):
    # CSV 로드
    meta_data = load_csv(data_path, data_type, mode)

    # 기본 출력 경로: /data/augment/train/normal/
    base_output_dir = os.path.join(output_path, mode, data_type)
    os.makedirs(base_output_dir, exist_ok=True)

    # cropped 디렉토리
    cropped_dir = os.path.join(data_path, data_type, "cropped")
    cropped_files = [f for f in os.listdir(cropped_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not cropped_files:
        raise FileNotFoundError(f"No cropped images found in {cropped_dir}")

    for _, row in tqdm(meta_data.iterrows(), desc="Validating", total=len(meta_data)):
        file_name = row['file']
        folder = row.get('folder', '')  # folder 정보가 없는 경우 빈 문자열 사용
        x1, y1, x2, y2 = row[['x1', 'y1', 'x2', 'y2']]

        # 좌표 검증
        if pd.isnull(x1) or pd.isnull(y1) or pd.isnull(x2) or pd.isnull(y2):
            print(f"Skipping {file_name}: Missing bounding box coordinates")
            continue

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Excluded 이미지 로드
        excluded_path = os.path.join(data_path, data_type, "excluded", f"excluded_{os.path.splitext(file_name)[0]}.png")
        if not os.path.exists(excluded_path):
            print(f"Excluded image not found: {excluded_path}")
            continue

        excluded_img = Image.open(excluded_path).convert("RGB")

        bbox_width = x2 - x1
        bbox_height = y2 - y1
        if bbox_width <= 0 or bbox_height <= 0:
            print(f"Skipping {file_name}: Invalid bounding box size")
            continue

        # 현재 row 제외한 나머지 meta_data에서 랜덤 선택
        neg_candidates = meta_data[meta_data['file'] != file_name]['file'].values
        if len(neg_candidates) == 0:
            print(f"No negative candidates found for {file_name}. Skipping.")
            continue

        # 현재 file_name용 폴더 생성
        base_name = os.path.splitext(file_name)[0]
        sample_dir = os.path.join(base_output_dir, base_name)
        anchor_dir = os.path.join(sample_dir, "anchor")
        pos_dir = os.path.join(sample_dir, "pos")
        neg_dir = os.path.join(sample_dir, "neg")

        os.makedirs(anchor_dir, exist_ok=True)
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)

        # anchor 이미지 저장 (excluded 이미지)
        anchor_path = os.path.join(anchor_dir, "anchor.png")
        excluded_img.save(anchor_path)

        # pos 이미지 생성 (cropped_{file_name}.png)
        cropped_file_name = f"cropped_{os.path.splitext(file_name)[0]}.png"
        cropped_path = os.path.join(cropped_dir, cropped_file_name)
        if os.path.exists(cropped_path):
            cropped_img = Image.open(cropped_path).convert("RGB").resize((bbox_width, bbox_height), Image.BILINEAR)
            combined_pos_img = excluded_img.copy()
            combined_pos_img.paste(cropped_img, (x1, y1))
            pos_original_path = os.path.join(pos_dir, "pos_original.png")
            combined_pos_img.save(pos_original_path)
            # 1000x1000 리사이즈
            resize_to_1000(pos_original_path)

            # 색조 변조된 pos 이미지 생성 및 저장
            for j in range(num_color_jitter):
                jittered_img = random_color_jitter(cropped_img)
                combined_jittered_img = excluded_img.copy()
                combined_jittered_img.paste(jittered_img, (x1, y1))
                jittered_path = os.path.join(pos_dir, f"pos_{j}.png")
                combined_jittered_img.save(jittered_path)
                resize_to_1000(jittered_path)
        else:
            print(f"Pos image not found: {cropped_path}. Skipping pos image for {file_name}.")

        # Negative 이미지 생성 및 저장 (옵션 활성화 시)
        if generate_negatives:
            num_to_generate = min(len(neg_candidates), num_pseudo)
            selected_neg_files = random.sample(list(neg_candidates), num_to_generate)

            for i, neg_file in enumerate(selected_neg_files):
                neg_cropped_file_name = f"cropped_{os.path.splitext(neg_file)[0]}.png"
                neg_cropped_path = os.path.join(cropped_dir, neg_cropped_file_name)
                if not os.path.exists(neg_cropped_path):
                    print(f"Negative image not found: {neg_cropped_path}, skipping this candidate.")
                    continue

                try:
                    neg_img = Image.open(neg_cropped_path).convert("RGB")
                except Exception as e:
                    print(f"Error loading negative image {neg_cropped_path}: {e}")
                    continue

                resized_neg = neg_img.resize((bbox_width, bbox_height), Image.BILINEAR)
                pseudo_img = excluded_img.copy()
                pseudo_img.paste(resized_neg, (x1, y1))

                pseudo_path = os.path.join(neg_dir, f"neg_{i}.png")
                pseudo_img.save(pseudo_path)
                # 1000x1000 리사이즈
                resize_to_1000(pseudo_path)

        # anchor 이미지도 후처리 리사이즈 (필요하다면)
        resize_to_1000(anchor_path)

    print(f"Augmentation images saved in {base_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented images with optional negatives.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--data_type", type=str, required=True, help="Data type (e.g., normal).")
    parser.add_argument("--mode", type=str, required=True, help="Mode (e.g., train, test).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save augmented data.")
    parser.add_argument("--num_pseudo", type=int, default=100, help="Number of pseudo negative images to generate.")
    parser.add_argument("--num_color_jitter", type=int, default=20, help="Number of color jittered images to generate.")
    parser.add_argument("--generate_negatives", action="store_true", help="Flag to generate negatives.")

    args = parser.parse_args()

    validate_excluded_images(
        data_path=args.data_path,
        data_type=args.data_type,
        mode=args.mode,
        output_path=args.output_path,
        num_pseudo=args.num_pseudo,
        num_color_jitter=args.num_color_jitter,
        generate_negatives=args.generate_negatives
    )
