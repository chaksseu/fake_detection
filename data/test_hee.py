import os
import pandas as pd
from tqdm import tqdm
import random
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import cv2
import easyocr


logging.basicConfig(
    filename='augmentation.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def load_csv(data_path, data_type, mode):
    file_path = os.path.join(data_path, data_type, f"{mode}.csv")
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    logging.info(f"Loaded CSV file: {file_path}")
    return pd.read_csv(file_path)


def random_jitter(image):
    """
    OCR(숫자만)으로 감지한 영역에 대해 색상, 회전, 이동, 스케일 등의 변화(jitter)를 주고
    다시 원래 위치에 재배치한 뒤, 전체 이미지에 대한 color jitter 및 affine 변환을 적용한다.
    """
    try:
        # PIL Image -> OpenCV 이미지 변환
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # EasyOCR 초기화 (영어 숫자 검출)
        reader = easyocr.Reader(['en'], gpu=True)

        # OCR 수행
        results = reader.readtext(cv_image, detail=1)

        # OCR 결과를 바탕으로 숫자 영역만 변환 적용
        augmented_image = image.copy()
        for bbox, text, confidence in results:
            # 신뢰도 및 숫자 여부 확인
            if confidence < 0.6 or not text.isdigit():
                continue

            (top_left, top_right, bottom_right, bottom_left) = bbox
            x_min = int(min(top_left[0], bottom_left[0]))
            y_min = int(min(top_left[1], top_right[1]))
            x_max = int(max(bottom_right[0], top_right[0]))
            y_max = int(max(bottom_right[1], bottom_left[1]))

            # 숫자 영역 크롭(PIL 기준)
            cropped_text_region = augmented_image.crop((x_min, y_min, x_max, y_max))

            # 개별 숫자 영역에 Color Jitter 적용
            if random.random() < 0.75:
                color_enhancer = ImageEnhance.Color(cropped_text_region)
                cropped_text_region = color_enhancer.enhance(random.uniform(0.85, 1.15))

            if random.random() < 0.75:
                brightness_enhancer = ImageEnhance.Brightness(cropped_text_region)
                cropped_text_region = brightness_enhancer.enhance(random.uniform(0.85, 1.15))

            if random.random() < 0.75:
                contrast_enhancer = ImageEnhance.Contrast(cropped_text_region)
                cropped_text_region = contrast_enhancer.enhance(random.uniform(0.85, 1.15))
                
            if random.random() < 0.75:
                sharpness_enhancer = ImageEnhance.Sharpness(cropped_text_region)
                cropped_text_region = sharpness_enhancer.enhance(random.uniform(0.85, 1.15))
                
            # 숫자 영역 회전
            if random.random() < 0.75:
                angle = random.uniform(-2, 2)
                cropped_text_region = cropped_text_region.rotate(angle, expand=True, fillcolor=(255, 255, 255))

            # 숫자 영역 스케일 조정
            if random.random() < 0.75:
                scale_factor = random.uniform(0.9, 1.1)
                new_w = max(1, int(cropped_text_region.width * scale_factor))
                new_h = max(1, int(cropped_text_region.height * scale_factor))
                cropped_text_region = cropped_text_region.resize((new_w, new_h), Image.BICUBIC)

            # 숫자 영역 평행이동
            translate_x = 0
            translate_y = 0
            if random.random() < 0.7:
                translate_x = random.randint(-2, 2)
                translate_y = random.randint(-2, 2)

            # 이동된 좌표 계산 (이미지 경계를 넘지 않도록 보정)
            paste_x = max(0, min(augmented_image.width - cropped_text_region.width, x_min + translate_x))
            paste_y = max(0, min(augmented_image.height - cropped_text_region.height, y_min + translate_y))

            # 변형된 숫자 영역을 다시 붙이기
            augmented_image.paste(cropped_text_region, (paste_x, paste_y))

        # 전체 이미지에 대한 Color Jitter
        image = augmented_image
        if random.random() < 0.5:
            color_enhancer = ImageEnhance.Color(image)
            image = color_enhancer.enhance(random.uniform(0.8, 1.2))
            
        if random.random() < 0.5:
            brightness_enhancer = ImageEnhance.Brightness(image)
            image = brightness_enhancer.enhance(random.uniform(0.8, 1.2))
            
        if random.random() < 0.5:
            contrast_enhancer = ImageEnhance.Contrast(image)
            image = contrast_enhancer.enhance(random.uniform(0.8, 1.2))
            
        if random.random() < 0.5:
            sharpness_enhancer = ImageEnhance.Sharpness(image)
            image = sharpness_enhancer.enhance(random.uniform(0.8, 1.2))

        # Affine 변환(회전, 스케일, 이동) 적용
        angle = 0
        translate_x, translate_y = 0, 0
        scale_factor = 1.0

        if random.random() < 0.5:
            angle = random.uniform(-2, 2)

        if random.random() < 0.5:
            translate_x = random.uniform(-2, 2)
            translate_y = random.uniform(-2, 2)

        if random.random() < 0.5:
            scale_factor = random.uniform(0.96, 1.04)

        angle_rad = np.deg2rad(angle)
        cos_theta = np.cos(angle_rad) * scale_factor
        sin_theta = np.sin(angle_rad) * scale_factor

        width, height = image.size
        center_x, center_y = width / 2, height / 2

        a = cos_theta
        b = sin_theta
        c = (1 - cos_theta) * center_x - sin_theta * center_y + translate_x
        d = -sin_theta
        e = cos_theta
        f = sin_theta * center_x + (1 - cos_theta) * center_y + translate_y

        affine_matrix = (a, b, c, d, e, f)

        image = image.transform(
            image.size,
            Image.AFFINE,
            affine_matrix,
            resample=Image.BICUBIC,
            fillcolor=(255, 255, 255)
        )

        return image

    except Exception as e:
        logging.error(f"Error during image augmentation with OCR-based text transformation: {e}")
        return image






def process_row(row, data_path, data_type, mode, output_path, num_color_jitter, generate_negatives, margin=50):
    try:
        file_name = row['file']
        x1, y1, x2, y2 = row[['x1', 'y1', 'x2', 'y2']]

        if pd.isnull(x1) or pd.isnull(y1) or pd.isnull(x2) or pd.isnull(y2):
            logging.warning(f"Skipping {file_name}: Missing bounding box coordinates")
            return

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        if bbox_width <= 0 or bbox_height <= 0:
            logging.warning(f"Skipping {file_name}: Invalid bounding box size")
            return

        excluded_path = os.path.join(data_path, data_type, "excluded", f"excluded_{os.path.splitext(file_name)[0]}.png")
        if not os.path.exists(excluded_path):
            logging.warning(f"Excluded image not found: {excluded_path}")
            return

        with Image.open(excluded_path).convert("RGB") as excluded_img:
            excluded_img = excluded_img.copy()
            img_width, img_height = excluded_img.size

            # Crop box with margin
            crop_x1 = max(x1 - margin, 0)
            crop_y1 = max(y1 - margin, 0)
            crop_x2 = min(x2 + margin, img_width)
            crop_y2 = min(y2 + margin, img_height)
            crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)

            base_output_dir = os.path.join(output_path, mode, data_type)
            sample_dir = os.path.join(base_output_dir, os.path.splitext(file_name)[0])
            anchor_dir = os.path.join(sample_dir, "anchor")
            pos_dir = os.path.join(sample_dir, "pos")
            neg_dir = os.path.join(sample_dir, "neg")

            os.makedirs(anchor_dir, exist_ok=True)
            os.makedirs(pos_dir, exist_ok=True)
            if generate_negatives:
                os.makedirs(neg_dir, exist_ok=True)

            # Anchor: no paste, just crop from excluded image with margin
            anchor_cropped = excluded_img.crop(crop_box)
            anchor_path = os.path.join(anchor_dir, "anchor.png")
            anchor_cropped.save(anchor_path)

            cropped_file_name = f"cropped_{os.path.splitext(file_name)[0]}.png"
            cropped_path = os.path.join(data_path, data_type, "cropped", cropped_file_name)

            if not os.path.exists(cropped_path):
                logging.warning(f"Pos image not found: {cropped_path}. Skipping pos image for {file_name}.")
                return

            with Image.open(cropped_path).convert("RGB") as cropped_img:
                # Resize cropped image to fit bbox
                resized_cropped = cropped_img.resize((bbox_width, bbox_height), Image.BILINEAR)

                # Pos: paste cropped image and then crop with margin
                combined_pos_img = excluded_img.copy()
                combined_pos_img.paste(resized_cropped, (x1, y1))
                pos_cropped = combined_pos_img.crop(crop_box)
                pos_original_path = os.path.join(pos_dir, "pos_original.png")
                pos_cropped.save(pos_original_path)

                # Neg: apply jitter, paste, then crop with margin
                if generate_negatives:
                    for j in range(num_color_jitter):
                        jittered_img = random_jitter(resized_cropped)
                        combined_jittered_img = excluded_img.copy()
                        combined_jittered_img.paste(jittered_img, (x1, y1))
                        neg_cropped = combined_jittered_img.crop(crop_box)
                        jittered_path = os.path.join(neg_dir, f"neg_{j}.png")
                        neg_cropped.save(jittered_path)

    except Exception as e:
        logging.error(f"Unexpected error processing row {row.get('file', 'Unknown')}: {e}")

def validate_excluded_images(data_path, data_type, mode, output_path, num_color_jitter=5, generate_negatives=True, num_workers=4):
    try:
        meta_data = load_csv(data_path, data_type, mode)
        cropped_dir = os.path.join(data_path, data_type, "cropped")
        if not os.path.isdir(cropped_dir):
            logging.error(f"Cropped directory not found: {cropped_dir}")
            raise FileNotFoundError(f"Cropped directory not found: {cropped_dir}")

        cropped_files = [f for f in os.listdir(cropped_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not cropped_files:
            logging.error(f"No cropped images found in {cropped_dir}")
            raise FileNotFoundError(f"No cropped images found in {cropped_dir}")

        base_output_dir = os.path.join(output_path, mode, data_type)
        os.makedirs(base_output_dir, exist_ok=True)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            process_func = partial(
                process_row,
                data_path=data_path,
                data_type=data_type,
                mode=mode,
                output_path=output_path,
                num_color_jitter=num_color_jitter,
                generate_negatives=generate_negatives
            )
            for _, row in meta_data.iterrows():
                futures.append(executor.submit(process_func, row))

            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                pass

        logging.info(f"Augmentation completed. Images saved in {base_output_dir}")
    except Exception as e:
        logging.error(f"Error in validate_excluded_images: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate augmented images with optional negatives.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--data_type", type=str, required=True, help="Data type (e.g., normal).")
    parser.add_argument("--mode", type=str, required=True, help="Mode (e.g., train, test).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save augmented data.")
    parser.add_argument("--num_color_jitter", type=int, default=10, help="Number of color jittered images to generate.")
    parser.add_argument("--generate_negatives", action="store_true", help="Flag to generate negatives.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of worker processes for parallel processing.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    validate_excluded_images(
        data_path=args.data_path,
        data_type=args.data_type,
        mode=args.mode,
        output_path=args.output_path,
        num_color_jitter=args.num_color_jitter,
        generate_negatives=args.generate_negatives,
        num_workers=args.num_workers
    )
