import argparse
import os
import pandas as pd
from PIL import Image
import random
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Offline Augmentation for Pseudo Negative")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the processed dataset base")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to the raw dataset base")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save augmented data and csv")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "valid", "test"], help="Dataset mode")
    parser.add_argument("--data_type", type=str, default="normal", choices=["normal", "fraud"], help="Data type")
    parser.add_argument("--num_aug", type=int, default=1, help="Number of augmentations per original sample")
    parser.add_argument("--num_neg", type=int, default=1, help="Number of different negative segments per original sample")
    args = parser.parse_args()
    return args

def load_csv(data_path, data_type, mode):
    file_path = os.path.join(data_path, data_type, f"{mode}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def load_raw_image(raw_data_path, folder, filename):
    file_path = os.path.join(raw_data_path, folder, filename) if folder else os.path.join(raw_data_path, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw file not found: {file_path}")
    img = Image.open(file_path).convert("RGB")
    return img

def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    try:
        meta_data = load_csv(args.data_path, args.data_type, args.mode)
    except FileNotFoundError as e:
        print(str(e))
        return

    indices = list(range(len(meta_data)))

    aug_dir = os.path.join(args.output_path, args.data_type, args.mode, "augmented")
    os.makedirs(aug_dir, exist_ok=True)

    new_rows = []

    for i in tqdm(range(len(meta_data)), desc="Augmenting"):
        row = meta_data.iloc[i]
        file_name = row.get('file', None)
        if file_name is None:
            print("Skipping: 'file' column missing or invalid.")
            continue

        # folder 컬럼이 없으면 빈 문자열 사용
        folder = row.get('folder', '')

        anc_base_name = os.path.splitext(file_name)[0]

        # Anchor(excluded) 이미지 로드
        anc_path = os.path.join(args.data_path, args.data_type, 'excluded', f"excluded_{anc_base_name}.png")
        if not os.path.exists(anc_path):
            print(f"Excluded image not found: {anc_path}, skipping.")
            continue

        try:
            anc_pil_img = Image.open(anc_path).convert("RGB")
        except Exception as e:
            print(f"Error loading excluded image {anc_path}: {e}")
            continue

        # Positive(raw) 이미지 로드
        try:
            positive_img = load_raw_image(args.raw_data_path, folder, file_name)
        except FileNotFoundError as e:
            print(str(e))
            continue
        except Exception as e:
            print(f"Error loading raw image: {e}")
            continue

        for neg_i in range(args.num_neg):
            negative_idx = random.choice(indices)
            while negative_idx == i:
                negative_idx = random.choice(indices)

            negative_row = meta_data.iloc[negative_idx]
            negative_file_name = negative_row.get('file', None)
            if negative_file_name is None:
                print("Skipping negative sample: 'file' missing.")
                continue

            neg_base_name = os.path.splitext(negative_file_name)[0]

            neg_path = os.path.join(args.data_path, args.data_type, 'cropped', f"cropped_{neg_base_name}.png")
            if not os.path.exists(neg_path):
                print(f"Negative image not found: {neg_path}, skipping.")
                continue

            try:
                neg_pil_img = Image.open(neg_path).convert("RGB")
            except Exception as e:
                print(f"Error loading negative image {neg_path}: {e}")
                continue

            # bounding box 정보 가져오기
            if any(col not in negative_row for col in ['x1', 'y1', 'x2', 'y2']):
                print(f"Skipping {negative_file_name}: Missing bounding box coordinates")
                continue

            x1, y1, x2, y2 = negative_row[['x1', 'y1', 'x2', 'y2']]
            if pd.isnull(x1) or pd.isnull(y1) or pd.isnull(x2) or pd.isnull(y2):
                print(f"Skipping {negative_file_name}: Invalid bounding box coordinates")
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # negative 이미지를 bbox 크기로 resize
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                print(f"Skipping {negative_file_name}: Invalid bounding box size ({width}x{height})")
                continue

            resized_neg = neg_pil_img.resize((width, height), Image.BILINEAR)

            # negative 삽입된 anchor 이미지
            anc_with_neg_pil = anc_pil_img.copy()
            anc_with_neg_pil.paste(resized_neg, (x1, y1))

            for aug_i in range(args.num_aug):
                # 원본 anchor
                aug_anchor = anc_pil_img.copy()
                # negative 삽입 anchor
                aug_anchor_neg = anc_with_neg_pil.copy()
                # positive 이미지
                aug_positive = positive_img.copy()

                anchor_name = f"{anc_base_name}_neg{neg_i}_aug{aug_i}_anchor.png"
                anchor_neg_name = f"{anc_base_name}_neg{neg_i}_aug{aug_i}_anchor_neg.png"
                positive_name = f"{anc_base_name}_neg{neg_i}_aug{aug_i}_positive.png"

                aug_anchor.save(os.path.join(aug_dir, anchor_name))
                aug_anchor_neg.save(os.path.join(aug_dir, anchor_neg_name))
                aug_positive.save(os.path.join(aug_dir, positive_name))

                new_rows.append({
                    'file': file_name,
                    'folder': folder,
                    'anchor_aug': anchor_name,
                    'anchor_neg_aug': anchor_neg_name,
                    'positive_aug': positive_name
                })

    new_csv_path = os.path.join(args.output_path, args.data_type, f"{args.mode}_augmented.csv")
    os.makedirs(os.path.dirname(new_csv_path), exist_ok=True)
    pd.DataFrame(new_rows).to_csv(new_csv_path, index=False)
    print(f"Augmented dataset saved at: {new_csv_path}")

if __name__ == "__main__":
    main()
