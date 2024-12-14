import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import random
from torchvision import transforms

# class FD_Dataset(Dataset):
#     def __init__(self, data_path, raw_data_path, mode, data_type='normal', transform=None):
#         """
#         Initialize the dataset.

#         Args:
#             data_path (str): The base path to the processed dataset.
#             raw_data_path (str): The base path to the raw dataset.
#             mode (str): One of 'train', 'valid', or 'test'.
#             data_type (str): 'normal' 또는 'fraud' 중 하나를 선택
#             transform (callable, optional): Transform to apply to the images.
#         """
#         if mode not in ['train', 'valid', 'test']:
#             raise ValueError("mode must be one of 'train', 'valid', or 'test'")

#         if data_type not in ['normal', 'fraud']:
#             raise ValueError("data_type must be either 'normal' or 'fraud'")

#         self.data_path = data_path
#         self.raw_data_path = raw_data_path
#         self.mode = mode
#         self.data_type = data_type

#         # transform이 None인 경우 기본 transform을 적용
#         self.transform = transform if transform is not None else transforms.ToTensor()

#         # data_type에 따라 해당 CSV 로드
#         file_path = os.path.join(data_path, data_type, f'{mode}.csv')
#         self.meta_data = self._load_csv(file_path)

#     def _load_csv(self, file_path):
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found: {file_path}")
#         return pd.read_csv(file_path)

#     def _load_raw_image(self, folder, filename):
#         """
#         Load a raw image from the raw data folder and apply transforms.
#         """
#         file_path = os.path.join(self.raw_data_path, folder, filename)
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"Raw file not found: {file_path}")

#         img = Image.open(file_path).convert("RGB")
#         if callable(self.transform):
#             img = self.transform(img)
#         return img

#     def __len__(self):
#         return len(self.meta_data)

#     def __getitem__(self, idx):
#         row = self.meta_data.iloc[idx]
#         file_name = row['file']  # CSV에 'file' 열이 있다고 가정

#         # folder 열 체크
#         if 'folder' not in row:
#             raise KeyError("CSV must contain 'folder' column for positive image loading.")

#         # Anchor 이미지: ./data/processed/<data_type>/excluded/excluded_<anc_base_name>.png
#         anc_base_name = os.path.splitext(file_name)[0]
#         anc_path = os.path.join(self.data_path, self.data_type, 'excluded', f"excluded_{anc_base_name}.png")
#         if not os.path.exists(anc_path):
#             raise FileNotFoundError(f"Image not found: {anc_path}")
#         anc_pil_img = Image.open(anc_path).convert("RGB")

#         # Positive 이미지 로드 (raw)
#         positive_image = self._load_raw_image(row['folder'], file_name)

#         # Negative 이미지 인덱스 선택
#         negative_idx = random.randint(0, len(self.meta_data) - 1)
#         while negative_idx == idx:
#             negative_idx = random.randint(0, len(self.meta_data) - 1)
#         negative_row = self.meta_data.iloc[negative_idx]
#         negative_file_name = negative_row['file']

#         # Negative 이미지 로드
#         neg_base_name = os.path.splitext(negative_file_name)[0]
#         neg_path = os.path.join(self.data_path, self.data_type, 'cropped', f"cropped_{neg_base_name}.png")
#         if not os.path.exists(neg_path):
#             raise FileNotFoundError(f"Negative image not found: {neg_path}")
#         neg_pil_img = Image.open(neg_path).convert("RGB")

#         # bounding box 정보 확인 및 정수 변환
#         if not all(col in negative_row for col in ['x1', 'y1', 'x2', 'y2']):
#             raise KeyError("CSV must contain 'x1', 'y1', 'x2', 'y2' columns for cropping.")
#         x1, y1, x2, y2 = negative_row[['x1', 'y1', 'x2', 'y2']]
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#         cropped_pil = neg_pil_img.crop((x1, y1, x2, y2))

#         # anchor 이미지에 negative cropped 이미지를 paste
#         anc_with_neg_pil = anc_pil_img.copy()
#         anc_with_neg_pil.paste(cropped_pil, (x1, y1))

#         # Transform 적용
#         anchor_original = self.transform(anc_pil_img) if callable(self.transform) else transforms.ToTensor()(anc_pil_img)
#         anchor_with_negative = self.transform(anc_with_neg_pil) if callable(self.transform) else transforms.ToTensor()(anc_with_neg_pil)

#         # cropped_image는 사용하지 않으므로 제거했습니다.

#         return {
#             'anchor': anchor_original,
#             'positive': positive_image,
#             'negative': anchor_with_negative
#         }

class ANP_Dataset(Dataset):
    def __init__(self, augmented_file_path, mode, transform=None):
        """
        ANP_Dataset 초기화

        Args:
            augmented_file_path (str): Augmented 파일들의 상위 경로
            mode (str): train, valid, test 중 하나
            transform (callable, optional): 이미지에 적용할 transform
        """
        self.augmented_file_path = augmented_file_path
        self.mode = mode
        self.transform = transform

        # 해당 mode의 normal 폴더 내의 모든 개체 디렉토리 리스트
        self.sample_dirs = [
            os.path.join(augmented_file_path, mode, "normal", sample_dir)
            for sample_dir in os.listdir(os.path.join(augmented_file_path, mode, "normal"))
            if os.path.isdir(os.path.join(augmented_file_path, mode, "normal", sample_dir))
        ]

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        """
        idx에 따라 anchor, positive, negative 이미지를 로드하고 반환

        Args:
            idx (int): 데이터 인덱스

        Returns:
            dict: {'anchor': anchor_image, 'positive': positive_image, 'negative': negative_image}
        """
        sample_dir = self.sample_dirs[idx]

        # anchor, positive, negative 폴더 경로
        anchor_dir = os.path.join(sample_dir, "anchor")
        pos_dir = os.path.join(sample_dir, "pos")
        neg_dir = os.path.join(sample_dir, "neg")

        # anchor 이미지 로드
        anchor_files = [os.path.join(anchor_dir, f) for f in os.listdir(anchor_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        anchor_path = random.choice(anchor_files)
        anchor_image = Image.open(anchor_path).convert("RGB")

        # positive 이미지 로드
        pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        pos_path = random.choice(pos_files)
        positive_image = Image.open(pos_path).convert("RGB")

        # negative 이미지 로드
        neg_files = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        neg_path = random.choice(neg_files)
        negative_image = Image.open(neg_path).convert("RGB")

        # transform 적용
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return {
            'anchor': anchor_image,
            'positive': positive_image,
            'negative': negative_image
        }

class eval_AP_Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        AP_Dataset 초기화

        Args:
            data_path (str): Augmented 파일들의 상위 경로
            transform (callable, optional): 이미지에 적용할 transform

        데이터 로드:
        - train/fraud
        - valid/fraud
        - valid/normal
        """
        self.data_path = data_path
        self.transform = transform

        # 처리할 카테고리별 디렉토리 설정
        categories = {
            "train/fraud": "fraud",
            "valid/fraud": "fraud",
            "valid/normal": "normal"
        }

        self.sample_dirs = []
        for sub_path, label in categories.items():
            cat_dir = os.path.join(data_path, sub_path)
            if not os.path.exists(cat_dir):
                continue
            for sample_dir in os.listdir(cat_dir):
                full_path = os.path.join(cat_dir, sample_dir)
                if os.path.isdir(full_path):
                    # sample_dirs에 (샘플 경로, 라벨) 추가
                    self.sample_dirs.append((full_path, label))

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir, cat = self.sample_dirs[idx]

        # anchor, pos 폴더 경로
        anchor_dir = os.path.join(sample_dir, "anchor")
        pos_dir = os.path.join(sample_dir, "pos")

        # anchor 이미지 로드
        anchor_files = [os.path.join(anchor_dir, f) for f in os.listdir(anchor_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        anchor_path = random.choice(anchor_files)
        anchor_image = Image.open(anchor_path).convert("RGB")

        # positive 이미지 로드
        pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        pos_path = random.choice(pos_files)
        positive_image = Image.open(pos_path).convert("RGB")

        # transform 적용
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)

        # label: 'fraud' 또는 'normal'
        label = cat

        return {
            'anchor': anchor_image,
            'image': positive_image,
            'label': label
        }