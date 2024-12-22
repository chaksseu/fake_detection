import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import random
from torchvision import transforms

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
        pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir) if f.lower().endswith(('original.png', '.jpg', '.jpeg'))]
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
            "valid/normal": "normal",
            "test/fraud": "fraud",
            "test/normal": "normal"
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
                      if f.lower().endswith(('original.png', '.jpg', '.jpeg'))]
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