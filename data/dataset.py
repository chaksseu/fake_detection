import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import random
from torchvision import transforms  # 추가

class FD_Dataset(Dataset):
    def __init__(self, data_path, raw_data_path, mode, transform=None):
        """
        Initialize the dataset.

        Args:
            data_path (str): The base path to the processed dataset.
            raw_data_path (str): The base path to the raw dataset.
            mode (str): One of 'train', 'valid', or 'test'.
            transform (callable, optional): Transform to apply to the images.
        """
        if mode not in ['train', 'valid', 'test']:
            raise ValueError("mode must be one of 'train', 'valid', or 'test'")

        self.data_path = data_path
        self.raw_data_path = raw_data_path
        self.mode = mode
        # transform이 None인 경우 기본 transform을 적용
        self.transform = transform if transform is not None else transforms.ToTensor()

        # Paths to normal and fraud data folders
        normal_path = os.path.join(data_path, 'normal', f'{mode}.csv')
        fraud_path = os.path.join(data_path, 'fraud', f'{mode}.csv')

        # Load data from CSV files
        self.meta_data = self._load_csv(normal_path)

    def _load_csv(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_csv(file_path)

    def _load_image(self, folder, prefix, filename):
        base_name = os.path.splitext(filename)[0]  # Remove original extension
        file_path = os.path.join(self.data_path, folder, f"{prefix}{base_name}.png")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image not found: {file_path}")

        image = Image.open(file_path).convert("RGB")
        # self.transform이 callable한지 확인 후 적용
        if callable(self.transform):
            image = self.transform(image)
        return image

    def _load_raw_image(self, folder, filename):
        """
        Load a raw image from the raw data folder and apply transforms.
        """
        file_path = os.path.join(self.raw_data_path, folder, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Raw file not found: {file_path}")

        img = Image.open(file_path).convert("RGB")
        if callable(self.transform):
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        row = self.meta_data.iloc[idx]
        file_name = row['file']  # Assuming the CSV has a column named 'file'

        # 원본 anchor 이미지 로드 (transform 적용 전)
        anc_base_name = os.path.splitext(file_name)[0]
        anc_path = os.path.join(self.data_path, 'excluded', f"excluded_{anc_base_name}.png")
        if not os.path.exists(anc_path):
            raise FileNotFoundError(f"Image not found: {anc_path}")
        anc_pil_img = Image.open(anc_path).convert("RGB")

        # Positive 이미지 로드 (raw)
        positive_image = self._load_raw_image(row['category'], file_name) 

        # Negative 이미지를 위한 랜덤 선택
        negative_idx = random.randint(0, len(self.meta_data) - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.meta_data) - 1)
        negative_row = self.meta_data.iloc[negative_idx]
        negative_file_name = negative_row['file']

        # Negative 이미지 로드 (transform 적용 전)
        neg_base_name = os.path.splitext(negative_file_name)[0]
        neg_path = os.path.join(self.data_path, 'cropped', f"cropped_{neg_base_name}.png")
        if not os.path.exists(neg_path):
            raise FileNotFoundError(f"Negative image not found: {neg_path}")
        neg_pil_img = Image.open(neg_path).convert("RGB")
        x1, y1, x2, y2 = negative_row[['x1', 'y1', 'x2', 'y2']]
        cropped_pil = neg_pil_img.crop((x1, y1, x2, y2))

        # anchor 이미지에 negative cropped 이미지를 paste한 PIL 이미지 만들기
        anc_with_neg_pil = anc_pil_img.copy()
        anc_with_neg_pil.paste(cropped_pil, (x1, y1))

        # 이제 transform 적용
        # 원본 anchor 이미지 transform
        if callable(self.transform):
            anchor_original = self.transform(anc_pil_img)
        else:
            anchor_original = transforms.ToTensor()(anc_pil_img)

        # negative가 붙은 anchor transform
        if callable(self.transform):
            anchor_with_negative = self.transform(anc_with_neg_pil)
        else:
            anchor_with_negative = transforms.ToTensor()(anc_with_neg_pil)

        # cropped negative image transform
        if callable(self.transform):
            cropped_image = self.transform(cropped_pil)
        else:
            cropped_image = transforms.ToTensor()(cropped_pil)

        # 여기서는 negative를 anchor_with_negative로 사용할 것이므로,
        # return에서 'negative': anchor_with_negative 로 반환
        return {
            'anchor': anchor_original,
            'positive': positive_image,
            'negative': anchor_with_negative
        }
