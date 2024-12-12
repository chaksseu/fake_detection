import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import random

class triplet_Dataset(Dataset):
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
        self.transform = transform

        # Paths to normal and fraud data folders
        normal_path = os.path.join(data_path, 'normal', f'{mode}.csv')
        fraud_path = os.path.join(data_path, 'fraud', f'{mode}.csv')

        # Load data from CSV files
        self.meta_data = self._load_csv(normal_path)

    def _load_csv(self, file_path):
        """
        Load a CSV file and return a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        return pd.read_csv(file_path)

    def _load_image(self, folder, prefix, filename):
        """
        Load an image from a specific folder with a given prefix and filename.

        Args:
            folder (str): Folder containing the images.
            prefix (str): Prefix to prepend to the filename.
            filename (str): Name of the file (with original extension, e.g., .tif, .jpg).

        Returns:
            Image: Loaded image.
        """
        base_name = os.path.splitext(filename)[0]  # Remove the original extension
        file_path = os.path.join(self.data_path, folder, f"{prefix}{base_name}.png")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image not found: {file_path}")

        image = Image.open(file_path)
        if self.transform:
            image = self.transform(image)

        return image

    def _load_raw_file(self, folder, filename):
        """
        Load a raw file from the raw data folder.

        Args:
            folder (str): Folder containing the raw files.
            filename (str): Name of the raw file (with extension).

        Returns:
            str: Path to the raw file.
        """
        file_path = os.path.join(self.raw_data_path, folder, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Raw file not found: {file_path}")

        return file_path

    def __len__(self):
        """Return the total number of samples."""
        return len(self.meta_data)

    def __getitem__(self, idx):
        """
        Get the data at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Contains anchor, positive, and negative images.
        """
        row = self.meta_data.iloc[idx]
        file_name = row['file']  # Assuming the CSV has a column named 'file'

        # Load anchor (excluded image)
        anchor_image = self._load_image('excluded', 'excluded_', file_name)

        # Load positive (raw image)
        positive_image = self._load_raw_file(row['category'], file_name)  # Assuming 'category' column specifies folder

        # Load negative image (cropped image with random insertion into anchor)
        negative_idx = random.randint(0, len(self.meta_data) - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.meta_data) - 1)
        negative_row = self.meta_data.iloc[negative_idx]
        negative_file_name = negative_row['file']
        
        # Load the cropped image for negative
        cropped_image = self._load_image('cropped', 'cropped_', negative_file_name)
        x1, y1, x2, y2 = negative_row[['x1', 'y1', 'x2', 'y2']]  # Assuming these columns exist
        cropped_image = cropped_image.crop((x1, y1, x2, y2))

        # Paste the cropped negative image onto the anchor image
        anchor_with_negative = anchor_image.copy()
        anchor_with_negative.paste(cropped_image, (x1, y1))

        return {
            'anchor': anchor_image,
            'positive': positive_image,
            'negative': anchor_with_negative
        }
