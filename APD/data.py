import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict
from torch.utils.data import Dataset
from PIL import Image
import os
import pickle
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from torchvision import transforms
import cv2
from albumentations import (
    Compose, RandomBrightnessContrast, GaussNoise, ShiftScaleRotate, Blur
)


@dataclass
class Word:
    id: str
    file_path: Path
    writer_id: str
    transcription: str

    def __getstate__(self):
        """Return state values to be pickled."""
        return {
            'id': self.id,
            'file_path': str(self.file_path),  # Convert Path to string
            'writer_id': self.writer_id,
            'transcription': self.transcription
        }

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.id = state['id']
        # Convert string back to Path
        self.file_path = Path(state['file_path'])
        self.writer_id = state['writer_id']
        self.transcription = state['transcription']


@dataclass
class APDProcessorOutput:
    pixel_values: Optional[torch.FloatTensor] = None
    input_ids: Optional[Union[torch.LongTensor, np.ndarray, List[int]]] = None
    attention_mask: Optional[Union[torch.FloatTensor,
                                   np.ndarray, List[int]]] = None
    labels: Optional[Union[torch.LongTensor, np.ndarray, List[int]]] = None


def get_words_from_xml(xml_file: str, word_image_files: List[str]) -> List[Word]:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    root_id = root.get('id')
    writer_id = root.get('writer-id')
    xml_words = []
    for line in root.findall('handwritten-part')[0].findall('line'):
        for word in line.findall('word'):
            image_file = Path(
                [f for f in word_image_files if f.endswith(word.get('id') + '.png')][0])
            try:
                with Image.open(image_file) as _:
                    xml_words.append(
                        Word(
                            id=root_id,
                            file_path=image_file,
                            writer_id=writer_id,
                            transcription=word.get('text')
                        )
                    )
            except Exception:
                pass
    return xml_words


def load_dataset(dataset_path: Path) -> Tuple[List[Word], List[Word], List[Word]]:
    """Load the full IAM words dataset"""
    processed_file = dataset_path / 'processed_dataset.pkl'
    if os.path.exists(processed_file):
        print(f"Loading processed dataset from {processed_file}")
        with open(processed_file, 'rb') as f:
            try:
                data = pickle.load(f)
                train_words = data['train_words']
                validation_words = data['validation_words']
                test_words = data['test_words']
            except (AttributeError, pickle.UnpicklingError):
                print("Error loading processed dataset. Regenerating...")
                train_words, validation_words, test_words = generate_dataset(
                    dataset_path)
                save_dataset(dataset_path, train_words,
                             validation_words, test_words)
    else:
        train_words, validation_words, test_words = generate_dataset(
            dataset_path)
        save_dataset(dataset_path, train_words, validation_words, test_words)

    print(f'Loaded dataset - Train size: {len(train_words)}; '
          f'Validation size: {len(validation_words)}; '
          f'Test size: {len(test_words)}')
    return train_words, validation_words, test_words


def generate_dataset(dataset_path: Path) -> Tuple[List[Word], List[Word], List[Word]]:
    """Generate dataset from raw files"""
    xml_files = sorted(glob.glob(str(dataset_path / 'xml' / '*.xml')))
    word_image_files = sorted(
        glob.glob(str(dataset_path / 'words' / '**' / '*.png'), recursive=True))
    print(f"{len(xml_files)} XML files and {len(word_image_files)} word image files")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        words_from_xmls = list(
            tqdm(
                pool.imap(partial(get_words_from_xml, word_image_files=word_image_files),
                          xml_files),
                total=len(xml_files),
                desc='Building dataset'
            )
        )

    words = [word for words in words_from_xmls for word in words]

    # Load splits
    with open(dataset_path / 'splits' / 'train.uttlist') as fp:
        train_ids = set(line.strip() for line in fp)
    with open(dataset_path / 'splits' / 'test.uttlist') as fp:
        test_ids = set(line.strip() for line in fp)
    with open(dataset_path / 'splits' / 'validation.uttlist') as fp:
        validation_ids = set(line.strip() for line in fp)

    train_words = [word for word in words if word.id in train_ids]
    validation_words = [word for word in words if word.id in validation_ids]
    test_words = [word for word in words if word.id in test_ids]

    return train_words, validation_words, test_words


def save_dataset(dataset_path: Path, train_words: List[Word],
                 validation_words: List[Word], test_words: List[Word]):
    """Save processed dataset to pickle file"""
    data = {
        'train_words': train_words,
        'validation_words': validation_words,
        'test_words': test_words
    }
    with open(dataset_path / 'processed_dataset.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"Dataset saved to {dataset_path / 'processed_dataset.pkl'}")


class IAMDatasetForDBNet(Dataset):
    def __init__(self, words: List[Word], config, is_training: bool = True):
        self.words = words
        self.config = config
        self.is_training = is_training

        # Basic image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Advanced augmentations for training
        if is_training:
            self.augmentor = Compose([
                RandomBrightnessContrast(p=0.5),
                GaussNoise(p=0.3),
                Blur(blur_limit=3, p=0.3),
                ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=config.scale_range,
                    rotate_limit=config.rotation_range,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=config.aug_prob
                ),
            ])

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        image = Image.open(word.file_path).convert('RGB')
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        # Get word bounding box (normalized coordinates)
        bbox = np.array([0, 0, w, 0, w, h, 0, h],
                        dtype=np.float32).reshape(-1, 2)
        bbox = bbox / np.array([w, h])[None, :]

        # Apply augmentations if in training mode
        if self.is_training and hasattr(self, 'augmentor'):
            augmented = self.augmentor(image=image_np)
            image_np = augmented['image']

        # Resize image
        image_np = cv2.resize(image_np, self.config.image_size[::-1])

        # Prepare targets
        target = self._prepare_target(bbox, image_np.shape[:2])

        # Transform image
        image_tensor = self.transform(image_np)

        return image_tensor, target

    def _prepare_target(self, bbox: np.ndarray, image_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """Create DBNet targets"""
        h, w = image_size
        prob_map = np.zeros((h, w), dtype=np.float32)
        thresh_map = np.zeros((h, w), dtype=np.float32)
        binary_map = np.zeros((h, w), dtype=np.float32)

        bbox_pixels = bbox * np.array([w, h])[None, :]
        bbox_pixels = bbox_pixels.astype(np.int32)

        cv2.fillPoly(prob_map, [bbox_pixels], 1.0)
        cv2.fillPoly(thresh_map, [bbox_pixels], 1.0)
        cv2.fillPoly(binary_map, [bbox_pixels], 1.0)

        return {
            'prob_map': torch.from_numpy(prob_map).float(),
            'thresh_map': torch.from_numpy(thresh_map).float(),
            'binary_map': torch.from_numpy(binary_map).float()
        }
