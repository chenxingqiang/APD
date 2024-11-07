import pickle
import os
import sys
import glob
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Dict
import multiprocessing as mp
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.optim import AdamW
import matplotlib.patches as patches
from torchvision import transforms
from typing import Optional
from torch.optim.lr_scheduler import CosineAnnealingLR
from albumentations import (
    Compose, RandomBrightnessContrast, GaussNoise, ShiftScaleRotate, Blur
)

from torch.utils.tensorboard import SummaryWriter
from shapely.geometry import Polygon
import torch.nn.functional as F

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

if True:
    from APD.config import APDConfig
    from APD.processor import APDProcessor
    from APD.model import APDModel


@dataclass
class Word:
    id: str
    file_path: Path
    writer_id: str
    transcription: str


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


class IAMDataset(Dataset):
    def __init__(self, words: List[Word], config: APDConfig):
        self.words = words
        self.processor = APDProcessor(
            config, add_eos_token=True, add_bos_token=True)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        inputs = self.processor(
            images=Image.open(self.words[item].file_path).convert('RGB'),
            texts=self.words[item].transcription,
            padding='max_length',
            return_tensors="pt",
            return_labels=True,
        )
        return {
            'pixel_values': inputs.pixel_values[0],
            'input_ids': inputs.input_ids[0],
            'attention_mask': inputs.attention_mask[0],
            'labels': inputs.labels[0]
        }


def save_dataset(dataset_path: Path, train_words: List[Word], validation_words: List[Word], test_words: List[Word]):
    data = {
        'train_words': train_words,
        'validation_words': validation_words,
        'test_words': test_words
    }
    with open(dataset_path / 'processed_dataset.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"Dataset saved to {dataset_path / 'processed_dataset.pkl'}")


def load_dataset(dataset_path: Path) -> Tuple[List[Word], List[Word], List[Word]]:
    processed_file = dataset_path / 'processed_dataset.pkl'
    if os.path.exists(processed_file):
        print(f"Loading processed dataset from {processed_file}")
        with open(processed_file, 'rb') as f:
            data = pickle.load(f)
        train_words = data['train_words']
        validation_words = data['validation_words']
        test_words = data['test_words']
        print(
            f'Loaded dataset - Train size: {len(train_words)}; Validation size: {len(validation_words)}; Test size: {len(test_words)}')
    else:
        print("Processed dataset not found. Generating new dataset.")
        train_words, validation_words, test_words = generate_dataset(
            dataset_path)
        save_dataset(dataset_path, train_words, validation_words, test_words)

    return train_words, validation_words, test_words


def generate_dataset(dataset_path: Path) -> Tuple[List[Word], List[Word], List[Word]]:
    xml_files = sorted(glob.glob(str(dataset_path / 'xml' / '*.xml')))
    word_image_files = sorted(
        glob.glob(str(dataset_path / 'words' / '**' / '*.png'), recursive=True))
    print(f"{len(xml_files)} XML files and {len(word_image_files)} word image files")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        words_from_xmls = list(
            tqdm.tqdm(
                pool.imap(partial(get_words_from_xml,
                          word_image_files=word_image_files), xml_files),
                total=len(xml_files),
                desc='Building dataset'
            )
        )

    words = [word for words in words_from_xmls for word in words]

    # Load train/validation/test splits
    with open(dataset_path / 'splits' / 'train.uttlist') as fp:
        train_ids = set(line.strip() for line in fp)
    with open(dataset_path / 'splits' / 'test.uttlist') as fp:
        test_ids = set(line.strip() for line in fp)
    with open(dataset_path / 'splits' / 'validation.uttlist') as fp:
        validation_ids = set(line.strip() for line in fp)

    train_words = [word for word in words if word.id in train_ids]
    validation_words = [word for word in words if word.id in validation_ids]
    test_words = [word for word in words if word.id in test_ids]

    print(
        f'Generated dataset - Train size: {len(train_words)}; Validation size: {len(validation_words)}; Test size: {len(test_words)}')
    return train_words, validation_words, test_words


def create_dataloaders(train_words: List[Word], validation_words: List[Word], test_words: List[Word], config: APDConfig, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_data = IAMDataset(words=train_words, config=config)
    validation_data = IAMDataset(words=validation_words, config=config)
    test_data = IAMDataset(words=test_words, config=config)

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=mp.cpu_count())
    validation_dataloader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False, num_workers=mp.cpu_count())
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=mp.cpu_count())

    return train_dataloader, validation_dataloader, test_dataloader


class HybridTrainer:
    def __init__(
        self,
        model,
        config,
        train_dataset,
        val_dataset=None,
        device='cuda',
        learning_rate=2e-4,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4
        ) if val_dataset else None

        self.optimizer = AdamW(model.parameters(), lr=learning_rate)

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm.tqdm(self.train_loader, desc="Training"):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
            )

            dbnet_loss = outputs.dbnet_output['loss']
            lm_loss = outputs.hidden_states.mean()

            total_batch_loss = dbnet_loss + lm_loss

            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()

        return total_loss / len(self.train_loader)


class HybridInference:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def detect_and_generate(
        self,
        image: np.ndarray,
        prompt: str = "",
        max_length: int = 50
    ) -> Dict:
        processed_image = self._preprocess_image(image)
        pixel_values = processed_image.unsqueeze(0).to(self.device)

        input_ids = self._tokenize_prompt(prompt)

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            use_cache=True
        )

        text_regions = self._process_dbnet_output(
            outputs.dbnet_output,
            original_image=image
        )

        generated_text = self._generate_text(
            outputs.hidden_states,
            outputs.past_key_values,
            max_length=max_length
        )

        return {
            'detected_regions': text_regions,
            'generated_text': generated_text
        }

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        h, w = self.config.image_size
        image = cv2.resize(image, (w, h))
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)
        return image

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        return torch.tensor([1]).to(self.device)

    def _process_dbnet_output(
        self,
        dbnet_output: Dict,
        original_image: np.ndarray
    ) -> List[Dict]:
        text_regions = []
        predictions = dbnet_output['predictions']

        for pred in predictions:
            coords = self._rescale_coordinates(
                pred['polygon'],
                original_image.shape[:2]
            )

            text_regions.append({
                'polygon': coords,
                'confidence': pred['confidence']
            })

        return text_regions

    def _generate_text(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Tuple,
        max_length: int
    ) -> str:
        return "Generated text"

    def _rescale_coordinates(
        self,
        coords: np.ndarray,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        h, w = self.config.image_size
        orig_h, orig_w = original_size

        coords[:, 0] *= (orig_w / w)
        coords[:, 1] *= (orig_h / h)

        return coords


class IAMDatasetForDBNet(Dataset):
    def __init__(self, words: List[Word], config: APDConfig, is_training: bool = True):
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
        # Normalize coordinates to [0, 1]
        bbox = bbox / np.array([w, h])[None, :]

        # Apply augmentations if in training mode
        if self.is_training:
            augmented = self.augmentor(image=image_np)
            image_np = augmented['image']

        # Resize image
        image_np = cv2.resize(image_np, self.config.image_size[::-1])

        # Prepare targets
        target = self._prepare_target(bbox, image_np.shape[:2])

        # Transform image
        image_tensor = self.transform(image_np)

        return {
            'image': image_tensor,
            'prob_map': target['prob_map'],
            'thresh_map': target['thresh_map'],
            'binary_map': target['binary_map']
        }

    def _prepare_target(self, bbox: np.ndarray, image_size: Tuple[int, int]) -> Dict:
        """Create DBNet targets"""
        h, w = image_size

        # Create empty target maps
        prob_map = np.zeros((h, w), dtype=np.float32)
        thresh_map = np.zeros((h, w), dtype=np.float32)
        binary_map = np.zeros((h, w), dtype=np.float32)

        # Convert normalized bbox back to pixel coordinates
        bbox_pixels = bbox * np.array([w, h])[None, :]
        bbox_pixels = bbox_pixels.astype(np.int32)

        # Generate probability map
        cv2.fillPoly(prob_map, [bbox_pixels], 1.0)

        # Generate threshold map (simplified)
        cv2.fillPoly(thresh_map, [bbox_pixels], 1.0)

        # Generate binary map
        cv2.fillPoly(binary_map, [bbox_pixels], 1.0)

        return {
            'prob_map': torch.from_numpy(prob_map).float(),
            'thresh_map': torch.from_numpy(thresh_map).float(),
            'binary_map': torch.from_numpy(binary_map).float()
        }


class DBNetTrainer:
    def __init__(self, model, config, train_dataset, val_dataset=None, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4
        ) if val_dataset else None

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs,
            eta_min=config.min_learning_rate
        )

        # Initialize metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_hmean': [],
            'val_precision': [],
            'val_recall': []
        }

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0

        # Learning rate warmup
        if epoch < self.config.warmup_epochs:
            warmup_factor = (epoch + 1) / self.config.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * warmup_factor

        for batch in tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}"):
            self.optimizer.zero_grad()

            # Changed 'pixel_values' to 'image' to match dataset output
            images = batch['image'].to(self.device)
            targets = {
                'prob_map': batch['prob_map'].to(self.device),
                'thresh_map': batch['thresh_map'].to(self.device),
                'binary_map': batch['binary_map'].to(self.device)
            }

            # Forward pass
            outputs = self.model(images)

            # Calculate losses
            loss_dict = self._compute_losses(outputs, targets)
            total_loss = sum(loss_dict.values())

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

        # Update learning rate
        self.scheduler.step()

        return total_loss / len(self.train_loader)

    def _compute_losses(self, outputs, targets):
        losses = {}

# Add unsqueeze to match dimensions
        prob_map = targets['prob_map'].unsqueeze(1)  # Add channel dimension
        thresh_map = targets['thresh_map'].unsqueeze(1)
        binary_map = targets['binary_map'].unsqueeze(1)

        losses['prob_loss'] = F.binary_cross_entropy_with_logits(
            outputs['prob_map'],
            prob_map
        )

        losses['thresh_loss'] = F.l1_loss(
            outputs['thresh_map'],
            thresh_map
        )

        losses['binary_loss'] = F.binary_cross_entropy_with_logits(
            outputs['binary_map'],
            binary_map
        )

        return losses

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss"""
        smooth = 1e-6
        pred = torch.sigmoid(pred)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice

    def validate(self):
        self.model.eval()
        total_metrics = []

        with torch.no_grad():
            for batch in tqdm.tqdm(self.val_loader, desc="Validating"):
                # Changed 'pixel_values' to 'image' to match dataset output
                images = batch['image'].to(self.device)
                targets = {
                    'prob_map': batch['prob_map'].to(self.device),
                    'thresh_map': batch['thresh_map'].to(self.device),
                    'binary_map': batch['binary_map'].to(self.device)
                }

                # Forward pass
                outputs = self.model(images)

                # Calculate metrics
                batch_metrics = self._calculate_metrics(outputs, targets)
                total_metrics.append(batch_metrics)

        # Average metrics across all batches
        avg_metrics = {}
        for key in total_metrics[0].keys():
            avg_metrics[key] = sum(m[key]
                                   for m in total_metrics) / len(total_metrics)

        return avg_metrics

    def _calculate_metrics(self, outputs, targets):
        # Calculate precision, recall, and hmean using local implementation
        metric = calculate_hmean(outputs, targets)
        return {
            'precision': metric['precision'],
            'recall': metric['recall'],
            'hmean': metric['hmean']
        }


class Visualizer:
    @ staticmethod
    def visualize_detection(image: np.ndarray, detected_regions: List[Dict],
                            save_path: Optional[str] = None):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        # Plot each detected region
        for region in detected_regions:
            bbox = region['bbox']
            confidence = region['confidence']

            # Create polygon patch
            polygon = patches.Polygon(
                bbox,
                fill=False,
                edgecolor='red',
                linewidth=2
            )
            plt.gca().add_patch(polygon)

            # Add confidence score
            center = bbox.mean(axis=0)
            plt.text(
                center[0], center[1],
                f'{confidence:.2f}',
                color='red',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7)
            )

        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()


class DBNetInference:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model.eval()

    @ torch.no_grad()
    def detect(self, image: np.ndarray) -> List[Dict]:
        # Preprocess image
        processed_image = self._preprocess_image(image)

        # Get model predictions
        outputs = self.model(processed_image.unsqueeze(0).to(self.device))

        # Post-process predictions to get bounding boxes
        regions = self._post_process(outputs, image.shape[:2])

        return regions

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        image = cv2.resize(image, self.config.image_size[::-1])
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)
        return image

    def _post_process(self, outputs: Dict, original_size: Tuple[int, int]) -> List[Dict]:
        prob_map = torch.sigmoid(outputs['prob_map']).cpu().numpy()[0]
        binary_map = (
            prob_map > self.config.min_text_confidence).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            bbox = contour.reshape(-1, 2)
            confidence = float(prob_map[bbox[:, 1], bbox[:, 0]].mean())

            # Rescale coordinates to original image size
            bbox = self._rescale_coordinates(bbox, original_size)

            regions.append({
                'bbox': bbox,
                'confidence': confidence
            })

        return regions

    def _rescale_coordinates(self, coords: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        h, w = self.config.image_size
        orig_h, orig_w = original_size

        coords = coords.astype(float)
        coords[:, 0] *= (orig_w / w)
        coords[:, 1] *= (orig_h / h)

        return coords


def crop_polygon(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Simple polygon cropping function"""
    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    cropped = image[y:y+h, x:x+w].copy()

    # Get mask
    points = points - points.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32)], (255))

    # Apply mask
    result = cv2.bitwise_and(cropped, cropped, mask=mask)
    return result


def calculate_hmean(predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
    """Simple implementation of hmean metric"""
    eps = 1e-6
    true_positives = 0
    num_predictions = len(predictions)
    num_targets = len(targets)

    for pred in predictions:
        for target in targets:
            iou = calculate_iou(pred['bbox'], target['bbox'])
            if iou > 0.5:  # IOU threshold
                true_positives += 1
                break

    precision = true_positives / (num_predictions + eps)
    recall = true_positives / (num_targets + eps)
    hmean = 2 * precision * recall / (precision + recall + eps)

    return {
        'precision': precision,
        'recall': recall,
        'hmean': hmean
    }


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate IoU between two polygons"""
    polygon1 = Polygon(box1)
    polygon2 = Polygon(box2)

    if not polygon1.is_valid or not polygon2.is_valid:
        return 0

    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area

    return intersection / union if union > 0 else 0


def main():
    dataset_path = Path('./datasets/iam_words')
    config = APDConfig()

    # Load dataset with new DBNet dataset class
    train_words, validation_words, test_words = load_dataset(dataset_path)

    train_dataset = IAMDatasetForDBNet(train_words, config, is_training=True)
    val_dataset = IAMDatasetForDBNet(
        validation_words, config, is_training=False)
    test_dataset = IAMDatasetForDBNet(test_words, config, is_training=False)

    # Create model
    model = APDModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize trainer
    trainer = DBNetTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device
    )

    # Add tensorboard logging
    writer = SummaryWriter('runs/dbnet_experiment')

    # Training loop with enhanced logging
    best_hmean = 0
    for epoch in range(config.max_epochs):
        # Train
        train_loss = trainer.train_one_epoch(epoch)

        # Validate
        val_metrics = trainer.validate()

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Metrics/precision', val_metrics['precision'], epoch)
        writer.add_scalar('Metrics/recall', val_metrics['recall'], epoch)
        writer.add_scalar('Metrics/hmean', val_metrics['hmean'], epoch)
        writer.add_scalar('LR', trainer.scheduler.get_last_lr()[0], epoch)

        # Save best model
        if val_metrics['hmean'] > best_hmean:
            best_hmean = val_metrics['hmean']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'best_hmean': best_hmean,
            }, 'best_model.pth')

    writer.close()

    # Initialize inferencer
    inferencer = DBNetInference(model=model, config=config, device=device)
    visualizer = Visualizer()

    # Test inference with visualization
    output_dir = Path('output_visualizations')
    output_dir.mkdir(exist_ok=True)

    for i, test_word in enumerate(test_words[:5]):
        image = Image.open(test_word.file_path).convert('RGB')
        image_np = np.array(image)

        # Detect text regions
        detected_regions = inferencer.detect(image_np)

        # Visualize and save results
        save_path = output_dir / f'detection_{i}.png'
        visualizer.visualize_detection(
            image_np,
            detected_regions,
            save_path=str(save_path)
        )

        print(f"Original text: {test_word.transcription}")
        print(f"Detected regions: {len(detected_regions)}")
        print(f"Visualization saved to {save_path}")
        print("---")

    # Add final test evaluation after training loop
    test_metrics = trainer.validate(DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4))
    print(f"Final test metrics: {test_metrics}")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

    main()
