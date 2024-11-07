import torch
import numpy as np
import cv2
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional


class DBNetInference:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> List[Dict]:
        # Preprocess image
        processed_image = self._preprocess_image(image)

        # Get model predictions
        outputs = self.model(processed_image.unsqueeze(0).to(self.device))

        # Post-process predictions to get bounding boxes
        regions = self._post_process(outputs, image.shape[:2])

        return regions

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        # Resize image to model's expected size
        image = cv2.resize(image, self.config.image_size[::-1])
        # Convert to float and normalize
        image = torch.from_numpy(image).float() / 255.0
        # Change from HWC to CHW format
        image = image.permute(2, 0, 1)
        return image

    def _post_process(self, outputs: Dict, original_size: Tuple[int, int]) -> List[Dict]:
        # Get probability map and convert to numpy
        prob_map = torch.sigmoid(outputs['prob_map']).cpu().numpy()[
            0, 0]  # Get first batch, first channel

        # Threshold to get binary map
        binary_map = (prob_map > self.config.threshold).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(
            binary_map,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            # Get confidence score from probability map
            mask = np.zeros_like(binary_map)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            confidence = float(np.mean(prob_map[mask == 1]))

            # Convert contour to polygon
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            bbox = approx.reshape(-1, 2)

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


class Visualizer:
    @staticmethod
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
