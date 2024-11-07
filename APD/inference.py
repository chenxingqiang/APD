import torch
import cv2
import numpy as np
from typing import List, Tuple, Dict


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
        """
        Detect text in image and generate response using GPT-2
        """
        # Preprocess image for DBNet
        processed_image = self._preprocess_image(image)
        pixel_values = processed_image.unsqueeze(0).to(self.device)

        # Create input ids from prompt
        input_ids = self._tokenize_prompt(prompt)

        # Model inference
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            use_cache=True
        )

        # Process DBNet outputs
        text_regions = self._process_dbnet_output(
            outputs.dbnet_output,
            original_image=image
        )

        # Generate text using GPT-2
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
        """Preprocess image for model input"""
        # Resize image
        h, w = self.config.image_size
        image = cv2.resize(image, (w, h))

        # Normalize and convert to tensor
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC to CHW

        return image

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Convert prompt to input ids"""
        # Implement tokenization based on your tokenizer
        # This is a placeholder
        return torch.tensor([1]).to(self.device)

    def _process_dbnet_output(
        self,
        dbnet_output: Dict,
        original_image: np.ndarray
    ) -> List[Dict]:
        """Process DBNet detection output"""
        # Get predicted text regions
        text_regions = []
        predictions = dbnet_output['predictions']

        # Process each detected region
        for pred in predictions:
            # Convert coordinates to original image scale
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
        """Generate text using GPT-2"""
        # Implement text generation logic
        # This is a placeholder
        return "Generated text"

    def _rescale_coordinates(
        self,
        coords: np.ndarray,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """Rescale coordinates to original image size"""
        h, w = self.config.image_size
        orig_h, orig_w = original_size

        coords[:, 0] *= (orig_w / w)
        coords[:, 1] *= (orig_h / h)

        return coords
