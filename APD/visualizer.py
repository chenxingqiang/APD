import cv2
import numpy as np


class Visualizer:
    def visualize_detection(self, image, detected_regions, save_path=None):
        """
        Visualize detected text regions on an image.

        Args:
            image: numpy array of the original image
            detected_regions: list of detected region coordinates
            save_path: optional path to save the visualization
        """
        vis_image = image.copy()

        # Draw each detected region
        for region in detected_regions:
            points = np.array(region).astype(np.int32)
            cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

        return vis_image
