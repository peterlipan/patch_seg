import cv2
import torch
import random
import skimage
import numpy as np
from PIL import Image
from numpy.random import default_rng

class RandomImageMaskShuffle:
    """Randomly shuffle a proportion of the images and corresponding masks in a batch.

    Args:
        size (int): Size of the input images and masks.
        ratio_up (float): The upper bound of the ratio of the image to be shuffled.
        ratio_low (float): The lower bound of the ratio of the image to be shuffled.
        p (float): The probability of shuffling the images and masks.
    """
    def __init__(self, size: int, ratio_up: float = 0.8, ratio_low: float = 0.1, p: float = 0.5):
        self.size = size
        self.ratio_up = ratio_up
        self.ratio_low = ratio_low
        self.p = p
        self.image_cache = None
        self.mask_cache = None


    def _generate_random_shape(self) -> np.ndarray:
        """Generate a random boolean mask for shuffling."""
        rng = default_rng()
        
        # Create random noise image
        noise = rng.integers(0, 255, (self.size, self.size), np.uint8, True)
        # Blur the noise image to control the size
        blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)
        # Stretch the blurred image to full dynamic range
        stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)
        
        # Generate a random proportion for thresholding
        ratio = random.uniform(self.ratio_low, self.ratio_up)
        upper_bound = int(np.percentile(stretch, ratio * 100))  # Adjusted for percentile

        # Apply cv2.threshold to the stretched image
        _, thresh = cv2.threshold(stretch, upper_bound, 255, cv2.THRESH_BINARY_INV)

        # Apply morphology open and close to smooth out and make the mask boolean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        final_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = final_mask.astype(bool)

        return final_mask

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> (np.ndarray, np.ndarray):
        """Shuffle the image and mask based on the generated mask."""
        if random.random() < self.p:
            if self.image_cache is None and self.mask_cache is None:
                self.image_cache = image.copy()
                self.mask_cache = mask.copy()
            else:
                random_idx = self._generate_random_shape()
                temp_image, temp_mask = image.copy(), mask.copy()
                image[random_idx] = self.image_cache[random_idx]
                mask[random_idx] = self.mask_cache[random_idx]
                self.image_cache, self.mask_cache = temp_image, temp_mask
        return image, mask
