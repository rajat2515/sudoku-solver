"""
PyTorch-based Digit Recognition Module for Sudoku Cells.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from model.network import DigitCNN


def grab_contours(cnts: Tuple) -> List[np.ndarray]:
    """Compatibility helper to retrieve contours across different OpenCV versions."""
    if len(cnts) == 2:
        return cnts[0]
    elif len(cnts) == 3:
        return cnts[1]
    return cnts


class DigitRecognizer:
    """
    Extracts digit candidates from cell ROIs and classifies them using PyTorch DigitCNN.
    """

    def __init__(self, model_path: Optional[str] = "model/digit_cnn.pt", device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = None
        self.is_ready = False
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        if model_path and os.path.exists(model_path):
            try:
                self.model = DigitCNN(num_classes=10)
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.is_ready = True
                print(f"[INFO] PyTorch model loaded successfully from: {model_path}")
            except Exception as e:
                print(f"[WARNING] Could not load model from {model_path}: {e}")
        else:
            print(f"[WARNING] Model weights not found at '{model_path}'. Running in detection-only mode.")
            print("[TIP] Run 'python model/train.py --dataset mnist' to train a fresh model.")

    def extract_cell_digit(self, digit_img: np.ndarray, cell_roi: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Locates the digit contour inside a cell ROI and crops it with safety margins.
        """
        r1, r2, c1, c2 = cell_roi
        cell_h = r2 - r1
        cell_w = c2 - c1

        # Margin buffer to ignore residual grid borders
        buf_r = cell_h // 9
        buf_c = cell_w // 9
        window = digit_img[r1 + buf_r:r2 - buf_r, c1 + buf_c:c2 - buf_c]
        if window.size == 0:
            return None

        min_area = (cell_h * cell_w) // 16

        contours = cv2.findContours(window.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = grab_contours(contours)
        if not contours:
            return None

        largest_cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        peri = cv2.arcLength(largest_cnt, True)
        approx = cv2.approxPolyDP(largest_cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)

        if (w * h) < min_area:
            return None

        # Square expansion for CNN input
        s = 2 * (max(w, h) // 2)
        r_start = max(0, r1 + y + (h // 2) - (s // 2) - (2 * buf_r))
        r_end = min(digit_img.shape[0], r1 + y + (h // 2) + (s // 2) + (3 * buf_r))
        c_start = max(0, c1 + x + (w // 2) - (s // 2) - (2 * buf_c))
        c_end = min(digit_img.shape[1], c1 + x + (w // 2) + (s // 2) + (3 * buf_c))

        patch = digit_img[r_start:r_end, c_start:c_end]
        if patch.size == 0 or patch.shape[0] < 5 or patch.shape[1] < 5:
            return None

        return patch

    def predict_patch(self, patch: np.ndarray) -> str:
        """Runs PyTorch inference on a 28x28 grayscale patch."""
        if not self.is_ready or self.model is None:
            return "0"

        try:
            tensor = self.transform(patch).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(tensor)
                predicted = output.argmax(dim=1).item()
                return str(predicted) if predicted != 0 else "0"
        except Exception:
            return "0"

    def recognize_grid(self, digit_img: np.ndarray, cells: List[Tuple[int, int, int, int]]) -> List[str]:
        """
        Iterates over all 81 cells and predicts digits.
        Returns a list of 81 strings ('0' for empty cells).
        """
        grid_digits = ["0"] * 81
        for i, cell in enumerate(cells[:81]):
            patch = self.extract_cell_digit(digit_img, cell)
            if patch is not None:
                grid_digits[i] = self.predict_patch(patch)
        return grid_digits
