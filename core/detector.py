"""
Sudoku Board Detection and Grid Extraction Pipeline using OpenCV.
Self-contained implementation without third-party wrapper dependencies.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


def resize_image(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    """Resize an image maintaining aspect ratio."""
    if width is None and height is None:
        return image
    h, w = image.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def grab_contours(cnts: Tuple) -> List[np.ndarray]:
    """Compatibility helper to retrieve contours across different OpenCV versions."""
    if len(cnts) == 2:
        return cnts[0]
    elif len(cnts) == 3:
        return cnts[1]
    return cnts


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2)

    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective warp to obtain a top-down birds-eye view."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_w = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_h = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, transform_matrix, (max_w, max_h))


@dataclass
class DetectionResult:
    """Holds information about a detected Sudoku board in a camera frame."""
    found: bool
    warped_color: Optional[np.ndarray] = None
    digit_image: Optional[np.ndarray] = None
    corners: Optional[np.ndarray] = None
    cells: Optional[List[Tuple[int, int, int, int]]] = None
    board_coords: Optional[np.ndarray] = None


class SudokuDetector:
    """
    Robust Sudoku grid detector using adaptive thresholding, morphological filters,
    and perspective transformation.
    """

    def __init__(self, target_width: int = 1000, aspect_ratio_tolerance: float = 0.25):
        self.target_width = target_width
        self.min_ar = 1.0 - aspect_ratio_tolerance
        self.max_ar = 1.0 + aspect_ratio_tolerance

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resize frame, convert to grayscale, and blur."""
        resized = resize_image(frame, width=self.target_width)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return resized, blurred

    def find_grid_contour(self, blurred: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Apply adaptive threshold, line reinforcement, and locate the 4-corner board contour."""
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
        )

        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        line_canvas = thresh.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_canvas, (x1, y1), (x2, y2), 255, 2)

        cnts = cv2.findContours(line_canvas.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                _, _, w, h = cv2.boundingRect(approx)
                ar = w / float(h)
                if self.min_ar <= ar <= self.max_ar:
                    return approx, line_canvas

        return None, line_canvas

    def extract_clean_digits(self, warped_lines: np.ndarray) -> np.ndarray:
        """Isolate digits by filtering out horizontal and vertical grid lines."""
        h, w = warped_lines.shape[:2]

        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
        horiz = cv2.erode(warped_lines, horiz_kernel)
        horiz = cv2.dilate(horiz, horiz_kernel)

        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 10))
        vert = cv2.erode(warped_lines, vert_kernel)
        vert = cv2.dilate(vert, vert_kernel)

        grid = cv2.bitwise_or(horiz, vert)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grid = cv2.dilate(grid, kernel)
        grid = cv2.bitwise_and(grid, warped_lines)

        return cv2.bitwise_xor(warped_lines, grid)

    def compute_cell_rois(self, height: int, width: int) -> List[Tuple[int, int, int, int]]:
        """Compute the bounding coordinate tuples for all 81 sub-grid cells."""
        step_r = (height // 9) - 1
        step_c = (width // 9) - 1
        cells = []

        for r in range(0, height - step_r, step_r):
            for c in range(0, width - step_c, step_c):
                cells.append((r, r + step_r, c, c + step_c))
                if len(cells) == 81:
                    return cells
        return cells

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        """Process a single image or video frame."""
        resized_frame, blurred = self.preprocess_frame(frame)
        board_contour, line_img = self.find_grid_contour(blurred)

        if board_contour is None:
            return resized_frame, DetectionResult(found=False)

        corners = board_contour.reshape(4, 2)
        warped_lines = four_point_transform(line_img, corners)
        warped_color = four_point_transform(resized_frame, corners)
        digit_img = self.extract_clean_digits(warped_lines)

        h, w = digit_img.shape[:2]
        cells = self.compute_cell_rois(h, w)

        if (corners[0][0]) ** 2 + (corners[0][1]) ** 2 < (corners[1][0]) ** 2 + (corners[1][1]) ** 2:
            board_coords = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)
        else:
            board_coords = np.array([[w, 0], [0, 0], [0, h], [w, h]], dtype=np.float32)

        return resized_frame, DetectionResult(
            found=True,
            warped_color=warped_color,
            digit_image=digit_img,
            corners=corners.astype(np.float32),
            cells=cells,
            board_coords=board_coords
        )
