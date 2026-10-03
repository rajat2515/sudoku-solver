"""
Augmented Reality (AR) Overlay and Visualization Pipeline using OpenCV.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np


def render_solution_overlay(
    frame: np.ndarray,
    warped_color: np.ndarray,
    cells: List[Tuple[int, int, int, int]],
    initial_digits: List[str],
    solution: Dict[str, str],
    board_coords: np.ndarray,
    frame_coords: np.ndarray,
    font_scale: float = 0.9,
    font_color: Tuple[int, int, int] = (0, 0, 255)  # BGR Red
) -> np.ndarray:
    """
    Renders the solved numbers onto the warped board and projects them back onto the frame
    using an inverse perspective homography.
    """
    canvas = warped_color.copy()
    solved_list = list(solution.values())
    height, width = frame.shape[:2]

    # Draw solved numbers in empty positions
    for idx in range(min(81, len(cells))):
        if idx < len(initial_digits) and initial_digits[idx] != "0":
            continue  # Keep original printed clues unchanged

        if idx < len(solved_list):
            val_text = solved_list[idx]
            r1, r2, c1, c2 = cells[idx]
            center_x = (c1 + c2) // 2 - 8
            bottom_y = r2 - 10

            cv2.putText(
                canvas,
                val_text,
                (center_x, bottom_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_color,
                thickness=2,
                lineType=cv2.LINE_AA
            )

    # Compute homography from warped board to original camera perspective
    h_matrix, _ = cv2.findHomography(board_coords, frame_coords)
    if h_matrix is None:
        return frame

    warped_back = cv2.warpPerspective(canvas, h_matrix, (width, height))

    # Mask to blend seamlessly back into original frame
    mask = np.zeros((warped_color.shape[0], warped_color.shape[1]), dtype=np.uint8)
    mask.fill(255)
    warped_mask = cv2.warpPerspective(mask, h_matrix, (width, height))
    warped_mask_3ch = cv2.merge([warped_mask, warped_mask, warped_mask])

    # Combine original frame outside the board with solved board inside
    background = cv2.bitwise_and(frame, cv2.bitwise_not(warped_mask_3ch))
    foreground = cv2.bitwise_and(warped_back, warped_mask_3ch)
    return cv2.add(background, foreground)


def draw_detection_hud(
    frame: np.ndarray,
    corners: np.ndarray,
    status_text: str = "Sudoku Detected",
    is_solved: bool = False
) -> np.ndarray:
    """Draws a clean augmented reality bounding polygon and HUD text on the frame."""
    hud_frame = frame.copy()
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    border_color = (0, 255, 0) if is_solved else (0, 215, 255)
    cv2.polylines(hud_frame, [pts], isClosed=True, color=border_color, thickness=3)

    cv2.putText(
        hud_frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        border_color,
        2,
        cv2.LINE_AA
    )
    return hud_frame
