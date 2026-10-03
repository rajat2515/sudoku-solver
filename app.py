#!/usr/bin/env python3
"""
Real-Time Augmented Reality Sudoku Solver with PyTorch and OpenCV.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

from core.detector import SudokuDetector
from core.recognizer import DigitRecognizer
from core.solver import solve_sudoku
from core.visualizer import render_solution_overlay, draw_detection_hud


# Sample classic puzzle for demonstration mode
DEMO_GRID = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Augmented Reality Sudoku Solver (PyTorch + OpenCV)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g. '0') or path to an image/video file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model/digit_cnn.pt",
        help="Path to PyTorch trained digit classifier weights (.pt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Enable demo mode with built-in test puzzle if model is missing or uncalibrated"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1000,
        help="Target processing width in pixels"
    )
    return parser.parse_args()


def open_media_source(source_str: str):
    """Detects whether source is a webcam index or file path and opens it."""
    if source_str.isdigit():
        cap = cv2.VideoCapture(int(source_str))
        is_stream = True
    else:
        path = Path(source_str)
        if not path.exists():
            print(f"[ERROR] Specified source file does not exist: {source_str}")
            sys.exit(1)
        # Try reading as static image first
        img = cv2.imread(str(path))
        if img is not None:
            return None, img, False
        cap = cv2.VideoCapture(str(path))
        is_stream = True

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source_str}")
        print("[TIP] Check camera permissions or specify an image file via --source <path>")
        sys.exit(1)

    return cap, None, is_stream


def run():
    args = parse_args()

    detector = SudokuDetector(target_width=args.width)
    recognizer = DigitRecognizer(model_path=args.model, device=args.device)

    cap, static_img, is_stream = open_media_source(args.source)
    print("\n" + "=" * 55)
    print("      REAL-TIME AR SUDOKU SOLVER (PyTorch)")
    print("=" * 55)
    print("Controls:")
    print("  [ESC] or [Q] : Exit application")
    print("  [S]          : Save current frame screenshot")
    print("  [D]          : Toggle demo puzzle mode")
    print("=" * 55 + "\n")

    demo_mode = args.demo
    last_solution = None
    last_grid_digits = None

    while True:
        if is_stream:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video stream ended or camera disconnected.")
                break
        else:
            frame = static_img.copy()

        # Step 1: Detect board
        resized_frame, result = detector.detect(frame)
        output_frame = resized_frame.copy()

        if result.found and result.warped_color is not None and result.digit_image is not None:
            # Step 2: Extract & recognize digits
            if demo_mode:
                grid_digits = list(DEMO_GRID)
            else:
                grid_digits = recognizer.recognize_grid(result.digit_image, result.cells)

            # Step 3: Solve grid
            # Only solve if we have clues or cached solution
            non_empty_count = sum(1 for d in grid_digits if d != "0")
            is_solved = False

            if non_empty_count >= 10:
                solution = solve_sudoku(grid_digits)
                if solution:
                    last_solution = solution
                    last_grid_digits = grid_digits
                    is_solved = True
            elif last_solution is not None:
                solution = last_solution
                grid_digits = last_grid_digits
                is_solved = True
            else:
                solution = None

            # Step 4: Render AR overlay
            if is_solved and solution is not None and result.cells is not None:
                output_frame = render_solution_overlay(
                    frame=resized_frame,
                    warped_color=result.warped_color,
                    cells=result.cells,
                    initial_digits=grid_digits,
                    solution=solution,
                    board_coords=result.board_coords,
                    frame_coords=result.corners
                )
                hud_status = "Sudoku Solved! (AR Active)"
            else:
                hud_status = f"Board Detected ({non_empty_count} clues detected)"

            output_frame = draw_detection_hud(
                output_frame,
                result.corners,
                status_text=hud_status,
                is_solved=is_solved
            )
        else:
            # Draw idle status
            cv2.putText(
                output_frame,
                "Searching for Sudoku grid...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("AR Sudoku Solver", output_frame)

        key = cv2.waitKey(1 if is_stream else 0) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("s"), ord("S")):
            cv2.imwrite("sudoku_screenshot.png", output_frame)
            print("[INFO] Screenshot saved as 'sudoku_screenshot.png'")
        elif key in (ord("d"), ord("D")):
            demo_mode = not demo_mode
            print(f"[INFO] Demo mode set to: {demo_mode}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
