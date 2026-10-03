"""
Core modules for real-time Sudoku detection, recognition, and solving.
"""

from .solver import solve_sudoku, parse_sudoku_grid, format_solution

__all__ = [
    "solve_sudoku",
    "parse_sudoku_grid",
    "format_solution"
]

# Optional CV & ML imports (available when opencv/torch are installed)
try:
    from .detector import SudokuDetector
    __all__.append("SudokuDetector")
except ImportError:
    pass

try:
    from .recognizer import DigitRecognizer
    __all__.append("DigitRecognizer")
except ImportError:
    pass

try:
    from .visualizer import render_solution_overlay
    __all__.append("render_solution_overlay")
except ImportError:
    pass
