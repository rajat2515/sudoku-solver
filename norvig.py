"""
Backward-compatibility wrapper for core.solver.
"""

from core.solver import (
    solve_sudoku as solve,
    parse_sudoku_grid as parse_grid,
    format_solution as display,
    DIGITS as digits,
    ROWS as rows,
    COLS as cols,
    SQUARES as squares,
    UNITS as units,
    PEERS as peers
)

__all__ = ["solve", "parse_grid", "display", "digits", "rows", "cols", "squares", "units", "peers"]