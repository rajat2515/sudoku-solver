"""
Sudoku constraint propagation and backtracking search solver.
Based on Peter Norvig's constraint propagation algorithm (norvig.com/sudoku.html).
"""

from typing import Dict, List, Optional, Union


def cross(a_list: str, b_list: str) -> List[str]:
    """Cartesian cross product of string elements."""
    return [a + b for a in a_list for b in b_list]


DIGITS: str = "123456789"
ROWS: str = "ABCDEFGHI"
COLS: str = DIGITS
SQUARES: List[str] = cross(ROWS, COLS)

UNIT_LIST: List[List[str]] = (
    [cross(ROWS, c) for c in COLS] +
    [cross(r, COLS) for r in ROWS] +
    [cross(rs, cs) for rs in ("ABC", "DEF", "GHI") for cs in ("123", "456", "789")]
)

UNITS: Dict[str, List[List[str]]] = {
    s: [u for u in UNIT_LIST if s in u] for s in SQUARES
}

PEERS: Dict[str, set] = {
    s: set(sum(UNITS[s], [])) - {s} for s in SQUARES
}


def parse_sudoku_grid(grid: Union[str, List[Union[str, int]]]) -> Union[Dict[str, str], bool]:
    """
    Convert a grid (string or list of 81 characters/numbers) into a dict of possible values.
    Returns False if a contradiction is detected.
    """
    if isinstance(grid, list):
        grid = "".join(str(x) for x in grid)

    # Initialize all squares with all possibilities
    values = {s: DIGITS for s in SQUARES}
    
    # Extract only valid digits or blank indicators (0 or .)
    chars = [c for c in grid if c in DIGITS or c in "0."]
    if len(chars) != 81:
        return False

    grid_map = dict(zip(SQUARES, chars))
    for s, d in grid_map.items():
        if d in DIGITS and not assign(values, s, d):
            return False  # Failed to assign initial digit
    return values


def assign(values: Dict[str, str], s: str, d: str) -> Union[Dict[str, str], bool]:
    """
    Eliminate all other candidate values from values[s] except d and propagate constraints.
    Returns values dict, or False on contradiction.
    """
    other_values = values[s].replace(d, "")
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    return False


def eliminate(values: Dict[str, str], s: str, d: str) -> Union[Dict[str, str], bool]:
    """
    Eliminate d from candidates in values[s]; propagate when single choices remain.
    Returns values dict, or False on contradiction.
    """
    if d not in values[s]:
        return values  # Already eliminated

    values[s] = values[s].replace(d, "")

    # Rule (1): If square s is reduced to one candidate, eliminate it from peers
    if len(values[s]) == 0:
        return False  # Contradiction: removed last candidate
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in PEERS[s]):
            return False

    # Rule (2): If a unit has only one place left for candidate d, assign it there
    for u in UNITS[s]:
        dplaces = [sq for sq in u if d in values[sq]]
        if len(dplaces) == 0:
            return False  # Contradiction: no place left for digit d
        elif len(dplaces) == 1:
            if not assign(values, dplaces[0], d):
                return False

    return values


def search(values: Union[Dict[str, str], bool]) -> Union[Dict[str, str], bool]:
    """Depth-first search with constraint propagation to find a valid solution."""
    if values is False:
        return False
    if all(len(values[s]) == 1 for s in SQUARES):
        return values  # Solved

    # Choose unfilled square with minimum candidate values (MRV heuristic)
    _, s = min((len(values[s]), s) for s in SQUARES if len(values[s]) > 1)
    
    for d in values[s]:
        result = search(assign(values.copy(), s, d))
        if result:
            return result
    return False


def solve_sudoku(grid: Union[str, List[Union[str, int]]]) -> Optional[Dict[str, str]]:
    """
    Public solver function.
    Takes 81-character string or list of 81 numbers/strings.
    Returns {square: digit} dict if solvable, None otherwise.
    """
    parsed = parse_sudoku_grid(grid)
    if parsed is False:
        return None
    res = search(parsed)
    return res if res is not False else None


def format_solution(values: Dict[str, str]) -> str:
    """Format solution dictionary into a human-readable 2D grid string."""
    width = 1 + max(len(values[s]) for s in SQUARES)
    line = "+".join(["-" * (width * 3)] * 3)
    output = []
    for r in ROWS:
        row_str = "".join(values[r + c].center(width) + ("|" if c in "36" else "") for c in COLS)
        output.append(row_str)
        if r in "CF":
            output.append(line)
    return "\n".join(output)
