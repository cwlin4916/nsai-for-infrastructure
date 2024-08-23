import random

import numpy as np

DIRECTIONS = ["HORIZONTAL", "VERTICAL", "SLOPE_UP", "SLOPE_DOWN"]

"Turn an ASCII representation of a grid, or a list of rows of such a representation, into the NumPy Boolean array representation"
def create_grid(rows):
    if type(rows) == str: return create_grid(rows.strip().split("\n"))
    return np.array([[c == "x" for c in row if c in ("x", "-")] for row in rows])

"Print the ASCII representation of the given NumPy Boolean grid"
def display_grid(grid):
    for row in grid:
        print(" ".join(["x" if v else "-" for v in row]))

"Return the indices of the starting cells of all `segment_length`-length runs of True in `row`. O(row*segment_length)"
def _find_1d_segments(row, segment_length):
    segments = []
    for i in range(len(row)-(segment_length-1)):
        if all(row[i:i+segment_length]): segments.append(i)
    return segments

assert _find_1d_segments([False, True, True, True, True, False], 3) == [1, 2]

"Apply `_find_1d_segments` to each row of a 2D grid and return the (row, col) coordinates of starting cells"
def _find_horiz_segments(grid, segment_length):
    segments = []
    for (row_n, row) in enumerate(grid):
        row_segments = _find_1d_segments(row, segment_length)
        segments.extend([((row_n, col_n), (row_n, col_n+(segment_length-1))) for col_n in row_segments])
    return segments

"""
'Shear' a grid down (direction = True) or up (direction = False) by one more pixel every
column, expanding the grid to fit and padding with False
"""
def _vertical_shear(grid, direction):
    rows, cols = np.shape(grid)
    result = np.zeros((rows*2-1, cols), bool)
    for col_n, col in enumerate(grid.T):
        first_offset = col_n if direction else rows-1-col_n
        result[:, col_n] = np.concat([np.zeros(first_offset, bool), col, np.zeros(rows-1-first_offset, bool)])
    return result

"""
Find all line segments in the grid. Return (1) a list of pairs of (row, col) endpoints where
the first point in the pair is the westmost > northmost end of the segment; and (2) a
corresponding list of directions, where each direction is one of "HORIZONTAL", "VERTICAL",
"SLOPE_UP", "SLOPE_DOWN".
"""
def find_all_segments(grid, segment_length = 2):
    rows, _ = np.shape(grid)
    segments = []
    directions = []
    # Horizontal segments
    these_segments = _find_horiz_segments(grid, segment_length)
    segments.extend(these_segments)
    directions.extend(["HORIZONTAL"]*len(these_segments))
    # Vertical segments
    these_segments = [((b, a), (d, c)) for ((a, b), (c, d)) in _find_horiz_segments(grid.T, segment_length)]
    segments.extend(these_segments)
    directions.extend(["VERTICAL"]*len(these_segments))
    # Upward-sloping diagonals
    these_segments = [((a-b, b), (c-d, d)) for ((a, b), (c, d)) in _find_horiz_segments(_vertical_shear(grid, True), segment_length)]
    segments.extend(these_segments)
    directions.extend(["SLOPE_UP"]*len(these_segments))
    # Downward-sloping diagonals
    these_segments = [((a-(rows-1)+b, b), (c-(rows-1)+d, d)) for ((a, b), (c, d)) in _find_horiz_segments(_vertical_shear(grid, False), segment_length)]
    segments.extend(these_segments)
    directions.extend(["SLOPE_DOWN"]*len(these_segments))
    return segments, directions

"Calculate whether the given `point` is within the given `grid`"
def is_in_grid(point, grid):
    rows, cols = np.shape(grid)
    return 0 <= point[0] < rows and 0 <= point[1] < cols

"""
Take a `segment` and `direction` and output the coordinates of the points on each end of the
segment that we'd want to extend
"""
def where_to_extend(segment, direction):
    (a, b), (c, d) = segment
    match direction:
        case "HORIZONTAL": return (a, b-1), (c, d+1)
        case "VERTICAL":   return (a-1, b), (c+1, d)
        case "SLOPE_DOWN": return (a-1, b-1), (c+1, d+1)
        case "SLOPE_UP":   return (a+1, b-1), (c-1, d+1)

"Verify that a grid is solved by checking that there are no more extendable segments"
def is_solved(grid):
    segments, directions = find_all_segments(grid)
    possible_moves = [point
                            for (segment, direction) in zip(segments, directions)
                                for point in where_to_extend(segment, direction)]
    possible_moves = list(filter(lambda point: is_in_grid(point, grid), possible_moves))
    possible_moves = list(filter(lambda point: not grid[point[0], point[1]], possible_moves))
    return len(possible_moves) == 0

"Helper function to mark all `points` coordinates as `True` in `grid`. Operates in place."
def _paint_points(grid, *points):
    for point in points:
        grid[point] = True

"""
Mark point1 and point2 as `True` on `grid`, then use `where_to_extend` to extend the segment
thereby implied until it hits the edge of the grid. Operates in place.
"""
def extend_line(grid, point1, point2, direction):
    _paint_points(grid, point1, point2)
    for j in range(max(*grid.shape)):
        if is_in_grid(point1, grid): grid[point1] = True
        if is_in_grid(point2, grid): grid[point2] = True
        point1, point2 = where_to_extend((point1, point2), direction)

"""
Generate a random problem by creating `n_three` three-pixel formations, `n_two` two-pixel
formations, and `n_one` one-pixel non-segment formations. Extend the segments and verify
that the resulting problem `is_solved`, else try again until `timeout` is reached.
"""
def generate_problem(rows, cols, n_three, n_two, n_one, segment_length = 2, timeout = 10_000):
    for t in range(timeout):
        problem = np.zeros((rows, cols), bool)
        answer = np.copy(problem)
        for i in range(n_two):
            point_a = (random.randrange(1, rows-1), random.randrange(1, cols-1))
            direction = random.choice(DIRECTIONS)
            point_b = where_to_extend((point_a, point_a), direction)[0]
            _paint_points(problem, point_a, point_b)
            _paint_points(answer, point_a, point_b)
            if segment_length <= 2: extend_line(answer, point_a, point_b, direction)
            else: _paint_points(answer, point_a, point_b)
        for i in range(n_three):
            direction = random.choice(DIRECTIONS)
            anchor = (random.randrange(1, rows-1), random.randrange(1, cols-1))
            point_a, point_b = where_to_extend((anchor, anchor), direction)
            _paint_points(problem, point_a, anchor, point_b)
            _paint_points(answer, point_a, anchor, point_b)
            if segment_length <= 3: extend_line(answer, point_a, point_b, direction)
            else: _paint_points(answer, point_a, point_b)
        for i in range(n_one):
            prow, pcol = (random.randrange(rows), random.randrange(cols))
            problem[prow, pcol] = True
            answer[prow, pcol] = True
        if is_solved(answer): return problem, answer
    raise TimeoutError(f"Could not generate a problem in {timeout} iterations")
