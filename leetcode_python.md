# Leetcode Python Problems (Tagged Microsoft)
## Spiral Matrix

https://leetcode.com/problems/spiral-matrix/description/

Main idea: We want to go right, down, left, then up. 
- To go right, we iterate through the top row, then increase our starting row once.
- To go down, we iterate through the rightmost column, then decrease our ending column.
- To go left, we iterate through the bottom row, then decrease our ending row.
- To go up, we backwards iterate through the leftmost column, then increase the starting column.

We have the additional `if` statements so we can avoid recounting/crossing pointers after going right.

**Code**
```python
def spiral_matrix(matrix):
    if not matrix:
        return []
    start_row, end_row, start_col, end_col = 0, len(matrix), 0, len(matrix[0])
    result = []
    while start_row < end_row or start_col < end_col:
        # Right
        if start_row < end_row:
            result.extend([matrix[start_row][i] for i in range(start_col, end_col)])
        start_row += 1
        # Down
        if start_col < end_col:
            result.extend(
                [matrix[i][end_col - 1] for i in range(start_row, end_row)]
            )
        end_col -= 1
        # Left
        if start_row - end_row:
            result.extend(
                [
                    matrix[end_row - 1][i]
                    for i in range(end_col - 1, start_col - 1, -1)
                ]
            )
        end_row -= 1
        # Up
        if start_col < end_col:
            result.extend(
                [
                    matrix[i][start_col]
                    for i in range(end_row - 1, start_row - 1, -1)
                ]
            )
        start_col += 1

    return result
```
