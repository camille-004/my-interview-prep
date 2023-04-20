# Leetcode Python Problems (Tagged Microsoft)

1. [Spiral Matrix](#spiral-matrix)
2. [Roman to Integer](#roman-to-integer)
3. [Longest Common Prefix](#longest-common-prefix)

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

## Roman to Integer

**Code**
```python
def romanToInt(self, s: str) -> int:
    symbols = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000
    }
    res = 0
    i = 0
    while i < len(s):
        curr_char = s[i]
        if i < len(s) - 1 and symbols[s[i + 1]] > symbols[curr_char]:
            val = symbols[s[i + 1]] - symbols[curr_char]
            i += 2
        else:
            val = symbols[curr_char]
            i += 1
        res += val

    return res
```

## Longest Common Prefix

Beats 85% in terms of speed, 77.17% in terms of memory.

```python
def longestCommonPrefix(self, strs: List[str]) -> str:
    res_str = ""
    for i in range(len(min(strs, key=lambda x: len(x)))):
        curr_chars = "".join(s[i] for s in strs)
        if len(set(curr_chars)) == 1:
            res_str += curr_chars[0]
        else:
            break

    return res_str
``` 

## Logger Rate Limiter

https://leetcode.com/problems/logger-rate-limiter/

```python
class Logger:

    def __init__(self):
        self.logged_messages = {}

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        if message not in self.logged_messages.keys():
            self.logged_messages[message] = timestamp
            return True
        else:
            if timestamp < self.logged_messages[message] + 10:
                return False
            else:
                self.logged_messages[message] = timestamp
                return True
```
