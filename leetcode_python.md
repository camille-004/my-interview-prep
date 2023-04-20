# Leetcode Python Problems (Tagged Microsoft, or Not)

1. [Spiral Matrix](#spiral-matrix)
2. [Roman to Integer](#roman-to-integer)
3. [Longest Common Prefix](#longest-common-prefix)
4. [Logger Rate Limiter](#logger-rate-limiter)
5. [LRU Cache](#lru-cache)
6. [Merge Binary Trees](#merge-binary-trees)

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

## LRU Cache

This is not an optimal solution.

```python
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.keys = []

    def get(self, key: int) -> int:
        if key in self.cache:
            # Have to move back to the end once accessed
            self.keys.remove(key)
            self.keys.append(key)
            return self.cache[key]
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.keys.remove(key)
        elif len(self.keys) == self.capacity:
            del self.cache[self.keys.pop(0)]
        self.cache[key] = value
        # Have to move key back to the end since we're "accessing" it again
        self.keys.append(key)
```

## Merge Binary Trees

```python
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        # Base case
        if not root1:
            return root2
        if not root2:
            return root1

        merged = TreeNode(root1.val + root2.val)
        merged.left = self.mergeTrees(root1.left, root2.left)
        merged.right = self.mergeTrees(root1.right, root2.right)
        return merge
```

First, check if either `root1` or `root2` is `None`. If one of them is `None`, simply return the other tree. If both trees are not `None`, create a new `TreeNode` with the sum of their values, and recursively merge their left and right subtrees by calling `mergeTrees` with the corresponding nodes.

### Corresponding Node in Cloned Binary Tree

Return a reference to the node in a tree that is the clone of an input `original` tree.

```python

class Solution:
    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
        if not original:
            return None
        if original == target:
            return cloned

        left_copy = self.getTargetCopy(original.left, cloned.left, target)
        if left_copy:
            return left_copy
        
        right_copy = self.getTargetCopy(original.right, cloned.right, target)
        if right_copy:
            return right_copy
        
        return None
```
