# LeetCode Python Questions (Probability and Statistics)

## `rand10()` from `rand7()`
https://leetcode.com/problems/implement-rand10-using-rand7/description/

The main idea is to call `rand7()` twice, so that the probabilities amount to picking a random number from 1 to 49, the first power of 7 greater than 10. We would pick the number and then apply a specific modulo operation so it falls in the range of 1 to 10. This would minimize the amount of times we call `rand7()`.

We will have a matrix set up as follows:
```
    1  2  3  4  5  6  7
   ---------------------
1 | 1  2  3  4  5  6  7
2 | 8  9  10 11 12 13 14
3 | 15 16 17 18 19 20 21
4 | 22 23 24 25 26 27 28
5 | 29 30 31 32 33 34 35
6 | 36 37 38 39 40 41 42
7 | 43 44 45 46 47 48 49
```

Consider the range 11-20. If we take one of these numbers, like 13, 13 % 10 = 3. However, 20 % 10 will yield 0. To get in the range from 1 to 10 instead of 0 to 9, we would use `((n - 1) % 10) + 1`, instead of `n % 10`. (20 - 1) % 10 + 1 = 10.

This will work for the range 21-30 and 31-40 as well. However, since we don't have 50 in the matrix, this won't work for the range 41-49. Therefore, we should ignore 41-49 and focus on rows 1-6.

To get a random column index, just call `rand7()`. To get the row index, as it only include up to row 6, since the index cannot be greater than 40, call `rand7() - 1`. To get the value at this row, use `(rand7() - 1) * 7`. Then, add the column index: `(rand7() - 1) * 7 + rand7()`. Finally, apply `((n - 1) % 10) + 1` to the result.

**Code**
```python
def rand10():
    idx = 41
    while idx > 40:
      row = (rand7() - 1) * 7
      col = rand7()
      idx = row + col  # Value at current index
    
    return ((idx - 1) % 10) + 1
```

## Statistics from a Large Sample

https://leetcode.com/problems/statistics-from-a-large-sample/description/

The only component out of the ones we need to find that isn't trivial is the median. To do this, we need to take the average of the two middle elements (if the length of the dataset is even). To this end, we define a `left_median`, whose index increases until we reach the middle value, and `right_median`, whose index increases through the reversed frequency table until we reach the middle value. Then, we take the average of these values.

**Code**
```python
def sample_stats(count):
    minimum = next((i for i, x in enumerate(count) if x != 0), None)
    maximum = len(count) - next((i for i, x in enumerate(count[::-1]) if x != 0), None) - 1
    mean = sum(i * x for i, x in enumerate(count)) / sum(count)

    num_count = 0
    left_median = 0
    for i, c in enumerate(count):
        num_count += c
        if num_count >= sum(count) / 2:
            left_median = i
            break

    num_count = 0
    right_median = 0
    for i, c in reversed(list(enumerate(count))):
        num_count += c
        if num_count >= sum(count) / 2:
            right_median = i
            break

    median = (left_median + right_median) / 2
    mode = count.index(max(count))
    return [minimum, maximum, mean, median, mode]
```
