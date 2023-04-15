# Microsoft AI Interview Questions

https://medium.com/acing-ai/microsoft-ai-interview-questions-acing-the-ai-interview-be6972f790ea

## Merge `k` arrays (in this `k = 2`) arrays and sort them.

It is simple to concatenate the two arrays. Then, we can use the quicksort function, which has the highest (`O(nlogn)`) average case complexity. 

```python
def merge_arrays(arr1, arr2):
    # Concatenate two arrays.
    arr = arr1 + arr2
	
    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
    	    if arr[j] <= pivot:
    		    i += 1
    		    arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
    	return i + 1
    
    def quicksort(arr, low, high):
    	if low < high:
    	    pi = partition(arr, low, high)
    	    quicksort(arr, low, pi - 1)
    	    quicksort(arr, pi + 1, high)
    
    quicksort(arr, 0, len(arr) - 1)
    return arr
```

The `partition()` function pivots two arrays around a pivot element. The `quicksort()` function is defined to recursively sort the array using the `partition()` function. The `low` and `high` parameters define the indices of the subarray that needs to be sorted. We recursively sorted the subarrays to the left and right of the pivot.

`partition()` selects a pivot element, which is the element at the `high` index. The `i` variable keeps track of the index of the last element that is smaller than or equal to the pivot. The function then iterates over the subarray from `low` to `high - 1`, and for each element that is smaller than or equal to the pivot, it swaps it with the element at index `i + 1`. Finally, the pivot element is swapped with the element at index `i + 1`, placing the pivot element in its correct sorted position.

In `quicksort()`, if `low` is less than `high`, it calls the `partition()` function to partition the subarray, and then recursively calls the `quicksort()` function on the subarrays to the left and right of the pivot.

Alternatively, using bubble sort:

```python
def merge_arrays(arr1, arr2):
    arr = arr1 + arr2
    for i in range(len(arr) - 1):
        for j in range(i, len(arr)):
            if arr[i] > arr[j]:
                # Swap
                temp = arr[i]
                arr[i] = arr[j]
                arr[j] = temp
    return arr
```

Bubble sort isn't the most efficient, but it is easy to describe and implement. Starting from the first element, compare each adjaceny pair of elements and swap them if they are in the wrong order. After the firsrt pass, the largest element will be at the end of the list. Repeat the process, but exclude the last element from the comparison since it is already in the correct position. After each pass, the number of elements to compare decreases by 1, since the largest element is already in its correct position.

For `k > 2`:

```python
def merge_arrays(*args):
    arr = []
    for a in args:
        arr.extend(a)
    ...
```

## How to best select a representative sample of search queries from 5 million?

For an ML model, we want the minimum amount of information that is required to learn properly from a phenomenon. We don't want information redundancy, as that doesn't contain any business value. In order to take a small dataset, we must be sure we don't lose statistical significance with respect to the population. We want our sample to keep the probability distribution of the population under a reasonable significance level. The easiest thing to do is take a random sub-sample with a uniform distribution and check if it's significant or not.
- One simple approach considers each variable independently of the others. If each one of the single univariate histograms of the sample columns is comparable with the correspondent histogram of the populations, we can assume the sample is not biased. We would repeat this for all variables. We don't have to worry about the correlation between variables if we select our sample uniformly. 
- Comparing the sample and population: Compare categorical variable with the chi-square test, and numerical variables with the KS-test. Both statistical tests work under the null hypothesis that the sample has the same distribution as the population.
- Since a sample is made by many columns and we want all of them to be significant, we can reject the null hypothesis is the $p$-value of at least one of the tests is lower than the confidence level. We want every column to pass the significance test in order to accept the sample as valid.

## Three friends in Seattle told you it's rainy. Each has a probability of 1/3 of lying. What's the probability that Seattle is actually rainy? Assume a prior probability of 0.25 for Seattle being rainy.

For Seattle to be rainy, our friends have to be telling the truth, of which there is a probability of 2/3. So, we should simply find the probability that Seattle is rainy and our friends are telling the truth.

$$
P(\text{raining} | \text{all say yes}) = \frac{P(\text{all truth}) \cdot P(\text{rain})}{P(\text{all say yes})}
$$

There are two things that can happen when all three of our friends say yes: They could all be lying (saying yes and Seattle is not rainy) or all telling the truth (saying yes and Seattle is rainy). So, the probability of each of these will be P(all say yes).

$$
P(\text{raining} | \text{all say yes}) = \frac{P(\text{all truth}) \cdot P(\text{rain})}{P(\text{all lie and Seattle is not rainy}) + P(\text{all tell the truth and Seattle is rainy})}
$$

Plug in the values:

$$
P(\text{raining} | \text{all say yes}) = \frac{(2/3)^2 \cdot 0.25}{(1/3)^2 \cdot 0.75 + (2/3)^2 \cdot 0.25} \approx 0.57
$$

## Can you explain the fundamentals of Naive Bayes? How do you set the threshold?

Naive Bayes classifiers are a collection of classification algorithms. These classifiers are a family of algorithms that share a common principle. NB classifiers assume that the occurrence of absence of a feature does not influence the presence or absence of another feature ("naive" assumption). Bayes' Theorem is how we find a probability when we know other probabilities. In other words, it provides the posterior probability of a prior knowledge event. This theorem is a principled way of calcultaing conditional probabilities.

When the assumption of independence holds, they are easy to implement and yield better results than other sophisticated predictors. They are used in spam filtering, text analysis, and recommendation systems.

One approach for finding the optimal threshold is the ROC curve, of which the X-axis is the false positive rate and the Y-axis is the true positive rate. The ROC curve is a graphical representation of the performance of a classification mdoel at all thresholds. It has two thresholds: true positive rate and false positive rate. AUC (Area under the ROC curve) is simply, the area under the ROC curve. AUC measures the two-dimensional area underneath the ROC curve from (0, 0) to (1, 1). It is used as a performance metric for evaluating binary classification models. We could use the G-Mean as a metric, which is the geometric mean of precision and recall. It is defined as:

$$
\sqrt{\text{TPR}\cdot(1-\text{FPR})}
$$

The G-mean is an unbiased evaluation metric and the main focus of threshold moving. And we could visualize where the G-mean occurs on the ROC curve, which is calculated from the predicted probabilites.

## Can you explain what MapReduce is and how it works?

MapReduce facilitates concurrent processing by splitting petabytes of data into smaller chunks, and processing them in parallel on Hadoop commodity servers. At the crux of MapReduce are two functions: Map and Reduce. They are sequenced one after the other.
- **Map**: Takes input from the disk as <key, value> pairs, processes them, and produces another set of <key, value> pairs as output.
    - The input data is first split ito smaller blocks. Each block is then assigned to a mapper for processing.
    - If a file has 100 records to be processed, 100 mappers can run together to process one record each. The Haoop framework decides how many mappers to use, based on the size of the data to be processed and the memory block available to each mapper server.
- **Reduce**: Takes inputs as <key, value> pairs, and produces <key, value> pairs as ouput.
    - After all the mappers complete processing, the framework shuffles and sorts the results before passing them on to the reducers. A reducer cannot start while a mapper is still in progress. All the map output values that have the same key are assigned to a single reducer, which then aggregates the values for that key.

Two intermediate steps:
- **Combine**: Optional process. Reduces data on each mapper further to a simplified form before passing it downstream.
- **Partition**: Translates <key, value> pairs resulting from mappers to another set of <key, value> pairs to feed into the reducer. It decides how the data has to be presented to the reducer and assigns it to a particular reducer.
    - Default partitioner determines the hash value for the key, resulting from the mapper, and assigns a partition based on its hash value.

[MapReduce](#mapreduce.png)
