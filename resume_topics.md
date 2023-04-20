# Time Series Forecasting as Supervised Learning

- Use the previous time steps as input variables and use the next step as the output variable
- Order must be preserved

```
X, y
?, 100
100, 110
110, 108
108, 115
115, 120
120, ?
```

- Delete first and last rows
  - For multi-step forecasting, more `NaNs`, need to delete more rows
- Known as **sliding window method**, **window method**, or **lag method**
- If order is preserved, problem is now regression or classification, can use standard nonlinear or linear ML algorithms
- Can increase width of sliding window to include more previous time steps 
- In my case, classical methods fell short when I had multiple, grouped time series
- Validate with walk-forward validation, instead of $k$-fold, which could give optimistiaclly biased result
  - Dataset first split into train and test sets by selecting a cut point
  - Example: One-step forecast, evluate model by training on training dataset and predicting the first step in the test data
    - Then add real observation from test set to training dataset, refit model, and hae model predict second step in test dataset
  - Requires assumptions:
    - **Minimum number of observations**: Minimum number of observations required to train the model, width of sliding window if a sliding window is used
    - **Sliding or expanding window**: Whether model trained on all data it has available or only most recent observations, which determines whether sliding window will be used
  - Steps:
    1. Minimum number of samples in the window used to train the model, starting at beginning of series
    2. Model makes prediction for next time step
    3. Prediction stored or evaluated against known value
    4. Window expanded to include the known value and process is repeated
  - Benefits:
    - More robust estimation of how chosen model and parameters will perform in practice
  - Computational cost: Creating many models
- Can be done with multiple variables $\rightarrow$ Multi-output regression
- Discovered we could not use this in my case, since we don't have access to previous steps of the target variable in new/unseen cases

# Reinforcement Learning Group Testing Project

**Group Testing**: A combinatorial method for searching for and identifying "infected" a small amount infected inviduals in a population. When a small proportion are affected, we want an to achieve a more effective scheme than testing all individual subjects. We want to break them into groups.

Familiar examples and applications: String of light bulbs, syphilitic soldiers, COVID-19

*Adaptive* (what I used): Results of current test depend on those of tests in previous stage(s) (typically Markov setting)
*Nonadaptive*: All tests are known beforehand, pre-determined scheme (pooling design)

- Represent population and infected subsample as sparse binary vector, $x$, of length $N$, with $K$ ones (1 = infected)
- Deduce locations of ones through repeated measurements
- Use the Walsh-Hadamard matrix, $W$, for measurements
- Sample rows from the matrix, multiply with unknown $x$ to get $y_t$, where $t$ is the number of rows sampled from $W$
- More measurements to reduce the number of solutions, until there is one solution left

**Walsh-Hadamard Matrix**
- Square matrix of dimensions $2^n$ whose rows and columns are orthogonal, rows correspond to Walsh function
- Walsh function incorporates binary representations of natural numbers, least significant and most significant bits, results in fractal structure in matrix after sequential ordering

For the rest, see https://www.canva.com/design/DAEusK2jsb8/IwON0-jpiqlAx0byHbEckw/edit?analyticsCorrelationId=b7163428-71c3-4632-aa41-778f41e6a790

