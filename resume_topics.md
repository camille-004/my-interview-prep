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
