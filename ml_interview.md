# Bias and Variance
1. **What impact do bias and variance in data have on machine learning models?** 

Bias usually causes ML algorithms to underfit the data. Henceforce, the trained model has high training and testing errors.

Variance usually reuslts in ML algorithms overfitting the data. Therefore, the trained model exhibits a low error in training. However, it is hbound to have a high error in testing.

2. **Can ML models overcome underfitting on biased data and overfitting on data with variance? Does this guarantee correct results?**

Yes, they can. Underfitting can be overcome by utilizing ML models with greater emphasis on the features - increasing the number of features or placing greater weight on the features at play (using higher degree polynomials, for example). As for overfitting, the reverse can be done to eradicate it.

This does not guarantee plausible results in real life since they still may need to be based on data that have not been collected with the proper technique.\\

3. **How can you identify a high bias (low variance) model?**

A high bias model is due to a simple model and can simply be identified when the model contains:
1. A high training error.
2. A validation error or test error that is the same as the training error.

4. **How can you fix a high bias model?**

1. Add more input features.
2. Add more complexity by introducing polynomial features.
3. Decrease the regularization term.

5. **How can you identify a high variance (low bias) model?**

A high variance model is due to a complex model and can simply be identified when the model contains:
1. A low training error.
2. A validation error or test error that is high.

6. **How can you fix a high variance model?**

1. Reduce the input features.
2. Reduce the complexity by getting rid of the polynomial features.
3. Increase the regularization term.

7. **What is the bias and variance tradeoff?**
