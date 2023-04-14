# Table of Contents

1. [General ML Questions](#general-ml-questions)
2. [Feature Selcetion](#feature-selection)
3. [Bias and Variance](#bias-and-variance)
4. [Missing Data](#missing-data)
5. [Supervised and Unsupervised Learning](#supervised-and-unsupervised-learning)
6. [Linear Regression](#linear-regression)
7. [Logistic Regression](#logistic-regression)
8. [Decision Trees](#decision-trees)
9. [Random Forests](#random-forests)
10. [XGBoost](#xgboost)
11. [SVM](#svm)
12. [Naive Bayes](#naive-bayes)
13. [Clustering](#clustering)
14. [Dimensionality Reduction](#dimensionality-reduction)
15. [Association Rule Learning](#association-rule-learning)
16. [Neural Networks](#neural-networks)
17. [Validation and Metrics](#validation-and-metrics)
18. [Discriminative and Generative Models](#discriminative-and-generative-models)
19. [Parametric Models](#parametric-models)
20. [SQL](#sql)
21. [Data Structures](#data-structures)

# General ML Questions

## How do you choose a classifier based on a training set size?

For a small training set, a model with high bias and low variance is better, as it is less likely to overfit. An example is Naive Bayes.

For a large training set, a model with low bias and high variance is better, as it expresses more complex relationships. An example is logistic regression. 

## How do you ensure you are not overfitting a model?

There are three methods we can use to prevent overfitting:
1. Use **cross-validation** techniques.
2. Keep the model **simple** (i.e., take in fewer variables) to reduce variance.
3. Use **regularization techniques** (like LASSO) that penalize model parameters likely to cause overfitting.

## How do hyperparameters and model parameters differ?

A model parameter is a variable that is **internal to the model**. The value of a parameter is estimated from training data. A hyperparameter is a variable that is **external to the model**. The value cannot be estimated from data, and they are commonly used to estimate model parameters.

## What is a Box-Cox transformation?

Transforms the target variable in a way that our data follow the normal distribution. It involves the transformation of any non-linear or power law distribution to a normal distribution. Converting to normal distributions makes it easy to analyze the data based on central tendency of data distribution and we can extract the information from the confidence interval. This in turn enhances the predictive power of the model.

## How do verify this that a model is suffering from multicollinearity and build a better model?

You should create a correlation matrix to identify and remove variables with a correlation above 75%. Keep in mind that this threshold is subjective. 

You could also calculate **VIF (variance inflation favor)** to check.

We can't remove variables, so we should use a penalized regression model or add random noise in the correlated variables, but this approach is less ideal.

## For k-means or kNN, why do we use Euclidean distance over Manhattan distance?

We don't use Manhattan Distance, because it calculates distance horizontally or vertically only. It has dimension restrictions. On the other hand, the Euclidean metric can be used in any space to calculate distance.

# Feature Selection

## How do you select important variables for a dataset?

- Remove correlated variables before selecting important variables.
- Use a RF and plot a variable importance chart.
- Use Lasso regression.
- Use linear regression to select variables based on p-values.
- Use forward selection, stepwise selection, and backward selection.

# Bias and Variance

## What impact do bias and variance in data have on machine learning models?

Bias usually causes ML algorithms to underfit the data. Henceforce, the trained model has high training and testing errors.

Variance usually reuslts in ML algorithms overfitting the data. Therefore, the trained model exhibits a low error in training. However, it is hbound to have a high error in testing.

## Can ML models overcome underfitting on biased data and overfitting on data with variance? Does this guarantee correct results?

Yes, they can. Underfitting can be overcome by utilizing ML models with greater emphasis on the features - increasing the number of features or placing greater weight on the features at play (using higher degree polynomials, for example). As for overfitting, the reverse can be done to eradicate it.

This does not guarantee plausible results in real life since they still may need to be based on data that have not been collected with the proper technique.

## How can you identify a high bias (low variance) model?

A high bias model is due to a simple model and can simply be identified when the model contains:
- A high training error.
- A validation error or test error that is the same as the training error.

## How can you fix a high bias model?

- Add more input features.
- Add more complexity by introducing polynomial features.
- Decrease the regularization term.

## How can you identify a high variance (low bias) model?

A high variance model is due to a complex model and can simply be identified when the model contains:
- A low training error.
- A validation error or test error that is high.

## How can you fix a high variance model?

- Reduce the input features.
- Reduce the complexity by getting rid of the polynomial features.
- Increase the regularization term.
- We can use a bagging algorithm, which divides a dataset into subsets using randomized sampling. We use those samples to generate a set of models with a single learning algorithm.

## What is the bias and variance tradeoff?

![Tradeoff](bias_variance.png)

The above tradeoff in complexity is why there is a tradeoff between bias and variance. This means that an algorithm can't be more complex and less complex at the same time since increasing the bias decreases the variance, and increasing the variance decreases the bias. As an example, in k-nearest neighbors, a small k results in predictions with high variance and low bias, whilst a large k results in predictions with a small variance and large bias.

Simple models are stable but highly biased. Complex models are prone to overfitting but express the truth of the model. The optimal reduction of error requires a tradeoff of bias and variance to avoid both high variance and high bias.

## Would it be better if an ML algorithm exhibits a greater amount of bias or a greater amount of variance?

Either one does not have precedence over the other since they both lead to a model that gives inaccurate results, which could cause poor decision-making by the machine or humans at play.

# Missing Data

## Which imputation is better for numerical data with outliers, mean or median? What is the reason behind them?

When an outlier is present in the dataset, median imputation is preferred the most.

## You are given a data set with missing values that spread along 1 standard deviation from the median. What percentage of data would remain unaffected?

The data is spread across the median, so we can assume we're working with a normal distribution. This means that approximately 68% of the data lie at 1 standard deviation from the mean. Therefore, around 32% of the data are unaffected. 

## Your dataset has 50 variables, but 8 variables have missing values higher than 30%. How do you address this?

1. Remove them (not ideal)
2. Assign a unique category to the missing values to see if there is a trend generating this ssue
3. Check distribution with the target variable. If a pattern is found, keep the missing values, assign them to a new category, and remove the others.


# Supervised and Unsupervised Learning

## Explain the difference between supervised and unsupervised machine learning.

**Supervised learning** requires training labeled data. In other words, supervised learning uses a grount truth, meaning we have existing knowledge of our outputs and samples. The goal here is to learn a function that approximates a relationship between inputs and outputs.

**Unsupervised learning**, on the other hand, does not use labeled outputs. The goal here is to infer the natural structure in a dataset.

## What are the most common algorithms for supervised learning and unsupervised learning?
**Supervised learning algorithms:**
- Linear regression
- Logistic regression
- Decision trees
- Random forests
- Naive Bayes
- Neural networks

**Unsupervised algorithms:**
- Clustering: k-Means
- PCA
- t-SNE
- Association rule learning

# Linear Regression

| Pros      | Cons |
| ----------- | ----------- |
| Simple model      | Often too simplistic to model real-world complexity       |
| Computatiionally efficient, fast prediction when large dataset  | Linearity assumption |
| Interpretability of output: Relative influence of one or more predictor variables to the predicted value when the predictors are independent of each other   | Assumes homoskedasticity        |
|               | Severely affected by outliers, since best-fit line tries to minimize the MSE |
|               | Inability to determine feature importance. If we run stochastic linear regression multiple times, the result may be different weights for two features that are highly correlated in the case of multicollinearity. |

## What is linear regression?

In simple terms, linear regression is adopting a linear approach to modeling the relationship between a dependent variable and one or more explanatory variables. In case you have one explanatory variable, you can call it a simple linear regression

## What are the critical assumptions of linear regression?

1. Linear relationship between the dependent and independent variables. A scatter plot can prove handy to check this fact.
2. The explanatory variables should not exhibit multi-collinearity.
3. Homoscedasticity: equal distribution of errors. Heteroscedasticity can be rectified with a log function.
outliers.
## What is the primary difference between R squared and adjusted R squared?

In linear regression, you use both values for model validation. R squared accounts for the variation of all independent variables on the dependent variable. In the case of adjusted R squared, it accounts for the significant variables (p < 0.05) alone for indicating the percentage of variation in the model.

## What is one method of improving a linear regression models?

Outlier treatment, removing outliers. Depending on the distribution, that can be replaced with the mean, median, mode, or percentile.

## How do you interpret a Q-Q plot in a linear regression model?

Plots the quantiles of two distributions with respect to each other. You should concentrate on the y = x line. In case you witness a deviation from this line, one of the distributions could be skewed when compared to the other.

# Logistic Regression

# Decision Trees

## What is the default method for splitting in decision trees?

The default method is the **Gini index**, which is the measure of impurity of a particular node. Essentially, it calculates the probability of a specific feature that is classified incorrectly. When the elements are linked by a single class, we call this "pure". It is preferred because it is not computationally intensive and doesn't involve logarithm function.
ans or kNN, why do we use Euclidean distance over Manhattan distance?
Linear regression models are usually evaluated using Adjusted R² or an F value. How would you evaluate a logistic regression model?
Explain the difference between the normal soft margin SVM and SVM with a linear kernel.
# Random Forests

# XGBoost

## Why does XGBoost perform better than SVM?

XGBoost is an **ensemble method** that uses many trees. This means it improves as it repeats itself.

SVM is a **linear separator**. So, if our data is not linearly separable, SVM requires a kernel to get the data to a state where it can be separated. This can limit us, as there is not a perfect kernel for every given dataset.

# Naive Bayes

| Pros      | Cons |
| ----------- | ----------- |
| Fast, easily predict class of test dataset      | Will assign category in test dataset a probability of 0 if it is not present in the test set. "Zero Frequency" problem, smoothing technique needed to solve.       |
| Supports multi-class predicction   | Assumption of independence        |
| Performs better thna other models with less training data if independence assumption holds   | Known to be a lousy estimator        |
| Performs well on categorical input variables to compared to numeric   | Will rarely find a set of independent features in real life        |

## What is Bayes' Theorem? Why do we use it?

Bayes' Theorem is how we find a probability when we know other probabilities. In other words, it provides the **posterior probability** of a prior knowledge event. This theorem is a principled way of calcultaing conditional probabilities.

In ML, Bayes' theorem is used in a probability framework that fits a model to a training dataset and for building classification predictive modeling problems.

## What are Naive Bayes classifiers? Why do we use them?

Naive Bayes classifiers are a **collection of classification algorithms**. These classifiers are a family of algorithms that share a common principle. NB classifiers assume that the occurrence of absence of a feature does not influence the presence or absence of another feature ("naive" assumption).

When the assumption of independence holds, they are easy to implement and yield better results than other sophisticated predictors. They are used in spam filtering, text analysis, and recommendation systems.

# Clustering

# Dimensionality Reduction

# Association Rule Learning

# Neural Networks

| Pros      | Cons |
| ----------- | ----------- |
| Store data on the entire network      | Requires complex processors       |
| Distributed memory   | Duration of a network is somewhat unknown        |
| Great accuracy even with limited information   | We rely on error value too heavily        |
| Parallel Processing   | Black-box nature        |

## What is the exploding gradient problem when using the back-propagation technique?

In a deep neural network with n hidden layers, n derivatives will be multiplied together while performing back-propagation. If the derivatives are large enough, the gradient increases exponentially as we propagate backwards in the model. This will cause accumulation of large error gradients and they eventually become large in magnitudes. This is called the problem of cexploding gradients, which makes the model unstable by making it difficult to converge to ideal weight value. We can tackle this by reducing the number of layers or by initializing the weight



# Validation and Metrics

| Name      | Description |
| ----------- | ----------- |
| Precision      | TP / (TP + FP), How many of the positive values the model predicted are predicted correctly       |
| Recall  | TP / (TP + FN), How many of the positive samples in the dataset did the model predict correctly   |


## Explain the difference between a Type I and Type II error.

A Type I error is a **false positive** and a Type II error is a **false negative**.

## Explain the ROC Curve and AUC.

The ROC curve is a graphical representation of the performance of a classification mdoel at all thresholds. It has two thresholds: true positive rate and false positive rate. AUC (Area under the ROC curve) is simply, the area under the ROC curve. AUC measures the two-dimensional area underneath the ROC curve from (0, 0) to (1, 1). It is used as a performance metric for evaluating binary classification models.

## What evaluation approaches would you use to gauge the effectiveness of an ML model?

First, split the dataset into training and test sets. You could also use a cross-validation technique to segment the dataset. Then, you would select and implement performance metrics. For example, you could use the confusion matrix, the F1 score, and accuracy.

## What is a confusion matrix? Why do we need it?

The confusion matrix displays the number of true positives, true negatives, false positives, and false negatives for a given model. Each of these values represents the number of times a model has made a correct or incorrect prediction. It allows us to see how well a model is performing by showing us how often it is making correct or incorrect predictions.

## You must evaluate a regression model based on R², adjusted R² and tolerance. What are your criteria?

R^2 and adjusted R^2 should be close together. If they are far apart, the model could be overfitting. Also, the mean of the error terms has to be zero (intercept can always be adjusted to produce 0 mean residuals), and identical and independently distributed. There should be no correlation among the error terms.

# Discriminative and Generative Models

## What is the difference between generative and discriminative models?

A discriminative model learns **distinctions between different categories** of data. A generative model learns **categories of data**. Discriminative models generally perform better on classification tasks.

# Parametric Models

## What are parameteric models? Provide an example.

Parametric models have a **finite number of parameters**. You only need ot know the parameters of the model to make a data prediction. Common examples are as follows: linear SVMs, linear regression, and logistic regression.

Non-parameteric models have **unbounded number of parameters** to offer flexibility. For data predictions, you need the parameters of the model and the state of the observed data. Common examples are as follows: k-nearest neighbors, decision trees, and topic models.

# Time Series

## Which cross-validation technique would you choose for a time series dataset?

A time series is not randomly distributed but has chronological ordering. You want to use something like **forward chaining** so you can model based on past data before looking at future data. For example:

- Fold 1: training set [1], test [2]
- Fold 2: training [1, 2], test [3]
- Fold 3: training [1, 2, 3] test [4]
- Fold 4: training [1, 2, 3, 4], test [5]
- Fold 5: training [1, 2, 3, 4, 5], test [6]

# SQL

## In SQL, how are primary and foreign keys related?

Foreign keys allows you to **match and join tables** on the primary key of the corresponding table.

# Data Structures

## Explain the difference between an array and a linked list.

An array is an **ordered collection** of objects. It assumes that every element has the same size, since the entire array is stored in a contiguous block of memory. The size of an array is specified at the time of declaration and cannot be changed afterward. Search options for an array are linear search and binary search (if it's sorted).

A linked list if a **series of objects** with pointers. Different elements are stored at different memory locations, and data items can be added or removed when desired. The only search option for a linked list is linear.