# Table of Contents

1. [Bias and Variance](#bias-and-variance)
2. [Supervised and Unsupervised Learning](#supervised-and-unsupervised-learning)
3. [Linear Regression](#linear-regression)
4. [Logistic Regression](#logistic-regression)
5. [Decision Trees](#decision-trees)
6. [Random Forests](#random-forests)
7. [Naive Bayes](#naive-bayes)
8. [Clustering](#clustering)
9. [Dimensionality Reduction](#dimensionality-reduction)
10. [Association Rule Learning](#association-rule-learning)
11. [Type I and Type II Errors](#type-i-and-type-ii-errors)
12. [Discriminative and Generative Models](#discriminative-and-generative-models)
13. [Parametric Models](#parametric-models)
14. [Data Structures](#data-structures)


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

## What is the bias and variance tradeoff?

![Tradeoff](bias_variance.png)

The above tradeoff in complexity is why there is a tradeoff between bias and variance. This means that an algorithm can't be more complex and less complex at the same time since increasing the bias decreases the variance, and increasing the variance decreases the bias. As an example, in k-nearest neighbors, a small k results in predictions with high variance and low bias, whilst a large k results in predictions with a small variance and large bias.

Simple models are stable but highly biased. Complex models are prone to overfitting but express the truth of the model. The optimal reduction of error requires a tradeoff of bias and variance to avoid both high variance and high bias.

## Would it be better if an ML algorithm exhibits a greater amount of bias or a greater amount of variance?

Either one does not have precedence over the other since they both lead to a model that gives inaccurate results, which could cause poor decision-making by the machine or humans at play.

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

# Logistic Regression

# Decision Trees

# Random Forests

# Naive Bayes

## What is Bayes' Theorem? Why do we use it?

Bayes' Theorem is how we find a probability when we know other probabilities. In other words, it provides the **posterior probability** of a prior knowledge event. This theorem is a principled way of calcultaing conditional probabilities.

In ML, Bayes' theorem is used in a probability framework that fits a model to a training dataset and for building classification predictive modeling problems.

## What are Naive Bayes classifiers? Why do we use them?

Naive Bayes classifiers are a **collection of classification algorithms**. These classifiers are a family of algorithms that share a common principle. NB classifiers assume that the occurrence of absence of a feature does not influence the presence or absence of another feature ("naive" assumption).

When the assumption of independence holds, they are easy to implement and yield better results than other sophisticated predictors. They are used in spam filtering, text analysis, and recommendation systems.

# Clustering

# Dimensionality Reduction

# Association Rule Learning

# Type I and Type II Errors

## Explain the difference between a Type I and Type II error.

A Type I error is a **false positive** and a Type II error is a **false negative**.

# Discriminative and Generative Models

A discriminative model learns **distinctions between different categories** of data. A generative model learns **categories of data**. Discriminative models generally perform better on classification tasks.

# Parametric Models

## What are parameteric models? Provide an example.

Parametric models have a **finite number of parameters**. You only need ot know the parameters of the model to make a data prediction. Common examples are as follows: linear SVMs, linear regression, and logistic regression.

Non-parameteric models have **unbounded number of parameters** to offer flexibility. For data predictions, you need the parameters of the model and the state of the observed data. Common examples are as follows: k-nearest neighbors, decision trees, and topic models.

# Data Structures
