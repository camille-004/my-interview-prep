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
20. [Time Series Forecasting](#time-series-forecasting)
21. [SQL](#sql)
22. [Data Structures](#data-structures)

# General ML Questions

## How do you choose a classifier based on a training set size?

For a small training set, a model with high bias and low variance is better, as it is less likely to overfit (because we are more likely to overfit a small dataset). An example is Naive Bayes.

For a large training set, a model with low bias and high variance is better, as it expresses more complex relationships. An example is a decision tree. 

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

See this section for a more detailed answer: https://github.com/camille-004/my-interview-prep/blob/d3289cc6c1245a74f952e0b166c091d9eb2c6180/probability_stats.md#multicollinearity

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

## How can you identify a high bias (low variance) model (usually linear models)?

A high bias model is due to a simple model and can simply be identified when the model contains:
- A high training error.
- A validation error or test error that is the same as the training error.

## How can you fix a high bias model?

- Add more input features.
- Add more complexity by introducing polynomial features.
- Decrease the regularization term.

## How can you identify a high variance (low bias) model (usually non-linear models)?

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

| Pros      | Cons |
| ----------- | ----------- |
| Easy to implement, interpret, and efficient to train.     | Leads to overfitting if number of observations is less than the number of features       |
| No assumptions about distributions of classes in feature space | Constructs linear boundaries |
| Extends to multiple classes (multinomial regression)   | Assumes linearity between dependent and indepedent variables        |
| Provides a measure of how appropriate a predictor (coefficient magnitude) is, but also its direction of association              | Non-linear problems can't be solved with logistic regression because it has a linear decision surface |
| Good accuracy for many simple datasets as it performs well when the dataset is linearly separable              | Requires average or no multicollinearity between independent variables |
| It can interpret model coefficients as indicators of feature importances. | Tough to obtain complex relationships. |
| Can overfit in high-dimensional datasets, but less inclined to overfit. Can be fixed with regularization. | Independent variables should be linearly related to the log odds (log(p / (1 - p)).

## How would you make a prediction using logistic regression?

We are modeling the probability that and input X belongs to the default class Y = 1:

$$
P(X)=P(Y=1|X)
$$

P(X) values are given by the logistic function.

$$
P(X)=\frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}}
$$

The beta_0 and beta_1 values are estimated during the training stage using maximum likelihood estimation or gradient descent. Once we have it, we can make predictions by simply putting numbers into the logistic regression equation and claculating a result.

## When can logistic regression be used?

Logistic regression can be used in classification problems where the output or dependent variable is categorical or binary. However, in order to implement logistic regression correctly, the dataset must also satisfy the following properties:

1. There should not be a high correlation between the explanatory variables. In other words, the predictor variables should be independent of each other.
2. There should be a linear relationship between the logit of the outcome and each predictor variable. The logit function is given as logit(p) = log(p / (1 - p)), where p is the probability of the outcome.
3. The sample size must be large. How large depends on the number of independent variables of the model.

## Why is logistic regression called regression and not classification?

Logistic regression does not actually individually classify things for you: it just gives you probabilities (or log odds ratios in the logit form).

## Compare SVM and logistic regression in handling outliers.

- For logistic regression, outliers can have an unusually large effect on the estimate of logistic regression coefficeints. It will find a linear boundary if it exists to accomodate the outliers. To solve the problem of outliers, sometimes a sigmoid function is used in logistic regression.
- For SVM, outliers can make the decision boundary deviate severly from the optimal hyperplane. One way for SVM to get around the problem is to introduce slack variables. There is a penalty involved with using slack variables, and how SVM handles outliers depends on how this penalty is imposed.

## How is a logistic regression model trained?

The logistic model is trained through the **logistic function**:

$$
P(y)=\frac{1}{1 + e^{-wx}}
$$

P(y) is the probability that the output label belogns to one class. If for some input we got P(y) > 0.5, then the predicted output is 1, and otherwise would be 0. The training is based in estimation of the w vector. For this, in each training instance, we use Stochastic Gradient Descent to clculate a prediction using some initial values of the coefficients, and then claculate new coefficient values based on the error in the previous prediction. The process is repeated for a fixed number of iterations or until the model is accurate enough or cannot be made any more accurate. 

## Provide a mathematical intuition for logistic regression.

Logistic regression can be seen as a transformation from linear regression using the logistic function, also known as the sigmoid function.

$$
S(x) = \frac{1}{1 + e^{-x}}
$$

Given the linear model:

$$
y=b_0 + b_1x
$$

If we apply the sigmoid function to the above equation

$$
S(y) = \frac{1}{1 + e^{-y}}=p
$$

Where p is the probability and it takes values between 0 and 1. If we now apply the logit function to p, it results in:

$$
\text{logit}(p) = \text{log}(\frac{p}{1 - p})=b_0 + b_1x
$$

![LogisticRegression](logistic_regression.png)


# Decision Trees

| Pros      | Cons |
| ----------- | ----------- |
| Simple to understand, interpret, and visualize.     | Overfitting: keeps generating new nodes to fit the data, including noisy, making the tree complex to interpret. It is a high variance model |
| Used for both classification and regression | Unstable: Adding new data points can lead to the regeneration of the overall tree. |
| Handles both continuous and categorical variables   | Not suitable for large datasets (should use ensemble techniques instead) |
| No feature scaling required | |
| Handles nonlinear parameters efficiently. | |
| Handles missing values and outliers automatically | |
| Less training period than ensemble techniques | |

## What is the decision tree algorithm?

A decision tree is a supervised ML algorithm that can be used for both regression and classification problem statements. It divides the complete dataset into smaller subsets while, at the same time, an associated decision tree is incrementally developed.

A machine learning model like a decision tree can be easily trained on a dataset by finding the best splits to make at each node. The decision trees' final output is a tree with decision nodes and leaf nodes. A decision tree can operate on both categorical and numerical data.

Unlike deep learning, DTs are very easy to interpret and understand, making them a popular choice for decision-making applications.

![DecisionTree](decision_tree.png)

## What is the default method for splitting in decision trees?

The default method is the **Gini index**, which is the measure of impurity of a particular node. Essentially, it calculates the probability of a specific feature that is classified incorrectly. When the elements are linked by a single class, we call this "pure". It is preferred because it is not computationally intensive and doesn't involve logarithm function.

## List down some popular algorithms used for deriving Decision Trees and their attribute selection measures.

1. **ID3 (Iterative Dichotomiser)**: Uses Information Gain as an attribute selection measure.
2. **C4.5 (Successor of ID3)**: Uses Gain Ratio as an attribute selection measure.
3. **CART (Classification algorithm and Regression Trees)**: Uses Gini Index as an attribute selection measure.

## Explain the CART Algorithm for Decision Trees.

**Classification Algorithm and Regression Trees** is a greedy algorithm that greedily searches for an optimum split at the top level, then repeats the same process at each of the subsequent levels.

Moreover, it verifies whether the split will lead to the lowest impurity, and the solution provided by the greedy algorithm is not guaranteed to be optimal. It often procues a reasonably good solution since finding the optimal Tree is an NP-Complete problem requiring exponential time complexity.

As a result, it makes problem intractable even for small training sets. This is why we must choose a "reasonably good" solution instead of an optimal one.

## List down the attribute selection measures usde by the ID3 algorithm to construct a Decision Tree.

The mostly widely used algorithm for building a DT is called ID3. ID3 uses entropy and information gain as attribute selection measures to construct a decision tree.

1. **Entropy**: A DT is built top-down from a root node and involves the partitioning of data into homogeneous subsets. To check the homogeneity of a sample, ID3 uses entropy. Therefore, entropy is zero when the sample is completely ghomogeneous, and entropy of one when the sample is equally divided between different classes.
2. **Information Gain**: Information Gain is based on the decrease in entropy after splitting a dataset based on an attribute. The meaning of constructing a DT is all about finding the attributes having the highest information gain.

![InformationGain](information_gain.png)

$$
\text{Gain}(T, X)=\text{Entropy}(T)-\text{Entropy}(T, X)
$$

## Explain the difference between the CART and ID3 algorithms.

The CART algorithm produces only binary trees: non-leaf nodes always have two childrens (questions only have yes/no answers). On the contrary, other tree algorithms can produce DTs with nodes having more than two children.

## Which should be preferred among Gini impurity and Entropy?

Most of the time, it does not make a big difference. Gini impurity is a good default while implementing in `sklearn` since it is slightly faster. However, when they work differently, Gini impurity tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced trees.

## What do you understand about Information Gain? Also, explain the mathematical formulation associated with it.

Information gain is the difference between the entropy of a data sement before and after the split, i.e., reduction in impurity due to the selection of an attribute.

Some points to keep in mind about information gain:
- High difference represents high information gain.of children 
- High difference implies the lower entropy of all data segments resulting from the split.
- Thus, the higher the difference, the higher the information gain, and the better the feature used for the split.

Mathematically: Information Gain = E(S1) - E(S2)
- E(S1) denotes the entropy of data belonging to the node before the split.
- E(S2) denotes the weighted summation of entropy of children nodes by considering the weights as the proporation of data instances falling in specific childern nodes.

## Do we require feature scaling for decision trees?

They don't require feature scaling or centering (standardization). Such models are often called **white-box models**. DTs provide simple classification rules based on if and else statements that can even be applied manually if necessary.

## What are the disadvantages of information gain?

Information gain is defined as the reduction in entropy due to the selection of a particular attribute. Information gain biasses the decision tree against considering attrbites with a large number of distinct values, which might lead to overftiting. The **information gain ratio** is used to solve this problem.

## When are decision trees most suitable?
- Tabular data
- Discret outputs
- Explanations for decisions are reuqired
- Training data may contain errors and noisy data (outliers)
- Training data may contain missing feature values

## Explain the time and space complexity of training and testing for decision trees.

Sorting the data in training takes O(nlogn) time, following which we traverse the data points to find the right threshold, which takes O(n) time. Subsequently, for d dimensions, the total complexity would be:

$$
O(n\text{log}n*d) + O(n*d)
$$

Which is asymptotically

$$
O(n\text{log}n*d)
$$

While training the DT,we identify the nodes, which are typically stored in the form of if-else statement, due to which the training space complexity is O(nodes).

The testing time complexity is O(depth) as we have to traverse from the root to a leaf node in the decision tree, so the testing space complexity is O(nodes).

## How does a decision tree handle missing attribute values?

- Fill the missing attribute value with the most common value of that attribute.
- Fill in the missing value by assigning a probability to each of the possible values of the attribute based on other samples.

## How does a decision tree handle continuous features?

Decision Trees handle continuous features by converting features into a threshold-based boolean feature. We use information gain to choose the threshold.

## What is the inductive bias of a decision tree?

The ID3 algorithm prefers shorter trees over longer trees. In decision trees, attributes with high information gain are placed close to the root and are preferred over those without. In the case of decision trees, the depth of the trees is the inductive bias. If the depth of the tree is too low, then there is too much generalization in the model.

## Compare the different attribute selection measures.
1. **Information gain**: Biased towards multivalued attributes.
2. **Gain ratio**: Prefers unbalanced splits in which one data segment is much smaller than the other segment.
3. **Gini index**: Biased to multivalued attributes, has difficulty when the number of classes is too large, and tends to favor tests that result in equal-sized partitions and purity in both partitions.

## Is the Gini Impurity of a node lower or greater than that of its parent? Comment whether it is generally lower/greater or always lower/greater.

A node's Gini impurity is generally lower than that of its parent as the CART training algorithm cost function splits each of the nodes in a way that minimizes the weighted sum of its children's Gini impurities. However, sometimes it is also possible for a node to have a higher Gini impurity than its parent.

## Why do we require pruning in decision trees?

After we create a DT, we observe that most of the time, the leaf nodes have very high homoegeneity, i.e., properly classified data. However, this also leads to overfitting. Moreover, if enough partitioning is not carried out, it would lead to underfitting. Hence, the major challenge is finding the optimal trees that result in the appropriate classification having acceptable accuracy. We first make the DT and use the error rates to prune the trees appropriately. Boosting can also be used to increase the accuracy of the model by combining the predictions of weak learners into a stronger learner.

## Are decision trees affected by outliers?

Decision trees are not sensitive to noisy data or outliers since extreme values or outliers never cause much reduction in the residual sum of squares, because they are never involved in the split. DTs are generally robust to outliers. Due to their tendency to overfit, they are prone to sampling errors. If sampled training data is someon edifferent than evaluation data, then DTs tend not tto produce great results.

## What do you understand by pruning in a DT?

Pruning is the process of removing sub-nodes of a decision tree, and it is the opposite process of splitting. Two types:

**Post-pruning**:
- Used after the construction of the decision tree.
- Used when the DT has a tremendous depth and will show overfitting.
- Also known as backward pruning.
- Used when we have an infinitely grown decision tree.

**Pre-pruning**:
- This technique is used before the construction of the decision tree.
- Pre-pruning done using hyperparameter tuning.

# Random Forests


| Pros      | Cons |
| ----------- | ----------- |
| RF is unbiased as we train multiple decision trees and each tree is trained on a subset of the same training data. | Complexity: more computational resources are required and results in a large number of decision trees combined together |
| Stable: if we introduce new data points, it is pretty hard to impact all trees | Longer training time |
| Handles both continuous and categorical variables   | Not good at generalizing caseese with completely new data (example: if we know the cost of one ice cream is 1 dollar, 2 ice creams cost 2 dollars, then how much do 10 ice creams cost? Linear regression can easily figure this out, while a RF has no way of finding the answer) |
| Performs well, even with missing values | Biased towards categorical variables having multiple levels or categories: feature selection technique is based on the reduction in impurity and biased towards preferring variables with more categories |

## What is the random forest algorithm?

Ensemble technique that averages several decision trees on different parts of the same training set, with the objective of overcoming overfitting of individual decision trees. It's used for both classification and regression problem statements that operate by constructing a lot of decision trees at the same time.

![RandomForest](random_forest.png)

## Why is the random forest algorithm popular?

There are very few assumptions attached to it so data preparation is less challenging, saving time. Like many ensembling techniques, it empirically performs very well.

## What is bagging?

Bagging (**bootstrap aggregating**) involves generating K new training datasets. Each new training dataset picks a sample of data points with replacements (known as **bootstrap samples**) from the dataset. The K models are fitted using the K bootstrap samples formed and then for predictions we combine the result of all trees by averaging the output (for regression) or voting (for classification).

## Explain the working of the random forest algorithm.

1. Pick K random records from the dataset having a total of N records.
2. Build and train a decision tree model on these K records..
3. Choose the number of trees you want in your algorithm and repeat steps 1 and 2.
4. In the case of regression, for an unseen data point, each tree in the forest predicts a value for output. The final value can be calculated by taking the mean or average of the values.

## Why do we prefer a forest rather than a single tree?

The problem of overfitting takes place when we have a flexible model. A flexible model is having highvariance because the learned parameters like the structure of the decision tree will vary with the training. On the contrary, an inflexible model is said to have high bias as it makes assumptions about the training data and an inflexbiel model may not have the capacity to fit even the training data and in both situations, the model has high variance, and high bias implies the model is not able to generalize new and unseen data points properly. So, we have to build a model carefully by keeping the bias-variance tradeoff in mind.

Decision trees have unlimited flexibility, which means it keeps growing, unless, for every single observation, there is one leaf node present. Moreover, instead of limiting depth of the tree, which results in reduced variance and an increase in bias, we can combine many decision trees that eventually convert into a forest, known as a single ensemble model.

## What is out-of-bag error?

Out-of-bag is equivalent to validation or test data. In random forests, there is no need for a separate testing dataset to validate the result. It is calculated internally: as the forest is built on training data, each tree is tested on 1/3rd of the samples that are not used in building that tree. This is known as the out-of-bag error estimate which in short is an internal error estimate of a random forest as it is being constructed.

## What does 'random' refer to in "random forest"?

Refers to two processes:
- Random observations to grow each tree.
- Random variables selected for splitting at each node.

**Random record selection**: Each tree in the forest is trained on roughly 2/3rd of the total training data, and here, the data points are drawn at random with replacement from the original training dataset.

**Random variable selection**: Some independent variables, m, are selected at random out of all the predictor variables, and the best split on this m is used to split the node.
- By default, m is taken as the square root of the total number of predictors for classification, whereas for regression, m is the total number of all predictors divided  by 3.

## Why does the random forest algorithm not require split sampling methods?

Random Forest does not require a split sampling method to assess the accuracy of a model. This is because it performs internal testing on 2/3rd of the available training data used to grow each tree, and the remaining 1/3rd used to calculate out-of-bag error.

## What are the features of bagged trees?

1. Reduces variance by averaging the ensemble's results.
2. Resulting model uses the entire feature space when considering node splits.
3. Allows the trees to grow without pruning, reducing the tree-depth sizes which results in high variance but lower bias, which can help improve prediction power.

## What are the limitations of bagging trees?

The major limitation is that it uses the entire feature space when creating splits in the trees. Suppose from all the variables within the feature space, some are indicating certain predictions, so there is a risk of having a forest with correlated trees, which acutally increases bias and reduces variance.

## What does the forest error rate depend on?

- How correlated the two trees in the forest are (increasing the correlation increases the error rate)
- How strong each individual tree in the forest is (a tree with a low error rate is considered a strong classifier)

## How does a random forest algorithm give predictions on an unseen dataset?

After training the algorithm, each tree in the forest gives a classification on leftover data (OOB), and we say the tree "votes" for that class. For example, suppose we fit 500 trees, and a case is out-of-bag in 200 of them:
- 160 trees vote class 1
- 40 trees vote class 2

In this case, the RF score is class 1 since the probability for that case would be 0.8 (160/200). Similarly, it would be an average of the target variable for the regression problem.

## How do we determine the overall OOB score for the classification problem statements in RF?

For each tree, by using the leftover (33%) data, compute the misclassification rate, which is known as the OOB error rate. Finally, we aggregate all the errors from all trees and we will determine the overall OOB error rate for the misclassification

## What is the use of the proximity matrix in the RF algorithm?

- Missing value imputation
- Detection of outliers

## How does RF define the proximity between observations?
- Initialize proximities to zero.
- For any given tree, apply all the cases to the tree.
- If case i and case j both end up in the same node, then prox(ij) between i and j increases by 1.
- Accumulate over all trees and normalize by twice the number of trees in RF.

Finally, it creates a proximity matrix, i.e., a square matrix with entry as 1 on the diagonal and values between 0 and 1 in the off-diagonal positions. Proximities are close to 1 when the observations are "alike" and conversely the closer proximity to 0, the more dissimilar the cases are.

## How do random forests select the important features?
- **Mean decrease accuracy**: If we drop that variable, how much the model accuracy decreases.
- **Mean decrease Gini**: Calculation of splits in trees based on the Gini impurity index

## What are the steps of calculating variable importance in RF?
1. For each tree grown in a random forest, find the number of votes for the correct class in out-of-bag data.
2. Perform random permutation (order) of a predictor's values in the OOB data and then check the number of votes for the correct class.
3. At this steps, we subtract the number of votes for the correct class in the variable k-permuted data from the number of votes for the correct class in the original OOB data.
4. Now, the raw importance score for variable k is the average of this number over all trees in the forest. Then, normalize the score by taking the standard deviation.

## Bagging vs. boosting vs. stacking?

**Bagging**
- Randomly sample from i.i.d. training dataset using bootstrap
- Builds one model ("weak learner") for each random sample
- Aggregates predictions from each model
- **Random forest uses this**

| Pros      | Cons |
| ----------- | ----------- |
| Reduces variance and can correct overfitting | Large memory and CPU used when training in parallel |
| Parallel training is time efficient |  |
| Robust to missing data (each DT uses a subset of records and features, only portion of models affected by missing data |  | 


**Boosting**
- Train multiple model using same algorithm sequentially, each models focuses on errors made in the previous model
- More weight on wrong predictions
- **Gradient Boosted Trees use this.** Builds many shallow decision trees

| Pros      | Cons |
| ----------- | ----------- |
| Reduces prediction bias because focuses on correcting wrong predictions | Sequential training takes longer |
| One model at a time, so less memory and CPU needed compared to bagging |  |

**Stacking**
- Train using different algorithms independently
- Predictions from individual models are used as features/predictors of a "meta-model"
- Base models can be XGB, RF, NN. Meta-learner can be ridge regression

| Pros      | Cons |
| ----------- | ----------- |
| Diverse algorithms with different assumptions, so incorporating advantages of all over those | Longer time to train models and make predictions, not a good choice for production model that needs fast prediction |
| Stacking algorithms have higher accuracy, commonly used in competitions |  |

# XGBoost


| Pros      | Cons |
| ----------- | ----------- |
| XGB consists of a number of hyperparameters that can be tuned. | Boosting method so sensitive to outliers. |
| Handles missing values | Has to create dummy variables/label encoding for categorical features, unlike LightGBM |
| Provides intuitive features such as parallelism, distributed computing | |

## How does XGBoost work?

When using *gradient boosting for regression*, where the weak learners are considered to be regression trees, eaach regression tree maps an input data point to one of its leaves that includes a continuous score.
- Minimizes a regularized objective function that merges a convex loss function, which is based on the variation between target outputs and predicted outputs
- Training proceeds iteratively, adding new trees with capability to predict residuals as well as errors of prior trees $\rightarrow$ prior trees coupled with previous trees to make the final prediction

## What are the weights of XGB leaf nodes? How do we calculate them?

"Leaf weight" can be said as the model's predicted output associated with each leaf node. 
- Example: Test data point with `age = 10` and `gender = female`.
- To get prediction for data point, tree traversed from top to bottom
  - At each intermediate node, feature is needed to compare against a threshold
  - Test `age < 15` is performed first and then proceed to the left branch, then second test `gender = male` is performed, so we proceed to right branch
  - We end up at Leaf 2, whose output/leaf weight is 0.1.

## How does XGB calculate features?

Automatically provides estimations of feature importance from trained predictive model
- Retrieves feature importance scores for each attribute, after constructing boosting tree
- Feature importance contributes a score which indicates how valuable each feature was in construction of boosted decision trees

## Differences between XGBoost and LightGBM?

- Slower than LightGBM but it achieves faster training through the histogram binning process
- LightGBM is a newer tool, so less users and less documentation

## How does XGB handle missing values??

In tree algorithms, branch directions for missing values are learned during training.
- Some boosters treat missing values as zeros.
- During training time XGB decides whether missing values should fall into right node or left node
- Decision is taken to minimze the loss
- *If no missing values during training time, new points with missing values sent to right node by default* 

## How can we improve the performance of the gradient boosting algorithm?
- Increase maximum leaf nodes of the algorithm, i.e., increase the weak learners' tree depth
- Increase the number of iterations
- Choose a lower learning rate, maybe between 0.1 and 0.4
- Implement regularization techniques

# SVM

## Explain the SVM algorithm in detail.

The basic idea behind SVM is to find a hyperplane in the feature space that separates the two classes with the maximum margin. The margin is the distance between the hyperplane and the nearest data points of each class. The intuition is that a larger margin translates to a better generalization performance of the classifier on unseen data. For binary classification:
- SVMs use a kernel function to implicitly map the input data into a higher-dimensional feature space where the classes are more separable.
- The objective function of the optimization problem is to minimize the magnitude of the weights while ensuring that the hyperplane correctly separates the two classes.
- The data points that lie on the margin or on the wrong side of the hyperplane are called **support vectors**. They are the critical points that determine the location of the hyperplane. The SVM algorithm only depends on the support vectors and not on the entire dataset.

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

## When would we want to use Naive Bayes over another model?
- Naive Bayes is computationally efficient, making it well-suited for large datasets or real-time applications.
- Naive Bayes is easy to implement and interpret, requiring minimal hyperparameter tuning. It is also less prone to overfitting, making it less likely to perform poorly on unseen data.
- Naive Bayes can be trained with a small amount of data, making it useful when there are limited labeled examples.
- Naive Bayes works well with high-dimensional data, such as text data, where the number of features can be very large.

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

## Explain cross-validation.

Cross-validation is a technique to evaluate the performance of a model on unseen data. In k-fold cross validation, the data is divided into $k$ equal-sized partitions or folds. The process is repeated $k$ times, with each fold used as the validation set.

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

# Time Series Forecasting

## Which cross-validation technique would you choose for a time series dataset?

A time series is not randomly distributed but has chronological ordering. You want to use something like **forward chaining** so you can model based on past data before looking at future data. For example:

- Fold 1: training set [1], test [2]
- Fold 2: training [1, 2], test [3]
- Fold 3: training [1, 2, 3] test [4]
- Fold 4: training [1, 2, 3, 4], test [5]
- Fold 5: training [1, 2, 3, 4, 5], test [6]

Or walk-forward validation: See https://github.com/camille-004/my-interview-prep/blob/main/resume_topics.md

## What is the moving average?

A technique used to smooth out time series data by calculating the averge value of a given number of past observations.
- **Simple moving average (SMA)**: Calculates the average of the last $n$ observations
- **Exponential moving average (EMA)**: Give more weight to the recent observations and less to older ones

## What is Auto Regression (AR)?

Models the relationships between a dependent variable and its lagged values. In an AR model, the value of the dependent variable at a point in given time is regressed on one or more of its past values, creating a linear relationship between the variable and its own lagged values.

$$
Y_t=c+\phi_1Y_{t-1}+\phi_2Y_{t-2}+...+\phi_pY_{t-p}+\epsilon_t
$$

$\phi_1$ to $\phi_p$ are the coefficients of the lagged values, $p$ is the nubmer of lags in the model, and $\epsilon_t$ is an error term. The order of the AR model is determiend by the number of lags used in the equation.

## What is the difference between ARMA and ARIMA?

**ARMA**: Assumes stationarity (mean and variance are constant over time). Two components:
- *Auto-regressive (AR)*: Models dependence of series on its own past values
- *Moving-average (MA)*: Models dependence on past error terms

**ARIMA**: Non-stationary time series data
- *"Integrated"*: Differencing component, used to remove trends and seaonality

## Can you explain RNN and LSTM, and when you use each for TSA?

RNNs can take inputs of any length and use their internal state to remember informataion about previous inputs.
- Good for short-term dependencies and simple patterns
- Easier to train and optimize, simpler than LSTMs

LSTM is a variant of RNN that is particularly well-suited for time series analysis.
- Use memory cell to store information about past inputs and selectively forget/remember information based on current input
- Good for long-term dependencies and avoiding vanishing gradient problem (when weights become very small as they are propagated back through layers)
- Often the case with seasonal variations and trends that traditional models may struggle to capture

## How the IQR (Interquartile Range) is used in Time Series Forecasting?

- **IQR**: Differnece between the 75th and 25th percentiles of the dataset, range of middle 50% of data
- Can be used to detect outliers, which can then be removed or adjusted to improve accuracy of forecast
- Use IQR to identify trends, seaonality through different time periods

## What methods will you use to measure the similarity/difference between two time-series vectors?

- **Euclidean distance**: Straight-line distance between two time series vectors in Euclidean space, squared root of sum of squared differences between corresponding values
  - Sensitive to outliers, doesn't take into account shape, but simple to implement
- **Dynamic type warping (DTW)**: Find optimal alignmnent between two time series, each value paired with a value in the other series. More flexible for comparison of series of different lengths or shifted in time
  - Flexible to different series, robust to noise and outliers, but computationally expensive or not good with series of complex shapes or multiple peaks
- **Cosine similarity**
  - Fast and efficient, robust to differences in amplitude, but does not take into account temporal ordering of data
- **Pearson correlation coefficient**: Linear correlation, divde covariance of two vectors by product of their standard deviations
  - Assumes normal distribution and linear relation, but can be useful for identifying trends and patterns
- **Fourier transform**: Convert data from time domain to frequency domain, then calculate similarity on frequency components
  - Sensitive to outliers and non-periodic or irregular patterns, but can be good for identifying seasonality and frequency components

## Can non-sequential deep learning models outperform sequential models in time series forecasting?

Advantage of non-sequential models: Can capture temporal dependencies as they can learn to extract feature from local patterns, like CNN. Transformers can model long-term dependencies and patterns over time while CNNs can learn patterns in short segments of the time series.
- Also can be more computationally efficient and require less memory, good for large datasets or real-time predictions

## How do you normalize time series data?

- **Min-max normalization**: Minimmum value is 0 and maximum value is 1
- **Z-score normalization/standardization**: Mean of 0 and standard deviation of 1
- **Log transformation**: Logarithmic transformation to data to reduce the range of values and compress any extreme values, reduce skewness

## What is cross-correlation?

One signal shifted by time lag then multiplied point-wise with the other signal. Resulting product summed over all time points to give cross-correlation value for that time lag. Process repeated for different time lags to obtain full CC function.

$$
C(t)=\text{sum}(x_n\cdot y_{n+t})), \forall n 
$$

Gives info. about time delay and similarity between two signals
- Can also be used for feature extraction or pattern recognition
Similar: CC will have a peak at time lag corresponding to delay

## What is exponential smoothing and when do we need it?

Gives more weight to recent observations and less weight to older ones
- Suitable for TS data that exhibits a trend or seasonality
- Short-term forecasting, where data contains large amount of noiose or volatility, and traditional statistical methods might not be appropriate

## How would you prepare your data before time series forecasting?

- _Data cleaning_: Remove missing, duplicate, erroneous values
- _Transformation_: First or second differencing to make it more stationay, log or Box-Cox transformation to stabilize variance
- _Normalization_: See question above
- _Feature engineering_: Lagged variables, seasonality, external predictors

## What is seasonality in time series and how can you deal with different types of seasonality in time series modelling?

Periodic fluctuations that occur at regular intervals, such as daily, weekly, yearly
- Caused by factors such as weather, holidays, economic cycles

- **Additive seasonality**: Seasonal component is independent of the level, can be modeled by subtracting the seasonal component from the data (by computing average value for each time point across all seasons and subtracting from data)
- **Multiplicative seasonality**: Seasonal component varies with level, can be modeled by dividing the data by seasonal component
- Use SARIMA

# SQL

## In SQL, how are primary and foreign keys related?

Foreign keys allows you to **match and join tables** on the primary key of the corresponding table.

## Explain Star Schema.

Star schema organizes data into a central fact table and a set of related dimension tables, resembling a star shape. The fact table in the star schema contains the keys to the dimension tables. The dimension tables contain descriptive attributes or characteristics that provide context for the measurements in the fact table.
- Advantages: No need for complex joins when querying
- Disadvantages: Many-to-many not supported, denormalized (duplicated, inconsistent) data can cause integrity issues

# Data Structures

## Explain the difference between an array and a linked list.

An array is an **ordered collection** of objects. It assumes that every element has the same size, since the entire array is stored in a contiguous block of memory. The size of an array is specified at the time of declaration and cannot be changed afterward. Search options for an array are linear search and binary search (if it's sorted).

A linked list if a **series of objects** with pointers. Different elements are stored at different memory locations, and data items can be added or removed when desired. The only search option for a linked list is linear.
