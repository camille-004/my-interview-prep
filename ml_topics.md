# Random Variables
Quantity with an associated probability distribution: discrete (probability mass function) or continuous (probability density function)

Discrete:

$$
\sum_{x\in X}p(x)=1
$$

Continuous:

$$
\int_{-\infty}^{\infty} p(x)dx=1
$$

**Cumulative distribution function**

$$
F(x)=p(X\leq x)
$$

**Expectation (average value)**

$$
E\[X\]=\int_{-\infty}^{\infty} xp(x)dx
$$

**Variance**

$$
Var(X)=E\[(X-E\[X\])^2\]=E\[X^2\]-E\[X^2\]
$$

**Covariance**

For any given random variables X and Y, the covariance, a linear measure of relationship, is defined by:

$$
Cov(X)=E\[(X-E\[X\])(Y-E\[Y\])\]=E\[XY\]-E\[X\]E\[Y\]
$$

**Correlation**

The normalized covariance between X and Y:

$$
\rho(X,Y)=\frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}
$$

# Probability Distributions

Normal distribution, probability density for single variable:

$$
p(x|\mu, \sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\text{exp}-\left(\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

One general method for fitting the parameters (mean and variance, in this case): MLE. The goal in MLE is to estimate the most likely parameters given a likelihood function.

$$
\theta_{\text{MLE}} = \text{argmax}_{\theta}L(\theta), L(\theta)=p(x_1,...,x_n|\theta)
$$

The values of X are assumed to be i.i.d, so the likelihood function becomes:

$$
L(\theta)=\prod_{i=1}^{n}p(x_i|\theta)
$$

Take the log so othat is monotonically increasing:

$$
\text{log}L(\theta)=\sum_{i=1}^{n}\text{log}p(x_i|\theta)
$$

We would maximize this with an optimization algorithm like gradient descent. Another way is maximum a posteriori estimation (MAP), which assumes a prior distribution.

$$
\theta_{\text{MAP}} = \text{argmax}_{\theta}p(\theta)p(x_1,...,x_n|\theta)
$$

MAP comes up in a Bayesian setting, since we have priors for the parameters. On the other hand, MLE is frequentist, meaning the likelihood will speak for itself. 

## Bernoulli and Uniform

When there is a tossing of a coin, we think of Bernoulli's distribution. p would be the probability of the coin landing on heads or tails, respectively. The outcome of the experiment is boolean in nature.
