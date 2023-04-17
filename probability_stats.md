# Table of Contents

**Helpful cheatsheet**: https://static1.squarespace.com/static/54bf3241e4b0f0d81bf7ff36/t/55e9494fe4b011aed10e48e5/1441352015658/probability_cheatsheet.pdf

1. [Probability](#probability)
    1. [Probability of Flipping an Unfair Coin](#probability-of-flipping-an-unfair-coin)
    2. [Probability of HH Before TH](#probability-of-hh-before-th)
    3. [Seven Game Probability (Negative Binomial)](#seven-game-probability)
    4. [Accuracy of Spam Classification (Bayes Rule)](#accuracy-of-spam-classification-bayes-rule)
    5. [Probability of Intersecting Chords](#probability-of-intersecting-chords)
    6. [Disease Bayes Rule](#disease-bayes-rule)
    7. [Combinations of Cards](#combinations-of-cards)
    8. [Expected Dice Rolls](#expected-dice-rolls)
        1. [Coupon Collector Problem](#coupon-collector-problem)
    9. [Microsoft Seattle Rain Problem](#microsoft-seattle-rain-problem)
    10. [3 Dice in Strictly Increasing Order](#3-dice-in-strictly-increasing-order)
    11. [Expected Coin Flips to HH](#expected-coin-flips-to-hh)
    12. [Expected Number of Cards Before First Ace](#expected-number-of-cards-before-first-ace)
    13. [Make a Fair Coin from a Biased Coin](#make-a-fair-coin-from-a-biased-coin)
    14. [Sample from a Normal Distribution](#sample-from-a-normal-distribution)
    15. [Largest Dice Roll = r](#largest-dice-roll--r)

2. [Statistics](#statistics)
    1. [Confidence Interval Explanation](#confidence-interval-explanation)
    2. [Multicollinearity](#multicollinearity)
    3. [p-Value Explanation](#p-value-explanation)
    4. [Movie Ranking Comparison](#movie-ranking-comparison)
    5. [Statistical Power](#statistical-power)
    6. [A/B Testing](#ab-testing)
    7. [Confidence Interval from Coin Tosses](#confidence-interval-from-coin-tosses)
    8. [Uniform Distribution Mean and Variance](#uniform-distribution-mean-and-variance)
    9. [Expected Minimum of Two Uniform Distribuitions](#expected-minimum-of-two-uniform-distribuitions)
    10. [Sampling from a Uniform Distribution](#sampling-from-a-uniform-distribution)
    11. [Expected Days Drawing from a Normal Distribution](#expected-days-drawing-from-a-normal-distribution)
    12. [Biased Coin if 560/1000 Heads](#biased-coin-if-5601000-heads)
    13. [Difference Between MLE and MAP](#difference-between-mle-and-map)
    14. [Combined Mean and SD of Subsets](#combined-mean-and-sd-of-subsets)
    15. [Uniform Sampling from a Circle](#uniform-sampling-from-a-circle)
    16. [Normal Sample from Bernoulli Trials](#normal-sample-from-bernoulli-trials)

# Probability

## Probability of Flipping an Unfair Coin
**There is a fair coin and an unfair coin (both sides tails). You pick one at random, flip it five times. What is the chance that you are flipping the unfair coin?**

Use Bayes Theorem. Let U denote the case where we are flipping the unfair coin and F denote the case we are flipping the fair coin. Since the coin is chosen randomly, we know that P(U) = P(F) = 0.5. Let 5T denote the event where we flip 5 headas in a row. Then, we are interested in solving for P(U|5T).

We know that P(5T|U) = 1 since, by definition, the unfair coin will always result in tails. Additionally, we know that P(5T|F) = (1/2)^5 = 1/32 by definition of a fair coin. By Bayes Theorem we have,

$$
P(U|5T)=\frac{P(5T|U)\cdot P(U)}{P(5T|U)\cdot P(U) + P(5T|F) \cdot P(F)}=\frac{0.5}{0.5 + 0.5 \cdot 1/32}=0.97
$$

## Probability of HH Before TH
**You and your friend are playing a game. The two of you will continue to toss again until the sequence HH or TH shows up. If HH shows up first, you win. If TH shows up first, your friend wins. What is the probability of you winning?**

The first flip is either heads or tails. If the second flip is heads we have a winner no matter what. If the second flip is tails, we have no winner, but your friend will win eventually.

Flip three is either heads or tails. If it is heads, player 2 wins. Tails, no one wins. Flip four and each afterward either results in a heads and player wins or a tails and no one wins. given the last flip was tails, HH will never occur before TH.

We have a 1/2 chance of the game ending on the second flip. Assuming it ends on the second flip, each player wins 1/2 of the time. If the game does not end on the second flip, Player 2 wins.

Therefore, the probability of you winning would be 1/4.

*Drawing a probability tree might help!*

## Seven Game Probability
**What is the probability that a seven-game series goes to seven games?**

The series goes to Game 7 if and only if either team obotains its 4th win in the 7th game. The events "A wins 4th time in Game 7" and "B wins 4th time in Game 7" are mutually exclusive, so

$$
P(\text{Game 7 is played}) = P(\text{A obtains its 4th win in the 7th game}) + P(\text{B obtains its 4th win in the 7th game})

Each of these terms is the probability of the case team obtaining 4th wins after 3 losses. Equivalently, it is the probability of the losing team winning (exactly) 3 games before the 4th loss. As reasoned in the question, this is the probability of 3 in a negative-binomial distribution with parameters `p=0.5`, `r=4` (where `r` is the number of failures). Thus, the total probability:

$$
2 \cdot {4 + 3 - 1 \choose 3} \cdot 0.5^4 \cdot 0.5^3 \approx 0.31
$$

(We multiply by 2 here since either of two teams can win Game 7.)

This involves the **negative binomial distribution**, which models the number of failures in a sequence of i.i.d. Bernoulli trials before a specified number of successes, `r`, occurs. The PMF would be:

$$
{k + r - 1 \choose k} \cdot (1-p)^kp^r
$$

## Accuracy of Spam Classification (Bayes Rule)

**Facebook has a content team that labels pieces of content on the platform as spam or not spam. 90% of them are diligent raters and will label 20% of the content as spam and 80% as non-spam. The remaining 10% are non-diligent raters and will label 0% of the content as spam and 100% as non-spam. Assume the pieces of content are labeled independently from one another, for every rater. Given that a rater has labeled 4 pieces of content as good, what is the probability that they are a diligent rater?**

Let D be the event that the rater is diligent, and G is the event that 4 independent pieces of content are labeled as good.

$$
P(D|G)=\frac{P(G|D)\cdot P(D)}{P(G)}
$$

P(D) = probability that a rater is diligent, so this is 0.9

P(G|D) = 4 pieces of content, so if this is a diligent rater, the probability that the 4 pieces of content will be good is 0.8^4.

For P(G):

$$
P(G)=P(G \cap D) + P(G \cap N)=P(G|D)\cdot P(D) + P(G|N) \cdot P(N)
$$

P(G|D) = 0.8^4, P(G|N) = 1, since non-dilligent raters rate everything as good, so:

$$
P(G)=0.8^4\cdot 0.9 + 1 \cdot 0.1 = 0.46864
$$

Finally:

$$
P(D|G)=\frac{P(G|D)\cdot P(D)}{P(G)}=\frac{0.8^4\cdot 0.9}{0.46864}=0.7867
$$

## Probability of Intersecting Chords
**Say you draw a circle and choose two chords at random. What is the probability that those chords will intersect?**

Any two artbirary chords can be represented by four points chosen on the circle. If you choose to represent the first chord by two of the four points, then you have:

$$
{4 \choose 2} = 6
$$

ways of choosing two points to represent chord 1, and the other two will represent chord 2. However, we are duplicating the count of each chord twice since a chord with endpoints p1 and p2 is the same as a chord with endpoints p2 and p1. Therefore, the number of valid chords is 3.

Visuaizing, out of these 3 chord combinations, only one of them will have intersection chords. Hence, the desired probability is

$$
p = \frac{1}{3}
$$

## Disease Bayes Rule

**1/1000 people have a particular disease, and there is a test that is 98% correct if you have the disease. If you don’t have the disease, there is a 1% error rate. If someone tests positive, what are the odds they have the disease?**

Let $A$ be the event a person has the disease, and $B$ be the event of getting a positive test. We want to find $P(A|B)$.

$$
P(A|B) =  \frac{P(B|A)P(A)}{P(B|A)P(A)+P(B|\overline{A})P(\overline{A})}
$$

$P(B|A) = 0.98, P(A)=0.001,P(B|\overline{A})=0.1,P(\overline{A})=0.999$. Therefore,

$$
P(A|B) = \frac{0.98 \cdot 0.001}{0.98 \cdot 0.001 + 0.01 \cdot 0.999} \approx 0.09
$$

## Combinations of Cards
**There are 50 cards of 5 different colors. Each color has cards numbered between 1 to 10. You pick 2 cards at random. What is the probability that they are not the same color and also not the same number?**

Choose 2 colors out of 5: ${5 \choose 2} = 10$

There are 10 ways to get the first number, and 9 ways to get the second number.

The total combinations of 2 out of 50 cards: ${50 \choose 2} = 1225$

Thus, the probability is

$$
\frac{10\cdot 10\cdot 9}{1225} = 0.7347
$$

## Expected Dice Rolls
**What is the expected number of rolls needed to see all 6 sides of a fair die?**

This is a formulation of the *coupon collector problem*. The standard approach is to write the total waiting time as a sum of the waiting times to get "the next one". If we let $T$ be the total waiting time and $T_k$ be the waiting time to see the $k^\text{th}$ face not yet seen, then

$T=\sum_{k=1}^{6} T_k$, and $E(T)=\sum_{k=1}^{6}$, by linearity of expectation. Clearly, $E(T_1)=1$, and $E(T_k)=\frac{6}{7-k}$, for $k > 1$ (each of these is a geometric random variable), So, the final answer is:

$$
1+6/5+6/4+6/3+6/2+6/1=14.7
$$

### Coupon Collector Problem

Given $n$ coupons, how many coupons do you expect you need to draw with replacement before having drawn each coupon at least once?

Let time $T$ be the number of draws needed to collect all $n$ coupons, and let $t_i$ be the time to collect the $i^{\text{th}}$ coupon after $i-1$ coupons have been collected. Then, $T=t_1+...+t_n$. Think of $T$ and $t_i$ as random variables. From

$$
p-i=\frac{n-(i-1)}{n}=\frac{n-i+1}{n}
$$

we see that $t_i$ has a geometric distribution with expectation $\frac{1}{p_i}=\frac{n}{n-i+1}$. By linearity of expectations, we have

$$
E(T)=E(t_1+t_2+...+t_n)=E(t_1)+E(t_2)+...+E(t_n)=\frac{1}{p_1}+\frac{1}{p_2}+...+\frac{1}{p_n}=\frac{n}{n}+\frac{n}{n-1}+...+\frac{n}{1}
$$

## Microsoft Seattle Rain Problem

**Three friends in Seattle each told you it’s rainy, and each person has a 1/3 probability of lying. What is the probability that Seattle is rainy? Assume the probability of rain on any given day in Seattle is 0.25.**

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

## 3 Dice in Strictly Increasing Order
**Say you roll three dice, one by one. What is the probability that you obtain 3 numbers in a strictly increasing order?**

For each selection of 3 out of 6 numbers, there is exactly one way to arrange them in order, so there are ${6 \choose 3}=20$ different strictly ordered outcomes, which yields a probability of $20/6^3=5/54$

## Expected Coin Flips to HH
**What is the expected number of coin flips needed to get two consecutive heads?**

Let the expected number of coin flips be $x$. Break this down into cases:
1. If the first flip is a tails, then we have wasted a flip. The probability of event is 1/2 and the total number of flips required is $x + 1$.
2. If the first flip is a heads and the second flip is a tails, then we have wasted two flips. The probability of this event is 1/4 and the number of flips required is $x + 2$.
3. If the first flip is a heads and the second flip is also a heads, then we are done. The probability of this event is 1/4 and the number of flips required is 2.

Adding, the equation is $x=\frac{1}{2}(x+1)+\frac{1}{4}(x+2)+\frac{1}{4}\cdot 2$.

## Expected Number of Cards Before First Ace
**How many cards would you expect to draw from a standard deck before seeing the first ace?**

There are four aces, so for any card, such as a seven of clubs, the probability of that card coming before any of the aces is $1/5$. $1/5$ is the chance of picking that card out of a pile of the aces plus that card. And this will apply for all 48 non-ace cards. Thus, the average number of cards picked before any of the aces is $48 \cdot \frac{1}{5}=9.6$.

## A Having More Coins than B
**A and B are playing a game where A has n+1 coins, B has n coins, and they each flip all of their coins. What is the probability that A will have more heads than B?**

Either A will have more tails or more heads than B. Both possible cases are symmetric so both will have the same probability, so the answer is 1/2.

## Make a Fair Coin from a Biased Coin

https://www.xarg.org/2018/01/make-a-fair-coin-from-a-biased-coin/

We are going to consider the outcomes of tossing the biased coin twice. Let $p$ be the probability of the coin landing heads and $q$ be the probability of the coin landing tails, where $q = 1 - p$. Imagine, we have a coin with $p = 0.6$. If we toss the coin twice and the coin's faces are the same, the probability would be either $P(HH)=P(H)\cdot P(H)=0.36$ or $P(TT)=P(T)\cdot P(T)=0.16$, which does not reveal anything. But, if the tosses are different, the probabilities are the same, $P(HT)=0.24$ and $P(TH)=0.24$. 

That means when the two tosses are different, we can use the outcome of the first coin and throw away the second. If the two tosses are the same, we disregard them and start over until we find two different tosses. Since we do not consider all cases where the outcomes are the same, it doesn't change the result. So even if we have $P(HT)=P(TH)=0.24$, it doesn't matter. The outcome has an equivalent probability, even if we have to wait for it a little longer. Procedure:

1. Toss the coin twice.
2. If the outcome of both coins is the same (HH or TT), start over and disregard the current toss.
3. If the outcome of both coins is different (HT or TH), take the first coin as the result and forget the second (i.e., if the outcome is HT, just take H as the result).

## Sample from a Normal Distribution

**Say you have $N$ i.i.d. draws of a normal distribution with parameters μ and σ. What is the probability that $k$ of those draws are larger than some value $Y$?**

Let $X_1,...,X_N$ be i.i.d from $N(\mu, \sigma^2)$, and let $W$ be the number of draws that are larger than $y$, thus $W \sim \text{Bin}(N, p)$, where

$$
p=P(X>y)=1-\Phi\left(\frac{y-\mu}{\sigma}\right)
$$

hence, for exactly $k$, we have

$$
P(W=k)={n \choose k}p^k(1-p)^{n - k},
$$

and for "at least $k$", 

$$
P(W \geq k)=\sum_{i=k}^{N}{N \choose i}p^i(1-p)^{n-i}
$$

## Largest Dice Roll = r

**A fair die is rolled n times. What is the probability that the largest number rolled is r, for each r in 1..6?**

We will forbid any number larger than $r$. That is, we forbit $6 - r$ alues. The probability that your single roll does not show any of these values is \frac{6-r}{6}, and the probability that this happens each time during a series of $n$ rolls is obviously $(\frac{6-r}{6})^n$. However, this is only the case for $\text{largest} \leq r$, but what if $\text{largest} = r$? See https://miro.medium.com/v2/resize:fit:720/format:webp/1*vrkRGgXUs-W6XRawkHFVwg.png

# Statistics

## Confidence Interval Explanation

**How would you explain a confidence interval to a non-technical audience?**

It is the range of values where any sample value is likely to fall into a certain probability. It is calculated based on some sample from the entire population. For example, we want to figure out the average height of women in the U.S. Assume someone tells you the 95% confidence interval is (5'2, 5'7), that means if we randomly pick one woman from th crowd, there is 95% chance that the height of this woman is between 5'2 and 5'7. In other wods, if we pick 100 woman from a crowd, we are confident that the height of at least 95% of them are between 5'2 and 5'7.

## Multicollinearity

**Say you are running a multiple linear regression and believe there are several predictors that are correlated. How will the results of the regression be affected if they are indeed correlated? How would you deal with this problem?**

A concept to understand is the *variance inflation factor*. The VIF is how much larger variance of your regression coefficient is larger than it would otherwise have been if the variable had been completely uncorrelated with the other variables. The VIF is a multiplicative factor; if the variable in question is uncorrelated, the $VIF=1$. 

You could fit a model predicting a variable, $Y$, from all other variables in your model, $X$, and get a multiple $R^2$. The VIG for $Y$ would be $1(1/-R^2)$. If the VIF for $Y$ were 10, then the variance of the sampling distribution of the regression coefficient for $Y$ would be 10 times larger than it would have been if $Y$ had been completely uncorrelated with all the other variables in the model. 

How does this effect manifest itself in $p$-values of the regression coefficients when including only one or including both predictors in the model?
- Because the variance of the sampling distribution of the regression coefficient would be larger (by a factor of the VIF) if it were correlated with other variables in the model, the $p$-value would be higher than they otherwise would.

Note that multicollinearity affects the coefficients and $p$-values, but it does not influence the predictions, precision, and goodness-of-fit statistics. We don't need to reduce multicollinearity if our goal is to make predictions.

Some other effects/warning signs:
- A coefficient is not significant even though, theoretically, that variable should be highly correlated with $Y$
- When you add or delete a predictor, the regression coefficients change dramatically

Some other ways we can solve it:
- If we have two ore more factors with a high VIF, we could remove one from the model, because they supply redundant information.
- We could use partial least squares regression or PCA to cut the number of predicotrs to a smaller set of uncorrelated components.

## p-Value Explanation
**Describe $p$-values in layman's terms.**

It is the probability of obtaining given results if the null hypothesis is correct. To reject it, the $p$-value must be lower than a predetermined significant level $\alpha$. It is the calculated probability of making a Type I error (false positive).

## Movie Ranking Comparison 
**How would you build and test a metric to compare two user’s ranked lists of movie/TV show preferences?**

Use Spearman's rho (or other etsts that work with rankings) to assess dependence/congruence between 2 people's rankings. To find shows/movies to include in the measuremnt instrument, maybe do cluster analysis on a large number of viewer's viewing habits.

$$
\rho = 1 - \frac{6\sum_i d_i^2}{n(n^2-1)}
$$

A value of +1 means a perfect assocation between the rankings, and a value of -1 means a perfect negative association.

## Statistical Power
**Explain the statistical background behind power.**

The statistical power of a hypothesis is the probability that the test correctly rejects the null hypothesis, that is, the probability of a true positive result. Thus, it is only useful when the null hypothesis is rejected. The higher the statistical power for a given experiment, the lower the probability of making a Type II (false negative) error. That is, the higher the probability of detecting an effect when there is an effect.

$$
\text{Power}=1-\text{Type II Error}
$$

- **Low statistical power**: Large risk of committing Type II errors, e.g., false negative
- **High statistical power**: Small risk of committing Type II errors

## A/B Testing

**Describe A/B testing. What are some common pitfalls?**

A/B testing is essentially an experiment where two or more variants of a page are shown to users at random, and statistical analysis is used to determine which variation performs better for a given conversion goal.

The underlying statistical support for A/B tests is called hypothesis testing: we try to attribute the difference in the metric, if any, to the presence of the treatment instead of random noise.

_Power Analysis_ - Don't underestimate power analysis and end the experiment prematurely.
- Insufficient sample size
- Ending prematurely, mistake of rolling out an A/B test and immediately ending the experiment after seeing a better metric in the treatment group.
- Sample size depends on three parameters
    - Significance level, probability of false positive.
    - Statistical power, probability of identifying the effect when there is indeed an effect
    - The Minimum Affect. What is the smallest acceptable difference between treatment and control groups?

_Randomization_ - Companies can't guarantee the consistent cluster assignment and thus not a truly random process. A random assignment fails to distribute heavy users equally.
- Ensuring random assignment in the industry is easier said than done. To administer randomization, companies use hashing an a "cluster" system to randomly assign users into the treatment and control groups. Create a hashing value for each visitor and assign him/her directly to the treatment condition or through a "cluster" that is exposed to the treatment condition.
- Can't guarantee the consistent assignment of users to the same treatment condition,
    - If a user visits a website multiple times a day, he may be exposed to both the treatment and control conditions, invalidating any causal conclusions.
- Adopt the "cluster" system, and randomize at the cluster level, not the user level. 
- After conducting a power analysis, we need 1,000 samples for each variant, for example. Within the target population, heavy users roughly take up 5% but contribute to 60% of website traffic. Particularly, when the sample size is relatively small, a random assignment may result in more heavy users falling in one group over the other.
    - Set a threshold for heavy users and use a dummy variable to randomly assign them to the treatment and control groups. The heavy vs. light users ratio balance out in the experimental groups. 

3. _Source of Contamination_ - Is the _Ceteris Paribus_ assumption valid? That is, it's likely to introduce additional variations in one experiment group but not the other.
- Microsoft case study: Following the same A/B test logic, they administer the treatment condition to one group, not to the other. However, the treatment condition needs to run JavaScript to load its content, which creates a response delay that affects the outcome variable. No delay in the control group.
- If we fail to check the Ceteris Paribus assumptions, the difference between the treatment and control group is attributable to the true treatment effect and the extra response delay. `The overall difference = the true treatment effect + the delay effect` Economically speaking, companies lose out in the competition due ot the delay effect in the treatment group and not being able to identify the actual effect.
- The solution is to do the balance check before, at, and after the assignment. If the intervention introduces extra delay, we should introduce the same amount of delay time to the control group. 
- We also have to check the _spillover effect_ between experimental groups. Does our treeatment group interact with the control.
    - To address the _cross-contamination_ effect, we would look at space and time dimensions. Spatially, a standard way is to administer the administer the randomization process at a macro level (i.e., city) instead of an individual level (i.e., Uber driver) and pick two geographically remote cases.'
    - Time dimension: Administer one version to the entire population Week 1 and check if metric decreases or increases, then take the treatment away and re-record the metric in Week 2. The downside is it is difficult to give a precise estimate due to the lingering effects from past week.

4. _Post-Experiment Analysis_ - A pitfall is getting fouled by false positive while making multiple comparison.
- We can divide the alpha level by the number of comparison, and check for statistical significance at the divided level. 
- _Simpson's paradox_: For an experiment, Microsoft adopts a 99% control to 1% treatment ratio and observes a higher conversion rate in the treatment group on day 1. They bump the ratio to 50% vs 50% and observe a higher conversion in the treatment group on day 2. On day 1 and day 2, the treatment group performs better. Surprisingly, the pattern reverses at the aggregated level: the control group has a higher conversion rate (1.68%). A pitfall is that individual and aggregated patterns look different for ramp-up experiments. Solutions:
    - Paired $t$-tests on data when th proportions are stable, i.e., comparing the treatment in day 1 to the control in day 1 and the treatment in day 2 to the control in day 2.
    - Use weighted sum to adjust for different ratios of day 1 and 2
    - Throw away the data fr the ramp-up period, which is shorter relative to the experiment
- _Primacy and Novelty Effects_: Primacy effect refers to the experienced users adjustment to the new version. If a new version is very different, the useres might get confused and click the link out of cuiosity. Novelty effect is existing users wanting to try out all new functions, leading to an increase in the metrics.
    - The best way of teasing out both effects is to resort to new users who have not been exposed to the old version

Overall list of pitfalls:
- Don't understand power analysis and end the experiment prematurely.
- Companies can't guarantee the consistent cluster assignment and thus not a truly random process.
- A random assignment fails to distribute heavy users equally.
- Fail to check the ceteris paribus assumpton.
- Cross-contamination between treatment and control groups.
- Get fouled by false positives while making multiple comparisons
- Individual and aggregated patterns look different for ramp-up experiments
- Fail to check the primacy and novelty effects, biasing the treatment effect

## Confidence Interval from Coin Tosses

**How would you derive a confidence interval from a series of coin tosses?**

It's the binomial distribution. Considering $n$ tosses, with the probability of successes in each toss $p$. The mean is $np$.  The standard error is $\frac{\sqrt{np\cdot (1-p)}}{\sqrt{n}}$. 

$$
CI=\[\text{Mean}-Z_c\cdot \text{SEM}, \text{Mean}+Z_c\cdot \text{SEM}\]
$$

$Z_c$ is the critical value corresponding to the confidence level. $Z_c=1.96$ for 95% CI.

## Uniform Distribution Mean and Variance

**Derive the mean and variance of the uniform distribution $U(a, b)$.**

For $X \sim U(a,b)$, we have

$$
f_X(x)=\frac{1}{b-a}
$$

Therefore, we can calculate the mean as,

$$
E\[X\]=\int_a^bxf_X(x)dx=\int_a^b\frac{x}{b-a}dx=\frac{x^2}{2(b-a)}\Biggr|_{a}^{b}=\frac{a+b}{2}
$$

For the variance, we want

$$
Var(X)=E\[X^2\]-E\[X\]^2
$$

And we have:

$$
E\[X^2\]=\int_a^b x^2f_X(x)dx=\int_a^b\frac{x^2}{b-a}dx=\frac{x^3}{3(b-a)}\Biggr|_{a}^{b}=\frac{a^2+ab+b^2}{3}
$$

Therefore,

$$
Var(X)=\frac{a^2+ab+b^2}{3}-\left(\frac{a+b}{2}\right)^2=\frac{(b-a)^2}{12}
$$

## Expected Minimum of Two Uniform Distribuitions

**Say we have $X \sim U(0, 1)$ and $Y \sim U(0, 1)$. What is the expected value of the minimum of $X$ and $Y$?**

The question is to find the expected value of the minimum value of the joint probability of the random variable. Let $Z$ be the minimum value of $X$, $Y$, so $Z=\text{min(X,Y)}$. To find the expected value of $Z$, we need the PDF of $Z$.

$$
P(Z \leq z) = P(X \leq z, Y \leq z) = 1 - P(X > z, Y > z)
$$

Since $X \sim U(0, 1)$ and $Y \sim U(0, 1)$,

$$
P(X > z) = 1 - z, P(Y > z) = 1 - z
$$

Also, $X$ and $Y$ are i.i.d., thus

$$
P(X > z, Y > z) = P(X > z) \cdot P(Y > z) = (1 - z) \cdot (1 - z) = (1 - z)^2
$$

Then, we get $P(Z \leq z)$ as

$$
F_Z = P(Z \leq z) = 1 - (1 - z)^2
$$

$F_Z$ is the CDF of $Z$. Now, to find the PDF of $Z$, we need to differentiate the CDF of $Z$ with respect to $z$. Thus,

$$
P_Z = \frac{d(F_Z)}{dz} = 2(1 - z)
$$

This is the PDF of the random variable $Z$. Now, to find the expected value of $Z$, which is also distributed as standard uniform, we integrate the PDF.

$$
E\[Z\] = \int_0^1 zP_Zdz = 2\int_0^1 z(1 - z)dz = 2\left(\frac{1}{2}-\frac{1}{3}\right)=\frac{1}{3}
$$

Therefore, the expected value of the minimum of $X$ and $Y$ is \frac{1}{3}.

_Main ideas_: 
- The expected value from the PDF is:

$$
E\[X\] = \int xf(x)dx
$$

- The PDF is simply the derivative of the CDF. 

## Sampling from a Uniform Distribution

**You sample from a uniform distribution $\[0, d\]$ $n$ times. What is your best estimate of $d$?**

Suppose we have i.i.d. draws $X_1,X_2,...,X_n$ from $U\[0, d\]$. We could use maximum likelihood estimation, since we are estimating a parameter. If we use MLE, we would get $\text{max}(X_1,X_2,...,X_n)$

The likelihood function takes valuse 0 at the values of $d$ when there is an observation in the dataset that is impossible to come from $U\[0,d\]$, and that will be the case when some observation is higher than $d$, or equivalently $\text{max}_i x_i>d$. On the other hand, for values of $d > \text{max}_i x_i$, the likelihood function is falling in $d$, so the likelihood function is maximized at

$$
\hat{d}=\text{max}(x_1,x_2,...,x_n)
$$

This estimation is consistent, but not unbiased.

## Expected Days Drawing from a Normal Distribution

**You are drawing from a normally distributed random variable $X \sim N(0, 1)$ once a day. What is the approximate expected number of days until you get a value of more than 2?**

The probability of interest for a single day is

$$
\theta = P(X_i>2)=1-\Phi (2) \approx 0.0228
$$

Let $Y=\text{min}(n \in \mathbb{N} | X_n > 2)$ be the first day where we draw a value greater than two, this random variable has a geometric distribution $Y \sim \text{Geom}(\theta)$. The expected number of days until we draw a value greater than two is:

$$
E\[Y\]=\frac{1}{\theta}\approx 43.956
$$

## Biased Coin if 560/1000 Heads

**A coin was flipped 1000 times, and 560 times it showed up heads. Do you think the coin is biased? Why or why not?**

With a large number of independent Bernoulli trials, the sample proportion has an approximate normal distribution by the CLT. With $p=0.56$ and $SE(p)=\sqrt{\frac{p(1-p)}{1000}}\approx 0.015$, the sample test statistics for the proportion test of the hypothesis of the hypothesis $p=0.5$ corresponding to the fair coin is $Z \approx (0.56 - 0.50) / 0.015 \approx 4$. Using the normal approximation to the sampling distribution of the test statistic under the null hypothesis, the probability of observing 550 or more is less than 0.001, which is very strong evidence the coin is biased.

Using the $Z$-test here. Determines whether two population means are different when the variances are known and the sample size is large.

$$
Z=\frac{\overline{X}-\mu_0}{s}
$$

## Difference Between MLE and MAP

**What is the difference between MLE and MAP? Describe it mathematically.**

Let's say we have a likelihood function $P(X|\theta)$. Then, the MLE for $\theta$, the parameter we want to infer, is:

$$
\theta_{\text{MLE}} = \text{argmax}_{\theta}P(X|\theta)=\text{argmax}_{\theta}\prod_i P(x_i|\theta)
$$

Of course, take the log instead, as the logarithm is monotonically increasing, so miaximizing a functon is equal to maximizing the log of that function.

$$
\begin{align}
\theta_{\text{MLE}}=\text{argmax}_{\theta}\text{log}P(X|\theta) \\
=\text{argmax}_{\theta}\text{log}\prod_i P(x_i|\theta) \\
=\text{argmax}_{\theta}\sum_i \text{log} P(x_i|\theta)
\end{align}
$$

Now, for MAP, we are working in a Bayesian setting:

$$
P(\theta | X) = \frac{P(X | \theta)P(\theta)}{P(X)} \propto P(X|\theta)P(\theta)
$$

If we replace the likelihood in the MLE formula above with the posterior, we can get

$$
\begin{align}
\theta_{\text{MAP}} = \text{argmax}_{\theta}P(X|\theta)P(\theta) \\
= \text{argmax}_{\theta} \text{log}P(X|\theta)+\text{log}P(\theta) \\
= \text{argmax}_{\theta} \text{log}\prod_i P(x_i | \theta) + \text{log}P(\theta) \\
= \text{argmax}_{\theta} \sum_i \text{log} P(x_i | \theta) + \text{log}P(\theta)
\end{align}
$$

The only difference is the inclusion of the prior. Note that if we were to make the prior constant in the MAP equations, we would get the MLE equatio again.

## Combined Mean and SD of Subsets
**Say you have two subsets of a dataset for which you know their means and standard deviations. How do you calculate the blended mean and standard deviation of the total dataset? Can you extend it to $K$ subsets?**

If the first subset has $n_1$ and the second has $n_2$ elements, we can add them together to get $N$ elements total.

$$
\text{Sum}=n_1\cdot \text{mean}_1 + n_2\cdot \text{mean}_2
$$

The combined mean is: $\text{Sum} / N$. The sum of squares is:

$$
SS = n_1 \cdot (SD_1^2 + \text{mean}_1^2) + n_2 \cdot (SD_2^2 + \text{mean}_2^2)
$$

So, the combined standard deviation is

$$
\sqrt{\frac{SS}{N - \text{mean}^2}}
$$

Yes, this can be extended to $K$ subsets. (?)

## Uniform Sampling from a Circle

**How do you randomly sample a point uniformly from a circle with radius 1?**

The first idea that comes to mind: utilize random number generator to get random angle between 0 and $2\pi$ to set angular position of a point and utilize another instance of the random generator to get the random distance of the point from the origin of the circle.

However, following this strategy, the distribution will be more dense around hte origin. This is because as the distance from the origin increases then the area covered by the circular segments increases for fixed discretization size along the radius. 

If a circle has a radius $d$ and has area $A$, then, working out the math, a circle with $2d$ radius will have an area of $3A$. The area and density of points in a circular ring is inversely proportional, so the probability distribution of the distance from the origin should be linearly increasing between 0 and $R$ to compensate for the reduction in density. For the rest of the solution, see

https://meyavuz.wordpress.com/2018/11/15/generate-uniform-random-points-within-a-circle/

## Normal Sample from Bernoulli Trials

**Given a random Bernoulli trial generator, how do you return a value sampled from a normal distribution?**

Assume we have $n$ Bernoulli trials each with success probability of $p$: $x_1,x_3,...,x_n,x_i \sim Ber(p)$.

Assuming i.i.d. trials, we can compute the sample mean for $p$ from a large number of trials:

$$
\hat{\mu}=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

We know the expectation of this sample mean is:

$$
E \[\hat{\mu}\]=\frac{np}{n}=p
$$

Additionally, we can compute the variance of this sample mean:

$$
Var(\hat{\mu}) = \frac{np(1-p)}{n^2} = \frac{p(1-p)}{n}
$$

Assume we sample a large $n$. Due to the CLT, our sample mean will be normally distributed:

$$
\hat{\mu} \sim N \left(p, \frac{p(1-p)}{n} \right)
$$

Therefore, we can take a z-score of our sample mean as:

$$
z({\hat{\mu}})=\frac{\hat{\mu}-p}{\sqrt{\frac{p(1-p)}{n}}}
$$
