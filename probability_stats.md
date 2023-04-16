# Table of Contents

1. [Probability](#Probability)
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
