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



# Statistics
