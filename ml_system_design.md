**FIRST** ASK QUESTIONS! The interviewer is supposed to ask a vague question for these.

# Table of Contents

1. [General System Design](#general-system-design)
    1. [Performance and Capacity Consideration](#performance-and-capacity-consideration)
    2. [Considerations in a Large-Scale System](#considerations-in-a-large-scale-system)
    3. [Training Data Collection Strategies](#training-data-collection-strategies)
        1. [User's interaction with pre-existing system (online)](#users-interaction-with-pre-existing-system-online)

# General System Design

## Performance and Capacity Consideration

We need to consider performance and capacity along with optimization for ML task at hand, measure complexity at training and evaluation time. There are three different types of complexities for machine learning algorithms:
- **Training complexity**: Time taken to train the model for a given task.
- **Evaluation complexity**: Time taken to evaluate the input at testing time.
- **Sample complexity**: Total number of training samples required to learn a target function. This changes if the model capacity changes (for a deep neural network, the sample complexity is larger than decision trees and linear regression).

**Training and prediction complexities**:
- **Linear/logistic regression (batch)**: Training time is $O(nfe)$, where $n$ is the number of training samples, $f$ is the number of features, $e$ is the number of epochs. Evaluation time is $O(f)$.
    - Best choice if we want to save time on training and evaluation.
    - Same complexity of a single-layer neural network algorithm.
    - xample: Service level agreement (SLA) for ad prediction system says we need to select the relevant adds from a pool of adds in 300 ms. Given this request, we need a fast algorithm, so linear regression would serve the purpose here.
- **Neural networks**: More time in training and evaluation, and needs more training data. However, can handle more complex tasks such as language understanding and image segmentation. Also can give more accurate predictions in comparison to other models. Training time is exponential as it varies per implementation. The evaluation time is $O(fn_{l_1} + n_{l_1}n_{l_2} + ...)$.
    - Viable choice if capacity isn't a problem and it fits well for a task.
- **MART (Multiple Additive Regression Trees) and tree-based algorithms**: Training time is $O(ndfn_{\text{trees}})$, and evaluation time is $O(fdn_{\text{trees}})$, where $d$ is the max depth.
    - Greater computational cost than linear models, but faster than deep neural networks.
    - Able to generalize well using a moderately-sized dataset.
    - Good choice if training data is limited to a few million examples and capacity/performance is critical.

## Considerations in a Large-Scale System

Example would be a search system getting a query that matches 100 million web pages, and we want our ML system to respond with the most relevant web pages while meeting the constraints outlined in the SLA. Assume the SLA emphasizes performance and cpaacity.
- **Performance** SLA ensures that we return the results back within a given time frame (500 ms, etc.) for 99% of queries.
- **Capacity** SLA refers to the load that our system can handle, i.e., a system that can support 1,000 QPS (queries per second).

Evaluate our document using a relatively fast model (tree-based, linear regression) and it takes 1 $\mu$ s. The model would still take 100 seconds to run for 100 million matched documents $\rightarrow$ **distributed systems**! 

We distribute the load of a single query among **multiple shards** (i.e., among 1,000 machines) $\rightarrow$ 100s/1000, 100 millisecond execution time for our fast model!

If we decide to use a neural network, which won't meet performance SLA even with 1,000 shards, we use a **layered/funnel-based** modeling approach:
- Start with a relatively fast model when we have the most number of documents.
- Increase complexity in later stage and execution time, but with reduced dataset
- Apply neural network for only top 500 documents. If it takes 1 ms per document for evaluation, we would need 500 milliseconds on a single machine. We could divide into five shards for around 100 ms.

Example for ad prediction: ad selection through logistic regression, then ad prediction through neural network.

## Training Data Collection Strategies

### User's interaction with pre-existing system (online)

- Relevance or ranking early versions is a rule-based system, off of which we can build the ML system
- ML system can utilize user's interaction with pre-existing system to generate training data for new model
- Need *positive* examples (Example would be a search system getting a query that matches 100 million web pages, and we want our ML system to respond with the most relevant web pages while meeting the constraints outlined in the SLA. Assume the SLA emphasizes performance and capacity.

### Human labelers (offline)

- Someone in a self-driving car cannot generate the training data as they are not interacting with the system in a a way that would give segmentation labels for the images captured by the car camera.
- **Crowdsourcing**:
    - Outsource a task to a large group of people, sites such as Amazon Mechanical Turk
    - For simple tasks, flagging an email as spam or real requires no special training for the crowd workers
    - Can't use if we have privacy concerns, or need specialized training

- **Specialized labelers**:
    - Trained labelers who know how to use Label box software to segment images taken by car
    - Training can be costly and time-consuming.

- **Open-source datasets**
- Also can build the product in a way that it collects data from the user. User can name interests (user profile), boards they want to save (content profile)
- Data expansion using GANs

### Train, test, & validation splits

Want to perform hyperparameter tuning on the validation set, not training set (testing performance of a model on the same data it was trained on would not give a good estimation of model's generalization abilities).

For the test data, the outcome on this prat will allow us to make the final choice for model selection. If we had trained several models, we can then further see if their performance poost is significant enough to call for an online A/B test. We can't use the validation here as it still impacted the model hyperparameters. 

How might we split the data for a movie recommender system? We usually want to forecast future recommendations, so we might want some time features (i.e., user interaction patterns might differ throughout the week). We could train the model on data from one time interval and validate/test from its succeeding time interval.

The quantity of the training data depends on the modeling technique. We can plot the model's performance against the number of training samples to see where there is no longer a gain in the model's performance.

### Training Data Filtering

- Cleaning data
    - Missing data, outliers, duplicates, dropping out irrelevant features.
    - Identify patterns that may not be useful for the tasks! Example: For a search engine dataset, get rid of bot traffic apart from real users. Bot traffic would contain impressions and no clicks on a search result, and would result in a lot of false negatives. 

- Removing bias
    - Bias example: Netflix movie recommender showing recommendations based on popularity, so popular movies always appear first, and better movies are shown later if they are newer and don't show a lot of user engagement. The user would then never interact with these newer, better movies.
    - Hence, the model will keep considering the top recommendations to be the popular movies time and time again. 
    - We need to diversify the recommendations, employing an exploration technique that explores the whole content pool

- Bootstrapping new items
    - If new items are added frequently, the new items may not garner a lot of attention, so we need to boost them to increase their visibility. In movie recommendation systems, new movies face the "cold start problem".
    - Recommend new movies based on their similarity to the user's already watched movies. 

## Online Experimentation

### Running an Online Experiment

There are many ways to gauge success. The success of an advertising platform can be measured using the user's engagement rate with the advertisement and the overall revenue generated by the system. A search ranking system might take into account correctly ranked search results as a metric. 

- Example hypothesis: Does an increase in the neural network depth or width (increase in activation units) increase latency and capacity but still have an overall positive impact on engagement and net ad revenue?
- Do we deploy the next version of the ML system to the production environment? What if the hypothesis intuition is wrong and the mistake becomes costly?
- **A/B testing** useful for seeing impact of new features or changes in system on user experience
    - Two versions of a webpage or app compared
    - Webpage or app screen is modified to create a second version of the same page
    - $H_0$: Design change will not have an effect on variation. If we fail to reject the null hypothesis, we should not launch the new feature.
    - $H_{\text{alternative}}$: Design change will have an impact on variation. Variation will go into production if we reject the null hypothesis.
    - **Task**: Determine if the number of successes in the variant is significantly higher than the control. Before analyzing the results, conduct a poewr analysis test to determine how much overall traffic should be given to the system, i.e., the minimum sample size required to see the impact of conversion.
    - Send half to the control, half to the variation

### Measuring results and computing statistical significance

If an A/B test is run with the outcome of a significance level of 95\%, ($p \leq 0.05$), there is a 5\% probability that the variation we see is by chance. 

### Long-term effects: Back testing and long-running A/B tests

In some cases, we need to be more confident about the result of an A/B experiment when it is overly optimistic.
- Say overall system performance improves by 5% instead of the expected 2%.
- We can confirm whether we are overconfident, and perform a **backtest**. System A is the previous system B, and vice versa, so we are swapping the control and variation.
- Check if we lose gains. Is the gain caused by an A/B experiment equal to the loss by B/A experiment? If so, the changes made in the system improved performance.

Sometimes, A/B tests can run for a too short period of time, resulting in a “false negative” impact.
- Example: Ad prediction system, revenue goes up by 5% when we started showing more ads to users, but no effect on user retention. Users might start leaving the platform if we show them significantly more ads over a longer period of time.
- To understand this impact, we could have a long-running A/B experiment to understand the impact. This can also be done via a backtest. 

## Embeddings

Capture semantic information in a low-dimensional vector space via encoding. This can help identify related entities in the vector space. Embeddings are usually generated with neural networks.
- Example: Word2Vec of Wiki data and using them as spam-filtering models. 
- Example: Twitter can build an embedding for their users based on their organic feed interactions.

### Text Embeddings

#### Word2Vec
- Word2Vec generates embeddings with a shallow neural network (single hidden layer)
- Self-supervised: trains a model by predicting words from other words that appear in the sentence.
- Represents words with a dense vector, uses neighboring words to predict the current word and in the process, generates word embeddings.
- Two networks generate these embeddings:
    - **CBOW**: Continuous bag of words predicts current word from surrounding words by optimizing

$$
\text{Loss}=-\text{log}(p(w_t|w_{t-n},...w_{t-1},w_{t+1},w_{t+n}))
$$

- $n$ is the window size. It uses the entire contextual information as one observation while training the network.
- **Skipgram**: Predicts surrounding words from the current word. Optimizes

$$
\text{Loss}=-\text{log}(p(w_{t-n},...w_{t-1},w_{t+1},w_{t+n}|w_t))
$$

- Example: Predict whether a user is interested in a particular document given the documents that they have previously read.
    - Represent the user by taking the mean of the Word2Vec embeddings of document titles.
    - Represent the document by the mean of its title term embeddings.
    - Use the dot product between these two vectors in our ML model, or simply pass the user and document embedding vector to a neural network.

#### Context-Based Embeddings

- Word2Vec embeddings have a fixed vector, so don't take into account the context. Don't distinguish between “I'd like to eat an apple” and “Apple makes great products”, meaning doesn't change with Word2Vec.
- Context-based embeddings look at neighboring terms at embedding generation time.
- Two popular architectures to generate word context-based embeddings:
    - **Embeddings from Language Models (ELMo)**: Bidirectional LSTM model to capture words before and after
    - **Bidirectional Encoder Representations from Transformers (BERT)**: Uses an attention mechanism to see all the words in the context and use only the ones that help with prediction

### Visual Embeddings

#### Auto-Encoders

- Encoder compresses raw image pixel data into lower dimension
- Decoder regenerates the same input image from low-dimensional data, last layer of encoder determines dimension of the embedding
- Embedding dimension should be sufficiently large for decoder to capture enough information to reconstruct
- Encoder and decoder jointly minimize difference between original and generated pixels, using backpropagation to train
- After training the model, we use only the encoder to generate image embeddings.
- Self-supervised: Uses image dataset without any label

#### Visual supervised learning tasks

- Tasks would be object detection, image classification. Set up with convolution, FC, pooling layers, softmax/final classification layers.
- Example, VGG16 architecture embedding
- Example case: Find images similar to a given image
- Example case: Image search problem where we want to find the best images for given text terms, i.e., query “cats”. Use query term embedding and image embedding.

### Learning embeddings for a particular learning task

Example: predict whether a user will watch a particular movie based on their historical interactions. Here, utilizing movies that the user watched as well as their prior search terms can be beneficial. We can embed a sparse vector of movies and terms in the network itself.

### Network/Relationship-based Embedding

Systems usually have multiple entities that interact with each other. These would be a graph/set of interaction pairs.

- Retrieval and ranking of results for a particular user are mostly about predicting how close they are.
- Have an embedding model that projects these documents in the same embedding space can help in retrieval and ranking of tasks for recommendation, search, feed, etc.
- Can generate embeddings for both of the above pairs of entities in the same space by creating a two-tower neural network model that encodes each item using their raw features.
    - Optimize the inner product loss such that positive pairs from interactions have a higher score and random pairs have a lower score.
