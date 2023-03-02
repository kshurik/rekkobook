---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(chapter1_part4)=

# Validation and Metrics
## Validation Methods
Training a recommendation system is a complex process, requiring careful consideration
of a number of different aspects. One of the key steps in the recommendation system
training pipeline is the validation process too. This process is essential for ensuring
that the data used to train the model is accurate, and that the model is performing as expected.

Basically, validation is a process which assesses the performance of the model and can help
to detect any issues or bias in the data that could impact the performance of the model in
production. Without validation, it is impossible to know whether the model is correctly
capturing user preferences and providing accurate recommendations. The validation process
also helps to detect any potential problems that could arise during the training process
such as overfitting or under-fitting of the data, some erorrs in data collection etc.

In general, validation process typically involves splitting the data into train and
test sets. The train set is used to build and train the model, while the test set is
used to measure the performance of the model and detect any issues in the data.
The results of the validation process provide an indication of how well the model
is able to capture general patterns and provide accurate predictions.

Validation can be carried out using a number of different methods such as `holdout method`,
`k-fold`, `stratified k-fold`, `leave-p-out`, `time-based`. These methods are popular methods of
validation in which the data is divided into multiple subsets which are then used in multiple
rounds of training and testing.

- `holdout method` - we just divide our data in three parts: train, test and validation. Train will
be used for model training, test for performance estimation and validation as a finall check on unseen data;

- `k-fold` - we divide data into two sets: train and test. Then, we train the model using only train. K-fold
allows us to divide train set into *k* subsets. Then we iterate over each subset and leave it as hold out set
to estimate model performance and *k-1* is used for training;

- `stratified k-fold` - it is similar to classic *k-fold* with a modification that overcomes imbalanced target.
It samples data such that each fold have approximately the same number of distinct target values;

- `leave-p-out` - we use *p* observations for test and *(n - p)* as train set. Once the training is done on
*(n - p)*, *p* data points are used for validation.  All possible combinations of *p* are tested on the
model so as to get the highest model performance

All aforementioned approaches can be found in [scikit-learn library](https://scikit-learn.org/stable/modules/cross_validation.html).
Moreover, Machine Learning Simplified book provides great overview of the methods [here](https://code.themlsbook.com/chapter5/validation_methods.html)

- `time-based` - we need it more in time series problems. Intuitively, we define training window and 
test window. Then, we go through our data with slidiging window - compute loss at each step - average in the end.
Example of implementation is [here](https://towardsdatascience.com/time-based-cross-validation-d259b13d42b8) and
another implementation with cumulative approach in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

Finally, validation helps to ensure that the model is not biased in any way. This is especially
important in the case of recommendation systems, as they are often used to recommend products and
services to users. If the data used to train the model is biased, the model’s predictions may not
reflect user preferences accurately, leading to an inaccurate and potentially unfair recommendation
system. Validation helps to detect any potential bias in the data, which can then be addressed by
adjusting the model parameters or using a different data set.
So, what's the appropiate way for recommender systems? The answer is -- time-based split.
We define time interval for test set and use all data up to the test set start date.

```{code-cell} ipython3
import pandas as pd
import datetime as dt

TRAIN_MAX_DATE = dt.datetime(2023, 2, 14) # define last date to include in train set
TEST_INTERVAL_DAYS = 14 # define number of days to use for test
TEST_MAX_DATE = TRAIN_MAX_DATE + dt.timedelta(days = TEST_INTERVAL_DAYS)

# create artificial df
df = pd.DataFrame({'date_time': [], 'values': []})

train_set = df.loc[df['date_time'] <= TRAIN_MAX_DATE].reset_index(drop = True)
test_set = df.loc[(df['date_time'] > TRAIN_MAX_DATE) \
                  & (df['date_time'] <= (TEST_MAX_DATE))].reset_index(drop = True)
```

Also, we should consider `cold \ warm start` problems:
- Cold Start - we do not have any interactions in train and test sets;
- Warm Start - we did not have anything in train set, but interactions appear during test set

## Metrics
When developing machine learning models, evaluation metrics are an essential part of the process.
There are a variety of metrics that can be used, each with their own benefits and drawbacks, and
understanding them is key to creating successful models.

Firstly, evaluation metrics allow us to measure the performance of our models. Without these metrics,
we would not be able to gauge the success of our models, as there would be no objective way to 
measure how well our models were performing. Evaluation metrics also provide an objective method of
comparing different models so that we can select the best one for the task at hand.

Secondly, evaluation metrics can be used to identify the strengths and weaknesses of our models.
By using such metrics we can identify which areas of our model are performing well, as well as
which areas need improvement. This information can then be used to refine our models and improve their performance.

Finally, evaluation metrics can be used to assess the generalizability of our models. In other words,
how well the models will perform in unseen data. This is important for any machine learning model,
as the ultimate goal is to create models that can generalize well and successfully make predictions
on data that has not been seen before.

### Regression
Mean Absolute Error = $\frac{1}{N} \sum_{i=1}^{D}|x_i-y_i|$

Mean Squared Error = $\frac{1}{N} \sum_{i=1}^{D}(x_i-y_i)^2$

What is their baseline by the way? :)

### Classification (Confusion Matrix)
From classification tasks we can use standard metrics that are widely used. Below, there is
well-known confusion matrix. Based on that matrix we can calculate various matrix like Precision
and Recall. Their formulae and definition we will discuss later, but for now let's elaborate on
what each of the events mean in terms of recommendations.

|  |  Positive | Negative |
|---|---|---|
Positive | True Positive `(TP)` | False Positive `(FP)` |
Negative | False Negative `(FN)` | True Negative `(TN)` |

- `TP` - we recommended an item and user interacted;
- `FP` - we recommended an item and user did not interact;
- `FN` - we did not recommend an item, but user interacted with it;
- `TN` - we did not recommend an item and user did not interact

Now, let's define most popular metrics for recommendations based on classification metrics - Precision@K & Recall&K.
First, you need to understand what `@K` stands for. In recommendations, we return some list of items in a given order.
Thus, we want to know, how many interactions we got from that list and therefore some threshold must be set to cut the
list length. For example, we recommended 100 movies, but usually users does not scroll more thant 20 of them and we want
to estimate our metric only on subset of recommendations - top-20 positions and that would be Precision@20 & Recall@20.

- `Precision@K` - share of relevant items in a list. Formula is $\frac{TP}{TP + FP}$.
Also, $TP + FP$ is *K* - total number of items and the formula simplifies to $\frac{TP}{K}$
- `Recall@K` - share of relevant items in a list of recommendations. Formula is $\frac{TP}{TP + FN}$,
where $TP + FN$ is number of known interactions (relevant items).

### Ranking
Using regression or classification metrics we evaluate predicted values of the model, but not real relevance.
In recommendations, we need both positive interaction and relevant item to be as high as possible. This is not
possible using those metrics. Thus, ranking metrics have been incorporated for such tasks. In general,
they consider both positive interaction with higher weights for those items that are higher in order.
Most popular ones are `Mean Reciprocal Rank`, `Mean Average Precision`, `Normalized Discounted Cumulative Gain`.

- `Mean Reciprocal Rank (MRR)` is an average inverse rank. Formula is $\frac{1}{N} \sum_{i=1}^{N}\frac{1}{rank_i}$

| user_id | rekkos_list | interaction | rank | reciprocal rank |
|---|---|---|---|---|
| 1 | [batman, haryy potter, ozark] | batman | 1 | 1/1 |
| 2 | [ozark, thor, something] | something | 3 | 1/3 |
| 3 | [something, harry potter, batman] | None | 0 | 0 |

Then, according to our formula $MRR = (\frac{1}{1} + \frac{1}{3} + 0) / 3 = 0.44$. Keep in mind that only
the rank of the first relevant answer is considered, possible further relevant answers are ignored.

- `Mean Average Precision at K (MAP@K)` - average precision by users. Formula is divided into two parts:

- Average Precision at K by user (`AP@K`) = $\frac{1}{r_user} \sum_{i=1}^{K}Precision@i * rel_i$,
where $K$ - number of recommendations,  $r_user$ - number of releveant items for a user
- $MAP@K = \frac{1}{N} \sum_{i=1}^{N}AP@K(user_i)$

| user_id | movie | interaction | Precision@K |
|---|---|---|---|
| 1 | ozark | 1 | 1/1 |
| 2 | batman | 0 | 1/2 |
| 3 | harry | 0 | 1/3 |
| 4 | thor | 1 | 2/4 |
| 5 | something | 0 | 2/5 |
| 6 |  something2 | 0 | 2/6 |

AP@6 = $\frac{1}{2} * (\frac{1}{1} * 1 + \frac{1}{2} * 0 + \frac{1}{3} * 0 + \frac{2}{4} * 1 + \frac{2}{5} * 0 + \frac{2}{6} * 0)$ = 0.75
The total number of relevant items for a user is 2, therefore we multiply by 1/2.
Looking at the first rank, we see that user interacted with our recommendation and
according to our formula we get 1/1 and multiply by relevance 1. Then, in the following
one we do not have interaction, our Precision@2 is 1/2 and multiplying by relevance 0
we get 0. Further we do the same logic and come the resulting 0.75 MAP@6 (because we have only 1 user in example).

Now, what is MAP@3? :)

- `Normalized Discounted Cumulative Gain (NDCG)` - TBD
