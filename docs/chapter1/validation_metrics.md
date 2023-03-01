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

TODO add illustrations for all methods

*holdout method* - we just divide our data in three parts: train, test and validation. Train will
be used for model training, test for performance estimation and validation as a finall check on unseen data;

*k-fold* - we divide data into two sets: train and test. Then, we train the model using only train. K-fold
allows us to divide train set into *k* subsets. Then we iterate over each subset and leave it as hold out set
to estimate model performance and *k-1* is used for training;

*stratified k-fold* - it is similar to classic *k-fold* with a modification that overcomes imbalanced target.
It samples data such that each fold have approximately the same number of distinct target values;

*leave-p-out* - we use *p* observations for test and *(n - p)* as train set. Once the training is done on
*(n - p)*, *p* data points are used for validation.  All possible combinations of *p* are tested on the
model so as to get the highest model performance

All aforementioned approaches can be found in [scikit-learn library](https://scikit-learn.org/stable/modules/cross_validation.html)

*time-based* - we need it more in time series problems. Intuitively, we define training window and 
test window. Then, we go through our data with slidiging window - compute loss at each step - average in the end.
Example of implementation is [here](https://towardsdatascience.com/time-based-cross-validation-d259b13d42b8) and
another implementation with cumulative approach in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

