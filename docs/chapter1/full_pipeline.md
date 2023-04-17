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

(chapter1_part7)=

# Full Pipeline of the Two-level Recommender System

In this chapter, we will wrap up all steps from 1.2 to 1.5:
- Preprocess data with proper two-level validation;
- Develop candidate generation model with implicit library;
- Then, move to Catboost and get our reranker - second level model;
- Finally, evaluate our models: implicit vs implicit + reranker

First, let's recall what we discussed in [`Metrics & Validation`](https://rekkobook.com/chapter1/validation_metrics.html)
In recommender systems we have special data split to validate our model - we split data by time for candidates
and by users for reranker. Now, we move on to coding.

# 0. Configuration
```{code-cell} ipython3
# KION DATA
INTERACTIONS_PATH = 'https://drive.google.com/file/d/1MomVjEwY2tPJ845zuHeTPt1l53GX2UKd/view?usp=share_link'
ITEMS_METADATA_PATH = 'https://drive.google.com/file/d/1XGLUhHpwr0NxU7T4vYNRyaqwSK5HU3N4/view?usp=share_link'
USERS_DATA_PATH = 'https://drive.google.com/file/d/1MCTl6hlhFYer1BTwjzIBfdBZdDS_mK8e/view?usp=share_link'
```

# 1. Modules and functions
```{code-cell} ipython3
# just to make it available to download w/o SSL verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import shap
import numpy as np
import pandas as pd
import datetime as dt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from lightfm.data import Dataset
from lightfm import LightFM

from catboost import CatBoostClassifier

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', lambda x: '%.3f' % x)
```

## 1. 1. Helper functions to avoid copy paste
```{code-cell} ipython3
def read_parquet_from_gdrive(url, engine: str = 'pyarrow'):
    """
    gets csv data from a given url (taken from file -> share -> copy link)
    :url: example https://drive.google.com/file/d/1BlZfCLLs5A13tbNSJZ1GPkHLWQOnPlE4/view?usp=share_link
    """
    file_id = url.split('/')[-2]
    file_path = 'https://drive.google.com/uc?export=download&id=' + file_id
    data = pd.read_parquet(file_path, engine = engine)

    return data
```

# 2. Main
## 2.1. Load and preprocess data
`interactions` dataset shows list of movies that users watched, along with given `total_dur` in seconds and `watched_pct` proportion

```{code-cell} ipython3
# interactions data
interactions = read_parquet_from_gdrive(INTERACTIONS_PATH)
interactions.head()
```

`movies_metadata` dataset shows the list of movies existing on OKKO platform
```{code-cell} ipython3
# information about films etc
movies_metadata = read_parquet_from_gdrive(ITEMS_METADATA_PATH)
movies_metadata.head(3)
```

`users_data` contains basic info like gender, age group, income group and kids flag
```{code-cell} ipython3
users_data = read_parquet_from_gdrive(USERS_DATA_PATH)
users_data.head()
```

Now, a bit of preprocessing to avoid noisy data.
```{code-cell} ipython3
# remove redundant data points
interactions_filtered = interactions.loc[interactions['total_dur'] > 300].reset_index(drop = True)
print(interactions.shape, interactions_filtered.shape)
```

```{code-cell} ipython3
# convert to datetime
interactions_filtered['last_watch_dt'] = pd.to_datetime(interactions_filtered['last_watch_dt'])
```

### 2.1.2. Train / Test split

As we dicussed in Validation and metrics [chapter], we need time based split for candidates generation
to avoid look-ahead bias. Therefor, let's set date thresholds

```{code-cell} ipython3
# set dates params for filter
MAX_DATE = interactions_filtered['last_watch_dt'].max()
MIN_DATE = interactions_filtered['last_watch_dt'].min()
TEST_INTERVAL_DAYS = 14
TEST_MAX_DATE = MAX_DATE - dt.timedelta(days = TEST_INTERVAL_DAYS)

print(f"min date in filtered interactions: {MAX_DATE}")
print(f"max date in filtered interactions:: {MIN_DATE}")
print(f"test max date to split:: {TEST_MAX_DATE}")
```

```{code-cell} ipython3
# define global train and test
global_train = interactions_filtered.loc[interactions_filtered['last_watch_dt'] < TEST_MAX_DATE]
global_test = interactions_filtered.loc[interactions_filtered['last_watch_dt'] >= TEST_MAX_DATE]

global_train = global_train.dropna().reset_index(drop = True)
print(global_train.shape, global_test.shape)
```

Here, we define "local" train and test to use some part of the global train for ranker
```{code-cell} ipython3
local_train_thresh = global_train['last_watch_dt'].quantile(q = .7, interpolation = 'nearest')

print(local_train_thresh)
```

```{code-cell} ipython3
local_train = global_train.loc[global_train['last_watch_dt'] < local_train_thresh]
local_test = global_train.loc[global_train['last_watch_dt'] >= local_train_thresh]

print(local_train.shape, local_test.shape)
```

Final filter, we will focus on warm start -- remove cold start users
```{code-cell} ipython3
local_test = local_test.loc[local_test['user_id'].isin(local_train['user_id'].unique())]
print(local_test.shape)
```

### 2.1.2 LightFM Dataset setup
LightFM provides built-in Dataset class to work with and use in fitting the model.

```{code-cell} ipython3
# init class
dataset = Dataset()

# fit tuple of user and movie interactions
dataset.fit(local_train['user_id'].unique(), local_train['item_id'].unique())
```

Next, we will need mappers as usual, but with lightfm everything is easier and can be
extracted from initiated data class `dataset`
```{code-cell} ipython3
# now, we define lightfm mapper to use it later for checks
lightfm_mapping = dataset.mapping()
lightfm_mapping = {
    'users_mapping': lightfm_mapping[0],
    'user_features_mapping': lightfm_mapping[1],
    'items_mapping': lightfm_mapping[2],
    'item_features_mapping': lightfm_mapping[3],
}
print('user mapper length - ', len(lightfm_mapping['users_mapping']))
print('user features mapper length - ', len(lightfm_mapping['user_features_mapping']))
print('movies mapper length - ', len(lightfm_mapping['items_mapping']))
print('Users movie features mapper length - ', len(lightfm_mapping['item_features_mapping']))
```

```{code-cell} ipython3
# inverted mappers to check recommendations
lightfm_mapping['users_inv_mapping'] = {v: k for k, v in lightfm_mapping['users_mapping'].items()}
lightfm_mapping['items_inv_mapping'] = {v: k for k, v in lightfm_mapping['items_mapping'].items()}
```

```{code-cell} ipython3
# crate mapper for movie_id and title names
item_name_mapper = dict(zip(movies_metadata['item_id'], movies_metadata['title']))
```

```{code-cell} ipython3
# special iterator to use with lightfm
def df_to_tuple_iterator(df: pd.DataFrame):
    '''
    :df: pd.DataFrame, interactions dataframe
    returs iterator
    '''
    return zip(*df.values.T)

```

Finally, built dataset using `user_id` & `item_id`
```{code-cell} ipython3
# defining train set on the whole interactions dataset (as HW you will have to split into test and train for evaluation)
train_mat, train_mat_weights = dataset.build_interactions(df_to_tuple_iterator(local_train[['user_id', 'item_id']]))
```

```{code-cell} ipython3
train_mat
```

```{code-cell} ipython3
train_mat_weights
```

## 2.2. Fit the model

Set some default parameters for the model
```{code-cell} ipython3
# set params
NO_COMPONENTS = 64
LEARNING_RATE = .03
LOSS = 'warp'
MAX_SAMPLED = 5
RANDOM_STATE = 42
EPOCHS = 20
```

```{code-cell} ipython3
# init model
lfm_model = LightFM(
    no_components = NO_COMPONENTS,
    learning_rate = LEARNING_RATE,
    loss = LOSS,
    max_sampled = MAX_SAMPLED,
    random_state = RANDOM_STATE
    )
```

Run training pipeline
```{code-cell} ipython3
# execute training
for _ in tqdm(range(EPOCHS), total = EPOCHS):
    lfm_model.fit_partial(
        train_mat,
        num_threads = 4
    )
```

Let's make sense-check on the output model
```{code-cell} ipython3
top_N = 10
user_id = local_train['user_id'][100]
row_id = lightfm_mapping['users_mapping'][user_id]
print(f'Rekko for user {user_id}, row number in matrix - {row_id}')
```

```{code-cell} ipython3
# item indices
all_cols = list(lightfm_mapping['items_mapping'].values())
len(all_cols)

# predictions
pred = lfm_model.predict(
    row_id,
    all_cols,
    num_threads = 4)
pred, pred.shape

# sort and final postprocessing
top_cols = np.argpartition(pred, -np.arange(top_N))[-top_N:][::-1]
top_cols
```

```{code-cell} ipython3
# pandas dataframe for convenience
recs = pd.DataFrame({'col_id': top_cols})
recs['item_id'] = recs['col_id'].map(lightfm_mapping['items_inv_mapping'].get)
recs['title'] = recs['item_id'].map(item_name_mapper)
recs
```

In the end, we need to make predictions on all `local_test` users to use this sample to train reranker model.
As I have mentioned earlier, in reranker we split randomly by users.
```{code-cell} ipython3
# make predictions for all users in test
local_test_preds = pd.DataFrame({
    'user_id': local_test['user_id'].unique()
})
len(local_test_preds)
```

```{code-cell} ipython3
def generate_lightfm_recs_mapper(
        model: object,
        item_ids: list,
        known_items: dict,
        user_features: list,
        item_features: list,
        N: int,
        user_mapping: dict,
        item_inv_mapping: dict,
        num_threads: int = 4
        ):
    def _recs_mapper(user):
        user_id = user_mapping[user]
        recs = model.predict(
            user_id,
            item_ids,
            user_features = user_features,
            item_features = item_features,
            num_threads = num_threads)
        
        additional_N = len(known_items[user_id]) if user_id in known_items else 0
        total_N = N + additional_N
        top_cols = np.argpartition(recs, -np.arange(total_N))[-total_N:][::-1]
        
        final_recs = [item_inv_mapping[item] for item in top_cols]
        if additional_N > 0:
            filter_items = known_items[user_id]
            final_recs = [item for item in final_recs if item not in filter_items]
        return final_recs[:N]
    return _recs_mapper
```


```{code-cell} ipython3
# init mapper to get predictions
mapper = generate_lightfm_recs_mapper(
    lfm_model, 
    item_ids = all_cols, 
    known_items = dict(),
    N = top_N,
    user_features = None, 
    item_features = None, 
    user_mapping = lightfm_mapping['users_mapping'],
    item_inv_mapping = lightfm_mapping['items_inv_mapping'],
    num_threads = 20
)
```

```{code-cell} ipython3
# get predictions
local_test_preds['item_id'] = local_test_preds['user_id'].map(mapper)
```

Prettify predictions to use in catboost - make list to rows and add rank
```{code-cell} ipython3
local_test_preds = local_test_preds.explode('item_id')
local_test_preds['rank'] = local_test_preds.groupby('user_id').cumcount() + 1 
local_test_preds['item_name'] = local_test_preds['item_id'].map(item_name_mapper)
print(f'Data shape{local_test_preds.shape}')
local_test_preds.head()
```

```{code-cell} ipython3
# sense check for diversity of recommendations
local_test_preds.item_id.nunique()
```

## 2.3. CatBoostClassifier (ReRanker)
### 2.3.1. Data preparation

We need to creat 0/1 as indication of interaction:

- positive event -- 1, if watch_pct is not null;
- negative venet -- 0 otherwise

```{code-cell} ipython3
positive_preds = pd.merge(local_test_preds, local_test, how = 'inner', on = ['user_id', 'item_id'])
positive_preds['target'] = 1
positive_preds.shape
```

```{code-cell} ipython3
negative_preds = pd.merge(local_test_preds, local_test, how = 'left', on = ['user_id', 'item_id'])
negative_preds = negative_preds.loc[negative_preds['watched_pct'].isnull()].sample(frac = .2)
negative_preds['target'] = 0
negative_preds.shape
```

Random split by users to train reranker
```{code-cell} ipython3
train_users, test_users = train_test_split(
    local_test['user_id'].unique(),
    test_size = .2,
    random_state = 13
    )
```

Set up train/test set and shuffle samples
```{code-cell} ipython3
cbm_train_set = shuffle(
    pd.concat(
    [positive_preds.loc[positive_preds['user_id'].isin(train_users)],
    negative_preds.loc[negative_preds['user_id'].isin(train_users)]]
    )
)
```

```{code-cell} ipython3
cbm_test_set = shuffle(
    pd.concat(
    [positive_preds.loc[positive_preds['user_id'].isin(test_users)],
    negative_preds.loc[negative_preds['user_id'].isin(test_users)]]
    )
)
```

```{code-cell} ipython3
print(f'TRAIN: {cbm_train_set.describe()} \n, TEST: {cbm_test_set.describe()}')
```

```{code-cell} ipython3
# in this tutorial, I will not do any feature aggregation - use default ones from data
USER_FEATURES = ['age', 'income', 'sex', 'kids_flg']
ITEM_FEATURES = ['content_type', 'release_year', 'for_kids', 'age_rating']
```

Prepare final datasets - joins user and item features
```{code-cell} ipython3
cbm_train_set = pd.merge(cbm_train_set, users_data[['user_id'] + USER_FEATURES],
                         how = 'left', on = ['user_id'])
cbm_test_set = pd.merge(cbm_test_set, users_data[['user_id'] + USER_FEATURES],
                        how = 'left', on = ['user_id'])

```

```{code-cell} ipython3
# joins item features
cbm_train_set = pd.merge(cbm_train_set, movies_metadata[['item_id'] + ITEM_FEATURES],
                         how = 'left', on = ['item_id'])
cbm_test_set = pd.merge(cbm_test_set, movies_metadata[['item_id'] + ITEM_FEATURES],
                        how = 'left', on = ['item_id'])

print(cbm_train_set.shape, cbm_test_set.shape)
```

```{code-cell} ipython3
cbm_train_set.head()
```

Set necessary cols to filter out sample
```{code-cell} ipython3
ID_COLS = ['user_id', 'item_id']
TARGET = ['target']
CATEGORICAL_COLS = ['age', 'income', 'sex', 'content_type']
DROP_COLS = ['item_name', 'last_watch_dt', 'watched_pct', 'total_dur']
```

```{code-cell} ipython3
X_train, y_train = cbm_train_set.drop(ID_COLS + DROP_COLS + TARGET, axis = 1), cbm_train_set[TARGET]
X_test, y_test = cbm_test_set.drop(ID_COLS + DROP_COLS + TARGET, axis = 1), cbm_test_set[TARGET]
print(X_train.shape, X_test.shape)
```

Fill missing values with mode - just in case by default
```{code-cell} ipython3
X_train = X_train.fillna(X_train.mode().iloc[0])
X_test = X_test.fillna(X_test.mode().iloc[0])
```

### 2.3.2 Train the model

```{code-cell} ipython3
cbm_classifier = CatBoostClassifier(
    loss_function = 'CrossEntropy',
    iterations = 5000,
    learning_rate = .1,
    depth = 6,
    random_state = 1234,
    verbose = True
)
```

```{code-cell} ipython3
cbm_classifier.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds = 100, # to avoid overfitting,
    cat_features = CATEGORICAL_COLS,
    verbose = False
)
```

### 2.3.3. Model Evaluation
Let's make basic shapley plot to investigate feature importance. We expect that `rank` - predicted
order from LightFM - must be on top
```{code-cell} ipython3
explainer = shap.TreeExplainer(cbm_classifier)
shap_values = explainer.shap_values(X_train)
```

```{code-cell} ipython3
shap.summary_plot(shap_values, X_train, show = False, color_bar = False)
```

Let's see performance of the classifier 
```{code-cell} ipython3
# predictions on test
from sklearn.metrics import roc_auc_score
y_test_pred = cbm_classifier.predict_proba(X_test)

print(f"ROC AUC score = {roc_auc_score(y_test, y_test_pred[:, 1]):.2f}")
```

### 2.4. Evaluation on global test
Here, we compare predictions of two models - LightFM vs LightFM + CatBoost.
First, let's calculate predictions from both models - here we generate candidates via LightFM.
```{code-cell} ipython3
global_test_predictions = pd.DataFrame({
    'user_id': global_test['user_id'].unique()
        }
    )

# filter out cold start users
global_test_predictions = global_test_predictions.loc[global_test_predictions['user_id'].isin(local_train.user_id.unique())]
```


```{code-cell} ipython3
# set param for number of candidates
top_k = 100

# generate list of watched titles to filter
watched_movies = local_train.groupby('user_id')['item_id'].apply(list).to_dict()

mapper = generate_lightfm_recs_mapper(
    lfm_model, 
    item_ids = all_cols, 
    known_items = watched_movies,
    N = top_k,
    user_features = None, 
    item_features = None, 
    user_mapping = lightfm_mapping['users_mapping'],
    item_inv_mapping = lightfm_mapping['items_inv_mapping'],
    num_threads = 10
)

global_test_predictions['item_id'] = global_test_predictions['user_id'].map(mapper)
global_test_predictions = global_test_predictions.explode('item_id').reset_index(drop=True)
global_test_predictions['rank'] = global_test_predictions.groupby('user_id').cumcount() + 1 
```

Now, we can move to reranker to make predictions and make new order.
Beforehand, we need to prepare data for reranker
```{code-cell} ipython3
cbm_global_test = pd.merge(global_test_predictions, users_data[['user_id'] + USER_FEATURES],
                         how = 'left', on = ['user_id'])

cbm_global_test = pd.merge(cbm_global_test, movies_metadata[['item_id'] + ITEM_FEATURES],
                         how = 'left', on = ['item_id'])
cbm_global_test.head()
```

Fill missing values with the most frequent values
```{code-cell} ipython3
cbm_global_test = cbm_global_test.fillna(cbm_global_test.mode().iloc[0])
```

Predict scores to get ranks
```{code-cell}
cbm_global_test['cbm_preds'] = cbm_classifier.predict_proba(cbm_global_test[X_train.columns])[:, 1]
cbm_global_test.head()
```

```{code-cell} ipython3
# define cbm rank
cbm_global_test = cbm_global_test.sort_values(by = ['user_id', 'cbm_preds'], ascending = [True, False])
cbm_global_test['cbm_rank'] = cbm_global_test.groupby('user_id').cumcount() + 1
cbm_global_test.head()
```

Finally, let's move on to comparison
- define function to calculate matrix-based metrics;
- create table of metrics for both models

```{code-cell} ipython3
def calc_metrics(df_true, df_pred, k: int = 10, target_col = 'rank'):
    """
    calculates confusion matrix based metrics
    :df_true: pd.DataFrame
    :df_pred: pd.DataFrame
    :k: int, 
    """
    # prepare dataset
    df = df_true.set_index(['user_id', 'item_id']).join(df_pred.set_index(['user_id', 'item_id']))
    df = df.sort_values(by = ['user_id', target_col])
    df['users_watch_count'] = df.groupby(level = 'user_id')[target_col].transform(np.size)
    df['cumulative_rank'] = df.groupby(level = 'user_id').cumcount() + 1
    df['cumulative_rank'] = df['cumulative_rank'] / df[target_col]
    
    # params to calculate metrics
    output = {}
    num_of_users = df.index.get_level_values('user_id').nunique()

    # calc metrics
    df[f'hit@{k}'] = df[target_col] <= k
    output[f'Precision@{k}'] = (df[f'hit@{k}'] / k).sum() / num_of_users
    output[f'Recall@{k}'] = (df[f'hit@{k}'] / df['users_watch_count']).sum() / num_of_users
    output[f'MAP@{k}'] = (df["cumulative_rank"] / df["users_watch_count"]).sum() / num_of_users
    print(f'Calculated metrics for top {k}')
    return output
```

```{code-cell} ipython3
# first-level only - LightFM
lfm_metrics = calc_metrics(global_test, global_test_predictions)
lfm_metrics
```


```{code-cell} ipython3
# LightFM + ReRanker
full_pipeline_metrics = calc_metrics(global_test, cbm_global_test, target_col = 'cbm_rank')
full_pipeline_metrics
```

Prettify both metrics calculation results for convenience
```{code-cell} ipython3
metrics_table = pd.concat(
    [pd.DataFrame([lfm_metrics]),
    pd.DataFrame([full_pipeline_metrics])],
    ignore_index = True
)
metrics_table.index = ['LightFM', 'FullPipeline']

# calc relative diff
metrics_table = metrics_table.append(metrics_table.pct_change().iloc[-1].mul(100).rename('lift_by_ranker, %'))

metrics_table
```

Thus, with a few number of features we could improve our metrics using reranker.
Further, imagine how it can be improved if we add more features and fine tune the reranker

# Source & further recommendations
- [Kaggle Notebook for LightFM](https://www.kaggle.com/code/sharthz23/implicit-lightfm/notebook);
- [Recommended course from MTS RecSys team on ods.ai](https://ods.ai/tracks/mts-recsys-df2020)