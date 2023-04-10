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

(chapter1_part6)=

# Full Pipeline of the Two-level Recommender System

In this chapter, we will wrap up all steps from 1.2 to 1.5:
- Preprocess data with proper two-level validation;
- Develop candidate generation model with implicit library;
- Then, move to Catboost and get our reranker - second level model;
- Finally, evaluate our models: implicit vs implicit + reranker

First, let's recall what we discussed in [`Metrics & Validation`](https://rekkobook.com/chapter1/validation_metrics.html)
In recommender systems we have special data split to validate our model - we split data by time for candidates
and by users for reranker. Now, we move on to coding.


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

from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', lambda x: '%.3f' % x)
```