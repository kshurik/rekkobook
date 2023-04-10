# Full Pipeline of the Two-level Recommender System

In this chapter, we will wrap up all steps from 1.2 to 1.5:
- Develop candidate generation model with implicit library;
- Then, move to Catboost and get our reranker - second level model
- Finally, evaluate our models: implicit vs implicit + reranker

```
# just to make it available to download w/o SSL verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
import scipy.sparse as sp

```