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

(chapter1_part3)=

# Content-based and Collaborative Filterings in a Nutshell
In this section, we will go through 2 straightforward ways to generate candidates for recommendations.
As we mentioned before, these are *content-based* and *collaborative* filterings. Yet,
we will go through explanation of both methods with examples and finnally discuss various libraries
to implement them. Before that we have to define and understand embeddings. As you might have noticed,
we mentioned a lot "similar items", "similar users" etc and question arises -- how we define that similarity?
Speaking of calculation of similarity it is pretty straightforward -- we calcualte cosine between two arrays.
The intruging part is how do we get these arrays from our data.

## Embeddings Explained
The evolution of text processing started from one-hot encoding. When there was text data, Data Scientists
would preprocess them (lower case, remove symbols etc.) and then create one-hot representations of words or
n-grams (when we split words/text into 2-3-...-n parts by characters). Finally, use some ML model on top of it.
Notwithstanding the fact of easiness and interpretability of this approach, human language is sophisticated
and various words can mean different meanings depending on the context and such techinique fails in most cases.

Therefore, embeddings have become a next stage in text processing pipeline. It is type of word representation
that allows words with similar meaning to have a similar representation. Unlike methods such as one-hot encoding,
word embeddings provide a way to represent words in a more meaningful way, by mapping them to a vector of real
numbers in a continuous vector space. The idea behind word embedding is to use a neural network to learn
relationships between words in a dataset. The neural network is trained to assign a numeric vector to each word
in the dataset. The vector is typically of fixed length and the goal is to find a vector that accurately
represents the meaning of the word, in the context of the dataset. This allows for words in similar contexts
to have similar vector representations. 

For example, imagine a dataset of movie reviews. Let’s say that the neural network has been trained to assign
a vector to each word in the dataset. If the word “amazing” is used in a movie review, then the vector assigned
to “amazing” will be similar to the vector assigned to “incredible”. This is because the meanings of these two
words are similar and they are often used in similar contexts. Word embeddings can also be used to identify
relationships between words. For example, consider the words “man” and “woman”. If the neural network assigned
similar vectors to these two words, this would indicate that the two words are related.  In addition to
identifying relationships between words, word embeddings can also be used to classify documents. For example,
if a document contains the words “amazing” and “incredible”, then the neural network can assign an appropriate
vector to each of these words. If a second document contains similar words, then the neural network can assign
similar vectors to these words. This allows the neural network to accurately classify the documents as being similar. 

Finally, word embeddings can be used for data visualization. By plotting the vectors assigned to words in
a two-dimensional space, it is possible to see how words are related. This can be a useful tool for understanding
the relationships between words in a given dataset. In summary, word embeddings are a powerful tool
for representing words in a meaningful way. They can be used to identify relationships between words,
sclassify documents, and visualize data. 

Now, let's consider *content-based filttering* and use simple Word2Vec/Doc2Vec
model to get such recommendations.

## Content-based Filtering
Content-based filtering can be used in a variety of applications, from recommending films and music to suggesting
restaurants and travel destinations. In this part, we'll discuss how content-based filtering works and provide
some examples.

Content-based filtering is a type of recommender system that recommends items to users based on their past
preferences and behaviors. It works by analyzing a user's preferences, in terms of attributes such as genre,
director, actor, or even a combination of these, and then recommending other items that have similar attributes.
For example, if a user has previously watched romantic comedies with Julia Roberts, content-based filtering
would recommend other romantic comedies with Julia Roberts, or other films featuring similar actors or directors.

Content-based filtering is based on the assumption that users who liked one item will likely like similar items.
To generate recommendations, the system first identifies the attributes of the items that the user has previously
interacted with. It then identifies other items that have similar attributes and recommends them to the user.
For example, if a user has previously listened to Taylor Swift songs, the system will identify other Taylor Swift
songs as well as songs with similar attributes, such as a similar genre or artist. In industry, this type of
recommendations is showed with "Similar to ...". It is additional nudge to increase interest of a user
as recommendations with explanation seems to be really personalized from the user's point of view.

In conclusion, content-based filtering is a type of recommender system that recommends items to users based on their
past preferences and behaviors. Next, we jump to coding part and create simple Word2Vec model via [`gensim`](https://pypi.org/project/gensim/) library.
Well explained logic of Word2Vec model you can find [here](https://israelg99.github.io/2017-03-23-Word2Vec-Explained/).
Here, we will not discuss details of implementation.

### gensim: example of content-based recommendations based on Doc2Vec approach
Now, we move on to implementation of content-based recommender using `gensim` library and Doc2Vec. It is almost
the same as Word2Vec with sligh modification, but idea remains the same.

#### 0. Configuration
```{code-cell} ipython3
# links to shared data MovieLens
# source on kaggle: https://www.kaggle.com/code/quangnhatbui/movie-recommender/data
MOVIES_METADATA_URL = 'https://drive.google.com/file/d/19g6-apYbZb5D-wRj4L7aYKhxS-fDM4Fb/view?usp=share_link'
```

#### 1. Modules and functions
```{code-cell} ipython3
import re
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from ast import literal_eval
from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import warnings
warnings.filterwarnings('ignore')

# download stop words beforehand
nltk.download('stopwords')
```
##### 1.1. Helper functions to avoid copypaste
```{code-cell} ipython3
def read_csv_from_gdrive(url):
    """
    gets csv data from a given url (taken from file -> share -> copy link)
    :url: example https://drive.google.com/file/d/1BlZfCLLs5A13tbNSJZ1GPkHLWQOnPlE4/view?usp=share_link
    """
    file_id = url.split('/')[-2]
    file_path = 'https://drive.google.com/uc?export=download&id=' + file_id
    data = pd.read_csv(file_path)

    return data
```

```{code-cell} ipython3
# init lemmatizer to avoid slow performance
mystem = Mystem() 

def word_tokenize_clean(doc: str, stop_words: list):
    '''
    tokenize from string to list of words
    '''

    # split into lower case word tokens \w lemmatization
    tokens = list(set(mystem.lemmatize(doc.lower())))
  
    # remove tokens that are not alphabetic (including punctuation) and not a stop word
    tokens = [word for word in tokens if word.isalpha() and not word in stop_words \
              not in list(punctuation)]
    return tokens
```

#### 2. Main
##### 2.1. Data Preparation

```{code-cell} ipython3
# read csv information about films etc
movies_metadata = read_csv_from_gdrive(MOVIES_METADATA_URL)
movies_metadata.dtypes
```

To get accurate results we need to preprocess text a bit. The pipeline will be as follows:
- Filter only necessary columns from movies_metadada : id, original_title, overview;
- Define `model_index` for model to match back with `id` column;
- Text cleaning: removing stopwords & punctuation, lemmatization for further tokenization and tagged document creatin required for gensim.Doc2Vec

```{code-cell} ipython3
# filter cols
sample = movies_metadata[['id', 'original_title', 'overview']].copy()
sample.info()
```
```{code-cell} ipython3
# as you see from above, we have missing overview in some cases -- let's fill it with the original title
sample.loc[sample['overview'].isnull(), 'overview'] = sample.loc[sample['overview'].isnull(), 'original_title']
sample.isnull().sum()
```

```{code-cell} ipython3
# define model_index and make it as string
sample = sample.reset_index().rename(columns = {'index': 'model_index'})
sample['model_index'] = sample['model_index'].astype(str)
```

```{code-cell} ipython3
# create mapper with title and model_idnex to use it further in evaluation
movies_inv_mapper = dict(zip(sample['original_title'].str.lower(), sample['model_index'].astype(int)))
```

```{code-cell} ipython3
# preprocess by removing non-character data, stopwords
tags_corpus = sample['overview'].values
tags_corpus = [re.sub('-[!/()0-9]', '', x) for x in tags_corpus]
stop_words = stopwords.words('english')

tags_doc = [word_tokenize_clean(description, stop_words) for description in tags_corpus]
tags_corpus[:1]
```

```{code-cell} ipython3
# prepare data as model input for Word2Vec
## it takes some time to execute
tags_doc = [TaggedDocument(words = word_tokenize_clean(D, stop_words), tags = [str(i)]) for i, D in enumerate(tags_corpus)]
```

```{code-cell} ipython3
# let's check what do we have
## tag = movie index
tags_doc[1]
```

#### 2.2. Model Training and Evaluation

First, let's define some paramters for Doc2Vec model
```{code-cell} ipython3
VEC_SIZE = 50 # length of the vector for each movie
ALPHA = .02 # model learning param
MIN_ALPHA = .00025 model learning param
MIN_COUNT = 5 # min occurrence of a word in dictionary
EPOCHS = 20 # number of trainings
```

```{code-cell} ipython3
# initialize the model
model = Doc2Vec(vector_size = VEC_SIZE,
                alpha = ALPHA, 
                min_alpha = MIN_ALPHA,
                min_count = MIN_COUNT,
                dm = 0)
```

```{code-cell} ipython3
# generate vocab from all tag docs
model.build_vocab(tags_doc)
```

```{code-cell} ipython3
# train model
model.train(tags_doc,
            total_examples = model.corpus_count,
            epochs = EPOCHS)
```

Now, let's make some checks by defining parameters for model ourselves.
Assume that we watched movie `batman` and based on that generate recommendation similar to it's description.
To do that we need:
- To extract movie id from `movies_inv_mapper` we created to map back titles from model output
- Load embeddings from trained model
- Use built-in most_similar() method to get most relevant recommendations based on film embedding
- Finally, map title names for sense-check

```{code-cell} ipython3
# get id
movie_id = movies_inv_mapper['batman']
movie_id
```

```{code-cell} ipython3
# load trained embeddings 
movies_vectors = model.dv.vectors
```

```{code-cell} ipython3
movie_embeddings = movies_vectors[movie_id]
```

```{code-cell} ipython3
# get recommendations
similars = model.docvecs.most_similar(positive = [movie_embeddings], topn = 20)
output = pd.DataFrame(similars, columns = ['model_index', 'model_score'])
output.head()
```

```{code-cell} ipython3
# reverse values and indices to map names in dataframe
name_mapper = {v: k for k, v in movies_inv_mapper.items()}
```

```{code-cell} ipython3
output['title_name'] = output['model_index'].astype(int).map(name_mapper)
output
```

## Collaborative Filtering [WIP]
Collaborative filtering is a powerful method for recommendation systems used to predict user preferences or
interests. It is based on the notion that people who have similar tastes and preferences in one domain are likely
to have similar tastes and preferences in a different domain. The collaborative filtering technique seeks to identify
users who have similar tastes and preferences, based on their past interactions, and then use those users'
interactions of items to predict relevance of similar items for the user. The goal of collaborative filtering is
to use the opinions of other people to make predictions about a user’s preferences and interests.
This is done by finding users who have similar tastes and preferences as the user in question, and then using
those users’ ratings of items to make predictions about how the user would rate the same items.
There are two main approaches to collaborative filtering: memory-based and model-based. 

### Memory-based Collaborative Filtering
Memory-based collaborative filtering, also known as neighborhood-based collaborative filtering, is an approach
that relies on finding similar users or items based on their behavior or preferences. The basic idea is to use
the ratings or interactions of users with items to identify other users who have similar tastes, and then use
the ratings of those similar users to make recommendations to a target user. One common approach in memory-based
collaborative filtering is user-based collaborative filtering. In this approach, the similarity between users is
calculated based on their ratings for items. A similarity metric such as the cosine similarity or Pearson correlation
coefficient is often used to measure the similarity between two users. The similarity scores are then used to
identify the most similar users to the target user. Once the most similar users are identified, their ratings
for items are used to generate recommendations for the target user. Item-based collaborative filtering is another
common approach in memory-based collaborative filtering. In this approach, the similarity between items is calculated
based on the ratings of users who have rated both items. The similarity scores are then used to identify items that
are similar to the items that the target user has already rated highly. Once the similar items are identified,
they are recommended to the target user. One advantage of memory-based collaborative filtering is that it is easy
to implement and interpret. The algorithm is relatively simple and does not require a lot of computational resources.
Additionally, memory-based collaborative filtering can be effective when there is a lot of data available and the
user-item matrix is sparse. However, memory-based collaborative filtering also has several disadvantages.
One major limitation is that it is prone to the cold-start problem, which occurs when there is not enough data
available to identify similar users or items. Additionally, memory-based collaborative filtering can be
computationally expensive when there are a large number of users or items.

Let's consider an example with Pearson Correlation
Say, we have a dataset that contains the ratings of four users on five movies. The data looks like this:

|         | User A | User B | User C | User D |
|-------- | -------- | ------- | ------- | --------     |
|Movie 1 | 5 | 4 | 2 | 3 | 
|Movie 2 | 3 | 3 | 4 | 4 | 
|Movie 3 | 4 | 4 | 5 | 5 |
|Movie 4 | 1 | 2 | 1 | 2 |
|Movie 5 | 2 | 1 | 3 | 3 |


To apply collaborative filtering, we can compute the similarity between each pair of users based on their 
ratings. The similarity is calculated using the Pearson Correlation Coefficient (PCC). The PCC is a measure
of how well two sets of data are correlated, and it ranges from -1 (perfectly negatively correlated) to +1
(perfectly positively correlated). For example, let’s assume that we want to find the similarity between
User A and User B. The PCC is calculated by taking the average of the product of the ratings for each movie.
So, let's get PCC for User A and User B:

```{code-cell} ipython3
import numpy as np

user_a = [5, 3, 4, 1, 2]
user_b = [4, 3, 4, 2, 1]

print(f'Pearson Correlation for user A and B is: {np.corrcoef(user_a, user_b)[0, 1]}')
```

It indicates that `User A` and `User B` have a strong positive correlation in their ratings.
To find recommendations for `User A`, we can first identify the users who are most similar to `User A`.
In this example, that would be `User B` and `User C` (check it out by calculating other pairs).
Next, we can take the weighted average of the ratings from those users for the movies that `User A`
has not yet rated. For example, let’s assume that `User A` has not yet rated Movie 4. We can then take
the weighted average of the ratings for Movie 4 from `User B` and `User C`.  `User B` rated Movie 4 a 2,
and `User C` rated it a 1. We can then take the weighted average of those ratings, giving more weight 
to `User B` since they are more similar to `User A`. In this case, the weighted average would be around 2.
Therefore, based on the ratings from other users, it is likely that `User A` would rate Movie 4 a 2.

To wrap up, we can say that memory-based collaborative filtering as about calculating similarity
between rows or columns of interaction matrix. In our example, we took columns a.k.a user similarities
while we could take item-item similarities and use as recommendation.


### Model-based Collaborative Filtering
Model-based collaborative filtering is an approach that uses machine learning algorithms to learn a model from 
the ratings or interactions of users with items. The model is then used to make predictions about the relevance of
users for items that they have not yet interacted with. One common approach in model-based collaborative filtering
is matrix factorization. In this approach, the user-item matrix is decomposed into two lower-dimensional matrices:
a user matrix and an item matrix. The user matrix represents the latent preferences of users, and the item matrix
represents the latent attributes of items. The dot product of the user and item matrices gives the predicted relevance
for a user-item pair. Matrix factorization is typically performed using a technique called Singular Value Decomposition (SVD).
The example of how it is computed is showed below. Basically, we have interactions data where rows represent
users and columns their ratings/other interactions. Based on thatm we have find such matrices that would approximate
this relationship from our interactions data.

![](img/svd_example.png)
*Toy example with SVD decomposition*

However, SVD is computationally expensive and may not scale well to large datasets. Therefore, alternative techniques
such as Alternating Least Squares (ALS) or modification for implicit target iALS, Stochastic Gradient Descent (SGD)
are often used. Another common approach in model-based collaborative filtering is deep learning.
In this approach, a neural network is used to learn a representation of users and items.
The network takes as input the ratings or interactions of users with items and outputs a prediction of the rating for a
user-item pair. Deep learning has the advantage of being able to capture complex patterns in the data and can be used
to learn non-linear relationships between users and items. One of the popular examples is Extreeme Deep Factorization machines (xDeepFM).
One advantage of model-based collaborative filtering is that it can handle the cold-start problem by using the
learned model to make predictions about items that have not yet been rated by users. Additionally, model-based
collaborative filtering can be more accurate than memory-based collaborative filtering, especially when there are
a large number of users and items. Obviously, if we have enough data we can generate more accurate predictions minimizing our loss function
However, model-based collaborative filtering also has some disadvantages. One major limitation is that it can be
difficult to interpret the learned model and understand why certain recommendations are being made. Additionally,
model-based collaborative filtering can be computationally expensive and may require a lot  of computational resources,
especially when using deep learning techniques. Another disadvantage of model-based collaborative filtering is that it
requires a large amount of data to train the model effectively. This can be a challenge in some domains, where there
may be a limited amount of data available. In these cases, memory-based collaborative filtering may be a better choice.

**TODO ADD PYTHON CODE FOR COLLABORATIVE FILTERING HERE HERE**

### Hybrid Approaches
In practice, many recommender systems use a hybrid approach that combines both memory-based and model-based
collaborative filtering. In a hybrid approach, the strengths of both approaches are leveraged to improve the
accuracy and performance of the recommender system. One common approach in hybrid collaborative filtering is
to use a memory-based approach to generate initial recommendations and then refine the recommendations using
a model-based approach. This approach can be effective in situations where there is not enough data to train
a model effectively but there is enough data to identify similar users or items using a memory-based approach.
Another approach is to use a model-based approach to generate initial recommendations and then refine the
recommendations using a memory-based approach. This approach can be effective in situations where the user-item
matrix is very sparse and a model-based approach is needed to make accurate predictions.


## Python Libraries for Implementations
There are many Python libraries available for content-based filtering such as Surprise & LightFM.
One of the most popular libraries is Surprise, which is a Python machine learning library for
recommendation systems. It includes several algorithms for making predictions and performing content-based filtering.

Another library is LightFM, which is a Python library for building recommendation systems. It includes a set of
algorithms for content-based filtering, such as the weighting of item attributes and personalized rankings.


## TODO
- discuss more deeply python libraries implementation for collaborative filtering in terms features / time