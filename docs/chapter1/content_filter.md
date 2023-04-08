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

# Content-based Filtering in a Nutshell
In this section, we will go through a straightforward way to generate candidates for recommendations.
As we mentioned before, one of the methods is *content-based* filtering. We will go through an explanation of
this method with an example and finally discuss a particular library to implement it.
Before that, we have to define and understand embeddings. As you might have noticed, we mentioned a lot
"similar items", "similar users" etc and the question arises -- how do we define that similarity?
Speaking of the calculation of similarity it is pretty straightforward -- we calculate cosine between two arrays.
The intriguing part is how we get these arrays from our data.

## Embeddings Explained
The evolution of text processing started from one-hot encoding. When there was text data, Data Scientists
would preprocess them (lower case, remove symbols, etc.) and then create one-hot representations of words or
n-grams (when we split words/text into 2-3-...-n parts by characters). Finally, use some ML model on top of it.
Notwithstanding the fact of easiness and interpretability of this approach, human language is sophisticated
and various words can mean different meanings depending on the context and such a technique fails in most cases.

Therefore, embeddings have become the next stage in the text processing pipeline. It is the type of word representation
that allows words with similar meanings to have a similar representation. Unlike methods such as one-hot encoding,
word embeddings provide a way to represent words in a more meaningful way, by mapping them to a vector of real
numbers in a continuous vector space. The idea behind word embedding is to use a neural network to learn
relationships between words in a dataset. The neural network is trained to assign a numeric vector to each word
in the dataset. Typically, the vector is of fixed length and the goal is to find a vector that accurately
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
classify documents, and visualize data. 

Now, let's consider *content-based filtering* and use simple Word2Vec/Doc2Vec
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
recommendation is shown with "Similar to ...". It is an additional nudge to increase the interest of a user
as recommendations with explanations seem to be personalized from the user's point of view.

In conclusion, content-based filtering is a type of recommender system that recommends items to users based on their
past preferences and behaviors. Next, we jump to the coding part and create a simple Word2Vec model via [`gensim`](https://pypi.org/project/gensim/) library.
Well-explained the logic of the Word2Vec model you can find [here](https://jalammar.github.io/illustrated-word2vec/).
Here, we will not discuss the details of implementation.

### gensim: example of content-based recommendations based on Doc2Vec approach
Now, we move on to the implementation of a content-based recommender using `gensim` library and Doc2Vec. It is almost
the same as Word2Vec with slight modifications, but the idea remains the same.

#### 0. Configuration
```{code-cell} ipython3
# links to shared data MovieLens
# source on kaggle: https://www.kaggle.com/code/quangnhatbui/movie-recommender/data
MOVIES_METADATA_URL = 'https://drive.google.com/file/d/19g6-apYbZb5D-wRj4L7aYKhxS-fDM4Fb/view?usp=share_link'
```

#### 1. Modules and functions
```{code-cell} ipython3
# just to make it available to download w/o SSL verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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

To get accurate results we need to preprocess the text a bit. The pipeline will be as follows:
- Filter only necessary columns from movies_metadada : id, original_title, overview;
- Define `model_index` for the model to match back with `id` column;
- Text cleaning: removing stopwords & punctuation, lemmatization for further tokenization, and tagged document creation required for gensim.Doc2Vec

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
MIN_ALPHA = .00025 # model learning param
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

Now, let's make some checks by defining parameters for the model ourselves.
Assume that we watched the movie `batman` and based on that generate recommendations similar to its description.
To do that we need:
- To extract movie id from `movies_inv_mapper` we created to map back titles from the model output
- Load embeddings from the trained model
- Use the built-in most_similar() method to get the most relevant recommendations based on film embedding
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