# Content-based and Collaborative Filterings in a Nutshell
In this section, we will go through with 2 straightforward ways to generate candidates for recommendations.
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

**TODO ADD PYTHON CODE**

## Collaborative Filtering



## Python Implementations
There are many Python libraries available for content-based filtering such as Surprise & LightFM.
One of the most popular libraries is Surprise, which is a Python machine learning library for
recommendation systems. It includes several algorithms for making predictions and performing content-based filtering.

Another library is LightFM, which is a Python library for building recommendation systems. It includes a set of
algorithms for content-based filtering, such as the weighting of item attributes and personalized rankings.


# TODO
- about collaborative filtering
- discuss more deeply python libraries implementation for collaborative filtering in terms features / time