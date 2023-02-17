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

**TODO ADD PYTHON CODE FOR CONTENT BASED HERE**

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

User A | User B | User C | User D
--------------------------------------
Movie 1  5        4        2        3
Movie 2  3        3        4        4
Movie 3  4        4        5        5
Movie 4  1        2        1        2
Movie 5  2        1        3        3
--------------------------------------

To apply collaborative filtering, we can compute the similarity between each pair of users based on their 
ratings. The similarity is calculated using the Pearson Correlation Coefficient (PCC). The PCC is a measure
of how well two sets of data are correlated, and it ranges from -1 (perfectly negatively correlated) to +1
(perfectly positively correlated). For example, let’s assume that we want to find the similarity between
User A and User B. The PCC is calculated by taking the average of the product of the ratings for each movie.
 So, for User A and User B, the PCC is:

`(5*4 + 3*3 + 4*4 + 1*2 + 2*1) / sqrt((5-3.5)^2 + (4-3)^2 + (2-4)^2 + (3-2)^2 + (1-1)^2)`

This calculation yields a PCC of 0.713, which indicates that User A and User B have a strong positive
correlation in their ratings. To find recommendations for User A, we can first identify the users who are
most similar to User A. In this example, that would be User B and User C. Next, we can take the weighted
average of the ratings from those users for the movies that User A has not yet rated. For example,
let’s assume that User A has not yet rated Movie 4. We can then take the weighted average of the ratings
for Movie 4 from User B and User C.  User B rated Movie 4 a 2, and User C rated it a 1. We can then take
the weighted average of those ratings, giving more weight to User B since they are more similar to User A.
In this case, the weighted average would be 1.7. Therefore, based on the ratings from other users,
it is likely that User A would rate Movie 4 a 2. 


### Model-based Collaborative Filtering
Model-based collaborative filtering is an approach that uses machine learning algorithms to learn a model from 
the ratings or interactions of users with items. The model is then used to make predictions about the ratings of
users for items that they have not yet interacted with. One common approach in model-based collaborative filtering
is matrix factorization. In this approach, the user-item matrix is decomposed into two lower-dimensional matrices:
a user matrix and an item matrix. The user matrix represents the latent preferences of users, and the item matrix
represents the latent attributes of items. The dot product of the user and item matrices gives the predicted rating
for a user-item pair. Matrix factorization is typically performed using a technique called Singular Value Decomposition (SVD).
However, SVD is computationally expensive and may not scale well to large datasets. Therefore, alternative techniques
such as Alternating Least Squares (ALS) and Stochastic Gradient Descent (SGD) are often used. Another common approach
in model-based collaborative filtering is deep learning. In this approach, a neural network is used to learn a
representation of users and items. The network takes as input the ratings or interactions of users with items and
outputs a prediction of the rating for a user-item pair. Deep learning has the advantage of being able to capture
complex patterns in the data and can be used to learn non-linear relationships between users and items.
One advantage of model-based collaborative filtering is that it can handle the cold-start problem by using the
learned model to make predictions about items that have not yet been rated by users. Additionally, model-based
collaborative filtering can be more accurate than memory-based collaborative filtering, especially when there are
a large number of users and items. However, model-based collaborative filtering also


## Python Implementations
There are many Python libraries available for content-based filtering such as Surprise & LightFM.
One of the most popular libraries is Surprise, which is a Python machine learning library for
recommendation systems. It includes several algorithms for making predictions and performing content-based filtering.

Another library is LightFM, which is a Python library for building recommendation systems. It includes a set of
algorithms for content-based filtering, such as the weighting of item attributes and personalized rankings.


# TODO
- about collaborative filtering
- discuss more deeply python libraries implementation for collaborative filtering in terms features / time