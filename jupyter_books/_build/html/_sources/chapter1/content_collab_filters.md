# Content-based and Collaborative Filtering
In this section, we will go through with 2 straightforward ways to generate candidates for recommendations.
As we mentioned before, these are *content-based* and *collaborative* filterings. In this chapter,
we will go through explanation of both methods with examples and finnally discuss various libraries
to implement them. First, let's consider *content-based filttering*.

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
songs as well as songs with similar attributes, such as a similar genre or artist.

In conclusion, content-based filtering is a type of recommender system that recommends items to users based on their
past preferences and behaviors. 

## Collaborative Filtering



## Python Implementations
There are many Python libraries available for content-based filtering such as Surprise & LightFM.
One of the most popular libraries is Surprise, which is a Python machine learning library for
recommendation systems. It includes several algorithms for making predictions and performing content-based filtering.

Another library is LightFM, which is a Python library for building recommendation systems. It includes a set of
algorithms for content-based filtering, such as the weighting of item attributes and personalized rankings.


# TODO
- about collaborative filtering
- discuss more deeply python libraries implementation in terms features / time
- maybe add example with codes here -- not another markdown
