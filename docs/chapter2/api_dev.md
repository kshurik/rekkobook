# Application Programming Interface (`API`)
Previously, we discussed the process and tools of production code writing. In this chapter,
we will focus on Application Programming Interface - API which is used in any service to
communicate with each other. Let's dive into that.


API stands for Application Programming Interface. It is a set of rules and protocols that
allows different software applications to communicate with each other. In simpler terms,
an API is like a waiter in a restaurant who takes your order and communicates it to the kitchen.
The kitchen prepares your food and gives it back to the waiter who then serves it to you.

Similarly, when you use an app on your phone or computer, it communicates with the server
through an API. The app sends a request to the server, and the server responds with the required data.
For example, when you search for something on Google, the app sends a request to the Google server
through an API, and the server responds with the search results.

APIs are essential for software development as they allow developers to create software that works
with other applications. This means that developers do not need to create everything from scratch,
but instead, they can use APIs to connect their software to other applications, services, or data sources.

In the case of a two-level recommender system, the API would allow different components of the system
to communicate with each other. For example, the client requests user data, which is then
processed by the first-level model. The API would allow the model to communicate with the feature
store to enrich the data and then pass it on to the ranker model. Finally, the ranker model would
use the API to send back the sorted items to the client.

When it comes to building APIs in Python, Flask and FastAPI are one of the popular frameworks
that come to mind. Both frameworks have their own strengths and weaknesses, and choosing
between them depends on the specific needs of the project which we will discuss further

# API development
We dived into the world of APIs, web frameworks, and their application to building recommendation systems.
Now, we will introduce Flask, a popular Python web framework, and FastAPI, a newer and
faster web framework for building APIs. 


After a brief overview, we will dive into a hands-on example of building an API using Flask.
Specifically, we will use Flask to build a two-level recommender system that recommends movies to users based
on their preferences - this will include both training and inference pipeline. This project will
showcase how Flask can be used to build a scalable and efficient API for machine learning-based applications.

## Flask & FastAPI
`Flask` is a popular web framework for building APIs in Python that provides a simple and
flexible approach to building RESTful APIs. It is easy to learn, has a minimalistic design,
and is suitable for small to medium-sized projects. Flask is known for its flexibility,
simplicity, and intuitive API, which makes it easy to get started with building an API.
It is also a mature and well-established framework with a large community and a vast range
of third-party libraries available. Flask is also popular for its lightweight design and
excellent documentation, which makes it a popular choice for building small to medium-sized APIs.

`FastAPI`, on the other hand, is a relatively new framework that is built on top of the Starlette
web framework and uses the Pydantic library for data validation and serialization. It is designed
to be fast, efficient, and easy to use, making it ideal for large-scale projects. FastAPI is known
for its high performance and speed, which is built on top of the Starlette web framework that uses
asynchronous programming to handle requests more efficiently. This allows FastAPI to handle a large
number of requests with minimal resources, making it ideal for building high-performance APIs.
FastAPI also has built-in support for data validation and serialization using the Pydantic library.
This makes it easy to define the API schema and ensures that the data sent and received by the API
is valid and conforms to the specified schema.

The two frameworks have their own set of pros and cons. Flask is very flexible and easy to use.
It has a simple and intuitive API, making it easy to get started with building an API. FastAPI,
on the other hand, is known for its high performance and speed, which makes it ideal for building
high-performance APIs. One of the key differences between Flask and FastAPI is their approach to
asynchronous programming. While Flask supports asynchronous programming, it is not built with it
in mind and as a result, it may not be as efficient as FastAPI when handling large numbers of requests.
FastAPI, on the other hand, is designed to handle asynchronous programming efficiently 
and is built with performance in mind.

Another difference between the two frameworks is the level of abstraction they provide. 
Flask is a low-level framework that gives developers complete control over the API's functionality
and design. This level of control makes it easy to customize the API to meet specific requirements.
FastAPI, on the other hand, is a high-level framework that provides a lot of functionality out of
the box. This makes it easier to get started with building an API, but it may be
less flexible than Flask in some cases.

In terms of development time, Flask may be faster to set up and get started with due to its
simplicity and flexibility. FastAPI, on the other hand, may take longer to set up due to its more
complex design and additional dependencies. However, FastAPI's performance benefits may make up
for the additional development time in the long run.

Overall, both Flask and FastAPI are excellent choices for building APIs in Python. Flask is
a mature and well-established framework that is easy to learn and flexible. It is ideal for building
small to medium-sized APIs. FastAPI, on the other hand, is a high-performance framework that is built
with performance in mind. It is ideal for building large-scale APIs that require high levels of
performance and efficiency. The choice between Flask and FastAPI ultimately depends on the specific
requirements of the project, the level of control needed, and the expected levels of performance and scalability.

## RecSys Project Pipeline
In the previous section, we discussed the theory of API development and compared FlaskAPI and FastAPI
frameworks for building APIs. Now, we will apply this knowledge to develop an API for a two-level
recommender system. Our recommender system has four parts as we discussed [here](https://rekkobook.com/chapter2/intro_to_deployment.html#architecture-for-our-recsys-project): a client, first-level training and inference
for candidate generation, a feature store that stores user and item features as parquet files, and a
reranker as the second-level model. The output of our API will be recommendations along with status
(success or error) and a message (null if the status is success).

To develop this API, we will be using the Flask framework. Also, we will use the Flask [app context](https://flask.palletsprojects.com/en/1.1.x/appcontext/) to load both the first-level and second-level models when the app is run for the first time.
Additionally, we will add caching of data downloading to reduce the response time of the API.

First, let's look at the structure of the project illustrated below
```{image} ./img/api_example.png
:alt: fishy
:class: bg-primary mb-1
:width: 400px
:align: centre
```



We will use Flask's app context to load both models only once when the app is first run. This helps to improve the performance of the API by reducing the time it takes to load the models for each request.

To ensure that the API is efficient, we will also add caching of data downloading. This will help to reduce the number of requests that need to be made to the feature store and improve the response time of the API.

The API's output will consist of recommendations, status, and message. The status will indicate whether the request was successful or not, and the message will provide additional information about the request, such as an error message if the request failed. With these components in mind, we can start building our recommendation system API with Flask.
