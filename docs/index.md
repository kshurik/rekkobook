# Welcome To The World Of Recommendations and Ranking

In most companies like OKKO (my current workplace) or Netflix, we aim to promote relevant content
to users such that they can spend leisure time watching interesting titles. Also, we want to find
relevant titles by users' search query and put them on top so that minimum effort is needed
to start watching the title. Yet, in many other industries such as e-commerce, banking, etc.
recommending relevant products or helping to find necessary services significantly increase
user satisfaction and/or sales.

These two problems can be considered ranking problems. In the first case,
we have to rank titles based on users' preferences. In the second one, we have to rank the
most similar to the query and most likely to be clicked/watched. In this handbook, we will dive into
these two topics and learn how to code them as well.

## Self-Introduction

Hi! I am Shuhrat, as of today, Lead Machine Learning Engineer at online cinema OKKO.
During my journey as a Machine Learning Engineer, I had vast experience in various 
industries like banking, metal & mining, consulting, two-sided marketplace, and
finally -- online streaming platform OKKO.

I got my masters at [International School of Economics and Finance](https://www.hse.ru/en/ma/financial/)
(Higher School of Economics) and before that BSc in Economics minoring in Data Science

You can reach out to me via
- Telegram: @kshurik
- Email: khalilbekov92@gmail.com
- [LinkedIn](https://www.linkedin.com/in/shkhalilbekov/)

## Motivation behind this handbook
In most projects where I was involved classic ML models were used and there are so
many guides and handbooks about them that is very easy to get started researching.
However, recommendations and ranking problems take a special place among other
problems because they face many difficulties in production-ready usage: custom
backend architecture, data streaming, runtime latency, heavy models for embeddings, etc.
Unfortunately, I could not find a resource where all this knowledge would be up-to-date
with an overview of various algorithms explained in theory and backed up by code.
Therefore, I decided to refresh everything I did and learn and structure it into
a useful handbook. First of all, it will help me to understand and memorize most stuff
efficiently. Also, it may help many of you to get to know this field
in ML and even allow you to develop good-performing models in production.

In addition, I tried to simplify hard topics so that as many folks as possible would be able
to process the content. Well, I am not a math and tech guy myself as you can see
from my degrees :) (even though math in our uni was hell!)

Thus, in this handbook, you will find an understandable overview of the most popular topics
in recommendation and ranking with practical examples in jupyter notebooks.


## Few words about the structure
This book aims to provide an overview of the theoretical part of RecSys methods, its implementation, and further deployment & estimation of experiments. Thus, each section will contain:
- Theoretical explanation;
- Python code with explanation and outputs
In the book's [repository](https://github.com/kshurik/rekkobook) you will find notebooks and poetry.lock
with necessary dependencies to replicate the pipeline without copypasting from the book


In this handbook, we will cover 3 chapters:
- In Chapter 1 we will discuss the architecture of recommendation systems, models, and various implementations;
- Then, Chapter 2 is going to be all about deployment - building RecSys microservice using various frameworks
with an explanation of those frameworks, pros & cons, etc.;
- Finally, we are going dive into the product part - estimation of experiments for our models, metrics to consider
and other cool stuff that allows us to understand whether we bring value to business by developing different models
