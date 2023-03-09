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

# Ranking Problem
The ranking problem is a type of supervised or semi-supervised learning problem where the goal
is to predict the relative order of a set of items. It is commonly used in search engines,
recommender systems, and other applications where the order of the response is vital. Ranking
models usually try to predict a relevance score and sorting is made based on these scores.
Strictly speaking, $s = f(x)$, $s$ - stands for ranking model and for each
input $x = (q, d)$ where $q$ is a query and $d$ is a document we predict *relevance scores*

Depending on the context and business needs, prediction of relevance score can be considered as:
- Binary classification - whether user will click on recommended movie;
- Regression - prediction of watch time by a user for particular recommended movie;
- Learning-to-rank