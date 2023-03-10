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
- `Binary classification` - whether user will click on recommended movie;
- `Regression` - prediction of watch time by a user for particular recommended movie;
- `Learning-to-rank`

Considering that the first two tasks are quite widespread, here we will focus in learning-to-rank
class of models. So, why do we have distinct class of learning-to-rank models while it still?
The main difference between learning to rank and classification/regression models is that
classification/regression models predict a label or value for a single input,
while learning to rank models predict a ranking for a list of inputs. Basically, you have a
list of items and you can make pair comparison within this set and decide the order within that
set. In case of regression / classification, you will not be able to do that.

![](img/ranking_example_1.png)

# Training learnig-to-rank models
In learning-to-rank pipeline there are three approaches to train the model:
- `Pointwise`: basically, it takes single item from a list and computes loss using
only its information. It resembles the same as if classifier / regressor is trained;
Some examples: cosine distance between embeddings, logistic regression based some 
features with binary relevance target etc, BM25 (it is "advanced" version of tf-idf).

- `Pairwise`: in this approach a pair of documents is used to minimize the loss.
The idea is to minimize the number of swaps in final ordered list. Some of the most
popular methods are [RankNet, LambdaRank and LambdaMART](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf);

**RankNet**

This cost function tries to minimize the number of swaps in the final ordered list. Originally,
it was developed for optimization parameters in neural nets, but in genral the underlying model
can be set to any. The formula:

$P_{ij} \equiv P(U_{i}>U_{j}) \equiv \frac{1}{1 + \exp^{-\sigma(s_{i} - s{j})}}$

**LambdaRank**

It was derived after `RankNet`, researchers found that during the process you do not need values of costs,
but only the gradients of the cost w.r.t model score. The formula that explains the idea 
with NDCG is the following:

$\lambda_{ij} = \frac{\partial C(s_{i} - s_{j})}{\partial s_{i}} = \frac{-\alpha}{1 + \exp^{\alpha(s_{i} - s_{j})}}|\Delta NDCG|$,
where C is the cost function, $\delta NDCG$ stands for how much NDCG will change if swap *i* and *j*

**LambdaMART**

This approach combines `LambdaRank` &  `Multiple Additive Regression Trees (MART)`. The approach is
quite straightforward: we use gradient boosted trees for prediction task and incorporate `LambdaRank`
cost function into the model to convert it to solve ranking problem. On experimental data, this
method outperformed `RankNet` & `LambdaRank`

- `Listwise`: as in the naming, it takes the whole list of candidates at once and tries to rank
documents within optimally. Fot that it uses two approaches:
1. Direct optimization of information retrievals (IR) metric such as NDCG via approximation
with [SoftRank](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SoftRankWsdm08Submitted.pdf) / [AdaRank](https://www.semanticscholar.org/paper/AdaRank%3A-a-boosting-algorithm-for-information-Xu-Li/a489d95fb930401c1f4b7d92bb139d271d49abbf);
2. Minimization of the loss that is defined based on you domain knowledge of what you are trying to achieve.
These are ListNet, ListMLE losses.

**ListNet**

It is a listwise version of aforementioned `RankNet`. It usses cross-entropy loss function along
with gradient descent to optimize parameters for neural net. In cases when we have only two items
in a list the result of the `ListNet` coincides with `RankNet`.

**ListMLE**

TBD