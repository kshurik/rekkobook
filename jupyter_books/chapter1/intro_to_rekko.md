# Introduction to Recommendation Problem

The main idea of recommendation is to rank relevant items according to some criterion.
As it was mentioned earlier, recommender systems intersect with ranking problem:
smart news feed, contacts recommendation, search response ranking, suggesting point B in ride --
all this stuff can be reffered to personal ranking. Mostly, two-level architecture is used
in both problems. First, we generate candidates that we consider relevant for a user / search query.
Then, we try to rank such that the most relevant ones are in top. Thus, difference often occur
in the first-level models. In ranking part it is quite similar.

In recommendations, we need user-item interaction and meta data of both. 
- In the first level model, we use embeddings and some similarity metric to generate
candidates to get highest recall i.e. complete set of items that might be relevant;
- In the second level, we use ranking model to get highest recall i.e. positive event.

Usually, positive event can be divided into two types:
- Explicit target: rating, purchase event etc., but they are rare;
- Implicit target: some proxy of explicit target like click, start watch etc., we have more such data in general

There is no unique way to define target, but in industry we use combination of both with preliminary research on target.

Also, in recommendation we should consider special cases:
- Items which user has already interacted with - there is no need to recommend films or books again, but it is
ok for groceries;
- Cold Start problem -- user does not have any history on our platform

## Baseline
- https://www.kaggle.com/datasets/dev0914sharma/dataset
- https://www.kaggle.com/datasets/ashirwadsangwan/imdb-dataset
