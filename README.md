# Stochastic Gradient Descent

TL;DR Develop stochastic gradient descent algorithm in Python from scratch. Create a ratings matrix
from last.fm dataset and preprocess the data. Train the model using SGD and use it to recommend music artists.

### Basics

#### Recommender System
Recommender system is everywhere in our lives today, it has played a big role in our lives today in helping us to make decisions, so in this repo we will look at one of the way to recommender artist to our users

### Types of Recommender System
* Content Based -> Based on properties of items and recommends similar property items
* Collaborative filtering -> Recommends based on knowledge of user preferences on items

This recommender system uses Model-Based Collaborative Filtering which are based on matrix factorization. Goal of matrix factorization is to learn hidden user preferences and item attributes based on exisiting ratings.

### Singular Value Decomposition

```X = USV^T```

Given m x n matrix X:
  - U is (m * r) orthogonal matrix
  - S is (r * r) diagonal matrix with non-negative real numbers on the diagonal
  - V^T is (r * n) orthogonal matrix

where U represents feature vectors of the users, V represents feature vectors of the items and the
elements on the diagonal of S are known as singular values.

### Recommender Engine
View the [Jupyter Notebook](https://github.com/fadilparves/stoch_gradient_descent/blob/master/Artist_Recommender.ipynb) with descriptions and results

SGD from scratch [here](https://github.com/fadilparves/stoch_gradient_descent/blob/master/stoch_gradient_descent.py)

### How to run
1. Make sure you have all the libs installed (pandas, numpy, scikit-learn, seaborn, matplotlib)
2. Clone the repo
3. Open the `Artist_Recommender.ipynb` file
4. Restart and run all kernel

## Contributor
<a href="https://github.com/fadilparves/stoch_gradient_descent/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=fadilparves/stoch_gradient_descent" />
</a>
