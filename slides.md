---
theme: seriph
background: gray
title: Light GBM 101
info: |
  Light GBM
  By Elahe Dastan
class: text-center
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
favicon: "https://github.com/1995parham-me.png"
layout: cover
hideInToc: true
---

# Light GBM, a closer look 🤓

By Elahe Dastan

<div class="abs-br m-6 flex">
  <a href="https://github.com/1995parham-learning/lightgbm101" target="_blank" alt="GitHub" title="Open in GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

---

## Bagging vs Boosting

As we know, _Ensemble learning_ helps improve machine learning results by **combining several models**.

1. [Bagging]{style="color:cyan"}: It is a homogeneous weak learners’ model that learns from each other independently in parallel and combines them for determining the model average.
2. [Boosting]{style="color:cyan"}: It is also a homogeneous weak learners’ model but works differently from Bagging. In this model, learners learn sequentially and adaptively to improve model predictions of a learning algorithm.

---

## Bagging vs Boosting (Cont'd)

<img src="/Bagging.png" alt="leaf-wise" class="rounded shadow h-50 basis-1/4 m-auto p-1" />
<img src="/Boosting.png" alt="leaf-wise" class="rounded shadow h-50 basis-1/4 m-auto p-1" />

---

## Boosting

In machine learning, boosting is an _ensemble_ meta-algorithm for primarily reducing bias, variance.
It is used in _supervised learning_ and a family of machine learning algorithms that **convert weak learners
to strong ones**.

---

## Boosting (Cont'd)

While boosting is not algorithmically constrained, most boosting algorithms consist of iteratively learning weak
classifiers with respect to a distribution and adding them to a final strong classifier. When they are added, they are
weighted in a way that is related to the weak learners' accuracy.

The main variation between many boosting algorithms is their method of weighting training data points and hypotheses.

---

## LightGBM

LightGBM is a _gradient boosting framework_ that uses **tree based learning algorithms**.

---

## Installation ⚙️

The preferred way to install LightGBM is via pip:

```bash
pip install lightgbm
```

To verify your installation, try to import `lightgbm` in Python:

```python
import lightgbm as lgb
```

To load a `numpy` array into Dataset:

```python
rng = np.random.default_rng()

data = rng.uniform(size=(500, 10))  # 500 entities, each contains 10 features

label = rng.integers(low=0, high=2, size=(500, ))  # binary target

train_data = lgb.Dataset(data, label=label)

# specific feature names and categorical features0:
train_data = lgb.Dataset(data, label=label, feature_name=['c1', 'c2', 'c3'], categorical_feature=['c3'])
```

---

## Installation (Cont'd) ⚙️

Training a model requires a parameter list and data set:

```python
num_round = 10
bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
```

Training with 5-fold CV (Cross-Validation):

```python
lgb.cv(param, train_data, num_round, nfold=5)
```

After training, the model can be saved:

```python
bst.save_model('model.txt')
```

The trained model can also be dumped to JSON format:

```python
json_model = bst.dump_model()
```

A saved model can be loaded:

```python
bst = lgb.Booster(model_file='model.txt')  # init model
```

A model that has been trained or loaded can perform predictions on datasets:

```python
# 7 entities, each contains 10 features
rng = np.random.default_rng()
data = rng.uniform(size=(7, 10))
ypred = bst.predict(data)
```

If early stopping is enabled during training, you can get predictions from the best iteration with `bst.best_iteration`:

```python
ypred = bst.predict(data, num_iteration=bst.best_iteration)
```

---

## Features 🤩

### Optimization in Speed and Memory Usage

Many boosting tools use pre-sort-based algorithms (e.g. default algorithm in `xgboost`) for decision tree learning.
It is a simple solution, **but not easy to optimize**.

LightGBM uses **histogram-based algorithms**, which bucket continuous feature (attribute) values into discrete bins.
This speeds up training and reduces memory usage. Advantages of histogram-based algorithms include the following:

- _Reduced cost of calculating the gain for each split_
  - Pre-sort-based algorithms have time complexity `O(#data)`.
  - Computing the histogram has time complexity `O(#data)`, but this involves only a fast sum-up operation.
    Once the histogram is constructed, a histogram-based algorithm has time complexity `O(#bins)`,
    and `#bins` is far smaller than `#data`.

---

## Features (Cont'd) 🤩

- _Use histogram subtraction for further speedup_
  - To get one leaf’s histograms in a binary tree, use the histogram subtraction of its parent and its neighbor
  - So it needs to construct histograms for only one leaf (with smaller #data than its neighbor).
    It then can get histograms of its neighbor by histogram subtraction with small cost (`O(#bins)`)
- _Reduce memory usage_
  - Replaces continuous values with discrete bins. If `#bins` is small, can use small data type, e.g. `uint8_t`,
    to store training data
  - No need to store additional information for pre-sorting feature values
- _Reduce communication cost for distributed learning_

---

## Features (Cont'd) 🤩

### Sparse Optimization

Need only `O(2 * #non_zero_data)` to construct histogram for sparse features.

### Optimization in Accuracy

_Leaf-wise (Best-first) Tree Growth_

Most decision tree learning algorithms grow trees by level (depth)-wise.

<img src="/level-wise.webp" alt="level-wise" class="rounded shadow h-60 basis-1/4 m-auto" />

---

## Features (Cont'd) 🤩

LightGBM grows trees leaf-wise (best-first). It will choose the leaf with max delta loss to grow.
Leaf-wise may cause over-fitting when `#data` is small, so LightGBM includes the `max_depth` parameter
to limit tree depth. However, trees still grow leaf-wise even when `max_depth` is specified.

<img src="/leaf-wise.webp" alt="leaf-wise" class="rounded shadow h-60 basis-1/4 m-auto" />

---

### Optimal Split for Categorical Features

It is common to represent categorical features with one-hot encoding, but this approach is suboptimal for tree learners.
Particularly for high-cardinality categorical features, a tree built on one-hot features tends to be unbalanced and
needs to grow very deep to achieve good accuracy.

Instead of one-hot encoding, the optimal solution is to split on a categorical feature by partitioning its categories
into 2 subsets. If the feature has `k` categories, there are `2^(k-1) - 1` possible partitions. But there is an efficient
solution for regression trees. It needs about `O(k * log(k))` to find the optimal partition.

### Optimization in Network Communication

It only needs to use collective communication algorithms, like 'All reduce', 'All gather' and 'Reduce scatter',
in distributed learning of LightGBM. LightGBM implements state-of-the-art algorithms.
These collective communication algorithms can provide much better performance than point-to-point communication.
