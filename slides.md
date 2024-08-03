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

---

## Features 🤩

### Optimization in Speed and Memory Usage

Many boosting tools use pre-sort-based algorithms (e.g. default algorithm in xgboost) for decision tree learning.
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
