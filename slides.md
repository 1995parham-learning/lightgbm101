---
theme: seriph
background: /background.jpg
title: Drift in Machine Learning
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

# Light GBM, a closer look

By Elahe Dastan

<div class="abs-br m-6 flex">
  <a href="https://github.com/1995parham-learning/lightgbm101" target="_blank" alt="GitHub" title="Open in GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

---

## Features

### Optimization in Speed and Memory Usage

Many boosting tools use pre-sort-based algorithms (e.g. default algorithm in xgboost) for decision tree learning.
It is a simple solution, **but not easy to optimize**.

LightGBM uses histogram-based algorithms, which bucket continuous feature (attribute) values into discrete bins.
This speeds up training and reduces memory usage. Advantages of histogram-based algorithms include the following:

- Reduced cost of calculating the gain for each split
  - Pre-sort-based algorithms have time complexity O(#data)
  - Computing the histogram has time complexity O(#data), but this involves only a fast sum-up operation. Once the histogram is constructed, a histogram-based algorithm has time complexity O(#bins), and #bins is far smaller than #data.
- Use histogram subtraction for further speedup
  - To get one leaf’s histograms in a binary tree, use the histogram subtraction of its parent and its neighbor
  - So it needs to construct histograms for only one leaf (with smaller #data than its neighbor). It then can get histograms of its neighbor by histogram subtraction with small cost (O(#bins))

---

## Features (Cont'd)

- Reduce memory usage
  - Replaces continuous values with discrete bins. If #bins is small, can use small data type, e.g. uint8_t, to store training data
  - No need to store additional information for pre-sorting feature values
- Reduce communication cost for distributed learning
