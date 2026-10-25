# Study guide

These notes explain implemented methods; do not claim independent authorship or experiments beyond the recorded results.

## Why evaluate train accuracy without dropout or augmentation?

Matching train and test evaluation conditions avoids attributing stochastic training-mode degradation to a generalization gap.

## Does a smaller train–test gap imply a better model?

No. Underfitting can lower both training and test accuracy while narrowing the gap. Inspect absolute performance and learning curves.

## What does the three-seed standard deviation measure?

Variation from optimization seeds conditional on one data split and test cohort. It is not a confidence interval for population performance.
