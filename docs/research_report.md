# Regularization under a limited training budget

## Abstract
We executed controlled Fashion-MNIST experiments on a small CNN across five fixed treatments and three seeds. We report observed variation without assuming regularization improves accuracy. Baseline mean extended test accuracy was 0.7507; the combined treatment reached 0.7209. These are bounded exploratory experiments, not full-dataset state-of-the-art results.

## Introduction
Regularization can reduce overfitting but also change optimization and increase underfitting under short budgets. Separating these effects requires matched architectures, splits and training budgets.

## Literature review
Dropout stochastically masks activations (Srivastava et al., 2014). Decoupled weight decay is distinct from adding an L2 penalty to adaptive optimizer gradients (Loshchilov & Hutter, 2019). Augmentation encodes transformations intended to preserve class identity (Simard et al., 2003). Generalization is not explained solely by capacity or training error (Zhang et al., 2017). This study tests related mechanisms; it does not reproduce those papers' protocols.

## Research questions and hypotheses
Do dropout, weight decay, augmentation or their combination change generalization at a fixed compute budget? Hypothesis: effects depend on optimization and data scale. No directional improvement is assumed. Predetermined treatments are compared pairwise to baseline by seed, without selecting a winner for deployment.

## Experimental methodology
Fashion-MNIST from the official zalandoresearch GitHub mirror, using torchvision published MD5 resource checks. Small CNN with identical parameter shapes and initial weights across treatments per seed. Fixed data split seed 2026 and official test subset seed 2027; 3,000 train, 750 validation, 750 official test images. Three optimization seeds (42,43,44), eight epochs, batch 64, AdamW learning rate 0.001. Five predetermined treatments: baseline, dropout p=0.3, weight decay 0.01, random horizontal flip plus small affine augmentation, and combined. Select each checkpoint on validation loss. Evaluate training accuracy with dropout disabled and no augmentation, matching validation/test evaluation mode. Log initial parameter hashes, data indices, epoch losses/accuracies, runtime and paired accuracy differences. Pilot (1,000/250/250, three epochs) retained separately; the larger budget was an exploratory extension for compute coverage, not tuning treatment values to test results.

## Results
| Treatment | Test accuracy mean | Sample SD | Mean train–test gap |
|---|---:|---:|---:|
| baseline | 0.7507 | 0.0083 | 0.0173 |
| dropout | 0.7471 | 0.0198 | 0.0192 |
| weight_decay | 0.7493 | 0.0069 | 0.0181 |
| augmentation | 0.7222 | 0.0113 | 0.0091 |
| combined | 0.7209 | 0.0235 | 0.0106 |

Exact per-seed values, losses, selected epochs, timing and paired differences are in results/extended/metrics.json. Sample standard deviations use ddof=1. Pilot data are retained in results/cpu; no pilot numbers were replaced by extension results.

## Discussion
Accuracy differences must be interpreted alongside training accuracy, train–test gaps and learning curves. A smaller gap alone does not establish better generalization: it can arise from underfitting. Treatment effects here are descriptive. Identical initial parameter hashes isolate initialization but do not force stochastic optimization trajectories to remain identical after applying dropout/augmentation.

## Threats to validity
Small single-dataset study with three seeds and a fixed 750-image test subset. Sample SD describes optimization variation, not confidence in general population accuracy. No significance claims. A larger budget was selected after inspecting pilot metrics, so regard the extension as exploratory rather than preregistered. No hyperparameter search was conducted, but no untouched confirmatory benchmark remains for this budget. Data augmentation changes optimization noise and may slow convergence. Eight epochs can still underfit. These runs do not reproduce the cited papers. Full 48,000/12,000/10,000 runs and additional seeds remain pending.

## Conclusion
The implemented protocol supports reproducible ablations and reports both favorable and unfavorable outcomes. Confirmatory work should use more seeds, larger subsets and a newly specified evaluation protocol before interpreting differences as broad scientific evidence.

## References
Srivastava et al. (2014), *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR 15:1929–1958, https://jmlr.org/papers/v15/srivastava14a.html . Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization*, ICLR, arXiv:1711.05101. Simard, Steinkraus & Platt (2003), *Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis*, ICDAR, DOI 10.1109/ICDAR.2003.1227801. Zhang et al. (2017), *Understanding deep learning requires rethinking generalization*, ICLR, arXiv:1611.03530. Xiao, Rasul & Vollgraf (2017), *Fashion-MNIST*, arXiv:1708.07747; official repository https://github.com/zalandoresearch/fashion-mnist (MIT).
