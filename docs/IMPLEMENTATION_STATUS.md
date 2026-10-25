# Implementation status

Implemented but not fully benchmarked

Controlled CPU experiments on dropout, weight decay and image augmentation using Fashion-MNIST, repeated seeds, explicit split isolation and genuinely measured learning curves.

Four tests passed: identical initial parameter values across dropout configurations, deterministic fixed splits across seeds/treatments, invalid parameter/data bounds, repeated CPU training metric equality, checkpoint reload and CLI help. Tests use generated fixtures; committed research results use real Fashion-MNIST.

Small single-dataset study with three seeds and a fixed 750-image test subset. Sample SD describes optimization variation, not confidence in general population accuracy. No significance claims. A larger budget was selected after inspecting pilot metrics, so regard the extension as exploratory rather than preregistered. No hyperparameter search was conducted, but no untouched confirmatory benchmark remains for this budget. Data augmentation changes optimization noise and may slow convergence. Eight epochs can still underfit. These runs do not reproduce the cited papers. Full 48,000/12,000/10,000 runs and additional seeds remain pending.

Exact commands: [reproducibility](reproducibility.md). Push status and commit identity are recorded in the portfolio summary; remote CI success is not implied.

LaTeX manuscript compiled successfully with pdflatex; the generated PDF is included in docs/research_report.pdf.
