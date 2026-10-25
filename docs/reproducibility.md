# Reproduce

Python >=3.11; executed in Python 3.12 on CPU. From repository root:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m generalization.cli --data data --limit 1000 --epochs 3 --seeds 42 43 44 --output results/cpu
python -m generalization.cli --data data --limit 3000 --epochs 8 --seeds 42 43 44 --output results/extended
# Optional full benchmark:
python -m generalization.cli --data data --limit 48000 --epochs 30 --seeds 42 43 44 45 46 --output results/full
pytest -q
ruff check .
ruff format --check .
```

Use CPU PyTorch wheels from https://download.pytorch.org/whl/cpu if installing the deep learning dependencies on a CPU machine. Remote datasets and weights require separate access. Do not load untrusted pickle/joblib/checkpoint artifacts.
