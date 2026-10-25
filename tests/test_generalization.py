import subprocess
import sys

import pytest
import torch

from generalization.cli import experiment
from generalization.core import CNN, loaders, seed_all


def test_architecture_and_seed():
    seed_all(42)
    a = CNN(0.3)
    seed_all(42)
    b = CNN(0)
    for x, y in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(x, y)
    assert a(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    with pytest.raises(ValueError):
        CNN(1)


def test_fixed_splits_and_validation():
    _, a = loaders(".", 42, 30, False, True)
    _, b = loaders(".", 43, 30, True, True)
    assert a == b and not set(a["train"]) & set(a["validation"])
    with pytest.raises(ValueError):
        loaders(".", limit=0)


def test_controlled_experiment(tmp_path):
    a = experiment(
        str(tmp_path / "a"),
        epochs=1,
        limit=30,
        seeds=[42],
        synthetic=True,
        treatments=["baseline", "dropout"],
    )
    b = experiment(
        str(tmp_path / "b"),
        epochs=1,
        limit=30,
        seeds=[42],
        synthetic=True,
        treatments=["baseline", "dropout"],
    )
    assert a["summary"]["baseline"]["test_accuracy_sample_std"] is None
    assert (
        a["records"][0]["initial_parameter_sha256"] == a["records"][1]["initial_parameter_sha256"]
    )
    assert a["records"][0]["test"] == b["records"][0]["test"]
    assert (tmp_path / "a/comparison.png").exists()
    checkpoint = torch.load(tmp_path / "a/baseline-42.pt", weights_only=True)
    model = CNN(checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    assert torch.isfinite(model(torch.zeros(1, 1, 28, 28))).all()


def test_cli():
    assert (
        subprocess.run(
            [sys.executable, "-m", "generalization.cli", "--help"], check=False, capture_output=True
        ).returncode
        == 0
    )
