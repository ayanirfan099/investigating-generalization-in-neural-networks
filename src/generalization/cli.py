import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .core import CNN, epoch, loaders, seed_all

TREATMENTS = {
    "baseline": {"dropout": 0.0, "weight_decay": 0.0, "augment": False},
    "dropout": {"dropout": 0.3, "weight_decay": 0.0, "augment": False},
    "weight_decay": {"dropout": 0.0, "weight_decay": 0.01, "augment": False},
    "augmentation": {"dropout": 0.0, "weight_decay": 0.0, "augment": True},
    "combined": {"dropout": 0.3, "weight_decay": 0.01, "augment": True},
}


def experiment(
    output="results/cpu",
    data="data",
    epochs=3,
    limit=1000,
    seeds=(42, 43, 44),
    synthetic=False,
    treatments=None,
):
    if epochs < 1 or not seeds:
        raise ValueError("Positive epoch budget and nonempty seeds required")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    treatments = treatments or list(TREATMENTS)
    if set(treatments) - set(TREATMENTS):
        raise ValueError("Unknown treatment")
    config = {
        "epochs": epochs,
        "limit": limit,
        "seeds": list(seeds),
        "synthetic": synthetic,
        "split_seed": 2026,
        "test_subset_seed": 2027,
        "batch_size": 64,
        "learning_rate": 0.001,
        "treatments": {k: TREATMENTS[k] for k in treatments},
    }
    splits = None
    for seed in seeds:
        for name in treatments:
            seed_all(seed)
            setting = TREATMENTS[name]
            model = CNN(setting["dropout"])
            initial_hash = hashlib.sha256(
                b"".join(p.detach().numpy().tobytes() for p in model.parameters())
            ).hexdigest()
            (train, clean, val, test), ids = loaders(
                data, seed, limit, setting["augment"], synthetic
            )
            if splits is not None and ids != splits:
                raise RuntimeError("Data split changed across treatments")
            splits = ids
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.001, weight_decay=setting["weight_decay"]
            )
            history = []
            best = float("inf")
            path = out / f"{name}-{seed}.pt"
            start = perf_counter()
            for step in range(epochs):
                optimization = epoch(model, train, optimizer)
                tr = epoch(model, clean)
                va = epoch(model, val)
                history.append(
                    {
                        "epoch": step + 1,
                        "optimization": optimization,
                        "train_eval": tr,
                        "validation": va,
                    }
                )
                if va["loss"] < best:
                    best = va["loss"]
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "dropout": setting["dropout"],
                            "seed": seed,
                            "treatment": name,
                            "epoch": step + 1,
                        },
                        path,
                    )
            seconds = perf_counter() - start
            checkpoint = torch.load(path, weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
            tr = epoch(model, clean)
            va = epoch(model, val)
            te = epoch(model, test)
            record = {
                "treatment": name,
                "seed": seed,
                "initial_parameter_sha256": initial_hash,
                "selected_epoch": checkpoint["epoch"],
                "train": tr,
                "validation": va,
                "test": te,
                "generalization_gap_train_minus_test": tr["accuracy"] - te["accuracy"],
                "training_seconds": seconds,
                "history": history,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            records.append(record)
            (out / "checkpoint-progress.json").write_text(
                json.dumps({"config": config, "records": records}, indent=2)
            )
    summary = {}
    for name in treatments:
        rows = [r for r in records if r["treatment"] == name]
        values = [r["test"]["accuracy"] for r in rows]
        summary[name] = {
            "runs": len(values),
            "test_accuracy_mean": float(np.mean(values)),
            "test_accuracy_sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "generalization_gap_mean": float(
                np.mean([r["generalization_gap_train_minus_test"] for r in rows])
            ),
            "training_seconds_total": sum(r["training_seconds"] for r in rows),
        }
    baseline = {r["seed"]: r["test"]["accuracy"] for r in records if r["treatment"] == "baseline"}
    for name in treatments:
        if name != "baseline" and baseline:
            differences = [
                r["test"]["accuracy"] - baseline[r["seed"]]
                for r in records
                if r["treatment"] == name
            ]
            summary[name]["paired_test_accuracy_delta_vs_baseline"] = differences
    result = {
        "dataset": "synthetic software smoke only"
        if synthetic
        else "Fashion-MNIST official upstream, verified torchvision resource MD5",
        "config": config,
        "summary": summary,
        "records": records,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / "splits.json").write_text(json.dumps(splits))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    names = list(summary)
    ax.bar(
        names,
        [summary[n]["test_accuracy_mean"] for n in names],
        yerr=[summary[n]["test_accuracy_sample_std"] or 0 for n in names],
        capsize=3,
    )
    ax.set(ylabel="Test accuracy (mean ± sample SD)", title=result["dataset"])
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "comparison.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    for name in names:
        row = next(r for r in records if r["treatment"] == name)
        ax.plot([h["validation"]["loss"] for h in row["history"]], label=name)
    ax.set(
        xlabel="Epoch (0-based)", ylabel="Validation cross entropy", title=f"First seed {seeds[0]}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png")
    plt.close(fig)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/cpu")
    p.add_argument("--data", default="data")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--treatments", nargs="+", choices=list(TREATMENTS))
    result = experiment(**vars(p.parse_args()))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
