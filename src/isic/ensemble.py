"""Ensembles por promedio de probabilidades softmax, desde las predicciones guardadas (sin GPU).

Uso: uv run python -m isic.ensemble
Escribe results/ensembles.json y results/ensembles.csv.
"""
import json
from pathlib import Path

import pandas as pd
import torch

from isic.evaluate import compute_metrics

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
# entre arquitecturas con semilla 0, y entre semillas de una misma arquitectura como control
ENSEMBLES = {
    "arquitecturas_seed0": ["resnet50_seed0", "efficientnet_b0_seed0", "swin_t_seed0"],
    "semillas_swin_t": ["swin_t_seed0", "swin_t_seed1", "swin_t_seed2"],
    "semillas_efficientnet_b0": ["efficientnet_b0_seed0", "efficientnet_b0_seed1", "efficientnet_b0_seed2"],
}
PROB_COLS = [f"prob_{c}" for c in CLASSES]


def mean_probs(runs, split, runs_dir="runs"):
    dfs = [pd.read_csv(Path(runs_dir) / r / f"preds_{split}.csv").sort_values("image").reset_index(drop=True)
           for r in runs]
    for df in dfs[1:]:
        assert (df["image"].values == dfs[0]["image"].values).all(), "las predicciones no tienen las mismas imágenes"
        assert (df["label"].values == dfs[0]["label"].values).all()
    probs = sum(torch.tensor(df[PROB_COLS].values, dtype=torch.float32) for df in dfs) / len(dfs)
    return probs, torch.tensor(dfs[0]["label"].values)


def main():
    out, rows = {}, []
    for name, runs in ENSEMBLES.items():
        out[name] = {"runs": runs}
        for split in ["test", "test_official"]:
            probs, labels = mean_probs(runs, split)
            m = compute_metrics(probs, labels, CLASSES)
            out[name][split] = m
            rows.append({"ensemble": name, "split": split,
                         **{k: m[k] for k in ["accuracy", "f1_macro", "f1_weighted", "balanced_accuracy",
                                              "auroc_macro"]}})
        print(f"{name}: F1 macro interno {out[name]['test']['f1_macro']:.4f}, "
              f"oficial {out[name]['test_official']['f1_macro']:.4f}")
    Path("results/ensembles.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    pd.DataFrame(rows).to_csv("results/ensembles.csv", index=False)
    print("Escritos results/ensembles.json y results/ensembles.csv")


if __name__ == "__main__":
    main()
