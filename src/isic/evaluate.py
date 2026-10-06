"""Evaluación del mejor checkpoint en el test interno y en el test oficial (sin UNK).

Uso: uv run python -m isic.evaluate --run runs/resnet50_seed0
     uv run python -m isic.evaluate --all
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import timm
import torch
import yaml
from torch.utils.data import DataLoader
from torchmetrics.classification import (MulticlassAccuracy, MulticlassAUROC, MulticlassConfusionMatrix,
                                         MulticlassF1Score, MulticlassRecall)
from tqdm import tqdm

from isic.data import ISICDataset, build_transforms


@torch.no_grad()
def predict(model, csv, img_dir, cfg):
    df = pd.read_csv(csv)
    ds = ISICDataset(df, img_dir, build_transforms(False, cfg))
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["train"]["num_workers"], pin_memory=True)
    dtype = getattr(torch, cfg["train"]["amp_dtype"])
    probs = []
    for x, _ in tqdm(loader, desc=Path(csv).stem, leave=False):
        with torch.autocast("cuda", dtype=dtype):
            logits = model(x.cuda(non_blocking=True))
        probs.append(logits.float().softmax(1).cpu())
    return df, torch.cat(probs)


def compute_metrics(probs, labels, classes):
    n = len(classes)
    m = {
        "accuracy": MulticlassAccuracy(n, average="micro"),
        "f1_macro": MulticlassF1Score(n, average="macro"),
        "f1_weighted": MulticlassF1Score(n, average="weighted"),
        "balanced_accuracy": MulticlassRecall(n, average="macro"),
        "auroc_macro": MulticlassAUROC(n, average="macro"),
    }
    out = {k: round(f(probs, labels).item(), 5) for k, f in m.items()}
    f1_cls = MulticlassF1Score(n, average=None)(probs, labels)
    out["f1_per_class"] = {c: round(v, 5) for c, v in zip(classes, f1_cls.tolist())}
    out["confusion_matrix"] = MulticlassConfusionMatrix(n)(probs, labels).tolist()
    out["n_images"] = len(labels)
    return out


def training_summary(history_csv):
    h = pd.read_csv(history_csv)
    best = h.loc[h["val_f1_macro"].idxmax()]
    return {
        "best_epoch": int(best["epoch"]),
        "best_val_f1_macro": float(best["val_f1_macro"]),
        "epochs_run": len(h),
        "total_seconds": round(float(h["epoch_seconds"].sum()), 1),
        "mean_epoch_seconds": round(float(h["epoch_seconds"].mean()), 1),
        "peak_vram_mb": int(h["peak_vram_mb"].max()),
    }


def evaluate_run(run):
    run = Path(run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    d, classes = cfg["data"], cfg["data"]["classes"]
    ckpt = torch.load(run / "best.pt", map_location="cpu")
    # sin pesos de ImageNet: se cargan los del checkpoint
    model = timm.create_model(cfg["model"]["name"], pretrained=False, num_classes=d["num_classes"])
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()

    splits = Path(d["splits_dir"])
    sets = {"test": (splits / "test.csv", d["train_dir"]),
            "test_official": (splits / "test_official.csv", d["test_dir"])}
    result = {"run": run.name, "model": cfg["model"]["name"], "seed": cfg["seed"],
              "checkpoint_epoch": ckpt["epoch"]}
    for name, (csv, img_dir) in sets.items():
        df, probs = predict(model, csv, img_dir, cfg)
        labels = torch.tensor(df["label"].values)
        result[name] = compute_metrics(probs, labels, classes)
        preds = pd.DataFrame(probs.numpy(), columns=[f"prob_{c}" for c in classes]).round(6)
        preds.insert(0, "pred", probs.argmax(1).numpy())
        preds.insert(0, "label", df["label"].values)
        preds.insert(0, "image", df["image"].values)
        preds.to_csv(run / f"preds_{name}.csv", index=False)

    gt = pd.read_csv(Path(d["root"]) / "ISIC_2019_Test_GroundTruth.csv")
    result["test_official"]["n_unk_excluded"] = int((gt["UNK"] == 1).sum())
    result["training"] = training_summary(run / "history.csv")
    (run / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    t, o = result["test"], result["test_official"]
    print(f"{run.name}: F1 macro test interno {t['f1_macro']:.4f}, test oficial {o['f1_macro']:.4f} "
          f"(UNK excluidas: {o['n_unk_excluded']})")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--runs_dir", default="runs")
    args = ap.parse_args()

    if args.run:
        evaluate_run(args.run)
        return
    # los runs de smoke no forman parte de los resultados
    pending = [p.parent for p in sorted(Path(args.runs_dir).glob("*/best.pt"))
               if not p.parent.name.startswith("smoke_") and not (p.parent / "metrics.json").exists()]
    print(f"Runs por evaluar: {[p.name for p in pending]}")
    for run in pending:
        evaluate_run(run)


if __name__ == "__main__":
    main()
