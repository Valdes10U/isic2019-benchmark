"""Entrenamiento con el protocolo común.

Uso: uv run python -m isic.train --config configs/resnet50.yaml --seed 0 [--smoke]
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from tqdm import tqdm

from isic.data import build_loaders, class_weights
from isic.models import build_model


def merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path):
    # base.yaml vive junto a la config del modelo y esta solo sobreescribe lo que cambia
    path = Path(path)
    cfg = yaml.safe_load((path.parent / "base.yaml").read_text())
    return merge(cfg, yaml.safe_load(path.read_text()) or {})


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return f


def train_one_epoch(model, loader, criterion, optimizer, scheduler, dtype):
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=dtype):
            loss = criterion(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def validate(model, loader, criterion, dtype, num_classes):
    model.eval()
    acc = MulticlassAccuracy(num_classes, average="micro").cuda()
    f1 = MulticlassF1Score(num_classes, average="macro").cuda()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=dtype):
            logits = model(x).float()
        total += criterion(logits, y).item() * len(y)
        n += len(y)
        acc.update(logits, y)
        f1.update(logits, y)
    return total / n, acc.compute().item(), f1.compute().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--class-weight", choices=["inverse", "sqrt_inverse", "none"], default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg["smoke_run"] = args.smoke
    if args.smoke:
        cfg["train"]["max_epochs"] = cfg["smoke"]["epochs"]
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.class_weight is not None:
        cfg["loss"]["class_weight"] = args.class_weight
    # train.loss es descriptiva; se ajusta para que config.yaml no contradiga los pesos usados
    cfg["train"]["loss"] = "cross_entropy" if cfg["loss"]["class_weight"] == "none" else "weighted_cross_entropy"
    tag = Path(args.config).stem
    name = args.run_name or (f"smoke_{tag}" if args.smoke else f"{tag}_seed{cfg['seed']}")
    out = Path(cfg["out_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    set_seed(cfg["seed"])
    t = cfg["train"]
    torch.backends.cudnn.benchmark = t["cudnn_benchmark"]
    dtype = getattr(torch, t["amp_dtype"])
    num_classes = cfg["data"]["num_classes"]

    train_loader, val_loader, _ = build_loaders(cfg, smoke=args.smoke)
    model = build_model(cfg["model"]["name"], num_classes).cuda()
    mode = cfg["loss"]["class_weight"]
    weights = class_weights(Path(cfg["data"]["splits_dir"]) / "train.csv", num_classes, mode)
    print(f"Pesos de clase ({mode}): {None if weights is None else [round(w, 3) for w in weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=None if weights is None else weights.cuda())
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    steps = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(t["warmup_epochs"] * steps, t["max_epochs"] * steps))

    print(f"{cfg['model']['name']} seed {cfg['seed']}: {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val, salida en {out}")
    best_f1, best_epoch, bad = -1.0, 0, 0
    fields = ["epoch", "train_loss", "val_loss", "val_acc", "val_f1_macro", "epoch_seconds", "peak_vram_mb"]
    with open(out / "history.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, t["max_epochs"] + 1):
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, dtype)
            val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, dtype, num_classes)
            torch.cuda.synchronize()
            row = dict(epoch=epoch, train_loss=round(train_loss, 5), val_loss=round(val_loss, 5),
                       val_acc=round(val_acc, 5), val_f1_macro=round(val_f1, 5),
                       epoch_seconds=round(time.time() - start, 1),
                       peak_vram_mb=round(torch.cuda.max_memory_allocated() / 2**20))
            writer.writerow(row)
            fh.flush()
            print(row)

            if val_f1 > best_f1:
                best_f1, best_epoch, bad = val_f1, epoch, 0
                torch.save({"model": model.state_dict(), "model_name": cfg["model"]["name"],
                            "epoch": epoch, "val_f1_macro": val_f1}, out / "best.pt")
            else:
                bad += 1
                if bad >= t["patience"]:
                    print(f"Early stopping en la época {epoch}")
                    break
    print(f"Mejor F1 macro de val: {best_f1:.4f} en la época {best_epoch}")


if __name__ == "__main__":
    main()
