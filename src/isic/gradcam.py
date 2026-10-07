"""Grad-CAM de los tres modelos (semilla 0) sobre un conjunto fijo de 16 imágenes del test interno.

Uso: uv run python -m isic.gradcam
Escribe results/gradcam_images.csv, figures/gradcam/*.png y figures/gradcam_grid_{1,2}.{pdf,png}.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms as T

from isic.data import build_transforms
from isic.evaluate import load_model
from isic.models import gradcam_target_layer
from isic.plotstyle import CLASSES, MODELS, NAMES, TWO_COL, plt, save, setup

OUT = Path("figures/gradcam")


def select_images(runs_dir="runs", seed=42):
    """2 por clase: una bien y una mal clasificada por ResNet50 semilla 0."""
    df = pd.read_csv(Path(runs_dir) / "resnet50_seed0" / "preds_test.csv")
    rng = random.Random(seed)
    rows = []
    for k, c in enumerate(CLASSES):
        sub = df[df["label"] == k]
        for kind, mask in [("bien", sub["pred"] == k), ("mal", sub["pred"] != k)]:
            rows.append({"image": rng.choice(sorted(sub.loc[mask, "image"])), "label": c, "resnet50": kind})
    sel = pd.DataFrame(rows).rename(columns={"resnet50": "seleccion_resnet50"})
    for m in MODELS:
        preds = pd.read_csv(Path(runs_dir) / f"{m}_seed0" / "preds_test.csv").set_index("image")["pred"]
        sel[f"pred_{m}"] = [CLASSES[preds[i]] for i in sel["image"]]
    return sel


def compute_cams(sel, runs_dir="runs"):
    cams = {}
    for m in MODELS:
        run = Path(runs_dir) / f"{m}_seed0"
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        model = load_model(run, cfg)
        layer, reshape = gradcam_target_layer(model, cfg["model"]["name"])
        tf = build_transforms(False, cfg)
        x = torch.stack([tf(Image.open(Path(cfg["data"]["train_dir"]) / f"{i}.jpg").convert("RGB"))
                         for i in sel["image"]]).cuda()
        targets = [ClassifierOutputTarget(CLASSES.index(c)) for c in sel["label"]]
        # Grad-CAM en fp32, con la clase real como objetivo
        with GradCAM(model=model, target_layers=[layer], reshape_transform=reshape) as cam:
            cams[m] = cam(input_tensor=x, targets=targets)
        del model
        torch.cuda.empty_cache()
    return cams


def originals(sel, cfg):
    d = cfg["data"]
    crop = T.Compose([T.Resize(d["resize"]), T.CenterCrop(d["crop"])])
    return [np.asarray(crop(Image.open(Path(d["train_dir"]) / f"{i}.jpg").convert("RGB")), dtype=np.float32) / 255
            for i in sel["image"]]


def plot_grid(sel, imgs, overlays, path):
    n = len(sel)
    cell = 1.06  # pulgadas por miniatura; 8 filas caben en el alto de una página IEEE
    text_w = TWO_COL - 4 * cell - 0.2
    fig = plt.figure(figsize=(TWO_COL, n * cell + 0.25))
    gs = fig.add_gridspec(n, 5, width_ratios=[text_w] + [cell] * 4, wspace=0.04, hspace=0.06,
                          left=0, right=1, bottom=0, top=1 - 0.25 / (n * cell + 0.25))
    titles = ["Original"] + [NAMES[m] for m in MODELS]
    for r, (_, row) in enumerate(sel.iterrows()):
        ax = fig.add_subplot(gs[r, 0])
        ax.axis("off")
        lines = [f"Real: {row['label']}", f"({row['seleccion_resnet50']} clasificada por ResNet50)"]
        for m in MODELS:
            p = row[f"pred_{m}"]
            lines.append(f"{NAMES[m]}: {p} {'✓' if p == row['label'] else '✗'}")
        ax.text(0.97, 0.5, lines[0], transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                fontweight="bold")
        ax.text(0.97, 0.47, "\n".join(lines[1:]), transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                linespacing=1.3)
        ax.text(0.97, 0.53 + 0.17, row["image"], transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5,
                color="#555555")
        for j, im in enumerate([imgs[r]] + [overlays[m][r] for m in MODELS]):
            ax = fig.add_subplot(gs[r, j + 1])
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            wrong = j > 0 and row[f"pred_{MODELS[j - 1]}"] != row["label"]
            for s in ax.spines.values():
                s.set_visible(wrong)
                s.set_linewidth(2.0)
                s.set_color("black")
            if wrong:
                ax.text(0.03, 0.04, f"✗ {row[f'pred_{MODELS[j - 1]}']}", transform=ax.transAxes, fontsize=6.5,
                        fontweight="bold", va="bottom",
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
            if r == 0:
                ax.set_title(titles[j], fontsize=8, pad=3)
    save(fig, path)


def main():
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    sel = select_images()
    sel.to_csv("results/gradcam_images.csv", index=False)
    cfg = yaml.safe_load(Path("runs/resnet50_seed0/config.yaml").read_text())
    imgs = originals(sel, cfg)
    cams = compute_cams(sel)
    overlays = {m: [show_cam_on_image(imgs[i], cams[m][i], use_rgb=True) for i in range(len(sel))] for m in MODELS}
    for i, name in enumerate(sel["image"]):
        Image.fromarray((imgs[i] * 255).astype(np.uint8)).save(OUT / f"{name}_original.png")
        for m in MODELS:
            Image.fromarray(overlays[m][i]).save(OUT / f"{name}_{m}.png")
    half = len(sel) // 2
    for k, part in enumerate([slice(0, half), slice(half, None)], start=1):
        s = sel.iloc[part].reset_index(drop=True)
        plot_grid(s, imgs[part], {m: overlays[m][part] for m in MODELS}, f"figures/gradcam_grid_{k}")
    print(sel.to_string(index=False))


if __name__ == "__main__":
    main()
