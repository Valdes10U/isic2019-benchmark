"""Partición 70/15/15 estratificada por clase y agrupada por lesion_id, más el test oficial sin UNK.

Uso: uv run python scripts/make_splits.py
"""
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((REPO / "configs" / "base.yaml").read_text())
CLASSES = cfg["data"]["classes"]
DATA = REPO / cfg["data"]["root"]
OUT = REPO / cfg["data"]["splits_dir"]
SPLITS = ["train", "val", "test"]

lines = []


def log(s=""):
    print(s)
    lines.append(s)


def to_labels(gt):
    onehot = gt[CLASSES].to_numpy()
    assert (onehot.sum(1) == 1).all(), "hay filas sin exactamente una clase"
    return onehot.argmax(1)


def resolution_sample(folder, n=20):
    files = sorted(folder.glob("*.jpg"))
    sample = random.Random(cfg["split"]["seed"]).sample(files, n)
    sizes = Counter(Image.open(f).size for f in sample)
    return ", ".join(f"{w}x{h}: {c}" for (w, h), c in sizes.most_common())


def main():
    gt = pd.read_csv(DATA / "ISIC_2019_Training_GroundTruth.csv")
    md = pd.read_csv(DATA / "ISIC_2019_Training_Metadata.csv")
    te = pd.read_csv(DATA / "ISIC_2019_Test_GroundTruth.csv")
    log("Filas por CSV")
    log(f"  Training_GroundTruth: {len(gt)}")
    log(f"  Training_Metadata:    {len(md)}")
    log(f"  Test_GroundTruth:     {len(te)}")

    unk_train = int((gt["UNK"] == 1).sum())
    gt = gt[gt["UNK"] != 1]
    df = pd.DataFrame({"image": gt["image"], "label": to_labels(gt)})
    df = df.merge(md[["image", "lesion_id"]], on="image", how="left", validate="one_to_one")
    n_lesion = int(df["lesion_id"].notna().sum())
    n_unique = df["lesion_id"].nunique()
    # sin lesion_id, cada imagen es su propio grupo
    df["lesion_id"] = df["lesion_id"].fillna(df["image"])
    log(f"\nUNK descartadas en train: {unk_train}")
    log(f"Imágenes con lesion_id: {n_lesion} de {len(df)} ({100 * n_lesion / len(df):.1f}%), "
        f"lesiones únicas: {n_unique}, imágenes sin lesion_id (grupo propio): {len(df) - n_lesion}")

    s = cfg["split"]
    sgkf = StratifiedGroupKFold(n_splits=s["folds"], shuffle=True, random_state=s["seed"])
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label"], df["lesion_id"])):
        fold[idx] = k
    n_tr, n_va = s["train_folds"], s["val_folds"]
    df["split"] = np.where(fold < n_tr, "train", np.where(fold < n_tr + n_va, "val", "test"))

    groups = {sp: set(df.loc[df["split"] == sp, "lesion_id"]) for sp in SPLITS}
    assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]), \
        "hay lesion_id compartidos entre splits"
    log("Assert OK: ningún lesion_id aparece en dos splits")

    OUT.mkdir(parents=True, exist_ok=True)
    for sp in SPLITS:
        df.loc[df["split"] == sp, ["image", "label", "lesion_id"]].to_csv(OUT / f"{sp}.csv", index=False)

    unk_test = int((te["UNK"] == 1).sum())
    te = te[te["UNK"] != 1]
    off = pd.DataFrame({"image": te["image"], "label": to_labels(te)})
    off.to_csv(OUT / "test_official.csv", index=False)
    log(f"\nUNK descartadas en test oficial: {unk_test}")
    log(f"Evaluación externa (test oficial sin UNK): {len(off)} imágenes")

    # verifica que todas las imágenes de los splits existen ya preprocesadas
    for name, frame, folder in [("train", df, cfg["data"]["train_dir"]), ("test_official", off, cfg["data"]["test_dir"])]:
        missing = [i for i in frame["image"] if not (REPO / folder / f"{i}.jpg").exists()]
        assert not missing, f"faltan {len(missing)} imágenes de {name} en {folder}"

    counts = pd.DataFrame({"total": df["label"].value_counts()})
    for sp in SPLITS:
        counts[sp] = df.loc[df["split"] == sp, "label"].value_counts()
    counts["test_official"] = off["label"].value_counts()
    counts = counts.reindex(range(len(CLASSES))).fillna(0).astype(int)
    counts.index = pd.Index(CLASSES, name="class")
    counts.loc["TOTAL"] = counts.sum()
    counts.to_csv(OUT / "class_counts.csv")

    pct = counts.drop("TOTAL") / counts.drop("TOTAL").sum() * 100
    table = counts.astype(str)
    for c in counts.columns:
        table[c] = [f"{n} ({p:.1f}%)" for n, p in zip(counts[c], list(pct[c]) + [100.0])]
    log("\nConteo por clase (porcentaje dentro de cada columna)")
    log(table.to_string())
    sizes = counts.loc["TOTAL", SPLITS]
    log("Tamaño de cada split respecto al total: " + ", ".join(
        f"{sp} {n} ({100 * n / sizes.sum():.1f}%)" for sp, n in sizes.items()))

    log("\nResoluciones originales en una muestra de 20 imágenes por conjunto (semilla 42)")
    log(f"  train: {resolution_sample(DATA / 'ISIC_2019_Training_Input')}")
    log(f"  test oficial: {resolution_sample(DATA / 'ISIC_2019_Test_Input')}")

    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
