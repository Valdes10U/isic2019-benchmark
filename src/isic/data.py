from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class ISICDataset(Dataset):
    """Recibe la ruta del CSV del split (o un DataFrame con columnas image y label)."""

    def __init__(self, csv, img_dir, transform):
        df = csv if isinstance(csv, pd.DataFrame) else pd.read_csv(csv)
        self.images = df["image"].tolist()
        self.labels = df["label"].tolist()
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.img_dir / f"{self.images[i]}.jpg").convert("RGB")
        return self.transform(img), self.labels[i]


def build_transforms(train: bool, cfg):
    d, a = cfg["data"], cfg["augment"]
    norm = [T.ToTensor(), T.Normalize(d["mean"], d["std"])]
    if not train:
        return T.Compose([T.Resize(d["resize"]), T.CenterCrop(d["crop"]), *norm])
    return T.Compose([
        T.RandomResizedCrop(d["crop"], scale=tuple(a["crop_scale"])),
        T.RandomHorizontalFlip(a["hflip"]),
        T.RandomVerticalFlip(a["vflip"]),
        T.RandomRotation(a["rotation"]),
        T.ColorJitter(brightness=a["brightness"], contrast=a["contrast"]),
        *norm,
    ])


def class_weights(train_csv, num_classes=8, mode="inverse"):
    # inverse: 1/frecuencia; sqrt_inverse: 1/raíz de la frecuencia; ambos normalizados a media 1
    if mode == "none":
        return None
    counts = np.bincount(pd.read_csv(train_csv)["label"], minlength=num_classes)
    if mode == "inverse":
        w = 1.0 / counts
    elif mode == "sqrt_inverse":
        w = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Modo de pesos desconocido: {mode}")
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def build_loaders(cfg, smoke=False):
    d, t = cfg["data"], cfg["train"]
    splits = {sp: pd.read_csv(Path(d["splits_dir"]) / f"{sp}.csv") for sp in ["train", "val", "test"]}
    if smoke:
        splits["train"] = splits["train"].sample(n=cfg["smoke"]["n_train"], random_state=cfg["seed"])
        splits["val"] = splits["val"].sample(n=cfg["smoke"]["n_val"], random_state=cfg["seed"])

    def loader(sp, train):
        ds = ISICDataset(splits[sp], d["train_dir"], build_transforms(train, cfg))
        return DataLoader(ds, batch_size=t["batch_size"], shuffle=train, num_workers=t["num_workers"],
                          pin_memory=t["pin_memory"], persistent_workers=t["num_workers"] > 0)

    return loader("train", True), loader("val", False), loader("test", False)
