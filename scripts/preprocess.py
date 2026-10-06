"""Redimensiona ISIC 2019 a 256x256 (resize directo, sin recorte) y guarda JPEG calidad 95.

Uso: uv run python scripts/preprocess.py
"""
import os
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

DATA = Path(__file__).resolve().parents[1] / "data"
JOBS = [
    (DATA / "ISIC_2019_Training_Input", DATA / "img256" / "train"),
    (DATA / "ISIC_2019_Test_Input", DATA / "img256" / "test"),
]
SIZE = 256


def process(paths):
    src, dst = paths
    if dst.exists():
        return 0
    # se escribe a un temporal y se renombra, así un corte a medias no deja archivos truncados
    tmp = dst.with_suffix(".tmp")
    with Image.open(src) as img:
        img.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC).save(tmp, "JPEG", quality=95)
    os.replace(tmp, dst)
    return 1


def main():
    tasks = []
    for src_dir, dst_dir in JOBS:
        dst_dir.mkdir(parents=True, exist_ok=True)
        tasks += [(p, dst_dir / p.name) for p in sorted(src_dir.glob("*.jpg"))]
    with Pool(os.cpu_count()) as pool:
        written = sum(pool.imap_unordered(process, tasks, chunksize=64))
    print(f"Escritas {written}, ya existían {len(tasks) - written}")
    for _, dst_dir in JOBS:
        print(f"{dst_dir}: {len(list(dst_dir.glob('*.jpg')))} imágenes")


if __name__ == "__main__":
    main()
