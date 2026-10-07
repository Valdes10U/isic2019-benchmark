"""Costo computacional de los tres modelos: parámetros, MACs, throughput y tiempos de entrenamiento.

Uso: uv run python -m isic.cost
Escribe results/cost.csv.
"""
import time
from pathlib import Path

import pandas as pd
import timm
import torch
from fvcore.nn import FlopCountAnalysis
from torchinfo import summary

from isic.train import load_config

MODELS = ["resnet50", "efficientnet_b0", "swin_t"]
SEEDS = [0, 1, 2]
NOTE = ("fvcore cuenta una multiplicación-acumulación como una operación: gmacs es su salida directa "
        "y gflops = 2 x gmacs. Lo que la literatura suele llamar FLOPs corresponde a gmacs.")


@torch.inference_mode()
def throughput(model, batch, dtype, warmup=10, iters=50):
    x = torch.randn(batch, 3, 224, 224, device="cuda")
    with torch.autocast("cuda", dtype=dtype):
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
    return batch * iters / (time.perf_counter() - start)


def training_times(key, runs_dir="runs"):
    # la época 1 incluye la calibración de cudnn.benchmark, así que se excluye de s/época;
    # el tiempo total sí suma todas las épocas
    hs = [pd.read_csv(Path(runs_dir) / f"{key}_seed{s}" / "history.csv") for s in SEEDS]
    per_epoch = pd.Series([h.loc[h["epoch"] > 1, "epoch_seconds"].mean() for h in hs])
    total = pd.Series([h["epoch_seconds"].sum() for h in hs])
    return {
        "epoch_s_mean": round(per_epoch.mean(), 1), "epoch_s_sd": round(per_epoch.std(), 1),
        "train_total_s_mean": round(total.mean(), 1), "train_total_s_sd": round(total.std(), 1),
        "peak_vram_mb": int(max(h["peak_vram_mb"].max() for h in hs)),
    }


def main():
    torch.backends.cudnn.benchmark = True
    rows = []
    for key in MODELS:
        cfg = load_config(f"configs/{key}.yaml")
        name, t = cfg["model"]["name"], cfg["train"]
        model = timm.create_model(name, pretrained=False, num_classes=cfg["data"]["num_classes"]).eval()

        info = summary(model, input_size=(1, 3, 224, 224), device="cpu", verbose=0)
        flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224))
        flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        macs = flops.total()
        unsupported = ";".join(sorted(flops.unsupported_ops()))

        model.cuda()
        ips = throughput(model, t["batch_size"], getattr(torch, t["amp_dtype"]))
        rows.append({
            "model": key, "timm_name": name,
            "params_total": info.total_params, "params_trainable": info.trainable_params,
            "gmacs": round(macs / 1e9, 3), "gflops": round(2 * macs / 1e9, 3),
            "fvcore_unsupported_ops": unsupported,
            "throughput_img_s": round(ips, 1), "latency_ms_img": round(1000 / ips, 3),
            **training_times(key),
            "nota": NOTE,
        })
        print({k: v for k, v in rows[-1].items() if k != "nota"})
        del model
        torch.cuda.empty_cache()

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("results/cost.csv", index=False)
    print("Escrito results/cost.csv")


if __name__ == "__main__":
    main()
