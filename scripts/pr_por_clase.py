"""Precisión y recall por clase en el test interno y el oficial, desde la matriz de confusión de metrics.json.

Uso: uv run python scripts/pr_por_clase.py
Escribe results/pr_por_clase.csv con media y desviación estándar muestral entre las semillas 0, 1 y 2.
"""
import csv
import json
import statistics
from pathlib import Path

MODELS = ["resnet50", "efficientnet_b0", "swin_t"]
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]


def pr_from_confusion(cm):
    # filas = clase real, columnas = clase predicha (MulticlassConfusionMatrix de torchmetrics)
    pr = {}
    for k, c in enumerate(CLASSES):
        tp, real, pred = cm[k][k], sum(cm[k]), sum(row[k] for row in cm)
        pr[c] = (tp / pred if pred else 0.0, tp / real if real else 0.0)
    return pr


rows = []
for m in MODELS:
    for split in ["test", "test_official"]:
        per_seed = []
        for s in [0, 1, 2]:
            res = json.loads(Path(f"runs/{m}_seed{s}/metrics.json").read_text())[split]
            pr = pr_from_confusion(res["confusion_matrix"])
            # control: el F1 recalculado debe coincidir con el que guardó evaluate.py
            for c, (p, r) in pr.items():
                f1 = 2 * p * r / (p + r) if p + r else 0.0
                assert abs(f1 - res["f1_per_class"][c]) < 1e-3, (m, s, split, c, f1)
            per_seed.append(pr)
        for c in CLASSES:
            p = [x[c][0] for x in per_seed]
            r = [x[c][1] for x in per_seed]
            rows.append({"model": m, "split": split, "class": c,
                         "precision_mean": round(statistics.mean(p), 5), "precision_sd": round(statistics.stdev(p), 5),
                         "recall_mean": round(statistics.mean(r), 5), "recall_sd": round(statistics.stdev(r), 5)})

out = Path("results/pr_por_clase.csv")
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{out}: {len(rows)} filas")
