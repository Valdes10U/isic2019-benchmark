# Tiempos de la prueba de reproducibilidad (Fase 5)

Medidos el 14 de octubre de 2026 en la RTX 5070 Ti Laptop (WSL2). Clon limpio del repositorio, caché de uv vacío, datos y `img256` ya existentes en `~/data/isic2019/`.

| Paso | Comando | Tiempo |
|---|---|---|
| Clonar (desde el repo local) | `git clone` | 0.1 s |
| Instalación en frío | `uv sync` | 379.2 s (6 min 19 s) |
| Verificar GPU | `scripts/check_gpu.py` | 11.4 s |
| Splits | `scripts/make_splits.py` | 7.5 s |
| Smoke test ResNet50 | `isic.train --config configs/resnet50.yaml --seed 0 --smoke` | 48.3 s |
| Smoke test EfficientNet-B0 | `isic.train --config configs/efficientnet_b0.yaml --seed 0 --smoke` | 38.7 s |
| Smoke test Swin-T | `isic.train --config configs/swin_t.yaml --seed 0 --smoke` | 8.5 s |
| F1 de train de las 9 corridas | `isic.evaluate --all --split train` | 113.7 s |
| Costo | `isic.cost` | 27.6 s |
| Ensembles | `isic.ensemble` | 3.4 s |
| Grad-CAM | `isic.gradcam` | 12.2 s |
| Precisión y recall por clase | `scripts/pr_por_clase.py` | 0.1 s |
| Tablas y figuras | `scripts/make_figures.py` | 2.3 s |

Los pasos del 1 al 7 se corrieron en el clon limpio. Los comandos del paso 10 se corrieron en `~/isic2019-benchmark`, con los checkpoints de `runs/`. Después se restauró `results/` y `figures/` con `git checkout`, porque `cost.csv` y lo que depende de él cambian con cada medición. No se probó `preprocess.py` (escribiría en `~/data/isic2019/img256`), ni la descarga y el `unzip` de los datos, ni el clon desde GitHub.

## Arranque en frío de los smoke tests

Los tiempos de los smoke tests no son comparables entre sí. El primero en correr paga el arranque: la primera iteración de ResNet50 tardó 24 s y la de EfficientNet-B0 15 s, por el warmup de cudnn (`cudnn.benchmark = True`) y la lectura de disco sin caché. Swin-T corrió al final con todo caliente y su primera iteración tardó 1.9 s. Un primer smoke test en una máquina nueva toma unos 40 a 50 s.

## VRAM pico de los smoke tests

| Modelo | VRAM pico |
|---|---|
| ResNet50 | 1800 MB |
| EfficientNet-B0 | 1478 MB |
| Swin-T | 2672 MB |

## Splits

Tras correr `make_splits.py` en el clon limpio, `git diff --stat results/splits` dio salida vacía: la partición se reproduce idéntica (17,735 train, 3,798 val, 3,798 test, 6,191 en el test oficial sin UNK).
