# isic2019-benchmark

Español: [README.es.md](README.es.md)

Comparison of ResNet50, EfficientNet-B0 and Swin-T for skin lesion classification on ISIC 2019, with the same training and evaluation protocol for all three models.

## Main results

Mean ± standard deviation over seeds 0, 1 and 2, in percent. The internal test set is the 15% hold-out of our own split (3,798 images) and the official test set is the ISIC 2019 test set without the UNK class (6,191 images). The eight classes are melanoma (MEL), melanocytic nevus (NV), basal cell carcinoma (BCC), actinic keratosis (AK), benign keratosis (BKL), dermatofibroma (DF), vascular lesion (VASC) and squamous cell carcinoma (SCC).

| Model | Macro F1, internal | Macro F1, official | Balanced accuracy, internal | Balanced accuracy, official | Parameters (M) | GMACs | Inference (img/s) |
|---|---|---|---|---|---|---|---|
| ResNet50 | 52.0 ± 1.4 | 51.6 ± 0.9 | 56.6 ± 1.7 | 52.8 ± 1.1 | 23.5 | 4.1 | 1770 |
| EfficientNet-B0 | 53.2 ± 0.5 | 53.0 ± 1.2 | 54.1 ± 0.9 | 52.4 ± 2.1 | 4.0 | 0.4 | 3096 |
| Swin-T | 58.9 ± 1.5 | 58.5 ± 1.5 | 57.7 ± 0.9 | 55.6 ± 1.7 | 27.5 | 4.5 | 1189 |

The numbers come from `results/tables/tabla1_resultados.csv` and `results/tables/tabla2_costo.csv`. Inference was measured with batch 32 on an RTX 5070 Ti Laptop GPU. Figure labels are in Spanish, like the paper.

![Internal macro F1 against parameters and seconds per epoch](figures/fig5_f1_vs_costo.png)

Internal macro F1 against the number of parameters and the seconds per epoch (bars show the standard deviation across seeds).

![Grad-CAM of the three models on eight internal test images](figures/gradcam_grid_1.png)

Grad-CAM of each model's seed 0 on two images from each of the classes MEL, NV, BCC and AK (one classified correctly and one misclassified by ResNet50). The other four classes are in `figures/gradcam_grid_2.png`.

Swin-T has the best macro F1 on both test sets (58.9 internal and 58.5 official, against 52.0 and 53.2 for the CNNs) and wins in 7 of the 8 classes, at the cost of the slowest inference (1189 img/s) and 46.1 s per epoch. EfficientNet-B0 reaches 53.2 with 4.0 M parameters and 0.4 GMACs, about a tenth of the compute of the other two, and the ensemble of its three seeds rises to 56.7. Moving from the internal to the official test set costs less than 0.5 pp of macro F1 in all three models, but 1.7 to 3.8 pp of balanced accuracy. In Grad-CAM the CNNs focus on the lesion, while the Swin-T maps come out in 7x7 blocks that follow the window partition and often cover the background: the method can compare the CNNs with each other, not with Swin-T.

## Data

The project uses ISIC 2019, with five files downloaded from [challenge.isic-archive.com](https://challenge.isic-archive.com/data/):

| File | Content |
|---|---|
| `ISIC_2019_Training_Input.zip` | 25,331 training images |
| `ISIC_2019_Training_GroundTruth.csv` | training labels (8 classes plus UNK) |
| `ISIC_2019_Training_Metadata.csv` | metadata, including the `lesion_id` used by the split |
| `ISIC_2019_Test_Input.zip` | 8,238 official test images |
| `ISIC_2019_Test_GroundTruth.csv` | official test labels |

The code expects the data in `~/data/isic2019/`, with both zip files extracted in place:

```
~/data/isic2019/
  ISIC_2019_Training_Input/            # 25,331 JPEG
  ISIC_2019_Test_Input/                # 8,238 JPEG
  ISIC_2019_Training_GroundTruth.csv
  ISIC_2019_Training_Metadata.csv
  ISIC_2019_Test_GroundTruth.csv
  img256/                              # written by scripts/preprocess.py
```

ISIC 2019 is distributed under the [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license: it may be used with attribution and for non-commercial purposes only. The MIT license of this repository covers the code, not the data. The attribution required by ISIC is:

- BCN_20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona
- HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161
- MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006; https://arxiv.org/abs/1902.03368

The full references are in `paper/datasets.bib`. `figures/gradcam/` contains 16 ISIC 2019 images (the 224 px center crop of the 256 px version) and their 48 Grad-CAM maps; those images, like the grids `figures/gradcam_grid_*.png`, are redistributed under the same CC-BY-NC 4.0 license with the attribution above.

## Method

The 25,331 images are resized to 256x256 (bicubic, JPEG quality 95) and split 70/15/15 (17,735, 3,798 and 3,798 images), stratified by class and grouped by `lesion_id` with seed 42, so that images of the same lesion never end up in different splits. The official test set, without its 2,047 UNK images, is used only as external evaluation. The three models come from timm with ImageNet weights and a new 8-class head, and they are trained with the same protocol: AdamW (lr 1e-4, weight decay 1e-2), one warmup epoch and cosine decay, batch 32, at most 15 epochs with early stopping (patience 3 on validation macro F1), bf16 AMP, cross-entropy weighted by inverse class frequency, and light data augmentation (RandomResizedCrop 224 with scale 0.7 to 1.0, horizontal and vertical flips, rotation up to 20 degrees, brightness and contrast 0.1). Each model runs with three seeds, nine runs in total.

Complementary experiments, all with seed 0:

- Convergence: with up to 30 epochs, early stopping stopped at epochs 23, 19 and 18, with gains of +0.5, +1.1 and +3.1 pp of internal macro F1 (`results/tables/tabla5_convergencia.csv`).
- Class weight ablation: neither the square root of inverse frequency nor an unweighted loss improves macro F1 consistently, and without weights internal balanced accuracy drops 9.7, 4.3 and 3.4 pp (`results/tables/tabla6_ablacion_pesos.csv`).
- Ensembles by averaging softmax probabilities, with no extra training: the ensemble of the three architectures reaches 59.3 internal macro F1, below the three-seed Swin-T ensemble, which reaches 60.5 (`results/tables/tabla4_ensembles.csv`).

## Repository layout

```
isic2019-benchmark/
  configs/
    base.yaml            # common protocol: data, augmentation, optimizer, epochs, seeds
    resnet50.yaml        # each model config only changes model.name
    efficientnet_b0.yaml
    swin_t.yaml
  src/isic/
    data.py              # Dataset, transforms and split loading
    models.py            # timm models and the Grad-CAM target layer of each one
    train.py             # training, early stopping and smoke test
    evaluate.py          # metrics on the internal test, official test and train sets
    cost.py              # parameters, MACs, throughput and training times
    ensemble.py          # ensembles from the saved predictions
    gradcam.py           # Grad-CAM on 16 fixed internal test images
    plotstyle.py         # shared colors, markers and names for the figures
  scripts/
    check_gpu.py         # checks CUDA and bf16
    preprocess.py        # resizes to 256x256 into data/img256/
    make_splits.py       # 70/15/15 split grouped by lesion_id
    run_all.sh           # queue of the 9 main runs
    run_long.sh          # queue of the convergence experiment (30 epochs)
    run_ablation.sh      # queue of the class weight ablation
    status.sh            # status of the three queues
    pr_por_clase.py      # per-class precision and recall from the confusion matrices
    make_figures.py      # tables in CSV and LaTeX, figures in PDF and PNG, summary.json
  results/               # metrics, splits, cost, ensembles, tables and queue logs
  figures/               # paper figures and Grad-CAM images
  paper/                 # paper (articulo.tex, articulo.pdf) and bibliography
  NOTAS.md               # working notes and observations for the paper (Spanish)
  data -> ~/data/isic2019  # symlink, not tracked by git
  runs/                  # checkpoints, histories and logs of each run, not tracked by git
```

## Reproduce

Requirements:

- Linux, or WSL2 on Windows.
- An NVIDIA driver that supports CUDA 12.8 or later (RTX 50 series GPUs do not work with earlier versions) and at least 4 GB of VRAM: the measured peak was 2.6 GiB, for Swin-T.
- [uv](https://docs.astral.sh/uv/), which installs Python 3.11 or 3.12 and the dependencies in `uv.lock`.
- About 30 GB of disk for the zip files, the images, the environment and the checkpoints.
- tmux, to launch the long queues.

The steps, in order. Times are for the RTX 5070 Ti Laptop GPU and come from the logs in `results/`; steps without a time were not logged.

1. Clone and install:

   ```bash
   git clone https://github.com/Valdes10U/isic2019-benchmark.git
   cd isic2019-benchmark
   uv sync
   uv run python scripts/check_gpu.py
   ```

2. Download the five files to `~/data/isic2019/`, extract the zip files and link the data into the repository:

   ```bash
   cd ~/data/isic2019
   unzip -q ISIC_2019_Training_Input.zip
   unzip -q ISIC_2019_Test_Input.zip
   cd -
   ln -s ~/data/isic2019 data
   ```

3. Preprocess (writes 25,331 and 8,238 images to `data/img256/train` and `data/img256/test`):

   ```bash
   uv run python scripts/preprocess.py
   ```

4. Build the split in `results/splits/`. With the same data, `git diff --stat results/splits` should show no changes:

   ```bash
   uv run python scripts/make_splits.py
   ```

5. Smoke test (500 training images, 100 validation images and one epoch; writes to `runs/smoke_resnet50/`). It can be repeated with `configs/efficientnet_b0.yaml` and `configs/swin_t.yaml`:

   ```bash
   uv run python -m isic.train --config configs/resnet50.yaml --seed 0 --smoke
   ```

6. The 9 main runs, each followed by its evaluation on both test sets (1 h 22 min; training one run takes 8.6, 6.9 and 11.0 min on average for ResNet50, EfficientNet-B0 and Swin-T). The queue can be resumed: it skips runs that already have `metrics.json`.

   ```bash
   tmux new-session -d -s train "bash scripts/run_all.sh"
   ```

7. The convergence experiment (35 min). It can be launched together with the previous step, since it waits for the main queue to finish:

   ```bash
   tmux new-session -d -s long "bash scripts/run_long.sh"
   ```

8. The class weight ablation (44 min). This queue does not wait for the others, so launch it once the convergence queue has finished:

   ```bash
   tmux new-session -d -s ablation "bash scripts/run_ablation.sh"
   ```

9. Check the progress of the three queues:

   ```bash
   bash scripts/status.sh
   ```

10. Analysis, tables and figures, in this order:

    ```bash
    uv run python -m isic.evaluate --all --split train   # train F1 for the generalization gap
    uv run python -m isic.cost                           # results/cost.csv
    uv run python -m isic.ensemble                       # results/ensembles.csv and ensembles.json
    uv run python -m isic.gradcam                        # figures/gradcam/ and results/gradcam_images.csv
    uv run python scripts/pr_por_clase.py                # results/pr_por_clase.csv
    uv run python scripts/make_figures.py                # results/tables/, figures/ and results/summary.json
    ```

The queues write their logs and `env.txt` to `runs/`; the files in `results/` are the copies from the reported runs.

## Reproducibility notes

- Training uses `cudnn.benchmark = True` and bf16 AMP, so repeating a run on the same hardware does not give bit-identical results. Differences should be compared against the standard deviation across seeds.
- The main queue ran on commit `7fdf47a`, the convergence queue on `efe0280` and the ablation on `df720c8` (first line of each log in `results/`). The training code has not changed since `df720c8`, and the options those commits added (`--max-epochs`, `--run-name` and `--class-weight`) keep the original behavior when omitted.
- The GMACs column is the output of fvcore, which counts multiply-accumulates; in strict FLOPs the number doubles, and `results/cost.csv` has both columns. fvcore does not count some operations (softmax, GELU and additions in Swin-T; SiLU, sigmoid and multiplications in EfficientNet-B0; max_pool in ResNet50), so the MACs are a slight lower bound.
- Inference throughput was measured with synthetic tensors on the GPU, without a DataLoader, with batch 32. Seconds per epoch exclude the first epoch.
- On Windows, keep the repository and the data on the WSL file system (ext4) and not under `/mnt/c`, because reading from NTFS is several times slower. Inside WSL there were 23 GiB of RAM and 8 GiB of swap (`results/env.txt`); those limits are set in `%UserProfile%\.wslconfig`, with the `memory` and `swap` keys of the `[wsl2]` section. Train with the laptop plugged in and the Windows power mode on best performance: on battery the GPU lowers its power and the times per epoch are no longer comparable.

## Paper

The paper is in [`paper/articulo.pdf`](paper/articulo.pdf) (IEEE Conference format, in Spanish, 9 pages). To compile it with [tectonic](https://tectonic-typesetting.github.io/), which takes the figures from `figures/` and the bibliography from `paper/datasets.bib` and `paper/literatura.bib`:

```bash
cd paper
tectonic articulo.tex
```

## Limitations

- The three models share the same learning rate (1e-4), and with it ResNet50 underfits (train macro F1 of 72.0, against 90.2 for EfficientNet-B0 and 95.5 for Swin-T), so its place in the comparison depends on the protocol.
- The absolute level is below published results for ISIC 2019 (accuracy of 65.6 to 73.4); likely causes include the 224 px images, the 15 epochs and the absence of metadata and test-time augmentation.
- DF and VASC have 36 and 38 images in the internal test set: the standard deviation of DF F1 ranges from 5.3 to 6.6 pp, and differences between models in those classes are not interpreted.
- There are only three seeds per model, and the complementary experiments use a single one.
- In Swin-T the Grad-CAM maps inherit the window geometry and the target layer is a normalization, not a convolution, so they are not compared with those of the CNNs.

## Citation

```bibtex
@misc{valdes2026isic,
  author       = {Santiago Valdes Uriarte},
  title        = {Comparación de {ResNet50}, {EfficientNet-B0} y {Swin-T} para la clasificación de lesiones cutáneas en {ISIC} 2019 bajo un protocolo común},
  year         = {2026},
  howpublished = {\url{https://github.com/Valdes10U/isic2019-benchmark}},
  note         = {Master's in Artificial Intelligence and Data Science, Universidad Panamericana}
}
```

## Contact

Santiago Valdes Uriarte

- Email: [santiago.valdesu@gmail.com](mailto:santiago.valdesu@gmail.com)
- GitHub: [Valdes10U](https://github.com/Valdes10U)
- LinkedIn: [santiagovaldesuri](https://www.linkedin.com/in/santiagovaldesuri)
