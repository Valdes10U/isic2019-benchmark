#!/usr/bin/env bash
# Cola secuencial de las 9 corridas (3 modelos x 3 semillas), cada una seguida de su evaluación.
# Reanudable: salta las corridas que ya tienen metrics.json. Un fallo no detiene la cola.
# Lanzar con: tmux new-session -d -s train "bash scripts/run_all.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

LOG=runs/run_all.log
mkdir -p runs
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

# descripción del hardware y del software para el artículo
{
    echo "# $(date '+%Y-%m-%d %H:%M:%S'), commit $(git rev-parse --short HEAD)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    uv run python -c "import torch, timm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'timm', timm.__version__)"
    lscpu | grep "Model name"
    free -h
} > runs/env.txt 2>&1

log "COLA INICIADA commit $(git rev-parse --short HEAD)"
for seed in 0 1 2; do
    for model in resnet50 efficientnet_b0 swin_t; do
        run="${model}_seed${seed}"
        dir="runs/$run"
        if [[ -f "$dir/metrics.json" ]]; then
            log "SALTADA $run (ya tiene metrics.json)"
            continue
        fi
        mkdir -p "$dir"
        log "INICIO $run"
        if ! uv run python -m isic.train --config "configs/$model.yaml" --seed "$seed" 2>&1 | tee "$dir/train.log"; then
            log "FALLA $run en entrenamiento (ver $dir/train.log): $(tail -n 1 "$dir/train.log")"
            continue
        fi
        if ! uv run python -m isic.evaluate --run "$dir" 2>&1 | tee "$dir/eval.log"; then
            log "FALLA $run en evaluación (ver $dir/eval.log): $(tail -n 1 "$dir/eval.log")"
            continue
        fi
        log "FIN $run"
    done
done
log "COLA TERMINADA"
