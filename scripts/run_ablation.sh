#!/usr/bin/env bash
# Ablación de ponderación de clases: semilla 0 de los tres modelos, 15 épocas,
# primero con raíz de la frecuencia inversa (_sqrtw) y luego sin pesos (_now).
# Lanzar con: tmux new-session -d -s ablation "bash scripts/run_ablation.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

LOG=runs/run_ablation.log
mkdir -p runs
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

log "COLA ABLACION INICIADA commit $(git rev-parse --short HEAD)"
for variant in "sqrt_inverse sqrtw" "none now"; do
    read -r mode suffix <<< "$variant"
    for model in resnet50 efficientnet_b0 swin_t; do
        run="${model}_seed0_${suffix}"
        dir="runs/$run"
        if [[ -f "$dir/metrics.json" ]]; then
            log "SALTADA $run (ya tiene metrics.json)"
            continue
        fi
        mkdir -p "$dir"
        log "INICIO $run"
        if ! uv run python -m isic.train --config "configs/$model.yaml" --seed 0 --class-weight "$mode" --run-name "$run" \
                2>&1 | tee "$dir/train.log"; then
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
log "COLA ABLACION TERMINADA"
