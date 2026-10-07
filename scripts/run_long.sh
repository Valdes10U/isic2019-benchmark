#!/usr/bin/env bash
# Experimento complementario de convergencia: semilla 0 de los tres modelos
# con máximo 30 épocas. Espera a que termine la cola principal y luego corre en secuencia.
# Lanzar con: tmux new-session -d -s long "bash scripts/run_long.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

LOG=runs/run_long.log
mkdir -p runs
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

# espera a la cola principal; revisa cada minuto y avisa cada 10
i=0
until tail -n 1 runs/run_all.log 2>/dev/null | grep -q "COLA TERMINADA$"; do
    (( i % 10 == 0 )) && echo "$(date '+%H:%M') esperando a que termine la cola principal"
    i=$((i + 1))
    sleep 60
done

log "COLA LARGA INICIADA commit $(git rev-parse --short HEAD)"
for model in resnet50 efficientnet_b0 swin_t; do
    run="${model}_seed0_e30"
    dir="runs/$run"
    if [[ -f "$dir/metrics.json" ]]; then
        log "SALTADA $run (ya tiene metrics.json)"
        continue
    fi
    mkdir -p "$dir"
    log "INICIO $run"
    if ! uv run python -m isic.train --config "configs/$model.yaml" --seed 0 --max-epochs 30 --run-name "$run" \
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
log "COLA LARGA TERMINADA"
