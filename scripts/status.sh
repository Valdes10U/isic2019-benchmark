#!/usr/bin/env bash
# Estado de la cola principal y de la cola larga (_e30). Uso: bash scripts/status.sh
cd "$(dirname "$0")/.."
LOG=runs/run_all.log
LONG_LOG=runs/run_long.log
tmux has-session -t train 2>/dev/null && alive=1 || alive=0
tmux has-session -t long 2>/dev/null && long_alive=1 || long_alive=0

header() { printf "%-26s %-10s %6s %8s %7s %9s %9s\n" corrida estado epoca f1_val seg f1_test f1_ofic; }

# una fila de la tabla: run, log de su cola, 1 si la sesión de tmux de esa cola está viva
row() {
    local run=$1 log=$2 session=$3 dir="runs/$1" last state ep=- f1=- sec=- ft=- fo=-
    last=$(grep -E " (INICIO|FIN|FALLA) $run( |$)" "$log" 2>/dev/null | tail -n 1 | awk '{print $3}')
    if [[ -f "$dir/metrics.json" ]]; then state=terminada
    elif [[ $last == INICIO && $session == 1 ]]; then state="en curso"
    elif [[ $last == INICIO || $last == FALLA ]]; then state=fallida
    else state=pendiente
    fi
    if [[ -f "$dir/history.csv" ]] && (( $(wc -l < "$dir/history.csv") > 1 )); then
        IFS=, read -r ep _ _ _ f1 sec _ < <(tail -n 1 "$dir/history.csv")
    fi
    if [[ -f "$dir/metrics.json" ]]; then
        read -r ft fo < <(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['test']['f1_macro'], m['test_official']['f1_macro'])" "$dir/metrics.json")
    fi
    printf "%-26s %-10s %6s %8s %7s %9s %9s\n" "$run" "$state" "$ep" "$f1" "$sec" "$ft" "$fo"
}

header
for seed in 0 1 2; do
    for model in resnet50 efficientnet_b0 swin_t; do
        row "${model}_seed${seed}" "$LOG" "$alive"
    done
done

echo
echo "Experimento de convergencia (30 épocas, semilla 0):"
header
for model in resnet50 efficientnet_b0 swin_t; do
    row "${model}_seed0_e30" "$LONG_LOG" "$long_alive"
done

echo
echo "Sesiones tmux: train $([[ $alive == 1 ]] && echo activa || echo inactiva), long $([[ $long_alive == 1 ]] && echo activa || echo inactiva)"
echo "Últimas líneas de $LOG:"
if [[ -f $LOG ]]; then tail -n 5 "$LOG"; else echo "  (la cola no ha arrancado)"; fi
if [[ -f $LONG_LOG ]]; then
    echo "Últimas líneas de $LONG_LOG:"
    tail -n 5 "$LONG_LOG"
fi
echo
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader \
    | awk -F', ' '{print "GPU: uso " $1 ", VRAM " $2 " de " $3}'
