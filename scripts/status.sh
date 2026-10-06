#!/usr/bin/env bash
# Estado de la cola de entrenamiento. Uso: bash scripts/status.sh
cd "$(dirname "$0")/.."
LOG=runs/run_all.log
tmux has-session -t train 2>/dev/null && alive=1 || alive=0

printf "%-22s %-10s %6s %8s %7s %9s %9s\n" corrida estado epoca f1_val seg f1_test f1_ofic
for seed in 0 1 2; do
    for model in resnet50 efficientnet_b0 swin_t; do
        run="${model}_seed${seed}"
        dir="runs/$run"
        last=$(grep -E " (INICIO|FIN|FALLA) $run( |$)" "$LOG" 2>/dev/null | tail -n 1 | awk '{print $3}')
        if [[ -f "$dir/metrics.json" ]]; then state=terminada
        elif [[ $last == INICIO && $alive == 1 ]]; then state="en curso"
        elif [[ $last == INICIO || $last == FALLA ]]; then state=fallida
        else state=pendiente
        fi
        ep=-; f1=-; sec=-; ft=-; fo=-
        if [[ -f "$dir/history.csv" ]] && (( $(wc -l < "$dir/history.csv") > 1 )); then
            IFS=, read -r ep _ _ _ f1 sec _ < <(tail -n 1 "$dir/history.csv")
        fi
        if [[ -f "$dir/metrics.json" ]]; then
            read -r ft fo < <(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['test']['f1_macro'], m['test_official']['f1_macro'])" "$dir/metrics.json")
        fi
        printf "%-22s %-10s %6s %8s %7s %9s %9s\n" "$run" "$state" "$ep" "$f1" "$sec" "$ft" "$fo"
    done
done

echo
echo "Sesión tmux train: $([[ $alive == 1 ]] && echo activa || echo inactiva)"
echo "Últimas líneas de $LOG:"
if [[ -f $LOG ]]; then tail -n 5 "$LOG"; else echo "  (la cola no ha arrancado)"; fi
echo
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader \
    | awk -F', ' '{print "GPU: uso " $1 ", VRAM " $2 " de " $3}'
