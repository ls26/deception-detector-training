#!/bin/bash
# Final eval of a smoke run: bash run_smoke_eval.sh <run> (inside tmux). The run's adapter through run_eval.sh (a
# "smoke" run: the first 50 eval questions -> dry/evals/<run>), the base model on the same 50 questions scored with the
# run's start-of-run probe (-> dry/evals/base_<run>), then compare.py (-> dry/comparison_<run>).
# Log: logs/smoke_eval_<run>_<host>.log, ends "EXIT <code>".
set -u
RUN=$1
FOLDER=/workspace/qwen_experiment/261120_qwen_port_checks
source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh
cd $FOLDER
LOG=logs/smoke_eval_${RUN}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists" >&2; exit 1; }
RUN_DIR=$($PY -c "
import json, yaml
config = yaml.safe_load(open('$FOLDER/$RUN.yaml'))
run_id = json.load(open(config['output_dir'] + '/run_metrics_' + config['run_key'] + '.json'))['run_id']
print('/workspace/results/run_results/' + config['group'] + '/' + config['run_key'] + '_' + run_id)")
MODEL=$(cd $SETUP_DIR && $PY -c "from pathlib import Path; from spec import model_of_run; print(model_of_run(Path('$FOLDER/$RUN.yaml')))")
{
  echo "RUN_DIR $RUN_DIR MODEL $MODEL"
  bash $SETUP_DIR/run_eval.sh $FOLDER $RUN 1 && grep -q "^EXIT 0" logs/eval_${RUN}_a1_$HOST.log \
    && $PY $SETUP_DIR/eval_checkpoint.py generate --out-dir dry/evals/base_$RUN --model $MODEL --limit 50 \
    && $PY $SETUP_DIR/eval_checkpoint.py probes --out-dir dry/evals/base_$RUN --model $MODEL --probe-run-dir $RUN_DIR \
    && $PY $SETUP_DIR/eval_checkpoint.py labels --out-dir dry/evals/base_$RUN \
    && (cd $SETUP_DIR && $PY compare.py --base $FOLDER/dry/evals/base_$RUN --final $FOLDER/dry/evals/$RUN --run-dir $RUN_DIR --out-dir $FOLDER/dry/comparison_$RUN)
  echo "EXIT $?"
} > "$LOG" 2>&1
