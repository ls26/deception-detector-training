# Source (do not execute) before every sycophancy step: `source /workspace/probe-rl/sycophancy_setup/run_env.sh`.
# The environment of every task in the copy (../env.sh: PYTHONPATH, PATH, HF_HOME, WANDB_PROJECT, the secrets, $PY,
# $HOST), plus this package's folder as $SETUP_DIR.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || return 1
SETUP_DIR=$COPY/sycophancy_setup
