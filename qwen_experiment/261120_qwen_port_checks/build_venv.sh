#!/bin/bash
# Builds the new venv for Qwen3.5 and Llama runs. The uv download cache goes to /tmp (the pod's own disk) so the
# network volume's /workspace/.uv-cache does not grow; it is deleted at the end.
set -euo pipefail
export UV_CACHE_DIR=/tmp/uv-cache-qwen-venv
export UV_LINK_MODE=copy
export UV_PYTHON_INSTALL_DIR=/workspace/.uv-python
VENV=/workspace/sycophancy-rl/.venv
HERE=/workspace/qwen_experiment/261120_qwen_port_checks
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -r "$HERE/venv_requirements.in"
uv pip freeze --python "$VENV/bin/python" > /workspace/sycophancy-rl/requirements.txt
"$VENV/bin/python" -c "import torch, transformers, vllm, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| vllm', vllm.__version__, '| peft', peft.__version__, '| gpu', torch.cuda.is_available())"
rm -rf "$UV_CACHE_DIR"
du -sh "$VENV"
