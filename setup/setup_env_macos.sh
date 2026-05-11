#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
ENV_DIR="${ENV_DIR:-$SCRIPT_DIR/.venv_smolvlm}"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_smolvlm.txt"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-0}"

echo "==> Project root: $SCRIPT_DIR"
echo "==> Python: $PYTHON_BIN"
echo "==> Env dir: $ENV_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN"
  echo "Tip: on Apple Silicon you can install one with: brew install python@3.11"
  exit 1
fi

if [ ! -d "$ENV_DIR" ]; then
  echo "==> Creating virtual environment"
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

VENV_PYTHON="$ENV_DIR/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  echo "Virtualenv python not found: $VENV_PYTHON"
  exit 1
fi

echo "==> Upgrading pip/setuptools/wheel"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo "==> Installing core dependencies"
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"

echo "==> Installing PyTorch for macOS"
"$VENV_PYTHON" -m pip install torch torchvision torchaudio

echo "==> Validating packages"
"$VENV_PYTHON" -c "import platform, torch, transformers, PIL; print('python', platform.python_version()); print('torch', torch.__version__); print('mps_available', torch.backends.mps.is_available()); print('transformers', transformers.__version__); print('pillow', PIL.__version__)"

if [ "$RUN_SMOKE_TEST" = "1" ]; then
  echo "==> Running local model smoke test"
  "$VENV_PYTHON" -c "from transformers import AutoProcessor; p=AutoProcessor.from_pretrained(r'$SCRIPT_DIR/SmolVLM2_Weights', local_files_only=True, trust_remote_code=True); print(type(p).__name__)"
fi

echo "==> Environment setup complete"
echo "Activate with:"
echo "  source \"$ENV_DIR/bin/activate\""
echo "Then run:"
echo "  python \"$SCRIPT_DIR/quant_eval_smolvlm.py\" --dataset \"$SCRIPT_DIR/real_scienceqa_eval.jsonl\" --device mps --strategies fp32"
