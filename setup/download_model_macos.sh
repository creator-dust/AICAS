#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv_smolvlm/bin/python}"
MODEL_REPO="${MODEL_REPO:-HuggingFaceTB/SmolVLM2-500M-Video-Instruct}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/SmolVLM2_Weights}"
REVISION="${REVISION:-}"

resolve_hf_cli() {
  local venv_bin
  venv_bin="$(dirname "$PYTHON_BIN")"

  if [ -x "$venv_bin/hf" ]; then
    printf '%s\n' "$venv_bin/hf"
    return 0
  fi
  if [ -x "$venv_bin/huggingface-cli" ]; then
    printf '%s\n' "$venv_bin/huggingface-cli"
    return 0
  fi
  if command -v hf >/dev/null 2>&1; then
    command -v hf
    return 0
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    command -v huggingface-cli
    return 0
  fi

  return 1
}

echo "==> Project root: $SCRIPT_DIR"
echo "==> Python: $PYTHON_BIN"
echo "==> Model repo: $MODEL_REPO"
echo "==> Model dir: $MODEL_DIR"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not found: $PYTHON_BIN"
  echo "Run setup_env_macos.sh first or set PYTHON_BIN=/path/to/python"
  exit 1
fi

echo "==> Ensuring Hugging Face CLI is available"
"$PYTHON_BIN" -m pip install -U "huggingface_hub[cli]"

HF_CLI="$(resolve_hf_cli)" || {
  echo "Hugging Face CLI not found after installation."
  exit 1
}
echo "==> Using CLI: $HF_CLI"

mkdir -p "$MODEL_DIR"

DOWNLOAD_CMD=(
  "$HF_CLI" download
  "$MODEL_REPO"
  --repo-type model
  --local-dir "$MODEL_DIR"
)

if [ -n "$REVISION" ]; then
  DOWNLOAD_CMD+=(--revision "$REVISION")
fi

echo "==> Downloading model snapshot"
"${DOWNLOAD_CMD[@]}"

echo "==> Verifying local model files"
"$PYTHON_BIN" -c "from transformers import AutoProcessor; p=AutoProcessor.from_pretrained(r'$MODEL_DIR', local_files_only=True, trust_remote_code=True); print(type(p).__name__)"

echo "==> Done"
echo "Model is ready in: $MODEL_DIR"
