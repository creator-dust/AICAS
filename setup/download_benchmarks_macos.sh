#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv_smolvlm/bin/python}"
BENCH_DIR="${BENCH_DIR:-$SCRIPT_DIR/benchmarks}"
DATASETS="${DATASETS:-scienceqa ai2d chartqa}"

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

resolve_repo() {
  case "$1" in
    scienceqa) printf '%s\n' "lmms-lab/ScienceQA" ;;
    ai2d) printf '%s\n' "lmms-lab/ai2d" ;;
    chartqa) printf '%s\n' "lmms-lab/ChartQA" ;;
    docvqa) printf '%s\n' "lmms-lab/DocVQA" ;;
    textvqa) printf '%s\n' "lmms-lab/textvqa" ;;
    mathvista) printf '%s\n' "AI4Math/MathVista" ;;
    mmstar) printf '%s\n' "Lin-Chen/MMStar" ;;
    *) return 1 ;;
  esac
}

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

mkdir -p "$BENCH_DIR"
echo "==> Benchmarks will be downloaded into: $BENCH_DIR"

for name in $DATASETS; do
  repo="$(resolve_repo "$name")" || {
    echo "Unsupported dataset key: $name"
    exit 1
  }

  target_dir="$BENCH_DIR/$name"
  mkdir -p "$target_dir"
  echo "==> Downloading $name from $repo"
  "$HF_CLI" download \
    "$repo" \
    --repo-type dataset \
    --local-dir "$target_dir"
done

echo "==> Done"
echo "Downloaded datasets: $DATASETS"
