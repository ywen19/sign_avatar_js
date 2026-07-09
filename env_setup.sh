#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-signavatar}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# PyTorch wheel selector. Override when needed, for example:
#   TORCH_CHANNEL=cpu bash env_setup.sh
#   TORCH_CHANNEL=cu121 bash env_setup.sh
#   TORCH_CHANNEL=cu128 bash env_setup.sh
TORCH_CHANNEL="${TORCH_CHANNEL:-cu128}"

echo "Preparing conda environment: ${ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found on PATH."
    echo "Install Miniconda/Anaconda or open a shell where conda is initialized."
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda environment already exists: ${ENV_NAME}"
else
    echo "Creating conda environment: ${ENV_NAME} (python=${PYTHON_VERSION})"
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

echo "Activating environment: ${ENV_NAME}"
conda activate "${ENV_NAME}"

echo "Python executable: $(command -v python)"
python --version

echo "Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing PyTorch (${TORCH_CHANNEL})..."
if [ "${TORCH_CHANNEL}" = "cpu" ]; then
    python -m pip install --upgrade torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
else
    python -m pip install --upgrade torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"
fi

echo "Installing app.py runtime packages..."
python -m pip install --upgrade \
    transformers \
    huggingface_hub \
    gliner \
    num2words

echo "Checking GPU visibility..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
else
    echo "nvidia-smi not found; this is OK for CPU-only runs."
fi

echo "Verifying Python packages..."
python - <<PY
import torch
import transformers
import huggingface_hub
import gliner
from num2words import num2words

print("torch version:", torch.__version__)
print("transformers version:", transformers.__version__)
print("huggingface_hub version:", huggingface_hub.__version__)
print("gliner import: ok")
print("num2words import:", num2words(3))
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "Done."
echo "Run the main app with:"
echo "  conda activate ${ENV_NAME}"
echo "  python app.py"
