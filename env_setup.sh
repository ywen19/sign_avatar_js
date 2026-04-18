#!/usr/bin/env bash
ENV_NAME="signavatar-smollm"
PYTHON_VERSION="3.10"

# Change this if your PyTorch selector tells you to use a different one.
# Example values:
#   cu118
#   cu121
#   cpu
TORCH_CHANNEL="cu128"

# echo "Creating conda environment: ${ENV_NAME}"
# conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# echo "Activating environment: ${ENV_NAME}"
# shellcheck disable=SC1091
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "${ENV_NAME}"

echo "Python version:"
python --version

echo "Installing Transformers..."
pip install transformers

echo "Installing PyTorch..."
if [ "${TORCH_CHANNEL}" = "cpu" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"
fi

echo "Checking GPU visibility..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
else
    echo "nvidia-smi not found"
fi

echo "Installing spaCy..."
pip install -U spacy

echo "Installing contractions..."
pip install -U contractions

echo "Installing num2words..."
pip install -U num2words

echo "Verifying Python packages..."
python - <<'PY'
import torch
import transformers

print("torch version:", torch.__version__)
print("transformers version:", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "Done."
echo "Activate later with: conda activate ${ENV_NAME}"