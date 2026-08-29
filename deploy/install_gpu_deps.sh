#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Check NVIDIA driver
# -----------------------------
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA driver is not installed."
    exit 1
fi

CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)

if [ -z "$CUDA_VER" ]; then
    echo "ERROR: Could not parse CUDA version from nvidia-smi"
    nvidia-smi | head -6
    exit 1
fi

CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

echo "Detected CUDA $CUDA_VER"

# -----------------------------
# PyTorch index selection
# -----------------------------
TORCH_INDEX=""
TORCH_PACKAGES=(torch torchvision)

if [ "$CUDA_MAJOR" -ge 13 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    # Unpinned installs on the cu128 index can currently resolve to torch 2.11
    # with cu13 runtime deps, which breaks CUDA init on 12.8 hosts.
    TORCH_PACKAGES=("torch==2.10.0+cu128" "torchvision==0.25.0+cu128")
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 6 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu126"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif [ "$CUDA_MAJOR" -eq 12 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif [ "$CUDA_MAJOR" -eq 11 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
    echo "ERROR: Unsupported CUDA version $CUDA_VER for PyTorch"
    exit 1
fi

# -----------------------------
# Output config
# -----------------------------
echo "PyTorch index:    $TORCH_INDEX"
echo ""

# -----------------------------
# Install PyTorch
# -----------------------------
echo "Installing PyTorch..."
pip install "${TORCH_PACKAGES[@]}" --index-url "$TORCH_INDEX"

# Keep transitive model deps in the ranges required by depth-pro and moondream.
pip install "numpy<2" "pillow>=10.4.0,<11.0.0"

# -----------------------------
# Verify PyTorch
# -----------------------------
echo ""
echo "Verifying PyTorch GPU..."
python - <<'EOF'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("ERROR: PyTorch cannot see GPU")
EOF

echo ""
echo "GPU dependencies installed successfully in the current environment."
