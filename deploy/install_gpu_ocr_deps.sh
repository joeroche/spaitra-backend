#!/usr/bin/env bash
set -euo pipefail

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

PADDLE_VERSION="${PADDLE_VERSION:-3.3.0}"
PADDLE_INDEX=""

if [ "$CUDA_MAJOR" -ge 13 ]; then
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu130/"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 9 ]; then
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu129/"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 6 ]; then
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"
elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu118/"
else
    echo "ERROR: Unsupported CUDA version $CUDA_VER for Paddle GPU wheel"
    exit 1
fi

echo "Detected CUDA $CUDA_VER"
echo "Paddle index:     $PADDLE_INDEX"
echo "Paddle version:   $PADDLE_VERSION"

pip uninstall -y paddlepaddle paddlepaddle-gpu >/dev/null 2>&1 || true
pip install "paddlepaddle-gpu==${PADDLE_VERSION}" -i "$PADDLE_INDEX"

echo ""
echo "Verifying Paddle GPU..."
python - <<'EOF'
import paddle

print("paddle version:", paddle.__version__)
print("compiled_with_cuda:", paddle.device.is_compiled_with_cuda())
print("cuda_device_count:", paddle.device.cuda.device_count())
if not paddle.device.is_compiled_with_cuda():
    raise SystemExit("ERROR: Paddle was installed without CUDA support")
if int(paddle.device.cuda.device_count()) <= 0:
    raise SystemExit("ERROR: Paddle cannot see any CUDA devices")
EOF

echo ""
echo "OCR GPU dependencies installed successfully in the current environment."
