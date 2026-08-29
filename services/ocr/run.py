"""Run the OCR service in development mode."""

import os

import uvicorn


def main() -> None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ and "OCR_GPU_DEVICE_ID" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["OCR_GPU_DEVICE_ID"]
    host = os.environ.get("OCR_HOST", "127.0.0.1")
    port = int(os.environ.get("OCR_PORT", "8001"))
    uvicorn.run("services.ocr.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
