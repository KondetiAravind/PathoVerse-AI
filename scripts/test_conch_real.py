import time

import torch
from huggingface_hub import get_token

from conch.open_clip_custom import create_model_from_pretrained


MODEL_NAME = "conch_ViT-B-16"
MODEL_SOURCE = "hf_hub:MahmoodLab/conch"


def main():

    print("=" * 70)
    print("PATHOVERSE — REAL CONCH MODEL TEST")
    print("=" * 70)

    print("PyTorch       :", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU           :", torch.cuda.get_device_name(0))

    token = get_token()

    if not token:
        raise RuntimeError(
            "Hugging Face token not available. "
            "Run `hf auth login` first."
        )

    print("HF auth       : available")
    print()
    print("Loading CONCH...")
    print("This may download the pretrained checkpoint on first run.")
    print()

    start = time.perf_counter()

    model, preprocess = create_model_from_pretrained(
        MODEL_NAME,
        MODEL_SOURCE,
        device="cuda",
        hf_auth_token=token,
    )

    elapsed = time.perf_counter() - start

    model.eval()

    print()
    print("=" * 70)
    print("CONCH MODEL LOADED")
    print("=" * 70)
    print("Model         :", MODEL_NAME)
    print("Device        :", next(model.parameters()).device)
    print("Load time     :", f"{elapsed:.2f} sec")
    print("Model type    :", type(model).__name__)
    print("Preprocess    :", preprocess)
    print("=" * 70)


if __name__ == "__main__":
    main()