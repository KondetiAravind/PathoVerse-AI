import sys
import importlib
import platform
import subprocess


def check_import(package_name, import_name=None):
    import_name = import_name or package_name

    try:
        module = importlib.import_module(import_name)

        version = getattr(module, "__version__", "unknown")

        print(f"  ✓ {package_name:<20} {version}")
        return True

    except Exception as exc:
        print(f"  ✗ {package_name:<20} FAILED")
        print(f"      {exc}")
        return False


def main():
    print("=" * 78)
    print("PATHOVERSE AI — FINAL ENVIRONMENT VALIDATION")
    print("=" * 78)

    print("\nSYSTEM")
    print("-" * 78)

    print(f"Python       : {sys.version.split()[0]}")
    print(f"Platform     : {platform.platform()}")
    print(f"Architecture : {platform.machine()}")

    print("\nPYTHON VERSION CHECK")
    print("-" * 78)

    if sys.version_info[:2] == (3, 10):
        print("  ✓ Python 3.10")
    else:
        print(
            f"  ✗ Expected Python 3.10, "
            f"found Python {sys.version_info.major}.{sys.version_info.minor}"
        )

    print("\nCORE PACKAGES")
    print("-" * 78)

    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("Pillow", "PIL"),
        ("OpenCV", "cv2"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
        ("PyYAML", "yaml"),
        ("python-dotenv", "dotenv"),
        ("pydantic", "pydantic"),
        ("psutil", "psutil"),
    ]

    core_ok = True

    for package_name, import_name in packages:
        if not check_import(package_name, import_name):
            core_ok = False

    print("\nAI / DEEP LEARNING")
    print("-" * 78)

    ai_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("safetensors", "safetensors"),
        ("huggingface_hub", "huggingface_hub"),
        ("MONAI", "monai"),
    ]

    ai_ok = True

    for package_name, import_name in ai_packages:
        if not check_import(package_name, import_name):
            ai_ok = False

    print("\nPATHOLOGY / VISION")
    print("-" * 78)

    vision_packages = [
        ("OpenSlide", "openslide"),
        ("scikit-image", "skimage"),
        ("h5py", "h5py"),
        ("imageio", "imageio"),
        ("joblib", "joblib"),
        ("networkx", "networkx"),
    ]

    vision_ok = True

    for package_name, import_name in vision_packages:
        if not check_import(package_name, import_name):
            vision_ok = False

    print("\nSEARCH / EXPERIMENT / API")
    print("-" * 78)

    infrastructure_packages = [
        ("FAISS", "faiss"),
        ("MLflow", "mlflow"),
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("Streamlit", "streamlit"),
    ]

    infrastructure_ok = True

    for package_name, import_name in infrastructure_packages:
        if not check_import(package_name, import_name):
            infrastructure_ok = False

    print("\nPYTORCH / CUDA")
    print("-" * 78)

    cuda_ok = False

    try:
        import torch

        print(f"  ✓ PyTorch version : {torch.__version__}")
        print(f"  ✓ CUDA available  : {torch.cuda.is_available()}")
        print(f"  ✓ CUDA version    : {torch.version.cuda}")
        print(f"  ✓ GPU count       : {torch.cuda.device_count()}")

        if torch.cuda.is_available():

            cuda_ok = True

            for i in range(torch.cuda.device_count()):
                print(
                    f"  ✓ GPU {i}           : "
                    f"{torch.cuda.get_device_name(i)}"
                )

            print("\nCUDA MATRIX TEST")
            print("-" * 78)

            device = torch.device("cuda:0")

            size = 8192

            a = torch.randn(
                size,
                size,
                device=device,
                dtype=torch.float16,
            )

            b = torch.randn(
                size,
                size,
                device=device,
                dtype=torch.float16,
            )

            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            c = torch.matmul(a, b)

            end_event.record()

            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)

            allocated_gb = (
                torch.cuda.memory_allocated(device) / (1024 ** 3)
            )

            reserved_gb = (
                torch.cuda.memory_reserved(device) / (1024 ** 3)
            )

            print(f"  ✓ Matrix multiplication : PASSED")
            print(f"  ✓ Matrix size           : {size} x {size}")
            print(f"  ✓ Execution time        : {elapsed_ms:.2f} ms")
            print(f"  ✓ GPU memory allocated  : {allocated_gb:.2f} GB")
            print(f"  ✓ GPU memory reserved   : {reserved_gb:.2f} GB")

            del a
            del b
            del c

            torch.cuda.empty_cache()

        else:
            print("  ✗ CUDA is not available")

    except Exception as exc:
        print(f"  ✗ CUDA validation failed")
        print(f"      {exc}")

    print("\nGPU PROCESS CHECK")
    print("-" * 78)

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        print(result.stdout.strip())

    except Exception as exc:
        print(f"  ⚠ Unable to query nvidia-smi: {exc}")

    print("\nFINAL RESULT")
    print("=" * 78)

    all_ok = (
        core_ok
        and ai_ok
        and vision_ok
        and infrastructure_ok
        and cuda_ok
    )

    if all_ok:
        print("✓ CORE ENVIRONMENT : PASSED")
        print("✓ PYTORCH / CUDA   : PASSED")
        print("✓ AI STACK         : PASSED")
        print("✓ VISION STACK     : PASSED")
        print("✓ INFRASTRUCTURE   : PASSED")
        print()
        print("🚀 PATHOVERSE AI ENVIRONMENT IS READY")
    else:
        print("✗ ENVIRONMENT VALIDATION FAILED")
        print("Review the failed components above.")

    print("=" * 78)


if __name__ == "__main__":
    main()