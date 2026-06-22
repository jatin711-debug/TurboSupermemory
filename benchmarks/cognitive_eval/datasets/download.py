"""Download benchmark datasets from HuggingFace."""
import os
import json
import zipfile
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent / "data"


def download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download a file with progress."""
    print(f"Downloading {url} ...")
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total = int(response.headers.get("Content-Length", 0))
            chunk_size = 8192
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        print(f"\r  Progress: {pct}% ({downloaded}/{total} bytes)", end="")
            print(f"\n  Saved to {dest}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_longmemeval(force: bool = False) -> Optional[Path]:
    """Download LongMemEval dataset from HuggingFace.

    Uses the official MemoryAsModality/LongMemEval dataset.
    """
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    # The dataset files to download from HuggingFace
    # LongMemEval has train/test splits in parquet format
    base_url = "https://huggingface.co/datasets/MemoryAsModality/LongMemEval/resolve/main/data"
    files = [
        "train-00000-of-00001.parquet",
        "test-00000-of-00001.parquet",
    ]

    dataset_dir = data_dir / "longmemeval"
    dataset_dir.mkdir(exist_ok=True)

    for fname in files:
        dest = dataset_dir / fname
        if dest.exists() and not force:
            print(f"{fname} already exists, skipping.")
            continue

        url = f"{base_url}/{fname}"
        if not download_file(url, dest):
            print(f"Failed to download {fname}")
            return None

    print(f"LongMemEval dataset downloaded to {dataset_dir}")
    return dataset_dir


def download_locomo(force: bool = False) -> Optional[Path]:
    """Download LoCoMo dataset from HuggingFace.

    Uses the Percena/locomo-mc10 dataset which contains the actual LoCoMo-MC10 benchmark.
    """
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    # LoCoMo-MC10 dataset from HuggingFace
    url = "https://huggingface.co/datasets/Percena/locomo-mc10/resolve/main/data/locomo_mc10.json"

    dataset_dir = data_dir / "locomo"
    dataset_dir.mkdir(exist_ok=True)

    dest = dataset_dir / "locomo_mc10.json"
    if dest.exists() and not force:
        print(f"{dest.name} already exists, skipping.")
        return dataset_dir

    if not download_file(url, dest):
        print(f"Failed to download LoCoMo dataset")
        return None

    print(f"LoCoMo dataset downloaded to {dataset_dir}")
    return dataset_dir


def download_all(force: bool = False) -> dict:
    """Download all benchmark datasets."""
    results = {}
    print("=" * 60)
    print("Downloading LongMemEval dataset...")
    print("=" * 60)
    results["longmemeval"] = download_longmemeval(force=force)

    print("\n" + "=" * 60)
    print("Downloading LoCoMo dataset...")
    print("=" * 60)
    results["locomo"] = download_locomo(force=force)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download cognitive benchmark datasets")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files exist"
    )
    parser.add_argument(
        "--dataset",
        choices=["longmemeval", "locomo", "all"],
        default="all",
        help="Which dataset to download",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        download_all(force=args.force)
    elif args.dataset == "longmemeval":
        download_longmemeval(force=args.force)
    elif args.dataset == "locomo":
        download_locomo(force=args.force)
