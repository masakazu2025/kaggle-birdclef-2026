"""Push a local notebook to Kaggle Kernels.

Usage:
    poetry run python scripts/push_notebook.py <notebook_path>

Example:
    poetry run python scripts/push_notebook.py notebooks/inference/birdclef-2026-pytorch-baseline-inference.ipynb
"""

import argparse
import json
import sys
from pathlib import Path

KAGGLE_USERNAME = "masakazum"


def push_notebook(notebook_path: str) -> None:
    from kaggle import api

    api.authenticate()

    nb = Path(notebook_path)
    if not nb.exists():
        print(f"Error: {nb} not found")
        sys.exit(1)

    # ノートブックのあるフォルダがそのままカーネルディレクトリ
    kernel_dir = nb.parent

    # Create kernel-metadata.json if not exists
    meta_path = kernel_dir / "kernel-metadata.json"
    if not meta_path.exists():
        kernel_id = nb.stem.lower().replace("_", "-")
        meta = {
            "id": f"{KAGGLE_USERNAME}/{kernel_id}",
            "title": nb.stem.replace("-", " ").replace("_", " ").title(),
            "code_file": nb.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": [],
            "competition_sources": ["birdclef-2026"],
            "kernel_sources": [],
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Created {meta_path}")
        print("Edit the metadata if needed, then run again to push.")
        return

    print(f"Pushing {kernel_dir} ...")
    api.kernels_push(str(kernel_dir))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", help="Path to the .ipynb file")
    args = parser.parse_args()
    push_notebook(args.notebook)
