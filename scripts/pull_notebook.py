"""Pull the latest version of a notebook from Kaggle Kernels.

Usage:
    poetry run python scripts/pull_notebook.py <kernel_slug>

Example:
    poetry run python scripts/pull_notebook.py masakazum/birdclef-2026-pytorch-baseline-inference
"""

import argparse
import sys
from pathlib import Path


def pull_notebook(kernel_slug: str, output_dir: str = "notebooks") -> None:
    from kaggle import api

    api.authenticate()
    kernel_name = kernel_slug.split("/")[-1]
    out = Path(output_dir) / kernel_name
    out.mkdir(parents=True, exist_ok=True)

    print(f"Pulling {kernel_slug} into {out} ...")
    api.kernels_pull(kernel_slug, path=str(out), metadata=True)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_slug", help="Kaggle kernel slug (username/kernel-name)")
    parser.add_argument("--output-dir", default="notebooks", help="Local output directory")
    args = parser.parse_args()
    pull_notebook(args.kernel_slug, args.output_dir)
