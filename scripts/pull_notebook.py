"""Pull the latest version of a notebook from Kaggle Kernels.

Usage:
    poetry run python scripts/pull_notebook.py <kernel_slug>

Example:
    poetry run python scripts/pull_notebook.py your-username/eda-01-overview
"""

import argparse
import subprocess
import sys
from pathlib import Path


def pull_notebook(kernel_slug: str, output_dir: str = "notebooks") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "poetry", "run", "kaggle", "kernels", "output",
            kernel_slug, "-p", str(out),
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_slug", help="Kaggle kernel slug (username/kernel-name)")
    parser.add_argument("--output-dir", default="notebooks", help="Local output directory")
    args = parser.parse_args()
    pull_notebook(args.kernel_slug, args.output_dir)
