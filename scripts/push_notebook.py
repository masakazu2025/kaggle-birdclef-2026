"""Push a local notebook to Kaggle Kernels.

Usage:
    poetry run python scripts/push_notebook.py <notebook_path>

Example:
    poetry run python scripts/push_notebook.py notebooks/eda/eda-01-overview.ipynb
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def push_notebook(notebook_path: str) -> None:
    nb = Path(notebook_path)
    if not nb.exists():
        print(f"Error: {nb} not found")
        sys.exit(1)

    kernel_dir = nb.parent / nb.stem
    kernel_dir.mkdir(exist_ok=True)

    # Copy notebook into kernel dir
    dest = kernel_dir / nb.name
    dest.write_bytes(nb.read_bytes())

    # Create kernel-metadata.json if not exists
    meta_path = kernel_dir / "kernel-metadata.json"
    if not meta_path.exists():
        # Derive a kernel id from the notebook name
        kernel_id = nb.stem.lower().replace("_", "-")
        meta = {
            "id": f"YOUR_USERNAME/{kernel_id}",
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
        print(f"Created {meta_path} — edit 'id' to set your Kaggle username.")
        return

    result = subprocess.run(
        ["poetry", "run", "kaggle", "kernels", "push", "-p", str(kernel_dir)],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", help="Path to the .ipynb file")
    args = parser.parse_args()
    push_notebook(args.notebook)
