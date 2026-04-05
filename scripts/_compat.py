"""Windows環境でのUnicode互換設定。各スクリプトの先頭でimportする。

Usage:
    import _compat  # noqa: F401
"""
import os
import sys

os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
