"""JSONL run logger (PORT of efs/ml_workspace/kaggle/src/utils/logger.py, verbatim)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, run_dir: str | Path, run_name: str | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
        self.path = self.run_dir / f"{self.run_name}.jsonl"
        self._fp = self.path.open("a", buffering=1)

    def log(self, **fields: Any) -> None:
        record = {"ts": time.time(), **fields}
        line = json.dumps(record, default=str, ensure_ascii=False)
        self._fp.write(line + "\n")
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
