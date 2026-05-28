from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: str | None, *, logger: logging.Logger | None = None, log_prefix: str = ""):
        self.path = path
        self.logger = logger
        self.log_prefix = log_prefix

    def emit(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        if self.logger is not None:
            self.logger.info("%s %s", self.log_prefix, line)
        if not self.path:
            return
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
