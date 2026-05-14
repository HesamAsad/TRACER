"""Key-based lookup of dense spatial captions from a JSONL file.

JSONL lines must be objects with keys ``image`` and ``dense_caption``. Keys are
normalized so CSV ``filepath`` columns (absolute-ish, with ``./``/``data_location``
prefix) resolve against JSONL ``image`` keys (relative, no prefix).
"""

import json
import logging
from typing import Optional


class SpatialCaptionIndex:
    def __init__(self, jsonl_path: str, data_location: str):
        self.jsonl_path = jsonl_path
        self.data_location = (data_location or "").rstrip("/")
        self._index = {}

        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                img = entry.get("image")
                cap = entry.get("dense_caption")
                if img is None or cap is None:
                    continue
                key = self._normalize(img)
                self._index[key] = cap

    def _normalize(self, path: str) -> str:
        p = path
        if p.startswith("./"):
            p = p[2:]
        if self.data_location and p.startswith(self.data_location):
            p = p[len(self.data_location):]
        return p.lstrip("/")

    def get(self, filepath: str) -> Optional[str]:
        key = self._normalize(filepath)
        cap = self._index.get(key)
        if cap is not None:
            return cap
        # Suffix fallback handles extra prefix components (e.g. "ILSVRC2012/") that
        # appear in the CSV but not the JSONL.
        parts = key.split("/")
        if len(parts) >= 3:
            return self._index.get("/".join(parts[-3:]))
        return None

    def match_rate(self, filepaths) -> float:
        if not filepaths:
            return 0.0
        hits = sum(1 for fp in filepaths if self.get(fp) is not None)
        return hits / len(filepaths)

    def __len__(self) -> int:
        return len(self._index)


def log_spatial_index_stats(index: SpatialCaptionIndex, filepaths, logger=None) -> float:
    rate = index.match_rate(filepaths)
    msg = (
        f"[spatial-captions] loaded={len(index)} entries; "
        f"match rate against CSV filepaths = {rate:.2%}"
    )
    print(msg, flush=True)
    (logger.info if logger is not None else logging.info)(msg)
    if rate < 0.80:
        warn = (
            f"[spatial-captions] WARNING: match rate {rate:.2%} < 80%. "
            f"Check path normalization (data_location='{index.data_location}')."
        )
        print(warn, flush=True)
        (logger.warning if logger is not None else logging.warning)(warn)
    return rate
