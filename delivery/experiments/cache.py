"""Deterministic result cache with integrity hashing and secret protection (§6.1, M2.6).

Keyed by canonical SHA-256 hash of configuration, seed, source code hashes, and run mode.
Supports --fresh bypass flag and prevents accidental caching of API secrets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "results" / "cache"

FORBIDDEN_SECRET_KEYS = {"api_key", "secret", "password", "token", "auth_token", "access_token"}


def _check_no_secrets(obj: Any) -> None:
    """Recursively validates that cached payload contains no credential keys."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_SECRET_KEYS:
                raise ValueError(f"Security invariant violated: Attempted to cache secret key '{k}'!")
            _check_no_secrets(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _check_no_secrets(item)


def compute_cache_key(
    config: Dict[str, Any],
    seed: int,
    source_hashes: Optional[Dict[str, str]] = None,
    run_mode: str = "quick",
) -> str:
    """Computes deterministic 64-char hex SHA-256 digest over execution parameters."""
    canonical_repr = {
        "config": config,
        "seed": seed,
        "source_hashes": source_hashes or {},
        "run_mode": run_mode,
    }
    dumped = json.dumps(canonical_repr, sort_keys=True, default=str)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


class ResultCache:
    """Disk-backed JSON cache for expensive experiment stage results."""

    def __init__(self, cache_dir: Optional[Path] = None, fresh: bool = False):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.fresh = fresh
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def has(self, key: str) -> bool:
        """Returns True if cache key exists and --fresh is not set."""
        if self.fresh:
            return False
        return self._path_for_key(key).exists()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached result if present and fresh is False."""
        if self.fresh:
            return None
        target = self._path_for_key(key)
        if not target.exists():
            return None
        try:
            with open(target, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, data: Dict[str, Any]) -> Path:
        """Caches data payload to disk after validating absence of credentials."""
        _check_no_secrets(data)
        target = self._path_for_key(key)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=str)
        return target

    def clear(self) -> int:
        """Removes all cached files in the cache directory."""
        count = 0
        if self.cache_dir.exists():
            for p in self.cache_dir.glob("*.json"):
                p.unlink()
                count += 1
        return count
