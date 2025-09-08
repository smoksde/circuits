import json
import os
import tempfile
from pathlib import Path

def load(file_path: Path) -> dict:
    if not file_path.exists():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        bak = file_path.with_suffix(".bad")
        os.replace(file_path, bak)
        print(f"Corrupt cache backed up to {bak.name}")
        return {}

def save(file_path: Path, cache: dict):
    fd, tmp_path = tempfile.mkstemp(
        dir=file_path.parent, prefix=".cache_tmp_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
            json.dump(cache, tmpf, indent=2, sort_keys=True)
            tmpf.flush()
            os.fsync(tmpf.fileno())
        os.replace(tmp_path, file_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def update(file_path: Path, key1: str, key2: str, value: dict):
    cache = load(file_path)
    cache.setdefault(key1, {})[str(key2)] = value
    save(file_path, cache)

def update_field(file_path: Path, key1: str, key2: str, field: str, value: int):
    cache = load(file_path)
    entry = cache.setdefault(key1, {}).setdefault(str(key2), {})
    entry[field] = value
    save(file_path, cache)

def get(file_path: Path, key1: str, key2: str):
    cache = load(file_path)
    return cache.get(key1, {}).get(str(key2))