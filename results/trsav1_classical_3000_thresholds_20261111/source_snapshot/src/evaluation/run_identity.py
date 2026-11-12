"""Implementation identity for safe evaluation resume.

This covers source and installed package versions. A paper run must also
record model/resource hashes and its dataset/sampling configuration.
"""
import hashlib
import json
import platform
from importlib.metadata import distributions
from pathlib import Path


def implementation_fingerprint():
    source_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    packages = sorted((dist.metadata.get("Name", ""), dist.version) for dist in distributions())
    digest.update(json.dumps([platform.python_version(), packages], ensure_ascii=True).encode("utf-8"))
    return digest.hexdigest()


def ensure_run_manifest(path, metadata, cached_paths=()):
    """Refuse unknown/mismatched cached results before any prediction reuse."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Run manifest differs; use a new output directory.")
        return
    if any(Path(cached).exists() for cached in cached_paths):
        raise ValueError("Cached predictions have no run manifest; use a new output directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
