"""Record what produced a result, so that it can be reproduced later.

A number in an output file is only useful if the settings and the *inputs* behind it can
be recovered. Version strings are not enough: two runs of the same amaterasu version with
different pseudopotential files give different answers, and the file name alone does not
identify the file. Every dataset is therefore recorded with its MD5 digest -- the same
digest the PseudoDojo manifest publishes, so a mismatch can be traced to a specific
upstream revision.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__

__all__ = ["file_digest", "git_revision", "provenance"]


def file_digest(path: str | Path) -> str:
    """MD5 of a file, matching the digests published in the PseudoDojo manifests."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: str | Path | None = None) -> str:
    """The git commit amaterasu was run from, or ``"unknown"`` outside a repository."""
    directory = Path(root) if root else Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git missing
        return "unknown"
    return result.stdout.strip() or "unknown"


def provenance(pseudopotentials: dict[str, str] | None = None) -> dict[str, str]:
    """A flat, string-valued record suitable for HDF5 attributes."""
    record = {
        "amaterasu_version": __version__,
        "git_revision": git_revision(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for symbol, path in (pseudopotentials or {}).items():
        record[f"pseudopotential_{symbol}"] = str(path)
        try:
            record[f"pseudopotential_{symbol}_md5"] = file_digest(path)
        except OSError:  # pragma: no cover - the file was readable a moment ago
            record[f"pseudopotential_{symbol}_md5"] = "unavailable"
    return record
