"""Fetch a pinned source file for a vendored dataset, and refuse it unless its size and sha256 match the pin.

The shared step of the data scripts that download their own source (`scripts/diabetes.py`): a dataset script names the
URL, the byte count and the sha256 of the one file it was written against, calls `fetch`, and writes its TypeScript
module from the bytes it gets back. A download is kept in `.scratch/vendor/` (ignored by git) and reused while it still
matches the pin, so a rerun works offline; a source given on the command line (`--source`) is checked the same way. A
source that has changed upstream fails loudly instead of changing the vendored data.
"""

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".scratch" / "vendor"


@dataclass(frozen=True)
class Pin:
    """One source file: where it is, how long it is, and its sha256."""

    url: str
    size: int
    sha256: str

    @property
    def name(self) -> str:
        return self.url.rsplit("/", 1)[-1]


def verify(pin: Pin, data: bytes, origin: str) -> bytes:
    """The bytes, if they are the pinned file; otherwise exit with what differs."""
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != pin.size or digest != pin.sha256:
        raise SystemExit(
            f"{origin}: {len(data)} bytes with sha256 {digest}, "
            f"not the pinned {pin.size} bytes with sha256 {pin.sha256}"
        )
    return data


def fetch(pin: Pin, source: Path | None = None) -> bytes:
    """The pinned file's bytes: from `source` when given, else the cache, else a download (which is then cached)."""
    if source is not None:
        return verify(pin, source.read_bytes(), str(source))
    cached = CACHE / pin.name
    if cached.exists() and hashlib.sha256(cached.read_bytes()).hexdigest() == pin.sha256:
        return cached.read_bytes()
    request = urllib.request.Request(pin.url, headers={"User-Agent": "aifn-engine data scripts"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = verify(pin, response.read(), pin.url)
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    print(f"fetched {pin.url} ({pin.size} bytes, sha256 verified)")
    return data
