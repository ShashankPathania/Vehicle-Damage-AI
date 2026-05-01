"""Download sample car damage images for quick inference demos."""

from __future__ import annotations

from pathlib import Path
from typing import List

import requests


def get_sample_urls() -> List[str]:
    """Return a list of publicly accessible sample image URLs."""
    return [
        "https://images.unsplash.com/photo-1617704548623-340376564e68?auto=format&fit=crop&w=1280&q=80",
        "https://images.unsplash.com/photo-1519641471654-76ce0107ad1b?auto=format&fit=crop&w=1280&q=80",
        "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=1280&q=80",
        "https://images.unsplash.com/photo-1549924231-f129b911e442?auto=format&fit=crop&w=1280&q=80",
        "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1280&q=80",
        "https://images.unsplash.com/photo-1508974239320-0a029497e820?auto=format&fit=crop&w=1280&q=80",
    ]


def download_file(url: str, out_path: Path, timeout: int = 20) -> None:
    """Download one URL and save it to disk."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    out_path.write_bytes(response.content)


def main() -> None:
    """Download sample images into samples/ directory."""
    root = Path(__file__).resolve().parent
    samples_dir = root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    urls = get_sample_urls()
    print(f"Downloading {len(urls)} sample images...")
    for idx, url in enumerate(urls, start=1):
        out_file = samples_dir / f"sample_{idx:02d}.jpg"
        try:
            download_file(url, out_file)
            print(f"[OK] {out_file.name}")
        except Exception as exc:
            print(f"[FAILED] {url} -> {exc}")

    print(f"Done. Samples are in: {samples_dir}")


if __name__ == "__main__":
    main()
