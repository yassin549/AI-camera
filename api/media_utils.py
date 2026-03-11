"""Shared helpers for resolving media paths and thumbnails."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence


MEDIA_SAMPLE_SUBDIRS: Sequence[str] = (
    "faces",
    "body",
    "bodies",
    "samples/faces",
    "samples/bodies",
    "data/faces",
    "data/bodies",
)


def _dedupe_rel_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for rel in paths:
        key = rel.as_posix()
        if key in seen:
            continue
        seen.add(key)
        out.append(rel)
    return out


def _candidate_rel_paths(raw_path: Optional[str], media_root: Path) -> List[Path]:
    if not raw_path:
        return []
    path = Path(str(raw_path))
    candidates: List[Path] = []
    if path.is_absolute():
        try:
            candidates.append(path.relative_to(media_root))
        except Exception:
            candidates = []
    else:
        candidates.append(Path(str(path).replace("\\", "/").lstrip("/")))

    basename = path.name
    if basename:
        for subdir in MEDIA_SAMPLE_SUBDIRS:
            candidates.append(Path(subdir) / basename)
    return _dedupe_rel_paths(candidates)


def resolve_media_url(raw_path: Optional[str], media_root: Path) -> Optional[str]:
    """Return a /media/* URL if the path resolves within media_root."""
    for rel in _candidate_rel_paths(raw_path, media_root):
        candidate = media_root / rel
        try:
            if candidate.exists() and candidate.is_file():
                return f"/media/{rel.as_posix()}"
        except Exception:
            continue
    return None


def iter_media_candidates(raw_path: Optional[str], media_root: Path) -> Iterable[Path]:
    """Yield absolute filesystem candidates for a media path."""
    for rel in _candidate_rel_paths(raw_path, media_root):
        yield media_root / rel


def discover_face_thumb(identity_id: int, media_root: Path) -> Optional[str]:
    search_dirs = [
        media_root / "faces",
        media_root / "data" / "faces",
        media_root / "samples" / "faces",
    ]
    patterns = [
        f"{identity_id}_*.jpg",
        f"{identity_id}_*.jpeg",
        f"{identity_id}_*.png",
        f"face_*_t{identity_id}.jpg",
        f"face_*_t{identity_id}.jpeg",
    ]
    for directory in search_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if not matches:
                continue
            candidate = matches[0]
            try:
                rel = candidate.relative_to(media_root)
            except Exception:
                continue
            return f"/media/{rel.as_posix()}"
    return None
