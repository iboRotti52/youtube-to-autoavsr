"""Cloud sync for the shared Auto-AVSR dataset.

Every teammate runs the pipeline locally, then pushes their processed clips to a
single *private* Hugging Face dataset repo. Each contributor writes into their own
sub-folder (``data/<contributor>/...``) so nobody overwrites anyone else's clips.

At training time you pull the whole repo once and every contributor's data lands
under one root, ready to feed Auto-AVSR.

Auth: run ``huggingface-cli login`` once, or set the ``HF_TOKEN`` env var, or pass
``--token``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

# Files produced per segment by the pipeline.
CLIP_FILES = (
    "source.mp4",
    "active_speaker.mp4",
    "mouth.mp4",
    "audio.wav",
    "transcript.txt",
    "metadata.json",
)

VALID_STATUSES = ("accepted", "review", "rejected")


def _get_token(token: str | None) -> str | None:
    """Explicit token > HF_TOKEN env var > cached CLI login (returns None)."""
    return token or os.environ.get("HF_TOKEN") or None


def _slug(name: str) -> str:
    """Make a safe, path-friendly contributor folder name."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    return name.strip("-._") or "unknown"


def _default_contributor(token: str | None) -> str:
    """Fall back to the logged-in Hugging Face username."""
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).whoami()
        return _slug(info.get("name") or "unknown")
    except Exception:
        return "unknown"


def _iter_clip_dirs(workspace: Path, statuses: Iterable[str]) -> list[Path]:
    """Clip folders whose metadata.json quality_status is in `statuses`."""
    wanted = set(statuses)
    dirs: list[Path] = []
    for meta in sorted((workspace / "clips").glob("*/*/metadata.json")):
        try:
            rec = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("quality_status") in wanted:
            dirs.append(meta.parent)
    return dirs


def push(
    workspace: Path,
    repo_id: str,
    contributor: str | None = None,
    statuses: Iterable[str] = ("accepted", "review"),
    token: str | None = None,
    private: bool = True,
    include_source: bool = False,
    include_audio: bool = False,
) -> str:
    """Upload selected local clips + manifests to the shared HF dataset repo.

    Uploads only what visual lip-reading (VSR) needs: mouth.mp4, transcript.txt,
    metadata.json. The large raw ``source.mp4`` and the ``audio.wav`` track are
    skipped by default. Pass ``include_source=True`` and/or ``include_audio=True``
    to add them (audio is needed only if you train the audio-visual model).

    Returns the path-in-repo that was written to.
    """
    from huggingface_hub import HfApi

    token = _get_token(token)
    statuses = [s for s in statuses if s in VALID_STATUSES]
    if not statuses:
        raise ValueError(f"statuses must be a subset of {VALID_STATUSES}")

    workspace = Path(workspace)
    if not (workspace / "clips").exists():
        raise FileNotFoundError(
            f"No clips found under {workspace/'clips'}. Run the pipeline first."
        )

    contributor = _slug(contributor) if contributor else _default_contributor(token)
    clip_dirs = _iter_clip_dirs(workspace, statuses)
    if not clip_dirs:
        raise RuntimeError(
            f"No clips with status {statuses} found. Nothing to upload."
        )

    # Precise single-commit selection: only chosen clip folders + all manifests.
    allow_patterns = ["manifests/**"]
    for d in clip_dirs:
        rel = d.relative_to(workspace).as_posix()  # clips/<item>/<segment>
        allow_patterns.append(f"{rel}/**")

    # Skip files VSR training doesn't need. active_speaker.mp4 is never produced in
    # v0.7; source.mp4 is large debug-only; audio.wav is only for the AV model.
    ignore_patterns = ["**/active_speaker.mp4"]
    if not include_source:
        ignore_patterns.append("**/source.mp4")
    if not include_audio:
        ignore_patterns.append("**/audio.wav")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    path_in_repo = f"data/{contributor}"
    api.upload_folder(
        folder_path=str(workspace),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        commit_message=f"Add {len(clip_dirs)} clips from {contributor} ({','.join(statuses)})",
    )
    return f"{repo_id}:{path_in_repo} ({len(clip_dirs)} clips)"


def pull(
    repo_id: str,
    dest: Path,
    token: str | None = None,
    contributor: str | None = None,
) -> Path:
    """Download the shared dataset (all contributors, or just one) for training."""
    from huggingface_hub import snapshot_download

    token = _get_token(token)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    allow_patterns = None
    if contributor:
        allow_patterns = [f"data/{_slug(contributor)}/**"]

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        token=token,
        allow_patterns=allow_patterns,
    )
    return Path(local)
