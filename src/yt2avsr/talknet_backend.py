from __future__ import annotations

from dataclasses import dataclass
import os
import pickle
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np

from .config import TalkNetConfig


@dataclass
class TalkNetResult:
    status: str
    speaking_ratio: float
    mean_probability: float
    max_probability: float
    reason: str | None = None


def _flatten_scores(value) -> np.ndarray:
    values = []

    def visit(item):
        if isinstance(item, np.ndarray):
            values.append(item.astype(np.float32).reshape(-1))
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (int, float, np.integer, np.floating)):
            values.append(np.asarray([item], dtype=np.float32))

    visit(value)
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def run_talknet(video_path: Path, cfg: TalkNetConfig) -> TalkNetResult:
    repo = cfg.repo_dir.resolve()
    python = cfg.python_executable.resolve()

    if not (repo / "demoTalkNet.py").exists():
        raise RuntimeError("TalkNet repository missing. Run: yt2avsr setup-talknet")
    if not python.exists():
        raise RuntimeError("TalkNet environment missing. Run: yt2avsr setup-talknet")

    run_id = f"yt2avsr_{uuid.uuid4().hex[:10]}"
    demo_dir = repo / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    input_path = demo_dir / f"{run_id}.mp4"
    shutil.copy2(video_path, input_path)

    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    proc = subprocess.run(
        [str(python), "demoTalkNet.py", "--videoName", run_id],
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    result_dir = demo_dir / run_id
    scores_path = result_dir / "pywork" / "scores.pckl"

    try:
        if proc.returncode != 0 or not scores_path.exists():
            raise RuntimeError(f"TalkNet failed:\n{(proc.stdout or '')[-3000:]}")

        with scores_path.open("rb") as handle:
            scores = _flatten_scores(pickle.load(handle))

        if scores.size == 0:
            return TalkNetResult(
                status="rejected",
                speaking_ratio=0.0,
                mean_probability=0.0,
                max_probability=0.0,
                reason="no_on_screen_active_speaker",
            )

        if np.any(scores < 0.0) or np.any(scores > 1.0):
            scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))

        speaking_ratio = float(np.mean(scores >= cfg.min_speaking_probability))
        mean_probability = float(np.mean(scores))
        max_probability = float(np.max(scores))

        if speaking_ratio >= cfg.accept_min_speaking_ratio:
            status, reason = "accepted", None
        elif speaking_ratio >= cfg.review_min_speaking_ratio:
            status, reason = "review", "borderline_on_screen_speaker"
        else:
            status, reason = "rejected", "no_on_screen_active_speaker"

        return TalkNetResult(
            status=status,
            speaking_ratio=round(speaking_ratio, 5),
            mean_probability=round(mean_probability, 5),
            max_probability=round(max_probability, 5),
            reason=reason,
        )
    finally:
        input_path.unlink(missing_ok=True)
        if not cfg.keep_workdirs:
            shutil.rmtree(result_dir, ignore_errors=True)
