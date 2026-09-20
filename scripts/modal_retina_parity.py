#!/usr/bin/env python3
"""Strict baseline/current RetinaFace parity benchmark on the same Modal T4."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from yt2avsr.modal_dependency_pins import (
    AUTO_AVSR_COMMIT,
    FACE_ALIGNMENT_COMMIT,
    FACE_DETECTION_COMMIT,
)

BASELINE_COMMIT = "dfae6fbd1b24820adc3da880030f4c70c2375a8f"
VIDEO_SHA256 = "7c72a1e18a73d312e6bb93106f02ce43bdbe66bc060c2e0067cc94a063ce7dbc"
VIDEO_PATH = Path("/workspace/test_inputs/video_b.mp4")
REPO_URL = "https://github.com/iboRotti52/youtube-to-autoavsr.git"

try:
    import modal
except ImportError:
    modal = None


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


CURRENT_COMMIT = _current_commit()


if modal is not None:
    app = modal.App("modal-retina-parity-benchmark")
    volume = modal.Volume.from_name("avsr-workspace", create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "git", "git-lfs", "curl", "build-essential", "unzip")
        .run_commands(
            "git lfs install",
            "curl -fsSL https://deno.land/install.sh | sh",
            "ln -s /root/.deno/bin/deno /usr/local/bin/deno",
        )
        .pip_install(
            "torch",
            "torchvision",
            "torchaudio",
            "nvidia-cublas-cu12",
            "nvidia-cudnn-cu12",
            "typer>=0.12,<1",
            "pydantic>=2.7,<3",
            "PyYAML>=6,<7",
            "yt-dlp>=2025.1.15",
            "faster-whisper>=1.1,<2",
            "opencv-python-headless>=4.10,<5",
            "mediapipe==0.10.14",
            "protobuf>=3.20,<5",
            "numpy>=1.26,<3",
            "tqdm>=4.66,<5",
            "webvtt-py>=0.5,<1",
            "huggingface-hub>=0.24,<1",
            "scipy",
            "ffmpeg-python",
            "sentencepiece",
            "scikit-image",
        )
        .run_commands(
            "mkdir -p /root/youtube-to-autoavsr/external",
            f"git clone --no-checkout https://github.com/mpc001/auto_avsr.git /root/youtube-to-autoavsr/external/auto_avsr && cd /root/youtube-to-autoavsr/external/auto_avsr && git fetch --depth 1 origin {AUTO_AVSR_COMMIT} && git checkout --detach {AUTO_AVSR_COMMIT}",
            f"git clone --no-checkout https://github.com/hhj1897/face_detection.git /root/youtube-to-autoavsr/external/face_detection && cd /root/youtube-to-autoavsr/external/face_detection && git fetch --depth 1 origin {FACE_DETECTION_COMMIT} && git checkout --detach {FACE_DETECTION_COMMIT} && git lfs pull && pip install -e .",
            f"git clone --no-checkout https://github.com/hhj1897/face_alignment.git /root/youtube-to-autoavsr/external/face_alignment && cd /root/youtube-to-autoavsr/external/face_alignment && git fetch --depth 1 origin {FACE_ALIGNMENT_COMMIT} && git checkout --detach {FACE_ALIGNMENT_COMMIT} && git lfs pull && pip install -e .",
            f"git clone --no-checkout {REPO_URL} /root/baseline && cd /root/baseline && git fetch --depth 1 origin {BASELINE_COMMIT} && git checkout --detach {BASELINE_COMMIT}",
            "python3 -c 'from faster_whisper import WhisperModel; WhisperModel(\"large-v3-turbo\", device=\"cpu\", compute_type=\"int8\")'",
            "echo '/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib' > /etc/ld.so.conf.d/nvidia.conf && echo '/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib' >> /etc/ld.so.conf.d/nvidia.conf && ldconfig",
        )
        .env(
            {
                "PYTHONPATH": "/root/youtube-to-autoavsr/external/auto_avsr",
                "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib",
            }
        )
        .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/current/src")
        .add_local_dir(str(REPO_ROOT / "configs"), remote_path="/root/current/configs")
    )

    @app.function(
        image=image,
        gpu="T4",
        cpu=4.0,
        timeout=60 * 60 * 2,
        volumes={"/workspace": volume},
    )
    def run_retina_parity_benchmark() -> dict:
        import json
        import os
        import shutil
        import time

        def records(workspace: Path) -> dict[str, dict]:
            result = {}
            for path in sorted((workspace / "clips").glob("*/*/metadata.json")):
                metadata = json.loads(path.read_text(encoding="utf-8"))
                key = f"{metadata['item_id']}/{metadata['segment_id']}"
                result[key] = metadata
            return result

        def run_pipeline(source_root: Path, workspace: Path) -> float:
            shutil.rmtree(workspace, ignore_errors=True)
            workspace.mkdir(parents=True, exist_ok=True)
            runner = f"""
from pathlib import Path
from yt2avsr.config import load_config
from yt2avsr.pipeline import Pipeline

cfg = load_config(Path({str(source_root / 'configs' / 'retina_1080p.yaml')!r}))
cfg.workspace = Path({str(workspace)!r})
cfg.sources.processed_file = cfg.workspace / 'processed_sources.txt'
cfg.sources.auto_sync_hf = False
cfg.auto_avsr.repo_dir = Path('/root/youtube-to-autoavsr/external/auto_avsr')
cfg.auto_avsr.detector = 'retinaface'
cfg.normalization.max_height = 1080
cfg.download.format = 'bestvideo[height<=1080]+bestaudio/best[height<=1080]'
cfg.transcription.device = 'cuda'
Pipeline(cfg, force=True, profile='no_voiceover').process_local(Path({str(VIDEO_PATH)!r}))
"""
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{source_root / 'src'}:/root/youtube-to-autoavsr/external/auto_avsr"
            started = time.perf_counter()
            subprocess.run([sys.executable, "-c", runner], env=env, check=True)
            return time.perf_counter() - started

        volume.reload()
        if not VIDEO_PATH.exists():
            return {"passed": False, "error": f"Missing pinned video: {VIDEO_PATH}"}
        actual_hash = hashlib.sha256(VIDEO_PATH.read_bytes()).hexdigest()
        if actual_hash != VIDEO_SHA256:
            return {
                "passed": False,
                "error": f"Pinned video hash mismatch: expected {VIDEO_SHA256}, got {actual_hash}",
            }

        baseline_workspace = Path("/workspace/runs/retina_parity_baseline")
        current_workspace = Path("/workspace/runs/retina_parity_current")
        print(f"Baseline commit: {BASELINE_COMMIT}", flush=True)
        print(f"Current commit: {CURRENT_COMMIT}", flush=True)
        print(f"Pinned video SHA256: {actual_hash}", flush=True)
        print("Environment: NVIDIA T4, RetinaFace, configs/retina_1080p.yaml", flush=True)

        baseline_time = run_pipeline(Path("/root/baseline"), baseline_workspace)
        print(f"Baseline finished in {baseline_time:.1f}s", flush=True)
        current_time = run_pipeline(Path("/root/current"), current_workspace)
        print(f"Current branch finished in {current_time:.1f}s", flush=True)

        baseline = records(baseline_workspace)
        current = records(current_workspace)
        baseline_ids = set(baseline)
        current_ids = set(current)
        status_mismatches = []
        text_mismatches = []
        original_text_mismatches = []
        timestamp_mismatches = []

        for clip_id in sorted(baseline_ids & current_ids):
            expected = baseline[clip_id]
            actual = current[clip_id]
            if expected.get("quality_status") != actual.get("quality_status"):
                status_mismatches.append(clip_id)
            if expected.get("text") != actual.get("text"):
                text_mismatches.append(clip_id)
            if expected.get("original_text") != actual.get("original_text"):
                original_text_mismatches.append(clip_id)
            if (
                abs(float(expected.get("start", 0)) - float(actual.get("start", 0))) > 0.001
                or abs(float(expected.get("end", 0)) - float(actual.get("end", 0))) > 0.001
            ):
                timestamp_mismatches.append(clip_id)

        result = {
            "passed": (
                baseline_ids == current_ids
                and not status_mismatches
                and not text_mismatches
                and not original_text_mismatches
                and not timestamp_mismatches
            ),
            "baseline_commit": BASELINE_COMMIT,
            "current_commit": CURRENT_COMMIT,
            "video_sha256": actual_hash,
            "baseline_count": len(baseline),
            "current_count": len(current),
            "missing_clip_ids": sorted(baseline_ids - current_ids),
            "extra_clip_ids": sorted(current_ids - baseline_ids),
            "quality_status_mismatches": status_mismatches,
            "text_mismatches": text_mismatches,
            "original_text_mismatches": original_text_mismatches,
            "timestamp_mismatches_over_1ms": timestamp_mismatches,
            "baseline_seconds": baseline_time,
            "current_seconds": current_time,
        }
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        shutil.rmtree(baseline_workspace, ignore_errors=True)
        shutil.rmtree(current_workspace, ignore_errors=True)
        volume.commit()
        return result


@app.local_entrypoint()
def main():
    result = run_retina_parity_benchmark.remote()
    print("Parity result:")
    print(result)
    if not result.get("passed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
