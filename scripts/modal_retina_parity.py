#!/usr/bin/env python3
"""Modal Production-Path Parity Benchmark: RetinaFace on NVIDIA T4 GPU.

Compares the perf branch output on pinned regression video 'video_b.mp4'
against the pinned baseline regression snapshot (dfae6fb) using the official
RetinaFace detector and T4 GPU in Modal.

Verifies:
1. Exact segment count and IDs.
2. Exact quality_status (accepted / review / rejected).
3. Exact transcript text (text and original_text).
4. Timestamp tolerance <= 1ms (0.001s).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repository root (never raise or validate paths at module import level!)
REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    import modal
except ImportError:
    modal = None

APP_NAME = "modal-retina-parity-benchmark"

if modal is not None:
    app = modal.App(APP_NAME)
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
            "nvidia-cublas-cu12",
            "nvidia-cudnn-cu12",
        )
        .run_commands(
            "mkdir -p /root/youtube-to-autoavsr/external",
            "git clone --depth 1 https://github.com/mpc001/auto_avsr.git /root/youtube-to-autoavsr/external/auto_avsr",
            "git clone https://github.com/hhj1897/face_detection.git /root/youtube-to-autoavsr/external/face_detection && cd /root/youtube-to-autoavsr/external/face_detection && git lfs pull && pip install -e .",
            "git clone https://github.com/hhj1897/face_alignment.git /root/youtube-to-autoavsr/external/face_alignment && cd /root/youtube-to-autoavsr/external/face_alignment && git lfs pull && pip install -e .",
            "python3 -c 'from faster_whisper import WhisperModel; WhisperModel(\"large-v3-turbo\", device=\"cpu\", compute_type=\"int8\")'",
            "echo '/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib' > /etc/ld.so.conf.d/nvidia.conf && echo '/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib' >> /etc/ld.so.conf.d/nvidia.conf && ldconfig",
        )
        .env({
            "PYTHONPATH": "/root/youtube-to-autoavsr/external/auto_avsr:/root/perf/src",
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib",
        })
        .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/perf/src")
        .add_local_dir(str(REPO_ROOT / "configs"), remote_path="/root/perf/configs")
    )


    @app.function(
        image=image,
        gpu="T4",
        cpu=4.0,
        timeout=60 * 60,  # 60 min
        volumes={"/workspace": volume},
    )
    def run_retina_parity_benchmark() -> dict:
        """Runs perf branch with RetinaFace on T4 GPU and validates 100% strict parity against pinned baseline snapshot."""
        import json
        import os
        import shutil
        import sys
        import tarfile
        import time
        from pathlib import Path

        print("\n" + "=" * 70)
        print("🔍 PRODUCTION-PATH PARITY BENCHMARK: RETINAFACE ON NVIDIA T4")
        print("=" * 70, flush=True)

        # 1. Path Validation inside function (never crashes module import)
        video_path = Path("/workspace/test_inputs/video_b.mp4")
        if not video_path.exists():
            err = f"Fatal: Input video not found at {video_path}"
            print(f"[modal-bench] {err}", flush=True)
            return {"passed": False, "error": err}

        # 2. Locate / extract pinned baseline snapshot metadata from volume
        volume.reload()
        tar_path = Path("/workspace/snapshots/baseline_b_metadata.tar.gz")
        target_dir = Path("/workspace/snapshots/baseline_video_b")
        if tar_path.exists() and not (target_dir / "video_b").exists():
            print(f"[modal-bench] Extracting {tar_path} into {target_dir}...", flush=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(target_dir)

        snapshot_candidates = [
            target_dir / "video_b",
            target_dir,
            Path("/workspace/snapshots/baseline_video_b/video_b"),
            Path("/workspace/snapshots/baseline_video_b"),
        ]
        base_snapshot_dir = next(
            (c for c in snapshot_candidates if c.exists() and any(c.glob("*/metadata.json"))),
            None,
        )
        if not base_snapshot_dir:
            err = f"Fatal: Baseline regression snapshot not found in candidates: {[str(c) for c in snapshot_candidates]}"
            print(f"[modal-bench] {err}", flush=True)
            return {"passed": False, "error": err}

        # 3. Load pinned baseline snapshot records (DO NOT RE-RUN BASELINE)
        print(f"[modal-bench] Loading pinned baseline snapshot from: {base_snapshot_dir}...", flush=True)
        base_records = {}
        for p in sorted(base_snapshot_dir.glob("*/metadata.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                base_records[rec["segment_id"]] = rec
            except Exception as e:
                print(f"[modal-bench] Warning reading baseline {p}: {e}", flush=True)

        print(f"[modal-bench] Pinned baseline snapshot loaded: {len(base_records)} clips.", flush=True)
        if not base_records:
            return {"passed": False, "error": "No baseline records found in snapshot directory"}

        # 4. Run PERF pipeline with RetinaFace on T4 GPU
        print("\n" + "=" * 70)
        print("🚀 RUNNING PERF BRANCH WITH RETINAFACE ON T4 GPU...")
        print("=" * 70, flush=True)

        run_ws = Path("/workspace/runs/retina_parity_perf_run")
        shutil.rmtree(run_ws, ignore_errors=True)
        run_ws.mkdir(parents=True, exist_ok=True)

        sys.path.insert(0, "/root/perf/src")
        from yt2avsr.config import load_config
        from yt2avsr.pipeline import Pipeline

        cfg = load_config(Path("/root/perf/configs/retina_1080p.yaml"))
        cfg.workspace = run_ws
        cfg.sources.processed_file = run_ws / "processed_sources.txt"
        cfg.sources.auto_sync_hf = False
        cfg.auto_avsr.repo_dir = Path("/root/youtube-to-autoavsr/external/auto_avsr")
        cfg.auto_avsr.detector = "retinaface"
        cfg.transcription.device = "cuda"

        t0_perf = time.perf_counter()
        pipe = Pipeline(cfg, force=True, profile="no_voiceover")
        item = pipe.process_local(video_path)
        perf_duration = time.perf_counter() - t0_perf
        print(f"✅ Perf pipeline processing finished in {perf_duration:.1f}s. Item: {item.get('id')}", flush=True)

        # 5. Load PERF records
        perf_records = {}
        for p in sorted((run_ws / "clips").glob("*/*/metadata.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                perf_records[rec["segment_id"]] = rec
            except Exception as e:
                print(f"[modal-bench] Warning reading perf {p}: {e}", flush=True)

        print(f"[modal-bench] Perf output loaded: {len(perf_records)} clips.", flush=True)

        # 6. Verify STRICT Parity
        print("\n" + "=" * 70)
        print("🔍 VERIFYING 100% STRICT PARITY (BASELINE SNAPSHOT VS PERF ON T4)")
        print("=" * 70, flush=True)

        base_keys = sorted(base_records.keys())
        perf_keys = sorted(perf_records.keys())

        if set(base_keys) != set(perf_keys):
            missing = sorted(set(base_keys) - set(perf_keys))
            extra = sorted(set(perf_keys) - set(base_keys))
            err = f"Clip count mismatch: baseline has {len(base_keys)}, perf has {len(perf_keys)}. Missing: {missing[:5]}, Extra: {extra[:5]}"
            print(f"❌ {err}", flush=True)
            return {"passed": False, "error": err}

        status_mismatches = []
        text_mismatches = []
        orig_text_mismatches = []
        timestamp_mismatches = []

        base_counts = {"accepted": 0, "review": 0, "rejected": 0}
        perf_counts = {"accepted": 0, "review": 0, "rejected": 0}

        for sid in base_keys:
            b = base_records[sid]
            p = perf_records[sid]

            b_st = b.get("quality_status")
            p_st = p.get("quality_status")
            base_counts[b_st] = base_counts.get(b_st, 0) + 1
            perf_counts[p_st] = perf_counts.get(p_st, 0) + 1

            if b_st != p_st:
                status_mismatches.append((sid, b_st, p_st))

            b_txt = b.get("text", "").strip()
            p_txt = p.get("text", "").strip()
            if b_txt != p_txt:
                text_mismatches.append((sid, b_txt, p_txt))

            b_orig = b.get("original_text", "").strip()
            p_orig = p.get("original_text", "").strip()
            if b_orig != p_orig:
                orig_text_mismatches.append((sid, b_orig, p_orig))

            b_start, b_end = float(b.get("start", 0)), float(b.get("end", 0))
            p_start, p_end = float(p.get("start", 0)), float(p.get("end", 0))
            if abs(b_start - p_start) > 0.001 or abs(b_end - p_end) > 0.001:
                timestamp_mismatches.append((sid, (b_start, b_end), (p_start, p_end)))

        print(f"  • Baseline clip distribution: {base_counts}")
        print(f"  • Perf clip distribution:     {perf_counts}")
        print(f"  • Wall-clock processing time: {perf_duration:.1f}s")

        passed = True
        if status_mismatches:
            print(f"❌ Quality Status mismatches ({len(status_mismatches)}): {status_mismatches[:5]}", flush=True)
            passed = False
        if text_mismatches:
            print(f"❌ Text mismatches ({len(text_mismatches)}): {text_mismatches[:5]}", flush=True)
            passed = False
        if orig_text_mismatches:
            print(f"❌ Original Text mismatches ({len(orig_text_mismatches)}): {orig_text_mismatches[:5]}", flush=True)
            passed = False
        if timestamp_mismatches:
            print(f"❌ Timestamp mismatches >1ms ({len(timestamp_mismatches)}): {timestamp_mismatches[:5]}", flush=True)
            passed = False

        if passed:
            print(f"\n🎉 100% STRICT PARITY VERIFIED ON T4 GPU WITH RETINAFACE! ({len(base_keys)} clips identical)", flush=True)

        # Cleanup run directory on volume
        try:
            shutil.rmtree(run_ws, ignore_errors=True)
            volume.commit()
        except Exception:
            pass

        return {
            "passed": passed,
            "total_clips": len(base_keys),
            "base_counts": base_counts,
            "perf_counts": perf_counts,
            "perf_duration_s": perf_duration,
            "status_mismatches": status_mismatches,
            "text_mismatches": text_mismatches,
            "orig_text_mismatches": orig_text_mismatches,
            "timestamp_mismatches": timestamp_mismatches,
        }


@app.local_entrypoint()
def main():
    result = run_retina_parity_benchmark.remote()
    print("\nBenchmark Result Summary:")
    print(result)
    if not result.get("passed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
