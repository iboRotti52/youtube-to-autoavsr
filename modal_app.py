"""Modal App for YouTube -> Auto-AVSR with RetinaFace + 1080p on Serverless GPU.

Runs the official Auto-AVSR preprocessing pipeline (RetinaFace landmark tracker,
25fps normalization, 1080p resolution, Whisper large-v3-turbo transcription,
visual quality checks) on a cloud GPU (e.g. NVIDIA T4 or A10G).

Processed clips are uploaded directly from the cloud container to the team's
shared Hugging Face dataset (configs/default.yaml: cloud.repo_id).
"""
from __future__ import annotations

import io
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent

# Ensure src/ is on local python path so helpers can be imported locally
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import modal
except ImportError:
    modal = None  # Handled in main / CLI when executed without modal installed


APP_NAME = "youtube-to-autoavsr"

if modal is not None:
    app = modal.App(APP_NAME)

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "git", "git-lfs", "curl", "unzip", "build-essential")
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
            "git clone --depth 1 https://github.com/mpc001/auto_avsr.git /root/youtube-to-autoavsr/external/auto_avsr",
            "git clone https://github.com/hhj1897/face_detection.git /root/youtube-to-autoavsr/external/face_detection && cd /root/youtube-to-autoavsr/external/face_detection && git lfs pull && pip install -e .",
            "git clone https://github.com/hhj1897/face_alignment.git /root/youtube-to-autoavsr/external/face_alignment && cd /root/youtube-to-autoavsr/external/face_alignment && git lfs pull && pip install -e .",
            "python3 -c 'import os, site; pths = [os.path.join(s, \"nvidia\", sub, \"lib\") for s in site.getsitepackages() for sub in (\"cublas\", \"cudnn\") if os.path.exists(os.path.join(s, \"nvidia\", sub, \"lib\"))]; open(\"/etc/ld.so.conf.d/nvidia.conf\", \"w\").write(\"\\n\".join(pths) + \"\\n\"); os.system(\"ldconfig\")'",
            "python3 -c 'from faster_whisper import WhisperModel; WhisperModel(\"large-v3-turbo\", device=\"cpu\", compute_type=\"int8\")'",
        )
        .env({
            "PYTHONPATH": "/root/youtube-to-autoavsr/src:/root/youtube-to-autoavsr/external/auto_avsr",
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib",
        })
        .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/youtube-to-autoavsr/src")
        .add_local_dir(str(REPO_ROOT / "configs"), remote_path="/root/youtube-to-autoavsr/configs")
    )


    volume = modal.Volume.from_name("avsr-workspace", create_if_missing=True)

    @app.function(
        image=image,
        gpu="T4",
        cpu=4.0,
        timeout=60 * 60 * 2,  # up to 2 hours per video
        volumes={"/workspace": volume},
    )
    def process_single_video_modal(job: dict[str, Any]) -> dict[str, Any]:
        """Runs video processing for a single video or playlist in an isolated Modal GPU worker container."""
        import json
        import traceback

        os.chdir("/root/youtube-to-autoavsr")

        url = job["url"]
        profile = job.get("profile", "no_voiceover")
        run_id = job.get("run_id", "default")
        video_id = job.get("video_id", "unknown")
        config_path = job.get("config_path", "configs/retina_1080p.yaml")
        hf_token = job.get("hf_token")
        is_playlist = bool(job.get("is_playlist", False))
        shard_tuple = job.get("shard")

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        from yt2avsr.config import load_config
        from yt2avsr.pipeline import Pipeline

        video_workspace = Path(f"/workspace/runs/{run_id}/{video_id}")
        video_workspace.mkdir(parents=True, exist_ok=True)

        cfg = load_config(Path(config_path))
        cfg.workspace = video_workspace
        # Worker isolation: isolate processed_sources to this worker workspace and disable auto sync in worker
        cfg.sources.processed_file = video_workspace / "processed_sources.txt"
        cfg.sources.auto_sync_hf = False
        passed_processed = job.get("processed_ids")
        if passed_processed:
            from yt2avsr.sources import append_processed_sources
            append_processed_sources(passed_processed, cfg.sources.processed_file)
        cfg.auto_avsr.repo_dir = Path("/root/youtube-to-autoavsr/external/auto_avsr")
        cfg.auto_avsr.detector = "retinaface"
        cfg.normalization.max_height = 1080
        cfg.download.format = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"

        cookies_content = job.get("cookies_content")
        if cookies_content:
            c_file = video_workspace / "cookies.txt"
            c_file.write_text(cookies_content, encoding="utf-8")
            cfg.download.cookies_file = str(c_file)

        print(
            f"[modal-worker] [{video_id}] Starting {url} (profile={profile}, playlist={is_playlist}) in {video_workspace}...",
            flush=True,
        )

        try:
            try:
                volume.reload()
            except Exception:
                pass
            pipe = Pipeline(cfg, force=False, profile=profile, shard=shard_tuple)
            if url.startswith("file://") or url.startswith("/"):
                clean_path = Path(url.replace("file://", "").strip())
                print(f"[modal-worker] [{video_id}] Processing local path {clean_path} (exists={clean_path.exists()})", flush=True)
                item = pipe.process_local(clean_path)
                items = [item] if item else []
            else:
                items = pipe.process_url(url, playlist=is_playlist)
            if not items:
                print(f"[modal-worker] [{video_id}] Warning: No items downloaded from {url}.", flush=True)
                return {
                    "success": False,
                    "url": url,
                    "video_id": video_id,
                    "profile": profile,
                    "workspace": str(video_workspace),
                    "error": "No items downloaded (video inaccessible, private, or download blocked)",
                    "accepted_clips": 0,
                    "review_clips": 0,
                    "rejected_clips": 0,
                }

            # Discover actual item directories created in clips
            clips_dir = video_workspace / "clips"
            item_ids = [p.name for p in clips_dir.iterdir() if p.is_dir()] if clips_dir.exists() else []

            # Count clips
            accepted_count = 0
            review_count = 0
            rejected_count = 0

            for meta_p in sorted(clips_dir.glob("*/*/metadata.json")):
                try:
                    rec = json.loads(meta_p.read_text(encoding="utf-8"))
                    st = rec.get("quality_status")
                    if st == "accepted":
                        accepted_count += 1
                    elif st == "review":
                        review_count += 1
                    elif st == "rejected":
                        rejected_count += 1
                except Exception:
                    pass

            print(
                f"[modal-worker] [{video_id}] Finished: items={item_ids}, {accepted_count} accepted, {review_count} review, {rejected_count} rejected.",
                flush=True,
            )

            # Explicit volume commit from worker; FAILURE IS A FATAL WORKER FAILURE
            try:
                volume.commit()
            except Exception as e:
                print(f"[modal-worker] [{video_id}] Fatal: volume.commit() failed: {e}", flush=True)
                return {
                    "success": False,
                    "url": url,
                    "video_id": video_id,
                    "profile": profile,
                    "workspace": str(video_workspace),
                    "is_playlist": is_playlist,
                    "error": f"volume.commit() failed: {e}",
                    "accepted_clips": 0,
                    "review_clips": 0,
                    "rejected_clips": 0,
                }

            return {
                "success": True,
                "url": url,
                "video_id": video_id,
                "profile": profile,
                "workspace": str(video_workspace),
                "item_ids": item_ids,
                "is_playlist": is_playlist,
                "accepted_clips": accepted_count,
                "review_clips": review_count,
                "rejected_clips": rejected_count,
            }
        except Exception as exc:
            print(f"[modal-worker] [{video_id}] Error processing {url}: {exc}", flush=True)
            traceback.print_exc()
            try:
                volume.commit()
            except Exception:
                pass
            return {
                "success": False,
                "url": url,
                "video_id": video_id,
                "profile": profile,
                "workspace": str(video_workspace),
                "is_playlist": is_playlist,
                "error": str(exc),
                "accepted_clips": 0,
                "review_clips": 0,
                "rejected_clips": 0,
            }


    @app.function(
        image=image,
        cpu=2.0,
        timeout=60 * 60 * 6,  # up to 6 hours
        volumes={"/workspace": volume},
    )
    def process_sources_on_modal(
        no_voiceover_lines: list[str],
        voiceover_lines: list[str],
        hf_token: str | None = None,
        contributor: str | None = None,
        shard: str | None = None,
        push_hf: bool = True,
        sync_hf: bool = True,
        config_path: str = "configs/retina_1080p.yaml",
        limit: int = 0,
        download_local: bool = False,
        gpu: str = "T4",
        cpu: float = 4.0,
        max_containers: int = 5,
        cleanup_volume: bool = False,
        cookies_content: str | None = None,
    ) -> dict[str, Any]:
        """Coordinator function: dispatches videos to parallel GPU workers, reloads volume, aggregates manifests, and uploads to HF."""
        import json
        import shutil
        import uuid

        os.chdir("/root/youtube-to-autoavsr")

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        from yt2avsr.config import load_config
        from yt2avsr.manifest import rebuild
        from yt2avsr.cloud import push, append_processed_sources
        from yt2avsr.utils import safe_id
        from yt2avsr.sources import (
            sync_processed_from_hf,
            load_processed_ids,
            get_source_key,
            deduplicate_source_lines,
            partition_sources,
            is_playlist_source,
            is_source_processed,
        )

        cfg = load_config(Path(config_path))
        # UUID-based run_id prevents volume collisions across concurrent runs
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run_root = Path(f"/workspace/runs/{run_id}")
        run_root.mkdir(parents=True, exist_ok=True)

        # 1. Sync already processed videos from Hugging Face
        if sync_hf and cfg.cloud.repo_id:
            try:
                print(f"[modal-coord] Syncing processed sources from HF dataset '{cfg.cloud.repo_id}'...", flush=True)
                sync_processed_from_hf(
                    repo_id=cfg.cloud.repo_id,
                    token=hf_token,
                    path=cfg.sources.processed_file,
                )
            except Exception as e:
                print(f"[modal-coord] Warning: HF sync encountered an issue: {e}", flush=True)

        processed_ids = load_processed_ids(cfg.sources.processed_file)
        print(f"[modal-coord] Found {len(processed_ids)} already processed video ID(s).", flush=True)

        shard_tuple = None
        if shard:
            from yt2avsr.sources import parse_shard, get_shard_owner

            shard_tuple = parse_shard(shard)
            owner = get_shard_owner(shard_tuple)
            owner_info = f" ({owner})" if owner else ""
            print(f"[modal-coord] Active sharding: shard {shard_tuple[0]}/{shard_tuple[1]}{owner_info}", flush=True)

        # 2. Deduplicate full source list across voiceover & no_voiceover
        dedup_vo, _ = deduplicate_source_lines(voiceover_lines)
        seen_vo_keys = {get_source_key(l) for l in dedup_vo if get_source_key(l)}
        dedup_nvo, _ = deduplicate_source_lines(no_voiceover_lines, seen_keys=seen_vo_keys)

        sources_pairs: list[tuple[str, str]] = []
        for line in dedup_nvo:
            sources_pairs.append(("no_voiceover", line))
        for line in dedup_vo:
            sources_pairs.append(("voiceover", line))

        # Shard partitioning on FULL deduplicated source list (guarantees stable ownership over time)
        if shard_tuple:
            orig_total = len(sources_pairs)
            sources_pairs = partition_sources(sources_pairs, shard_tuple)
            print(
                f"[modal-coord] Shard {shard_tuple[0]}/{shard_tuple[1]} assigned {len(sources_pairs)} of {orig_total} total sources.",
                flush=True,
            )

        # Filter already-processed sources within assigned shard
        sources_pairs = [pair for pair in sources_pairs if not is_source_processed(pair[1], processed_ids)]
        total_pending = len(sources_pairs)
        print(f"[modal-coord] Pending unique videos to process in this shard: {total_pending}", flush=True)

        if total_pending == 0:
            return {
                "success": True,
                "message": "All submitted videos have already been processed.",
                "total_processed": 0,
                "accepted_clips": 0,
                "review_clips": 0,
                "rejected_clips": 0,
                "newly_processed_records": [],
                "push_result": None,
                "contributor": contributor,
                "tar_bytes": None,
                "run_id": run_id,
            }

        # Apply limit if specified
        if limit > 0 and total_pending > limit:
            print(f"[modal-coord] Limiting total videos to {limit}", flush=True)
            sources_pairs = sources_pairs[:limit]

        # 3. Build job items for per-video workers
        job_items = []
        used_vids: set[str] = set()
        for i, (profile, url) in enumerate(sources_pairs):
            if url.startswith("file://") or url.startswith("/"):
                raw_stem = Path(url.replace("file://", "").strip()).stem
                base_vid = safe_id(raw_stem) or f"local_{i:04d}"
            else:
                raw_key = get_source_key(url) or f"video_{i:04d}"
                # Strip prefix like 'video:' or 'playlist:' for safe directory naming
                base_vid = raw_key.split(":", 1)[-1] if ":" in raw_key else raw_key
                base_vid = safe_id(base_vid)

            clean_vid = base_vid
            if clean_vid in used_vids:
                clean_vid = f"{base_vid}_{i:04d}"
            used_vids.add(clean_vid)

            is_playlist = is_playlist_source("auto", url) if not (url.startswith("file://") or url.startswith("/")) else False
            job_items.append({
                "url": url,
                "profile": profile,
                "run_id": run_id,
                "video_id": clean_vid,
                "config_path": config_path,
                "hf_token": hf_token,
                "is_playlist": is_playlist,
                "shard": shard_tuple,
                "cookies_content": cookies_content,
                "processed_ids": list(processed_ids),
            })

        print(
            f"[modal-coord] Launching distributed execution on Modal: {len(job_items)} item(s) "
            f"across workers (gpu={gpu}, cpu={cpu}, max_containers={max_containers})...",
            flush=True,
        )

        worker_fn = process_single_video_modal.with_options(
            gpu=gpu,
            cpu=cpu,
            max_containers=max_containers,
        )

        raw_results = list(worker_fn.map(job_items, order_outputs=False, return_exceptions=True))

        successful_results = []
        failed_jobs = []
        for res in raw_results:
            if isinstance(res, Exception):
                failed_jobs.append(str(res))
            elif not res.get("success"):
                failed_jobs.append(f"{res.get('url')}: {res.get('error')}")
            else:
                successful_results.append(res)

        print(
            f"[modal-coord] Worker results: {len(successful_results)} succeeded, {len(failed_jobs)} failed.",
            flush=True,
        )

        # 4. Reload Volume and aggregate worker outputs (with explicit duplicate item detection)
        print("[modal-coord] Reloading volume to aggregate worker outputs...", flush=True)
        volume.reload()

        aggregated_workspace = run_root / "aggregated"
        aggregated_clips = aggregated_workspace / "clips"
        aggregated_clips.mkdir(parents=True, exist_ok=True)

        seen_items: dict[str, str] = {}
        for res in successful_results:
            src_url = res.get("url", "unknown")
            worker_clips = Path(res["workspace"]) / "clips"
            if worker_clips.exists():
                for item_dir in worker_clips.iterdir():
                    if item_dir.is_dir():
                        item_id = item_dir.name
                        dst_clips = aggregated_clips / item_id

                        # Determine canonical source for this item from metadata or worker URL
                        first_meta = next(item_dir.glob("*/metadata.json"), None)
                        item_source = src_url
                        if first_meta and first_meta.exists():
                            try:
                                m_data = json.loads(first_meta.read_text(encoding="utf-8"))
                                item_source = m_data.get("source_url") or src_url
                            except Exception:
                                pass

                        if item_id in seen_items:
                            existing_source = seen_items[item_id]
                            k_exist = get_source_key(existing_source) or existing_source
                            k_new = get_source_key(item_source) or item_source
                            if k_exist == k_new:
                                print(
                                    f"[modal-coord] Safe deduplication: duplicate item_id '{item_id}' from same source '{item_source}' already aggregated. Skipping redundant copy.",
                                    flush=True,
                                )
                                continue
                            else:
                                err = (
                                    f"Fatal: duplicate item_id collision detected across workers! "
                                    f"Item ID '{item_id}' belongs to two different sources: "
                                    f"'{existing_source}' and '{item_source}'. Aborting run to prevent corrupt dataset."
                                )
                                print(f"[modal-coord] {err}", flush=True)
                                raise RuntimeError(err)

                        seen_items[item_id] = item_source
                        shutil.copytree(item_dir, dst_clips)

        # Manifests and records rebuilt directly from aggregated disk state
        aggregated_records = rebuild(aggregated_workspace)

        # Write append-only completion ledger: data/<contributor>/completions/<run_id>.jsonl
        completions_dir = aggregated_workspace / "completions"
        completions_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = completions_dir / f"{run_id}.jsonl"
        with ledger_path.open("w", encoding="utf-8") as f_ledger:
            for r in successful_results:
                ledger_entry = {
                    "run_id": run_id,
                    "url": r.get("url"),
                    "video_id": r.get("video_id"),
                    "item_ids": r.get("item_ids", []),
                    "contributor": contributor,
                    "profile": r.get("profile"),
                    "is_playlist": bool(r.get("is_playlist", False)),
                    "accepted_clips": r.get("accepted_clips", 0),
                    "review_clips": r.get("review_clips", 0),
                    "rejected_clips": r.get("rejected_clips", 0),
                    "status": "completed",
                    "timestamp": time.time(),
                }
                f_ledger.write(json.dumps(ledger_entry, ensure_ascii=False) + "\n")

        total_accepted = sum(1 for r in aggregated_records if r.get("quality_status") == "accepted")
        total_review = sum(1 for r in aggregated_records if r.get("quality_status") == "review")
        total_rejected = sum(1 for r in aggregated_records if r.get("quality_status") == "rejected")

        print(
            f"[modal-coord] Aggregated clips: {total_accepted} accepted, {total_review} review, {total_rejected} rejected.",
            flush=True,
        )

        # 5. Push to Hugging Face (even if accepted+review == 0, push manifests so completion is archived)
        push_result = None
        newly_processed = []
        if push_hf and cfg.cloud.repo_id and successful_results:
            try:
                print(f"[modal-coord] Uploading clips & manifests to Hugging Face dataset '{cfg.cloud.repo_id}'...", flush=True)
                push_result = push(
                    aggregated_workspace,
                    repo_id=cfg.cloud.repo_id,
                    contributor=contributor,
                    statuses=["accepted", "review"],
                    token=hf_token,
                    private=cfg.cloud.private,
                )
                print(f"[modal-coord] Upload complete: {push_result}", flush=True)

                # On successful HF push, record all evaluated videos (including 100% rejected or zero-clip ones) in processed_sources.txt
                # Do NOT append playlist URL itself; track completion via real item IDs
                newly_processed = list(aggregated_records)
                for r in successful_results:
                    if r.get("url") and not r.get("is_playlist"):
                        newly_processed.append(r["url"])
                    for i_id in r.get("item_ids", []):
                        newly_processed.append(f"video:{i_id}")
                if newly_processed:
                    added = append_processed_sources(newly_processed, cfg.sources.processed_file)
                    print(f"[modal-coord] Recorded {added} newly processed video(s) in {cfg.sources.processed_file}.", flush=True)
            except Exception as e:
                print(f"[modal-coord] Error during Hugging Face upload: {e}", flush=True)
                push_result = f"Failed: {e}"

        # Commit volume in coordinator
        try:
            volume.commit()
        except Exception as e:
            print(f"[modal-coord] Warning: volume.commit() failed: {e}", flush=True)

        # 6. Local archive if requested
        tar_bytes = None
        if download_local and (aggregated_workspace / "clips").exists():
            print("[modal-coord] Packaging accepted & review clips for local download...", flush=True)
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                if (aggregated_workspace / "manifests").exists():
                    tar.add(str(aggregated_workspace / "manifests"), arcname="manifests")
                for d in (aggregated_workspace / "clips").glob("*/*"):
                    meta = d / "metadata.json"
                    if meta.exists():
                        try:
                            m = json.loads(meta.read_text(encoding="utf-8"))
                            if m.get("quality_status") in ("accepted", "review"):
                                tar.add(str(d), arcname=str(d.relative_to(aggregated_workspace)))
                        except Exception:
                            pass
            tar_bytes = buf.getvalue()

        # 7. Volume cleanup if requested
        if cleanup_volume:
            print(f"[modal-coord] Cleaning up volume run directory: {run_root}", flush=True)
            try:
                shutil.rmtree(run_root, ignore_errors=True)
                volume.commit()
                print(f"[modal-coord] Successfully cleaned up {run_root}", flush=True)
            except Exception as e:
                print(f"[modal-coord] Warning: Failed to cleanup volume run dir: {e}", flush=True)

        has_failures = bool(failed_jobs)
        upload_failed = bool(push_result and push_result.startswith("Failed:"))
        overall_success = (not has_failures) and (not upload_failed)

        return {
            "success": overall_success,
            "total_processed": len(successful_results),
            "failed_videos": len(failed_jobs),
            "accepted_clips": total_accepted,
            "review_clips": total_review,
            "rejected_clips": total_rejected,
            "push_result": push_result,
            "newly_processed_records": newly_processed,
            "contributor": contributor,
            "tar_bytes": tar_bytes,
            "run_id": run_id,
        }


    @app.local_entrypoint()
    def main(
        url: str = "",
        shard: str = "",
        voiceover: bool = False,
        no_push: bool = False,
        limit: int = 0,
        download_local: bool = False,
        gpu: str = "T4",
        cpu: float = 4.0,
        max_containers: int = 5,
        hf_token: str = "",
        contributor: str = "",
        config: str = "configs/retina_1080p.yaml",
        cleanup_volume: bool = False,
        sources_file: str = "",
        cookies_file: str = "",
    ):
        """CLI local entrypoint for running YouTube -> Auto-AVSR on Modal."""
        from yt2avsr.cloud import check_hf_login_or_warn, append_processed_sources

        # Read cookies if provided
        cookies_content = None
        if cookies_file:
            cp = Path(cookies_file)
            if not cp.is_absolute():
                cp = REPO_ROOT / cp
            if cp.exists():
                cookies_content = cp.read_text(encoding="utf-8")
                print(f"🍪 Using cookies file: {cp}")
            else:
                print(f"Warning: Cookies file not found: {cookies_file}", file=sys.stderr)

        # Pre-flight Hugging Face check
        resolved_token, detected_user = check_hf_login_or_warn(
            token=hf_token,
            required=(not no_push),
        )

        if not resolved_token and not no_push:
            print("Modal işlemi Hugging Face girişi yapılmadığı için durduruldu.", file=sys.stderr)
            print("Giriş yaptıktan sonra tekrar çalıştırın veya '--no-push' bayrağını ekleyin.", file=sys.stderr)
            sys.exit(1)

        active_contributor = contributor or detected_user or "unknown"

        # Determine source URLs
        no_voiceover_lines = []
        voiceover_lines = []

        if url:
            urls = [u.strip() for u in url.split(",") if u.strip()]
            if voiceover:
                voiceover_lines = urls
            else:
                no_voiceover_lines = urls
            print(f"🎯 Direct URL mode: {len(urls)} video(s) ({'voiceover' if voiceover else 'no_voiceover'})")
        elif sources_file:
            s_path = Path(sources_file)
            if not s_path.is_absolute():
                s_path = REPO_ROOT / s_path
            if not s_path.exists():
                print(f"Hata: Belirtilen kaynak dosyası bulunamadı: {sources_file}", file=sys.stderr)
                sys.exit(1)
            lines = [
                l.strip()
                for l in s_path.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.strip().startswith("#")
            ]
            if voiceover:
                voiceover_lines = lines
            else:
                no_voiceover_lines = lines
            print(f"📂 Sources file mode: {len(lines)} link(s) from {sources_file} ({'voiceover' if voiceover else 'no_voiceover'})")
        else:
            nvo_path = REPO_ROOT / "sources_no_voiceover.txt"
            vo_path = REPO_ROOT / "sources_voiceover.txt"
            if nvo_path.exists():
                no_voiceover_lines = [
                    l.strip()
                    for l in nvo_path.read_text(encoding="utf-8").splitlines()
                    if l.strip() and not l.strip().startswith("#")
                ]
            if vo_path.exists():
                voiceover_lines = [
                    l.strip()
                    for l in vo_path.read_text(encoding="utf-8").splitlines()
                    if l.strip() and not l.strip().startswith("#")
                ]
            print(f"📂 Sources mode: {len(no_voiceover_lines)} no_voiceover, {len(voiceover_lines)} voiceover links.")

        if not no_voiceover_lines and not voiceover_lines:
            print("Hata: İşlenecek link bulunamadı! sources_*.txt dosyalarına link ekleyin veya --url verin.", file=sys.stderr)
            sys.exit(1)

        shard_display = "Tüm liste (tek çalışan)"
        canonical_shard = None
        if shard:
            from yt2avsr.sources import parse_shard, get_shard_owner

            try:
                st = parse_shard(shard)
                if st:
                    canonical_shard = f"{st[0]}/{st[1]}"
                    owner = get_shard_owner(st)
                    shard_display = f"{st[0]}/{st[1]}" + (f" ({owner})" if owner else "")
            except ValueError as e:
                print(f"Hata: {e}", file=sys.stderr)
                sys.exit(1)

        print("\n" + "=" * 70)
        print("🚀 MODAL GPU DAĞITIK ÇALIŞTIRILIYOR (RetinaFace + 1080p)")
        print(f"  • GPU: {gpu}")
        print(f"  • CPU / Worker: {cpu}")
        print(f"  • Max Paralel Worker: {max_containers}")
        print(f"  • Shard: {shard_display}")
        print(f"  • Hugging Face Kullanıcısı: {active_contributor}")
        print(f"  • Hugging Face'e Yükle: {'Hayır (--no-push)' if no_push else 'Evet (Otomatik)'}")
        print(f"  • Volume Temizleme: {'Evet (--cleanup-volume)' if cleanup_volume else 'Hayır'}")
        print("=" * 70 + "\n")

        result = process_sources_on_modal.remote(
            no_voiceover_lines=no_voiceover_lines,
            voiceover_lines=voiceover_lines,
            hf_token=resolved_token,
            contributor=active_contributor,
            shard=canonical_shard,
            push_hf=(not no_push),
            sync_hf=True,
            config_path=config,
            limit=limit,
            download_local=download_local,
            gpu=gpu,
            cpu=cpu,
            max_containers=max_containers,
            cleanup_volume=cleanup_volume,
            cookies_content=cookies_content,
        )

        success = result.get("success", False)
        failed_count = result.get("failed_videos", 0)

        print("\n" + "=" * 70)
        if success:
            print("✅ MODAL İŞLEMİ BAŞARIYLA TAMAMLANDI!")
        else:
            print("⚠️ MODAL İŞLEMİ KISMİ VEYA TAM BAŞARISIZLIKLA SONLANDI!")
            if failed_count > 0:
                print(f"  • Başarısız Video Sayısı: {failed_count}")
        print(f"  • Başarılı Video Sayısı:  {result.get('total_processed', 0)}")
        print(f"  • Kabul Edilen Klip:      {result.get('accepted_clips', 0)}")
        print(f"  • İnceleme Klibi:         {result.get('review_clips', 0)}")
        print(f"  • Reddedilen Klip:        {result.get('rejected_clips', 0)}")
        if result.get("push_result"):
            print(f"  • Hugging Face Durumu:    {result.get('push_result')}")
        print("=" * 70)

        # Update local processed_sources.txt
        new_recs = result.get("newly_processed_records", [])
        if new_recs:
            proc_file = REPO_ROOT / "processed_sources.txt"
            added = append_processed_sources(new_recs, proc_file)
            if added:
                print(f"📝 Yerel {proc_file.name} dosyasına {added} yeni video eklendi.")

        # If downloaded local tar
        tar_bytes = result.get("tar_bytes")
        if tar_bytes and download_local:
            print("📦 Klipler yerel 'data/' klasörüne çıkartılıyor...")
            buf = io.BytesIO(tar_bytes)
            with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                tar.extractall(path=str(REPO_ROOT / "data"))
            print("✅ Yerel klasöre kopyalandı: data/")

        if not success:
            print("Modal görevi hata(lar) ile sonuçlandığı için işlem başarısız olarak işaretlendi.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    if modal is None:
        print(
            "Modal CLI bulunamadı.\n"
            "Kurulum adımları:\n"
            "  1. modal.com adresinde hesap açın\n"
            "  2. pip install modal\n"
            "  3. modal setup\n",
            file=sys.stderr,
        )
        sys.exit(1)
    print("Modal app loaded. To run:\n  modal run modal_app.py [OPTIONS]")
