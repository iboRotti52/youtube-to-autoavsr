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
        .apt_install("ffmpeg", "git", "git-lfs", "curl", "build-essential")
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
        )
        .run_commands(
            "mkdir -p /root/youtube-to-autoavsr/external",
            "git clone --depth 1 https://github.com/mpc001/auto_avsr.git /root/youtube-to-autoavsr/external/auto_avsr",
            "git clone https://github.com/hhj1897/face_detection.git /root/youtube-to-autoavsr/external/face_detection && cd /root/youtube-to-autoavsr/external/face_detection && git lfs pull && pip install -e .",
            "git clone https://github.com/hhj1897/face_alignment.git /root/youtube-to-autoavsr/external/face_alignment && cd /root/youtube-to-autoavsr/external/face_alignment && git lfs pull && pip install -e .",
            "python3 -c 'from faster_whisper import WhisperModel; WhisperModel(\"large-v3-turbo\", device=\"cpu\", compute_type=\"int8\")'",
        )
        .env({
            "PYTHONPATH": "/root/youtube-to-autoavsr/src:/root/youtube-to-autoavsr/external/auto_avsr",
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
        ephemeral_disk=50 * 1024,  # 50 GB
        volumes={"/workspace": volume},
    )
    def process_single_video_modal(job: dict[str, Any]) -> dict[str, Any]:
        """Runs video processing for a single video in an isolated Modal GPU worker container."""
        import json
        import traceback

        os.chdir("/root/youtube-to-autoavsr")

        url = job["url"]
        profile = job.get("profile", "no_voiceover")
        run_id = job.get("run_id", "default")
        video_id = job.get("video_id", "unknown")
        config_path = job.get("config_path", "configs/retina_1080p.yaml")
        hf_token = job.get("hf_token")

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        from yt2avsr.config import load_config
        from yt2avsr.pipeline import Pipeline

        video_workspace = Path(f"/workspace/runs/{run_id}/{video_id}")
        video_workspace.mkdir(parents=True, exist_ok=True)

        cfg = load_config(Path(config_path))
        cfg.workspace = video_workspace
        cfg.auto_avsr.repo_dir = Path("/root/youtube-to-autoavsr/external/auto_avsr")
        cfg.auto_avsr.detector = "retinaface"
        cfg.normalization.max_height = 1080
        cfg.download.format = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"

        print(f"[modal-worker] [{video_id}] Starting video {url} ({profile}) in {video_workspace}...", flush=True)

        try:
            pipe = Pipeline(cfg, force=False, profile=profile)
            pipe.process_url(url)

            # Collect results for this video
            recs = []
            accepted_count = 0
            review_count = 0
            rejected_count = 0

            for meta_p in sorted((video_workspace / "clips").glob("*/*/metadata.json")):
                try:
                    rec = json.loads(meta_p.read_text(encoding="utf-8"))
                    recs.append(rec)
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
                f"[modal-worker] [{video_id}] Finished: {accepted_count} accepted, {review_count} review, {rejected_count} rejected.",
                flush=True,
            )

            # Explicit volume commit from worker
            try:
                volume.commit()
            except Exception as e:
                print(f"[modal-worker] [{video_id}] Warning: volume.commit() failed: {e}", flush=True)

            return {
                "success": True,
                "url": url,
                "video_id": video_id,
                "profile": profile,
                "workspace": str(video_workspace),
                "accepted_clips": accepted_count,
                "review_clips": review_count,
                "rejected_clips": rejected_count,
                "records": recs,
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
                "error": str(exc),
                "accepted_clips": 0,
                "review_clips": 0,
                "rejected_clips": 0,
                "records": [],
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
    ) -> dict[str, Any]:
        """Coordinator function: dispatches videos to parallel GPU workers, reloads volume, aggregates manifests, and uploads to HF."""
        import json
        import shutil
        import time

        os.chdir("/root/youtube-to-autoavsr")

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        from yt2avsr.config import load_config
        from yt2avsr.manifest import rebuild
        from yt2avsr.cloud import push, append_processed_sources
        from yt2avsr.sources import (
            sync_processed_from_hf,
            read_processed_ids,
            get_source_key,
            deduplicate_source_lines,
            partition_sources,
        )

        cfg = load_config(Path(config_path))
        run_id = f"run_{int(time.time())}"
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

        processed_ids = read_processed_ids(cfg.sources.processed_file)
        print(f"[modal-coord] Found {len(processed_ids)} already processed video ID(s).", flush=True)

        shard_tuple = None
        if shard:
            from yt2avsr.sources import parse_shard, get_shard_owner

            shard_tuple = parse_shard(shard)
            owner = get_shard_owner(shard_tuple)
            owner_info = f" ({owner})" if owner else ""
            print(f"[modal-coord] Active sharding: shard {shard_tuple[0]}/{shard_tuple[1]}{owner_info}", flush=True)

        # 2. Filter & Deduplicate
        def is_unprocessed(line: str) -> bool:
            k = get_source_key(line)
            return bool(k and k not in processed_ids)

        dedup_vo, _ = deduplicate_source_lines(voiceover_lines)
        filt_vo = [l for l in dedup_vo if is_unprocessed(l)]

        seen_vo_keys = {get_source_key(l) for l in dedup_vo if get_source_key(l)}
        dedup_nvo, _ = deduplicate_source_lines(no_voiceover_lines, seen_keys=seen_vo_keys)
        filt_nvo = [l for l in dedup_nvo if is_unprocessed(l)]

        sources_pairs: list[tuple[str, str]] = []
        for line in filt_nvo:
            sources_pairs.append(("no_voiceover", line))
        for line in filt_vo:
            sources_pairs.append(("voiceover", line))

        # Shard partitioning if specified
        if shard_tuple:
            sources_pairs = partition_sources(sources_pairs, shard_tuple)

        total_pending = len(sources_pairs)
        print(f"[modal-coord] Pending unique videos to process: {total_pending}", flush=True)

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
            }

        # Apply limit if specified
        if limit > 0 and total_pending > limit:
            print(f"[modal-coord] Limiting total videos to {limit}", flush=True)
            sources_pairs = sources_pairs[:limit]

        # 3. Build job items for per-video workers
        job_items = []
        for i, (profile, url) in enumerate(sources_pairs):
            vid = get_source_key(url) or f"video_{i:04d}"
            job_items.append({
                "url": url,
                "profile": profile,
                "run_id": run_id,
                "video_id": vid,
                "config_path": config_path,
                "hf_token": hf_token,
            })

        print(
            f"[modal-coord] Launching distributed execution on Modal: {len(job_items)} video(s) "
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

        # 4. Reload Volume and aggregate worker outputs
        print("[modal-coord] Reloading volume to aggregate worker outputs...", flush=True)
        volume.reload()

        aggregated_workspace = run_root / "aggregated"
        aggregated_clips = aggregated_workspace / "clips"
        aggregated_clips.mkdir(parents=True, exist_ok=True)

        for res in successful_results:
            vid = res["video_id"]
            src_clips = Path(res["workspace"]) / "clips" / vid
            dst_clips = aggregated_clips / vid
            if src_clips.exists():
                shutil.copytree(src_clips, dst_clips, dirs_exist_ok=True)

        rebuild(aggregated_workspace)

        total_accepted = sum(r.get("accepted_clips", 0) for r in successful_results)
        total_review = sum(r.get("review_clips", 0) for r in successful_results)
        total_rejected = sum(r.get("rejected_clips", 0) for r in successful_results)

        print(
            f"[modal-coord] Aggregated clips: {total_accepted} accepted, {total_review} review, {total_rejected} rejected.",
            flush=True,
        )

        # 5. Push to Hugging Face
        push_result = None
        newly_processed = []
        if push_hf and cfg.cloud.repo_id and (total_accepted + total_review > 0):
            try:
                print(f"[modal-coord] Uploading clips to Hugging Face dataset '{cfg.cloud.repo_id}'...", flush=True)
                push_result = push(
                    aggregated_workspace,
                    repo_id=cfg.cloud.repo_id,
                    contributor=contributor,
                    statuses=["accepted", "review"],
                    token=hf_token,
                    private=cfg.cloud.private,
                )
                print(f"[modal-coord] Upload complete: {push_result}", flush=True)

                # Only on successful HF push: update newly processed records
                for res in successful_results:
                    for rec in res.get("records", []):
                        if rec.get("quality_status") in ("accepted", "review"):
                            newly_processed.append(rec)

                if newly_processed:
                    added = append_processed_sources(newly_processed, cfg.sources.processed_file)
                    print(f"[modal-coord] Recorded {added} newly processed video(s).", flush=True)
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

        return {
            "success": True,
            "total_processed": len(successful_results),
            "failed_videos": len(failed_jobs),
            "accepted_clips": total_accepted,
            "review_clips": total_review,
            "rejected_clips": total_rejected,
            "push_result": push_result,
            "newly_processed_records": newly_processed,
            "contributor": contributor,
            "tar_bytes": tar_bytes,
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
    ):
        """CLI local entrypoint for running YouTube -> Auto-AVSR on Modal."""
        from yt2avsr.cloud import check_hf_login_or_warn, append_processed_sources

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
            if voiceover:
                voiceover_lines = [url]
            else:
                no_voiceover_lines = [url]
            print(f"🎯 Single video mode: {url} ({'voiceover' if voiceover else 'no_voiceover'})")
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
        )

        print("\n" + "=" * 70)
        print("✅ MODAL İŞLEMİ TAMAMLANDI!")
        print(f"  • İşlenen Video Sayısı: {result.get('total_processed', 0)}")
        print(f"  • Kabul Edilen Klip:    {result.get('accepted_clips', 0)}")
        print(f"  • İnceleme Klibi:       {result.get('review_clips', 0)}")
        print(f"  • Reddedilen Klip:      {result.get('rejected_clips', 0)}")
        if result.get("push_result"):
            print(f"  • Hugging Face Durumu:  {result.get('push_result')}")
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
