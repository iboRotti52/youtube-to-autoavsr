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


    @app.function(
        image=image,
        gpu="T4",
        timeout=60 * 60 * 4,  # up to 4 hours
        ephemeral_disk=50 * 1024,  # 50 GB
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
    ) -> dict[str, Any]:
        """Runs video processing in the remote Modal GPU container."""
        import json
        import shutil

        os.chdir("/root/youtube-to-autoavsr")

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        from yt2avsr.config import load_config
        from yt2avsr.pipeline import Pipeline
        from yt2avsr.cloud import push, append_processed_sources
        from yt2avsr.sources import (
            sync_processed_from_hf,
            read_processed_ids,
            get_source_key,
            deduplicate_source_lines,
        )

        cfg = load_config(Path(config_path))
        cfg.auto_avsr.repo_dir = Path("/root/youtube-to-autoavsr/external/auto_avsr")
        cfg.auto_avsr.detector = "retinaface"
        cfg.normalization.max_height = 1080
        cfg.download.format = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"

        workspace = cfg.workspace
        workspace.mkdir(parents=True, exist_ok=True)

        # 1. Sync already processed videos from Hugging Face
        if sync_hf and cfg.cloud.repo_id:
            try:
                print(f"[modal] Syncing processed sources from HF dataset '{cfg.cloud.repo_id}'...", flush=True)
                sync_processed_from_hf(
                    repo_id=cfg.cloud.repo_id,
                    token=hf_token,
                    path=cfg.sources.processed_file,
                )
            except Exception as e:
                print(f"[modal] Warning: HF sync encountered an issue: {e}", flush=True)

        processed_ids = read_processed_ids(cfg.sources.processed_file)
        print(f"[modal] Found {len(processed_ids)} already processed video ID(s).", flush=True)

        shard_tuple = None
        if shard:
            from yt2avsr.sources import parse_shard, get_shard_owner

            shard_tuple = parse_shard(shard)
            owner = get_shard_owner(shard_tuple)
            owner_info = f" ({owner})" if owner else ""
            print(f"[modal] Active sharding: shard {shard_tuple[0]}/{shard_tuple[1]}{owner_info}", flush=True)


        # 2. Filter & Deduplicate
        def is_unprocessed(line: str) -> bool:
            k = get_source_key(line)
            return bool(k and k not in processed_ids)

        dedup_vo, _ = deduplicate_source_lines(voiceover_lines)
        filt_vo = [l for l in dedup_vo if is_unprocessed(l)]

        seen_vo_keys = {get_source_key(l) for l in dedup_vo if get_source_key(l)}
        dedup_nvo, _ = deduplicate_source_lines(no_voiceover_lines, seen_keys=seen_vo_keys)
        filt_nvo = [l for l in dedup_nvo if is_unprocessed(l)]

        jobs: list[tuple[str, list[str]]] = []
        if filt_nvo:
            jobs.append(("no_voiceover", filt_nvo))
        if filt_vo:
            jobs.append(("voiceover", filt_vo))

        total_pending = len(filt_nvo) + len(filt_vo)
        print(f"[modal] Pending unique videos to process: {total_pending} (no_vo: {len(filt_nvo)}, vo: {len(filt_vo)})", flush=True)

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
            print(f"[modal] Limiting total videos to {limit}", flush=True)
            allocated = 0
            limited_jobs = []
            for prof, lines in jobs:
                take = min(len(lines), limit - allocated)
                if take > 0:
                    limited_jobs.append((prof, lines[:take]))
                    allocated += take
                if allocated >= limit:
                    break
            jobs = limited_jobs

        # 3. Process jobs
        processed_records = []
        for profile, lines in jobs:
            job_file = Path(f"/root/youtube-to-autoavsr/job_{profile}.txt")
            job_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

            print(f"[modal] Starting {profile} pipeline on GPU with {len(lines)} video(s)...", flush=True)
            pipe = Pipeline(cfg, force=False, profile=profile, shard=shard_tuple)
            pipe.process_sources_file(job_file)

        # 4. Count clips & collect metadata
        recs = []
        accepted_count = 0
        review_count = 0
        rejected_count = 0

        for meta_p in sorted((workspace / "clips").glob("*/*/metadata.json")):
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
            f"[modal] Processing finished: {accepted_count} accepted, {review_count} review, {rejected_count} rejected clips.",
            flush=True,
        )

        # 5. Push to Hugging Face
        push_result = None
        newly_processed = []
        if push_hf and cfg.cloud.repo_id and (accepted_count + review_count > 0):
            try:
                print(f"[modal] Uploading clips to Hugging Face dataset '{cfg.cloud.repo_id}'...", flush=True)
                push_result = push(
                    workspace,
                    repo_id=cfg.cloud.repo_id,
                    contributor=contributor,
                    statuses=["accepted", "review"],
                    token=hf_token,
                    private=cfg.cloud.private,
                )
                print(f"[modal] Upload complete: {push_result}", flush=True)

                pushed_recs = [r for r in recs if r.get("quality_status") in ("accepted", "review")]
                added = append_processed_sources(pushed_recs, cfg.sources.processed_file)
                newly_processed = pushed_recs
                print(f"[modal] Recorded {added} newly processed video(s).", flush=True)
            except Exception as e:
                print(f"[modal] Error during Hugging Face upload: {e}", flush=True)
                push_result = f"Failed: {e}"

        # 6. Local archive if requested
        tar_bytes = None
        if download_local and (workspace / "clips").exists():
            print("[modal] Packaging accepted & review clips for local download...", flush=True)
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                if (workspace / "manifests").exists():
                    tar.add(str(workspace / "manifests"), arcname="manifests")
                for d in (workspace / "clips").glob("*/*"):
                    meta = d / "metadata.json"
                    if meta.exists():
                        try:
                            m = json.loads(meta.read_text(encoding="utf-8"))
                            if m.get("quality_status") in ("accepted", "review"):
                                tar.add(str(d), arcname=str(d.relative_to(workspace)))
                        except Exception:
                            pass
            tar_bytes = buf.getvalue()

        return {
            "success": True,
            "total_processed": len({r.get("item_id") for r in recs if r.get("item_id")}),
            "accepted_clips": accepted_count,
            "review_clips": review_count,
            "rejected_clips": rejected_count,
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
        print("🚀 MODAL GPU ÇALIŞTIRILIYOR (RetinaFace + 1080p)")
        print(f"  • GPU: {gpu}")
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
