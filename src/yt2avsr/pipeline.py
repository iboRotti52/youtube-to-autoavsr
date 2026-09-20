from __future__ import annotations

import time
from pathlib import Path

from .active_speaker import select_active_speaker
from .auto_avsr_crop import NoUsableFaceError, crop_with_official_auto_avsr
from .config import AppConfig
from .downloader import download, register_local
from .manifest import rebuild
from .media import extract_audio_clip, extract_clip, normalize
from .profiles import get_profile
from .scenes import detect_scene_cuts
from .segment import make_segments
from .sources import (
    append_processed_sources,
    get_source_key,
    is_playlist_source,
    is_source_processed,
    load_processed_ids,
    partition_sources,
)
from .state import StateDB
from .subtitles import save_youtube_transcript
from .transcribe import (
    transcribe,
    transcript_similarity,
    words_confidence,
    words_to_text,
)
from .utils import read_json, write_json
from .visual_quality import analyze_visual_quality


class Pipeline:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        force: bool = False,
        profile: str = "no_voiceover",
        shard: tuple[int, int] | None = None,
    ) -> None:
        self.cfg, self.workspace, self.force = cfg, cfg.workspace, force
        self.profile = get_profile(profile)
        self.shard = shard
        self.state = StateDB(self.workspace / "state.sqlite3")

    def process_url(self, url: str, *, playlist: bool = False):
        self.last_download_all_processed = False
        processed_ids = load_processed_ids(self.cfg.sources.processed_file)
        if not playlist and is_source_processed(url, processed_ids):
            print(f"[SKIP] {url}: Zaten işlenmiş ({self.cfg.sources.processed_file}), atlanıyor.")
            self.last_download_all_processed = True
            return []

        items = download(
            url,
            self.workspace / "raw",
            self.cfg.download,
            playlist=playlist,
            processed_ids=processed_ids,
            shard=self.shard,
        )
        self.last_download_all_processed = bool(getattr(items, "all_already_processed", False))
        for item in items:
            self._process_item(item)
            raw_vid = str(item.get("id", ""))
            if raw_vid:
                processed_ids.add(raw_vid)
                processed_ids.add(f"video:{raw_vid}")
                processed_ids.add(f"https://www.youtube.com/watch?v={raw_vid}")
        rebuild(self.workspace)
        if items:
            append_processed_sources(
                [item["metadata"] for item in items if "metadata" in item],
                self.cfg.sources.processed_file,
            )
        return items

    def process_sources_file(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Sources file not found: {path}")

        lines = path.read_text(encoding="utf-8").splitlines()
        sources = []
        for line_number, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # Optional per-line mode:
            # video https://...
            # playlist https://...
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].lower() in {"video", "playlist"}:
                mode, url = parts[0].lower(), parts[1].strip()
            else:
                mode, url = "auto", line

            if not url.startswith(("https://", "http://")):
                raise ValueError(
                    f"Invalid source on line {line_number}: {raw!r}. "
                    "Expected a YouTube URL."
                )
            sources.append((mode, url))

        if not sources:
            raise ValueError(f"No usable sources found in {path}")

        # Intra-file deduplication
        seen_keys: set[str] = set()
        deduped_sources = []
        for mode, url in sources:
            key = get_source_key(url)
            if key and key in seen_keys:
                print(f"[SKIP] Mükerrer link (aynı listede tekrar): {url}")
                continue
            if key:
                seen_keys.add(key)
            deduped_sources.append((mode, url))
        sources = deduped_sources

        if self.shard:
            from .sources import get_shard_owner

            original_count = len(sources)
            sources = partition_sources(sources, self.shard)
            owner = get_shard_owner(self.shard)
            owner_info = f" ({owner})" if owner else ""
            print(
                f"[shard {self.shard[0]}/{self.shard[1]}{owner_info}] Assigned {len(sources)} of {original_count} source(s)."
            )


        processed_ids = load_processed_ids(self.cfg.sources.processed_file)
        results = []
        failures = []
        for index, (mode, url) in enumerate(sources, start=1):
            playlist = is_playlist_source(mode, url)
            if not playlist and is_source_processed(url, processed_ids):
                print(
                    f"[SKIP] [{index}/{len(sources)}] {url}: "
                    f"Zaten işlenmiş ({self.cfg.sources.processed_file}), atlanıyor."
                )
                continue

            print(f"[{index}/{len(sources)}] Processing: {url}")
            try:
                items = download(
                    url,
                    self.workspace / "raw",
                    self.cfg.download,
                    playlist=playlist,
                    processed_ids=processed_ids,
                    shard=self.shard,
                )
                for item in items:
                    self._process_item(item)
                    raw_vid = str(item.get("id", ""))
                    if raw_vid:
                        processed_ids.add(raw_vid)
                        processed_ids.add(f"video:{raw_vid}")
                        processed_ids.add(f"https://www.youtube.com/watch?v={raw_vid}")
                results.extend(items)
                if items:
                    append_processed_sources(
                        [item["metadata"] for item in items if "metadata" in item],
                        self.cfg.sources.processed_file,
                    )
            except Exception as exc:
                failures.append((url, str(exc)))
                print(f"[ERROR] {url}: {exc}")

        rebuild(self.workspace)
        if failures:
            details = "\n".join(f"- {url}: {error}" for url, error in failures)
            raise RuntimeError(
                f"{len(failures)} source(s) failed. Successful sources were preserved:\n"
                f"{details}"
            )
        return results

    def process_local(self, path: Path):
        item = register_local(path, self.workspace/"raw")
        self._process_item(item); rebuild(self.workspace); return item

    def _process_item(self, item):
        item_start = time.perf_counter()
        iid = item["id"]
        normalized = self.workspace/"normalized"/iid/"normalized.mp4"
        words_path = self.workspace/"transcripts"/iid/"words.json"
        segments_path = self.workspace/"transcripts"/iid/"segments.json"
        
        t0 = time.perf_counter()
        self._stage(
            iid,
            "normalize",
            lambda: normalize(item["source"], normalized, self.cfg.normalization),
            outputs=[normalized],
        )
        t_normalize = time.perf_counter() - t0

        def transcript():
            if item.get("subtitle_path"):
                save_youtube_transcript(
                    item["subtitle_path"],
                    words_path,
                    item["metadata"].get("subtitle_language") or self.cfg.language,
                    automatic=bool(item.get("subtitle_automatic")),
                )
            elif self.cfg.transcription.use_whisper_when_no_manual_subtitles:
                transcribe(normalized, words_path, self.cfg.language, self.cfg.transcription)
            else:
                raise RuntimeError("No manual subtitle and Whisper fallback disabled")
        
        t0 = time.perf_counter()
        self._stage(iid, "transcribe", transcript, outputs=[words_path])
        t_transcribe = time.perf_counter() - t0

        def segment():
            payload = read_json(words_path)
            cut_times = (
                detect_scene_cuts(normalized, self.cfg.segmentation.scene_cut_threshold)
                if self.cfg.segmentation.split_on_scene_cut
                else []
            )
            segments = make_segments(
                payload["words"],
                self.cfg.segmentation,
                cut_times,
                strict_sentence_boundaries=self.profile.verify_lip_sync,
            )
            intro_cutoff = (
                self.cfg.visual_quality.reject_voiceover_segments_before_seconds
                if self.profile.verify_lip_sync
                else 0.0
            )
            if intro_cutoff > 0:
                segments = [
                    row for row in segments
                    if float(row["start"]) >= intro_cutoff
                ]
                for index, row in enumerate(segments):
                    row["segment_id"] = f"{index:06d}"
            for row in segments:
                row["transcript_source"] = payload.get("source", "unknown")
            write_json(segments_path, segments)
            
        t0 = time.perf_counter()
        self._stage(
            iid,
            f"segment_v4_{self.profile.name}",
            segment,
            outputs=[segments_path],
        )
        t_segment = time.perf_counter() - t0

        segments = read_json(segments_path)
        if not self.cfg.active_speaker.enabled and segments:
            try:
                from .auto_avsr_crop import _read_cached_landmarks
                _read_cached_landmarks(normalized, self.cfg.auto_avsr)
            except Exception as exc:
                print(f"[warning] Pre-caching landmarks skipped: {exc}", flush=True)

        t0_clips = time.perf_counter()
        workers_cfg = getattr(self.cfg, "processing", None)
        max_workers = (
            workers_cfg.workers_count
            if hasattr(workers_cfg, "workers_count")
            else 4
        )
        if max_workers > 1 and len(segments) > 1:
            # Respect cfg.transcription.num_workers (default 1) to prevent VRAM over-allocation on 16GB T4.
            if self.cfg.transcription.verify_clips:
                try:
                    from .transcribe import _load_model, resolve_device
                    dev, comp = resolve_device(self.cfg.transcription.device)
                    ctype = comp if self.cfg.transcription.compute_type == "auto" else self.cfg.transcription.compute_type
                    _load_model(self.cfg.transcription.model, dev, ctype, num_workers=self.cfg.transcription.num_workers)
                except Exception as exc:
                    print(f"[warning] Whisper model pre-loading failed: {exc}", flush=True)

            from concurrent.futures import ThreadPoolExecutor, as_completed
            import os
            total_cpus = os.cpu_count() or "?"
            print(f"[parallel] Processing {len(segments)} segments with {max_workers} worker threads (detected {total_cpus} CPU cores)...", flush=True)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._process_segment, item, normalized, seg): seg
                    for seg in segments
                }
                for f in as_completed(futures):
                    f.result()
        else:
            for seg in segments:
                self._process_segment(item, normalized, seg)
        t_clips_total = time.perf_counter() - t0_clips
        t_item_total = time.perf_counter() - item_start

        # Collect clip sub-stage timings
        clip_metas = sorted((self.workspace / "clips" / iid).glob("*/metadata.json"))
        extract_times = []
        whisper_times = []
        visual_times = []
        crop_times = []
        status_counts = {"accepted": 0, "review": 0, "rejected": 0}

        for meta_p in clip_metas:
            try:
                m = read_json(meta_p)
                st = m.get("quality_status", "unknown")
                status_counts[st] = status_counts.get(st, 0) + 1
                tm = m.get("timings", {})
                if "extract_s" in tm: extract_times.append(tm["extract_s"])
                if "whisper_s" in tm: whisper_times.append(tm["whisper_s"])
                if "visual_s" in tm: visual_times.append(tm["visual_s"])
                if "crop_s" in tm: crop_times.append(tm["crop_s"])
            except Exception:
                pass

        n_clips = len(clip_metas)
        sum_extract = sum(extract_times)
        sum_whisper = sum(whisper_times)
        sum_visual = sum(visual_times)
        sum_crop = sum(crop_times)

        print("\n" + "=" * 70, flush=True)
        print(f"⏱️  TIMING BREAKDOWN for item: {iid} ({n_clips} clips)", flush=True)
        print(f"  • Normalize:              {t_normalize:7.2f}s", flush=True)
        print(f"  • Full Transcription:     {t_transcribe:7.2f}s", flush=True)
        print(f"  • Segmentation:           {t_segment:7.2f}s", flush=True)
        print(f"  • Clip Processing Total:  {t_clips_total:7.2f}s", flush=True)
        if n_clips > 0:
            print(f"      - Clip Extract:       {sum_extract:7.2f}s (avg: {sum_extract/n_clips:.3f}s)", flush=True)
            print(f"      - Clip Whisper ASR:   {sum_whisper:7.2f}s (avg: {sum_whisper/n_clips:.3f}s)", flush=True)
            print(f"      - Visual Quality:     {sum_visual:7.2f}s (avg: {sum_visual/n_clips:.3f}s)", flush=True)
            print(f"      - Auto-AVSR Crop:     {sum_crop:7.2f}s (avg: {sum_crop/n_clips:.3f}s)", flush=True)
        print(f"  • Decisions: accepted={status_counts.get('accepted', 0)}, review={status_counts.get('review', 0)}, rejected={status_counts.get('rejected', 0)}", flush=True)
        print(f"  • Total Core Pipeline:    {t_item_total:7.2f}s", flush=True)
        print("=" * 70 + "\n", flush=True)

    def _process_segment(self, item, normalized, segment):
        iid, sid = item["id"], segment["segment_id"]
        key = f"{iid}/{sid}"
        out = self.workspace/"clips"/iid/sid
        source_clip, audio_clip = out/"source.mp4", out/"audio.wav"
        speaker_clip, mouth_clip = out/"active_speaker.mp4", out/"mouth.mp4"
        transcript_path, metadata_path = out/"transcript.txt", out/"metadata.json"
        clip_words_path = out/"clip_transcript.json"

        def build():
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            extract_audio_clip(
                normalized,
                segment["start"],
                segment["end"],
                audio_clip,
                self.cfg.normalization.audio_sample_rate,
            )
            t_extract = time.perf_counter() - t0

            original_text = segment["text"]
            label_text = original_text
            asr_conf = float(segment.get("asr_confidence", 1.0))
            transcript_source = segment.get("transcript_source")
            transcript_check = {
                "enabled": False,
                "status": "not_run",
                "similarity": None,
                "original_text": original_text,
                "verified_text": None,
            }
            t_whisper = 0.0
            if self.cfg.transcription.verify_clips:
                t0 = time.perf_counter()
                clip_words = transcribe(
                    audio_clip,
                    clip_words_path,
                    self.cfg.language,
                    self.cfg.transcription,
                )
                t_whisper = time.perf_counter() - t0
                verified_text = words_to_text(clip_words)
                similarity = transcript_similarity(original_text, verified_text)
                asr_conf = words_confidence(clip_words)
                mismatch = similarity < self.cfg.transcription.clip_min_similarity
                transcript_check = {
                    "enabled": True,
                    "status": "mismatch" if mismatch else "matched",
                    "similarity": similarity,
                    "minimum_similarity": self.cfg.transcription.clip_min_similarity,
                    "original_text": original_text,
                    "verified_text": verified_text,
                    "confidence": asr_conf,
                    "word_count": len(clip_words),
                }
                if self.cfg.transcription.replace_with_clip_transcript:
                    label_text = verified_text
                    transcript_source = "whisper_clip_verification"

            transcript_path.write_text(label_text+"\n", encoding="utf-8")

            # Early deterministic rejection: if transcript or ASR confidence guarantees rejection,
            # skip expensive MediaPipe FaceMesh and Auto-AVSR crop stages immediately.
            deterministic_reject = False
            deterministic_reasons = []
            if not bool(label_text):
                deterministic_reject = True
                deterministic_reasons.append("empty_transcript")
            if asr_conf < self.cfg.quality.min_asr_confidence:
                deterministic_reject = True
                deterministic_reasons.append("low_asr_confidence")
            if (
                transcript_check["status"] == "mismatch"
                and self.cfg.transcription.clip_mismatch_status == "rejected"
            ):
                deterministic_reject = True
                deterministic_reasons.append("transcript_mismatch")

            if deterministic_reject:
                for path in (source_clip, audio_clip, speaker_clip, mouth_clip, transcript_path):
                    if path.exists():
                        path.unlink()
                meta = item["metadata"]
                write_json(metadata_path, {
                    "item_id": iid, "segment_id": sid,
                    "video_path": "", "active_speaker_path": "",
                    "mouth_path": "", "audio_path": "",
                    "text": label_text, "original_text": original_text,
                    "start": segment["start"],
                    "end": segment["end"], "duration": segment["duration"],
                    "source_url": meta.get("source_url"), "title": meta.get("title"),
                    "channel": meta.get("channel"),
                    "transcript_source": transcript_source,
                    "transcript_check": transcript_check,
                    "asr_confidence": asr_conf,
                    "active_speaker_score": 1.0,
                    "face_coverage": 1.0,
                    "sharpness": 0.0,
                    "source_profile": self.profile.name,
                    "quality_status": "rejected",
                    "accepted": False,
                    "visual_quality": None,
                    "early_rejected": True,
                    "reasons": deterministic_reasons,
                    "timings": {
                        "extract_s": round(t_extract, 3),
                        "whisper_s": round(t_whisper, 3),
                        "visual_s": 0.0,
                        "crop_s": 0.0,
                    },
                })
                return

            if self.cfg.active_speaker.enabled:
                extract_clip(normalized, segment["start"], segment["end"],
                             source_clip, audio_clip, self.cfg.normalization)
                asd = select_active_speaker(source_clip, speaker_clip,
                    self.cfg.active_speaker, self.cfg.normalization.fps)
                crop_input = asd.video_path
                as_score, coverage = asd.score, asd.coverage
                crop_landmark_source = None
                crop_start_seconds = None
                crop_duration_seconds = None
                visual_source = crop_input
                visual_start = 0.0
                visual_duration = None
            else:
                crop_input = normalized
                as_score, coverage = 1.0, 1.0
                crop_landmark_source = normalized
                crop_start_seconds = float(segment["start"])
                crop_duration_seconds = float(segment["duration"])
                visual_source = normalized
                visual_start = float(segment["start"])
                visual_duration = float(segment["duration"])

            # Both profiles run the same mouth-visibility / scene / occlusion checks.
            # The ONLY profile difference: voiceover verifies lip-sync (rejects
            # segments whose audio doesn't match the visible mouth = external voice),
            # no_voiceover relaxes that check.
            t0 = time.perf_counter()
            visual = (
                analyze_visual_quality(
                    visual_source, audio_clip, self.cfg.visual_quality,
                    verify_lip_sync=self.profile.verify_lip_sync,
                    start_seconds=visual_start,
                    duration_seconds=visual_duration,
                )
                if self.cfg.visual_quality.enabled else None
            )
            t_visual = time.perf_counter() - t0

            visual_status = visual.status if visual is not None else "accepted"
            reasons = list(visual.reasons) if visual is not None else []

            if (
                self.profile.verify_lip_sync
                and float(segment["start"]) < self.cfg.visual_quality.reject_voiceover_segments_before_seconds
            ):
                visual_status = "rejected"
                reasons.append("voiceover_intro_window")
                if visual is not None:
                    visual.status = "rejected"
                    visual.reasons = reasons
            elif (
                self.profile.verify_lip_sync
                and self.cfg.visual_quality.reject_voiceover_review_segments
                and visual_status == "review"
            ):
                visual_status = "rejected"
                reasons.append("voiceover_review_rejected")
                if visual is not None:
                    visual.status = "rejected"
                    visual.reasons = reasons

            t_crop = 0.0
            if visual_status == "rejected":
                crop_sharpness = 0.0
                mouth_path_value = ""
            else:
                try:
                    t0 = time.perf_counter()
                    crop = crop_with_official_auto_avsr(
                        crop_input,
                        mouth_clip,
                        self.cfg.auto_avsr,
                        landmark_source=crop_landmark_source,
                        start_seconds=crop_start_seconds,
                        duration_seconds=crop_duration_seconds,
                    )
                    t_crop = time.perf_counter() - t0
                    crop_sharpness = crop.sharpness
                    mouth_path_value = str(mouth_clip)
                except NoUsableFaceError:
                    # A face-free clip is a normal dataset rejection, not a
                    # reason to abort every remaining item in a playlist.
                    visual_status = "rejected"
                    reasons.append("auto_avsr_no_face")
                    if visual is not None:
                        visual.status = "rejected"
                        visual.reasons = reasons
                    crop_sharpness = 0.0
                    mouth_path_value = ""

            base_ok = (
                bool(label_text) and
                asr_conf >= self.cfg.quality.min_asr_confidence and
                as_score >= self.cfg.quality.min_active_speaker_score and
                coverage >= self.cfg.quality.min_face_coverage
            )
            transcript_mismatch = transcript_check["status"] == "mismatch"

            if (
                not base_ok
                or visual_status == "rejected"
                or crop_sharpness < self.cfg.quality.min_sharpness
            ) or (
                transcript_mismatch
                and self.cfg.transcription.clip_mismatch_status == "rejected"
            ):
                quality_status = "rejected"
            elif visual_status == "review" or transcript_mismatch:
                quality_status = "review"
            else:
                quality_status = "accepted"

            accepted = quality_status == "accepted"
            if quality_status == "rejected":
                for path in (source_clip, audio_clip, speaker_clip, mouth_clip, transcript_path):
                    if path.exists():
                        path.unlink()
                video_path_value = ""
                active_speaker_path_value = ""
                audio_path_value = ""
                mouth_path_value = ""
            else:
                if self.cfg.quality.save_source_clip:
                    if not source_clip.exists():
                        extract_clip(normalized, segment["start"], segment["end"],
                                     source_clip, audio_clip, self.cfg.normalization)
                    video_path_value = str(source_clip)
                else:
                    if source_clip.exists():
                        source_clip.unlink()
                    video_path_value = ""
                active_speaker_path_value = str(crop_input)
                audio_path_value = str(audio_clip)

            meta = item["metadata"]
            write_json(metadata_path, {
                "item_id": iid, "segment_id": sid,
                "video_path": video_path_value,
                "active_speaker_path": active_speaker_path_value,
                "mouth_path": mouth_path_value, "audio_path": audio_path_value,
                "text": label_text, "original_text": original_text,
                "start": segment["start"],
                "end": segment["end"], "duration": segment["duration"],
                "source_url": meta.get("source_url"), "title": meta.get("title"),
                "channel": meta.get("channel"),
                "transcript_source": transcript_source,
                "transcript_check": transcript_check,
                "asr_confidence": asr_conf,
                "active_speaker_score": as_score,
                "face_coverage": coverage,
                "sharpness": crop_sharpness,
                "source_profile": self.profile.name,
                "quality_status": quality_status,
                "accepted": accepted,
                "visual_quality": visual.to_dict() if visual is not None else None,
                "timings": {
                    "extract_s": round(t_extract, 3),
                    "whisper_s": round(t_whisper, 3),
                    "visual_s": round(t_visual, 3),
                    "crop_s": round(t_crop, 3),
                },
            })
        self._stage(key, f"clip_v8_transcript_check_{self.profile.name}", build)

    def _stage(self, item_id, stage, fn, outputs: list[Path] | None = None):
        outputs_exist = all(path.exists() for path in outputs or [])
        if not self.force and self.state.done(item_id, stage) and outputs_exist:
            print(f"[stage] {item_id} {stage}: skipped", flush=True)
            return
        if not self.force and self.state.done(item_id, stage) and not outputs_exist:
            print(
                f"[stage] {item_id} {stage}: output missing, re-running",
                flush=True,
            )
        print(f"[stage] {item_id} {stage}: running", flush=True)
        started = time.monotonic()
        self.state.set(item_id, stage, "running")
        try: fn()
        except Exception as exc:
            elapsed = time.monotonic() - started
            print(f"[stage] {item_id} {stage}: failed ({elapsed:.1f}s) {exc}", flush=True)
            self.state.set(item_id, stage, "failed", str(exc)); raise
        self.state.set(item_id, stage, "done")
        elapsed = time.monotonic() - started
        print(f"[stage] {item_id} {stage}: done ({elapsed:.1f}s)", flush=True)
