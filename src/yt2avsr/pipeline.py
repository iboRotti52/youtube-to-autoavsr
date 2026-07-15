from __future__ import annotations
import time
from pathlib import Path
from typing import Any
from .active_speaker import select_active_speaker
from .auto_avsr_crop import crop_with_official_auto_avsr
from .config import AppConfig
from .downloader import download, register_local
from .manifest import rebuild
from .media import extract_clip, normalize
from .scenes import detect_scene_cuts
from .segment import make_segments
from .state import StateDB
from .subtitles import save_youtube_transcript
from .transcribe import transcribe
from .utils import read_json, write_json
from .visual_quality import analyze_visual_quality
from .profiles import get_profile

class Pipeline:
    def __init__(self, cfg: AppConfig, *, force: bool = False, profile: str = "no_voiceover") -> None:
        self.cfg, self.workspace, self.force = cfg, cfg.workspace, force
        self.profile = get_profile(profile)
        self.state = StateDB(self.workspace / "state.sqlite3")

    def process_url(self, url: str, *, playlist: bool = False):
        items = download(url, self.workspace/"raw", self.cfg.download, playlist=playlist)
        for item in items: self._process_item(item)
        rebuild(self.workspace)
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

        results = []
        failures = []
        for index, (mode, url) in enumerate(sources, start=1):
            print(f"[{index}/{len(sources)}] Processing: {url}")
            playlist = mode == "playlist" or (
                mode == "auto" and ("list=" in url or "/playlist" in url)
            )
            try:
                items = download(
                    url,
                    self.workspace / "raw",
                    self.cfg.download,
                    playlist=playlist,
                )
                for item in items:
                    self._process_item(item)
                results.extend(items)
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
        iid = item["id"]
        normalized = self.workspace/"normalized"/iid/"normalized.mp4"
        words_path = self.workspace/"transcripts"/iid/"words.json"
        segments_path = self.workspace/"transcripts"/iid/"segments.json"
        self._stage(
            iid,
            "normalize",
            lambda: normalize(item["source"], normalized, self.cfg.normalization),
            outputs=[normalized],
        )

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
        self._stage(iid, "transcribe", transcript, outputs=[words_path])

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
        self._stage(
            iid,
            f"segment_v3_{self.profile.name}",
            segment,
            outputs=[segments_path],
        )

        for seg in read_json(segments_path):
            self._process_segment(item, normalized, seg)

    def _process_segment(self, item, normalized, segment):
        iid, sid = item["id"], segment["segment_id"]
        key = f"{iid}/{sid}"
        out = self.workspace/"clips"/iid/sid
        source_clip, audio_clip = out/"source.mp4", out/"audio.wav"
        speaker_clip, mouth_clip = out/"active_speaker.mp4", out/"mouth.mp4"
        transcript_path, metadata_path = out/"transcript.txt", out/"metadata.json"

        def build():
            out.mkdir(parents=True, exist_ok=True)
            extract_clip(normalized, segment["start"], segment["end"],
                         source_clip, audio_clip, self.cfg.normalization)
            transcript_path.write_text(segment["text"]+"\n", encoding="utf-8")

            if self.cfg.active_speaker.enabled:
                asd = select_active_speaker(source_clip, speaker_clip,
                    self.cfg.active_speaker, self.cfg.normalization.fps)
                crop_input = asd.video_path
                as_score, coverage = asd.score, asd.coverage
            else:
                crop_input = source_clip
                as_score, coverage = 1.0, 1.0

            # Both profiles run the same mouth-visibility / scene / occlusion checks.
            # The ONLY profile difference: voiceover verifies lip-sync (rejects
            # segments whose audio doesn't match the visible mouth = external voice),
            # no_voiceover relaxes that check.
            visual = (
                analyze_visual_quality(
                    crop_input, audio_clip, self.cfg.visual_quality,
                    verify_lip_sync=self.profile.verify_lip_sync,
                )
                if self.cfg.visual_quality.enabled else None
            )

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

            if visual_status == "rejected":
                crop_sharpness = 0.0
                mouth_path_value = ""
            else:
                crop = crop_with_official_auto_avsr(
                    crop_input, mouth_clip, self.cfg.auto_avsr
                )
                crop_sharpness = crop.sharpness
                mouth_path_value = str(mouth_clip)

            asr_conf = float(segment.get("asr_confidence", 1.0))
            base_ok = (
                asr_conf >= self.cfg.quality.min_asr_confidence and
                as_score >= self.cfg.quality.min_active_speaker_score and
                coverage >= self.cfg.quality.min_face_coverage
            )

            if (
                not base_ok
                or visual_status == "rejected"
                or crop_sharpness < self.cfg.quality.min_sharpness
            ):
                quality_status = "rejected"
            elif visual_status == "review":
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
                video_path_value = str(source_clip)
                active_speaker_path_value = str(crop_input)
                audio_path_value = str(audio_clip)

            meta = item["metadata"]
            write_json(metadata_path, {
                "item_id": iid, "segment_id": sid,
                "video_path": video_path_value,
                "active_speaker_path": active_speaker_path_value,
                "mouth_path": mouth_path_value, "audio_path": audio_path_value,
                "text": segment["text"], "start": segment["start"],
                "end": segment["end"], "duration": segment["duration"],
                "source_url": meta.get("source_url"), "title": meta.get("title"),
                "channel": meta.get("channel"),
                "transcript_source": segment.get("transcript_source"),
                "asr_confidence": asr_conf,
                "active_speaker_score": as_score,
                "face_coverage": coverage,
                "sharpness": crop_sharpness,
                "source_profile": self.profile.name,
                "quality_status": quality_status,
                "accepted": accepted,
                "visual_quality": visual.to_dict() if visual is not None else None,
            })
        self._stage(key, f"clip_v6_{self.profile.name}", build)

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
