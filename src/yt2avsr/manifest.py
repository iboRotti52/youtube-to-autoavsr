from __future__ import annotations
import csv, json
from pathlib import Path

FIELDS = [
 "item_id","segment_id","video_path","active_speaker_path","mouth_path","audio_path",
 "text","start","end","duration","source_url","title","channel","transcript_source",
 "asr_confidence","active_speaker_score","face_coverage","sharpness",
 "source_profile","quality_status","talknet_status","talknet_speaking_ratio","talknet_reason","mouth_visible_ratio","scene_cut_ratio","static_speech_ratio",
 "speech_mouth_motion_ratio","lip_sync_correlation","mouth_opening_correlation",
 "max_missing_run_seconds","unstable_landmark_ratio","visual_quality_reasons","accepted"
]

def _flatten(record):
    row = dict(record)
    visual = row.pop("visual_quality", None) or {}
    talknet = row.pop("talknet", None) or {}
    row["talknet_status"] = talknet.get("status")
    row["talknet_speaking_ratio"] = talknet.get("speaking_ratio")
    row["talknet_reason"] = talknet.get("reason")
    row["mouth_visible_ratio"] = visual.get("mouth_visible_ratio")
    row["scene_cut_ratio"] = visual.get("scene_cut_ratio")
    row["static_speech_ratio"] = visual.get("static_speech_ratio")
    row["speech_mouth_motion_ratio"] = visual.get("speech_mouth_motion_ratio")
    row["lip_sync_correlation"] = visual.get("lip_sync_correlation")
    row["mouth_opening_correlation"] = visual.get("mouth_opening_correlation")
    row["max_missing_run_seconds"] = visual.get("max_missing_run_seconds")
    row["unstable_landmark_ratio"] = visual.get("unstable_landmark_ratio")
    row["visual_quality_reasons"] = "|".join(visual.get("reasons") or [])
    return row

def rebuild(workspace: Path):
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted((workspace/"clips").glob("*/*/metadata.json"))]
    out = workspace/"manifests"; out.mkdir(parents=True, exist_ok=True)

    with (out/"all.jsonl").open("w",encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")

    groups = {
        "all.csv": records,
        "accepted.csv": [r for r in records if r.get("quality_status") == "accepted"],
        "review.csv": [r for r in records if r.get("quality_status") == "review"],
        "rejected.csv": [r for r in records if r.get("quality_status") == "rejected"],
    }
    for name, rows in groups.items():
        with (out/name).open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction="ignore")
            w.writeheader()
            w.writerows(_flatten(r) for r in rows)
    return records
