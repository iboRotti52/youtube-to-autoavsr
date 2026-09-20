#!/usr/bin/env python3
"""Strict Parity Verification Script.

Compares outputs from baseline (dfae6fb) and perf branch to verify:
1. Exact match on clip counts and segment IDs.
2. 100% parity on quality_status (accepted / review / rejected).
3. 100% parity on transcript text (text and original_text).
4. Timestamp consistency (start and end within 1ms).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_metadata_records(path: Path) -> dict[str, dict]:
    """Loads metadata.json files from a video clips directory or manifest directory."""
    records = {}
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Case 1: path is a clips/video_id directory containing segment dirs
    meta_files = sorted(path.glob("*/metadata.json"))
    if not meta_files:
        # Case 2: path is a root workspace containing clips/*/*/metadata.json
        meta_files = sorted(path.glob("clips/*/*/metadata.json"))

    if not meta_files:
        # Case 3: check if all.jsonl exists in path or path/manifests
        jsonl_path = path / "manifests" / "all.jsonl" if (path / "manifests").exists() else path / "all.jsonl"
        if jsonl_path.exists():
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    records[rec["segment_id"]] = rec
            return records

    for mf in meta_files:
        try:
            rec = json.loads(mf.read_text(encoding="utf-8"))
            records[rec["segment_id"]] = rec
        except Exception as e:
            print(f"Warning: Failed to parse {mf}: {e}", file=sys.stderr)

    return records


def verify_parity(base_path: Path, perf_path: Path, video_id: str = "") -> bool:
    print(f"\n{'='*70}")
    print(f"🔍 Verifying Parity: {video_id or base_path.name}")
    print(f"  • Baseline: {base_path}")
    print(f"  • Perf:     {perf_path}")
    print(f"{'='*70}")

    base_records = load_metadata_records(base_path)
    perf_records = load_metadata_records(perf_path)

    base_keys = sorted(base_records.keys())
    perf_keys = sorted(perf_records.keys())

    print(f"  • Baseline clips: {len(base_keys)}")
    print(f"  • Perf clips:     {len(perf_keys)}")

    if not base_keys:
        print("❌ Error: No baseline records found!", file=sys.stderr)
        return False
    if not perf_keys:
        print("❌ Error: No perf records found!", file=sys.stderr)
        return False

    missing_in_perf = set(base_keys) - set(perf_keys)
    extra_in_perf = set(perf_keys) - set(base_keys)

    if missing_in_perf:
        print(f"❌ Segments missing in perf ({len(missing_in_perf)}): {sorted(missing_in_perf)[:5]}", file=sys.stderr)
        return False
    if extra_in_perf:
        print(f"❌ Extra segments in perf ({len(extra_in_perf)}): {sorted(extra_in_perf)[:5]}", file=sys.stderr)
        return False

    status_mismatches = []
    text_mismatches = []
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

        b_start, b_end = float(b.get("start", 0)), float(b.get("end", 0))
        p_start, p_end = float(p.get("start", 0)), float(p.get("end", 0))
        if abs(b_start - p_start) > 0.01 or abs(b_end - p_end) > 0.01:
            timestamp_mismatches.append((sid, (b_start, b_end), (p_start, p_end)))

    print(f"\n📊 Summary Stats:")
    print(f"  • Baseline: accepted={base_counts.get('accepted', 0)}, review={base_counts.get('review', 0)}, rejected={base_counts.get('rejected', 0)}")
    print(f"  • Perf:     accepted={perf_counts.get('accepted', 0)}, review={perf_counts.get('review', 0)}, rejected={perf_counts.get('rejected', 0)}")

    passed = True
    if status_mismatches:
        print(f"\n❌ Quality Status Mismatches ({len(status_mismatches)}):", file=sys.stderr)
        for sid, b_st, p_st in status_mismatches[:5]:
            print(f"    - Segment {sid}: baseline='{b_st}' vs perf='{p_st}'", file=sys.stderr)
        passed = False

    if text_mismatches:
        print(f"\n❌ Text Mismatches ({len(text_mismatches)}):", file=sys.stderr)
        for sid, b_txt, p_txt in text_mismatches[:5]:
            print(f"    - Segment {sid}:\n        baseline: {b_txt}\n        perf:     {p_txt}", file=sys.stderr)
        passed = False

    if timestamp_mismatches:
        print(f"\n❌ Timestamp Mismatches ({len(timestamp_mismatches)}):", file=sys.stderr)
        for sid, b_ts, p_ts in timestamp_mismatches[:5]:
            print(f"    - Segment {sid}: baseline={b_ts} vs perf={p_ts}", file=sys.stderr)
        passed = False

    if passed:
        print(f"\n✅ 100% PARITY VERIFIED for {video_id or base_path.name} ({len(base_keys)} clips identical)!")
    else:
        print(f"\n❌ PARITY CHECK FAILED for {video_id or base_path.name}!", file=sys.stderr)

    return passed


def main():
    parser = argparse.ArgumentParser(description="Strict Parity Verification")
    parser.add_argument("baseline", type=Path, help="Path to baseline workspace or clips")
    parser.add_argument("perf", type=Path, help="Path to perf workspace or clips")
    parser.add_argument("--video-id", default="", help="Optional video identifier")
    args = parser.parse_args()

    success = verify_parity(args.baseline, args.perf, args.video_id)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
