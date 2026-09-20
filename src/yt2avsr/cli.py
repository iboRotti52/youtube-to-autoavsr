from __future__ import annotations
import csv, subprocess, sys, shutil
from pathlib import Path
from typing import Annotated
import typer
from .config import load_config
from .manifest import rebuild
from .pipeline import Pipeline
from .sources import (
    add_sources_to_file,
    append_processed_sources,
    canonicalize_source,
    deduplicate_source_file,
    deduplicate_source_lines,
    get_source_key,
    is_playlist_source,
    is_source_processed,
    load_processed_ids,
    partition_sources,
    parse_shard,
    sync_processed_from_hf,
)

app=typer.Typer(no_args_is_help=True,help="Prepare permitted videos for Auto-AVSR.")

def _packaged_default_config() -> Path | None:
    candidate = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
    return candidate if candidate.is_file() else None


def load_cli_config(explicit: Path | None):
    """Team CLI config resolution (no shell hacks).

    Explicit --config always wins. Otherwise load configs/default.yaml
    (CWD first, then the copy packaged with this repo) so bare commands
    like `ytavsr process-both-sources --shard 0` behave exactly like
    `--config configs/default.yaml`. Falls back to code defaults only when
    no default.yaml can be found.
    """
    if explicit is not None:
        return load_config(explicit)
    for candidate in (Path("configs/default.yaml"), _packaged_default_config()):
        if candidate is not None and candidate.is_file():
            return load_config(candidate)
    print(
        "[config] configs/default.yaml bulunamadi; kod varsayilanlari kullaniliyor.",
        flush=True,
    )
    return load_config(None)

@app.command()
def process(
    url: Annotated[str, typer.Argument()],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    force: Annotated[bool, typer.Option()] = False,
    profile: Annotated[str, typer.Option("--profile", help="no_voiceover or voiceover")] = "no_voiceover",
    shard: Annotated[str | None, typer.Option("--shard", "-s", help="Shard index/total, e.g. 0/3")] = None,
    workers: Annotated[int | None, typer.Option("--workers", "-w", help="Number of worker threads (default: auto based on CPU cores)")] = None,
):
    cfg = load_cli_config(config)
    if workers is not None:
        cfg.processing.max_workers = workers
    shard_tuple = parse_shard(shard)
    Pipeline(cfg, force=force, profile=profile, shard=shard_tuple).process_url(url)
    typer.echo(f"Done: {cfg.workspace / 'manifests' / 'accepted.csv'}")

@app.command("process-playlist")
def playlist(
    url: Annotated[str, typer.Argument()],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    force: Annotated[bool, typer.Option()] = False,
    profile: Annotated[str, typer.Option("--profile", help="no_voiceover or voiceover")] = "no_voiceover",
    shard: Annotated[str | None, typer.Option("--shard", "-s", help="Shard index/total, e.g. 0/3")] = None,
    workers: Annotated[int | None, typer.Option("--workers", "-w", help="Number of worker threads (default: auto based on CPU cores)")] = None,
):
    cfg = load_cli_config(config)
    if workers is not None:
        cfg.processing.max_workers = workers
    shard_tuple = parse_shard(shard)
    Pipeline(cfg, force=force, profile=profile, shard=shard_tuple).process_url(url, playlist=True)

@app.command("process-local")
def local(path:Annotated[Path,typer.Argument(help="Video dosyası veya video klasörü (klasördeki .mp4/.mkv/.webm/.mov/.m4v dosyalarının hepsi işlenir)")],config:Annotated[Path|None,typer.Option("--config","-c")]=None,
          force:Annotated[bool,typer.Option()]=False,
          profile: Annotated[str, typer.Option("--profile", help="no_voiceover or voiceover")] = "no_voiceover",
          workers: Annotated[int | None, typer.Option("--workers", "-w", help="Number of worker threads (default: auto based on CPU cores)")] = None):
    cfg=load_cli_config(config)
    if workers is not None:
        cfg.processing.max_workers = workers
    Pipeline(cfg,force=force,profile=profile).process_local(path)


@app.command("process-sources")
def process_sources(
    sources: Annotated[
        Path,
        typer.Argument(help="Text file containing one YouTube URL per line"),
    ] = Path("sources_no_voiceover.txt"),
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(help="Re-run completed stages"),
    ] = False,
    profile: Annotated[
        str,
        typer.Option("--profile", help="no_voiceover or voiceover"),
    ] = "no_voiceover",
    shard: Annotated[
        str | None,
        typer.Option("--shard", "-s", help="Shard index/total, e.g. 0/3"),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option("--workers", "-w", help="Number of worker threads (default: auto based on CPU cores)"),
    ] = None,
):
    cfg = load_cli_config(config)
    if workers is not None:
        cfg.processing.max_workers = workers
    shard_tuple = parse_shard(shard)
    usable = _usable_source_lines(sources)
    deduped, duplicates = deduplicate_source_lines(usable)
    if duplicates:
        typer.echo(f"Skipped {len(duplicates)} duplicate source(s) within {sources.name}.")
    job_path = _write_filtered_sources(sources, deduped) if duplicates else sources
    Pipeline(cfg, force=force, profile=profile, shard=shard_tuple).process_sources_file(job_path)
    typer.echo(f"Done: {cfg.workspace / 'manifests' / 'accepted.csv'}")


@app.command("add")
def add_command(
    items: Annotated[
        list[str] | None,
        typer.Argument(
            help="One or more YouTube URLs, video IDs, or paths to .txt files containing links",
        ),
    ] = None,
    file: Annotated[
        list[Path] | None,
        typer.Option("--file", "-f", help="Read links from one or more text files (e.g. -f linkler.txt)"),
    ] = None,
    target: Annotated[
        Path | None,
        typer.Option("--target", "-t", help="Target source file (default: sources_no_voiceover.txt or sources_voiceover.txt)"),
    ] = None,
    voiceover: Annotated[
        bool,
        typer.Option("--voiceover", "-vo", help="Add to sources_voiceover.txt instead of sources_no_voiceover.txt"),
    ] = False,
    push: Annotated[
        bool,
        typer.Option("--push/--no-push", help="Automatically commit and push changes to GitHub"),
    ] = True,
    pull: Annotated[
        bool,
        typer.Option("--pull/--no-pull", help="Automatically pull latest changes from GitHub first"),
    ] = True,
    canonicalize: Annotated[
        bool,
        typer.Option("--canonicalize/--keep-raw", help="Standardize YouTube URLs (strip list/tracking params)"),
    ] = True,
):
    """Add YouTube links (directly or from text files) to source files with auto git sync & deduplication."""
    combined_items: list[str] = []
    if items:
        combined_items.extend(items)
    if file:
        combined_items.extend(str(f) for f in file)

    if not combined_items:
        typer.secho("Hata: Eklenecek video linki veya dosya belirtilmedi!", fg=typer.colors.RED, bold=True)
        typer.echo(
            "Örnek kullanım:\n"
            "  ytavsr add \"https://www.youtube.com/watch?v=VIDEO_ID\"\n"
            "  ytavsr add dosyam.txt\n"
            "  ytavsr add -f dosyam.txt\n"
            "  ytavsr add -f dosyam.txt --voiceover\n"
        )
        raise typer.Exit(1)

    if target:
        target_path = target
        other_path = None
    else:
        target_path = Path("sources_voiceover.txt") if voiceover else Path("sources_no_voiceover.txt")
        other_path = Path("sources_no_voiceover.txt") if voiceover else Path("sources_voiceover.txt")


    # Step 1: Git pull if requested
    if pull:
        try:
            res = subprocess.run(
                ["git", "pull", "--rebase"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                out_msg = res.stdout.strip()
                if "Already up to date" not in out_msg and out_msg:
                    typer.echo(f"[Git] Updated local repository: {out_msg}")
            else:
                typer.echo(f"[Git warning] git pull failed ({res.stderr.strip()}), proceeding locally.")
        except Exception as exc:
            typer.echo(f"[Git warning] git pull error: {exc}")

    # Step 2: Add sources
    result = add_sources_to_file(
        items=combined_items,
        target_path=target_path,
        other_source_path=other_path,
        canonicalize=canonicalize,
    )


    added = result["added"]
    duplicates = result["duplicates"]
    already_proc = result["already_processed"]
    cross_warn = result["cross_file_warnings"]
    invalid = result["invalid"]

    if added:
        typer.echo(f"\nSuccessfully added {len(added)} link(s) to {target_path.name}:")
        for a in added:
            typer.echo(f"  + {a}")

    if duplicates:
        typer.echo(f"\nSkipped {len(duplicates)} duplicate link(s) already in {target_path.name}:")
        for d in duplicates:
            typer.echo(f"  - {d}")

    if already_proc:
        typer.echo(f"\nSkipped {len(already_proc)} link(s) already processed in processed_sources.txt:")
        for p in already_proc:
            typer.echo(f"  - {p}")

    if cross_warn:
        typer.echo(f"\n[Note] {len(cross_warn)} of the added link(s) also exist in {other_path.name}:")
        for c in cross_warn:
            typer.echo(f"  * {c}")

    if invalid:
        typer.echo(f"\nSkipped {len(invalid)} invalid input(s):")
        for inv in invalid:
            typer.echo(f"  ! {inv}")

    if not added:
        typer.echo(f"\nNo new unique links were added to {target_path.name}.")
        return

    # Step 3: Git push if requested
    if push:
        try:
            subprocess.run(["git", "add", str(target_path)], check=True)
            commit_msg = f"chore(sources): add {len(added)} link(s) to {target_path.name}"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            push_res = subprocess.run(["git", "push"], capture_output=True, text=True, check=False)
            if push_res.returncode == 0:
                typer.echo(f"[Git] Pushed updates to GitHub: {commit_msg}")
            else:
                typer.echo(f"[Git warning] git push failed ({push_res.stderr.strip()}). You can push manually later.")
        except Exception as exc:
            typer.echo(f"[Git warning] git commit/push error: {exc}")


@app.command("check-downloader")
def check_downloader(
    url: Annotated[str, typer.Argument(help="A YouTube URL to test")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
):
    """Test YouTube extraction with the same automatic settings used by the pipeline."""
    from yt2avsr.downloader import _ydl_options
    import yt_dlp

    cfg = load_cli_config(config)
    options = _ydl_options(
        cfg.workspace / "_download_test",
        cfg.download,
        playlist=False,
    )
    options["skip_download"] = True
    options["quiet"] = False
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    typer.echo(
        f"Downloader OK: {info.get('id')} | {info.get('title')} | "
        f"{len(info.get('formats') or [])} formats"
    )


def _has_usable_sources(path: Path) -> bool:
    """True if the file exists and has at least one non-comment, non-blank line."""
    if not path.exists():
        return False
    return any(
        line.strip() and not line.strip().startswith("#")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _usable_source_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _source_url(line: str) -> str:
    parts = line.split(maxsplit=1)
    if len(parts) == 2 and parts[0].lower() in {"video", "playlist"}:
        return parts[1].strip()
    return line.strip()


def _write_filtered_sources(path: Path, lines: list[str]) -> Path:
    filtered = path.with_name(f".{path.stem}.filtered.txt")
    filtered.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return filtered


@app.command("process-both-sources")
def process_both_sources(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    force: Annotated[bool, typer.Option(help="Re-run completed stages")] = False,
    shard: Annotated[
        str | None,
        typer.Option("--shard", "-s", help="Shard: 0/3 (İbrahim Gözlükaya), 1/3 (Damla Kemal), 2/3 (İbrahim Billurcu)"),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option("--workers", "-w", help="Number of worker threads (default: auto based on CPU cores)"),
    ] = None,
):
    cfg = load_cli_config(config)
    if workers is not None:
        cfg.processing.max_workers = workers
    shard_tuple = parse_shard(shard)
    if shard_tuple:
        from .sources import get_shard_owner

        owner = get_shard_owner(shard_tuple)
        owner_info = f" ({owner})" if owner else ""
        typer.echo(f"Running shard {shard_tuple[0]}/{shard_tuple[1]}{owner_info}")


    if cfg.sources.auto_sync_hf and cfg.cloud.repo_id:
        try:
            typer.echo(f"Checking Hugging Face dataset '{cfg.cloud.repo_id}' for latest processed videos...")
            hf_records = sync_processed_from_hf(cfg.cloud.repo_id, path=cfg.sources.processed_file)
            if hf_records:
                typer.echo(f"Hugging Face sync: {len(hf_records)} total unique video(s) tracked.")
        except Exception as exc:
            typer.echo(f"[warning] Hugging Face sync skipped ({exc}), continuing with local list.")

    processed_ids = load_processed_ids(cfg.sources.processed_file)

    def _is_unprocessed(line: str) -> bool:
        url = _source_url(line)
        parts = line.split(maxsplit=1)
        mode = parts[0].lower() if len(parts) == 2 and parts[0].lower() in {"video", "playlist"} else "auto"
        if is_playlist_source(mode, url):
            return True
        return not is_source_processed(url, processed_ids)

    no_voiceover_path = Path("sources_no_voiceover.txt")
    voiceover_path = Path("sources_voiceover.txt")

    # 1. Deduplicate voiceover internally
    usable_voiceover = _usable_source_lines(voiceover_path)
    deduped_vo_lines, dups_vo = deduplicate_source_lines(usable_voiceover)
    skipped_dup_vo = len(dups_vo)

    # 2. Track keys seen in voiceover to deduplicate against no_voiceover
    seen_vo_keys = {get_source_key(l) for l in deduped_vo_lines if get_source_key(l)}

    # 3. Deduplicate no_voiceover internally AND against voiceover
    usable_no_voiceover = _usable_source_lines(no_voiceover_path)
    deduped_nvo_lines, dups_nvo = deduplicate_source_lines(
        usable_no_voiceover, seen_keys=set(seen_vo_keys)
    )

    # Distinguish internal duplicates in nvo vs cross-file duplicates with vo
    internal_nvo_keys: set[str] = set()
    skipped_internal_dup_nvo = 0
    skipped_cross_duplicates = 0
    for l in usable_no_voiceover:
        k = get_source_key(l)
        if not k:
            continue
        if k in seen_vo_keys:
            skipped_cross_duplicates += 1
        elif k in internal_nvo_keys:
            skipped_internal_dup_nvo += 1
        else:
            internal_nvo_keys.add(k)

    # Keep the full deduplicated source order across both profiles before sharding.
    all_sources = [
        *(("no_voiceover", line) for line in deduped_nvo_lines),
        *(("voiceover", line) for line in deduped_vo_lines),
    ]
    assigned_sources = partition_sources(all_sources, shard_tuple)
    if shard_tuple:
        typer.echo(
            f"Shard {shard_tuple[0]}/{shard_tuple[1]} assigned "
            f"{len(assigned_sources)} of {len(all_sources)} total source(s)."
        )

    pending_sources = []
    skipped_processed = {"no_voiceover": 0, "voiceover": 0}
    for profile, line in assigned_sources:
        if _is_unprocessed(line):
            pending_sources.append((profile, line))
        else:
            skipped_processed[profile] += 1

    jobs = []
    for profile, source_path in (
        ("no_voiceover", no_voiceover_path),
        ("voiceover", voiceover_path),
    ):
        assigned_lines = [line for job_profile, line in pending_sources if job_profile == profile]
        jobs.append((profile, _write_filtered_sources(source_path, assigned_lines)))

    if skipped_dup_vo > 0:
        typer.echo(f"Skipped {skipped_dup_vo} duplicate source(s) within sources_voiceover.txt.")

    if skipped_internal_dup_nvo > 0:
        typer.echo(f"Skipped {skipped_internal_dup_nvo} duplicate source(s) within sources_no_voiceover.txt.")

    if skipped_cross_duplicates > 0:
        typer.echo(
            f"Skipped {skipped_cross_duplicates} duplicate no_voiceover source(s) because they also "
            "exist in sources_voiceover.txt."
        )

    total_skipped_processed = sum(skipped_processed.values())
    if total_skipped_processed > 0:
        typer.echo(
            f"Skipped {total_skipped_processed} source(s) because they were already processed "
            f"({cfg.sources.processed_file})."
        )

    ran = []
    for profile, path in jobs:
        if not _has_usable_sources(path):
            typer.echo(f"Skipping {path.name}: no usable links, moving on.")
            continue
        Pipeline(cfg, force=force, profile=profile, shard=shard_tuple).process_sources_file(
            path,
            already_partitioned=True,
        )
        ran.append(path.name)

    if not ran:
        if shard_tuple:
            typer.echo("Bu shard için pending source yok; işlem yapılmadı.")
            return
        if total_skipped_processed > 0:
            typer.echo("All sources have already been processed.")
            return
        raise typer.BadParameter(
            "Neither sources_no_voiceover.txt nor sources_voiceover.txt has any "
            "usable links. Add at least one YouTube URL (one per line, no '#')."
        )
    typer.echo(f"Done ({', '.join(ran)}): {cfg.workspace / 'manifests'}")


@app.command("setup-external")
def setup_external(config:Annotated[Path|None,typer.Option("--config","-c")]=None):
    cfg=load_cli_config(config); repo=cfg.auto_avsr.repo_dir
    repo.parent.mkdir(parents=True,exist_ok=True)
    if not repo.exists():
        subprocess.run(["git","clone","--depth","1",
                        "https://github.com/mpc001/auto_avsr.git",str(repo)],check=True)
    subprocess.run([sys.executable,"-m","pip","install","-r",
                    str(repo/"preparation"/"requirements.txt")],check=True)
    typer.echo(f"Official Auto-AVSR installed at {repo}")


@app.command("setup-retinaface")
def setup_retinaface(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
):
    """Install the official RetinaFace tracker (torch + ibug) used by Auto-AVSR.

    Mirrors external/auto_avsr/preparation/tools: installs torch, then the
    ibug.face_detection and ibug.face_alignment packages (pretrained weights come
    via Git LFS). Run once after `setup-external`.
    """
    cfg = load_cli_config(config)
    ext = cfg.auto_avsr.repo_dir.parent
    ext.mkdir(parents=True, exist_ok=True)

    if shutil.which("git-lfs") is None and shutil.which("git") is not None:
        # git-lfs plugin is invoked as `git lfs`; a missing `git-lfs` binary means
        # the LFS weights won't download.
        raise RuntimeError(
            "Git LFS gerekli (ibug ağırlıkları için). Kur: "
            "brew install git-lfs && git lfs install"
        )

    subprocess.run([sys.executable, "-m", "pip", "install",
                    "torch", "torchvision"], check=True)

    repos = [
        ("face_detection", "https://github.com/hhj1897/face_detection.git"),
        ("face_alignment", "https://github.com/hhj1897/face_alignment.git"),
    ]
    for name, url in repos:
        target = ext / name
        if not target.exists():
            subprocess.run(["git", "clone", url, str(target)], check=True)
        subprocess.run(["git", "lfs", "pull"], cwd=str(target), check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(target)],
                       check=True)

    typer.echo("RetinaFace (ibug.face_detection + ibug.face_alignment) hazır.")


@app.command("setup-whisper")
def setup_whisper(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    model: Annotated[str | None, typer.Option("--model", help="Override the model to fetch")] = None,
):
    """Pre-download the Whisper model so the first run doesn't stall downloading it.

    Fetches the model named in the config (default large-v3-turbo) into the local
    Hugging Face cache. Run once during setup.
    """
    from faster_whisper import WhisperModel

    cfg = load_cli_config(config)
    name = model or cfg.transcription.model
    typer.echo(f"Downloading Whisper model '{name}' (once; cached afterwards)...")
    # cpu/int8 just triggers the download; the cache is reused on any device later.
    WhisperModel(name, device="cpu", compute_type="int8")
    typer.echo(f"Whisper model '{name}' ready.")

@app.command("push-data")
def push_data(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    repo: Annotated[str | None, typer.Option("--repo", help="HF dataset repo id, overrides config")] = None,
    contributor: Annotated[str | None, typer.Option("--contributor", help="Your sub-folder name (default: HF username)")] = None,
    include: Annotated[str, typer.Option("--include", help="Comma list: accepted,review,rejected")] = "accepted,review",
    include_source: Annotated[bool, typer.Option("--include-source", help="Also upload the large raw source.mp4 (default: skip)")] = False,
    include_audio: Annotated[bool, typer.Option("--include-audio", help="Also upload audio.wav (needed only for the audio-visual model)")] = False,
    token: Annotated[str | None, typer.Option("--token", help="HF token (else HF_TOKEN env or cached login)")] = None,
):
    """Upload your processed clips to the shared private Hugging Face dataset."""
    from .cloud import push

    cfg = load_cli_config(config)
    repo_id = repo or cfg.cloud.repo_id
    if not repo_id:
        raise typer.BadParameter("Set cloud.repo_id in the config or pass --repo")
    statuses = [s.strip() for s in include.split(",") if s.strip()]
    result = push(cfg.workspace, repo_id, contributor=contributor,
                  statuses=statuses, token=token, private=cfg.cloud.private,
                  include_source=include_source, include_audio=include_audio)
    typer.echo(f"Pushed: {result}")

    # Record successfully pushed video metadata into processed_sources.txt
    import json
    recs = []
    for p in sorted((cfg.workspace / "clips").glob("*/*/metadata.json")):
        try:
            recs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    pushed_records = [r for r in recs if r.get("quality_status") in statuses]
    added = append_processed_sources(pushed_records, cfg.sources.processed_file)
    if added:
        typer.echo(f"Recorded {added} new video(s) into {cfg.sources.processed_file}")


@app.command("sync-processed")
def sync_processed(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    repo: Annotated[str | None, typer.Option("--repo", help="HF dataset repo id, overrides config")] = None,
    token: Annotated[str | None, typer.Option("--token", help="HF token (else HF_TOKEN env or cached login)")] = None,
):
    """Sync previously processed videos from all teammates on Hugging Face into processed_sources.txt."""
    cfg = load_cli_config(config)
    repo_id = repo or cfg.cloud.repo_id
    if not repo_id:
        raise typer.BadParameter("Set cloud.repo_id in the config or pass --repo")
    typer.echo(f"Checking Hugging Face dataset '{repo_id}' for all contributors' videos...")
    records = sync_processed_from_hf(repo_id=repo_id, token=token, path=cfg.sources.processed_file)
    typer.echo(f"Sync complete. Found {len(records)} unique video(s) from HF, updated {cfg.sources.processed_file}.")


@app.command("pull-data")
def pull_data(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    repo: Annotated[str | None, typer.Option("--repo", help="HF dataset repo id, overrides config")] = None,
    dest: Annotated[Path, typer.Option("--dest", help="Where to download the dataset")] = Path("data_cloud"),
    contributor: Annotated[str | None, typer.Option("--contributor", help="Only pull one contributor's folder")] = None,
    token: Annotated[str | None, typer.Option("--token", help="HF token (else HF_TOKEN env or cached login)")] = None,
):
    """Download the shared dataset (everyone's clips) for training."""
    from .cloud import pull

    cfg = load_cli_config(config)
    repo_id = repo or cfg.cloud.repo_id
    if not repo_id:
        raise typer.BadParameter("Set cloud.repo_id in the config or pass --repo")
    local = pull(repo_id, dest, token=token, contributor=contributor)
    typer.echo(f"Pulled into: {local}")


@app.command()
def manifest(config:Annotated[Path|None,typer.Option("--config","-c")]=None):
    cfg=load_cli_config(config); typer.echo(f"Wrote {len(rebuild(cfg.workspace))} records")

@app.command()
def inspect(config:Annotated[Path|None,typer.Option("--config","-c")]=None,
            limit:Annotated[int,typer.Option("--limit","-n")]=20):
    cfg=load_cli_config(config); path=cfg.workspace/"manifests"/"all.csv"
    with path.open(encoding="utf-8") as f: rows=list(csv.DictReader(f))
    for r in rows[:limit]:
        status="ACCEPT" if r["accepted"].lower()=="true" else "REJECT"
        typer.echo(f"[{status}] {r['item_id']}/{r['segment_id']} "
                   f"ASR={r['asr_confidence']} ASD={r['active_speaker_score']} | {r['text'][:70]}")

@app.command("dedup-sources")
def dedup_sources(
    files: Annotated[
        list[Path] | None,
        typer.Argument(help="Specific source files to deduplicate"),
    ] = None,
    check: Annotated[
        bool,
        typer.Option("--check", help="Check for duplicates without modifying files"),
    ] = False,
    canonicalize: Annotated[
        bool,
        typer.Option("--canonicalize", help="Clean URLs to standard YouTube format"),
    ] = False,
):
    """Find and remove duplicate video/playlist entries from source text files."""
    target_files = files or [
        Path("sources_no_voiceover.txt"),
        Path("sources_voiceover.txt"),
    ]

    total_removed = 0
    for path in target_files:
        if not path.exists():
            continue

        count, dups = deduplicate_source_file(
            path,
            in_place=not check,
            canonicalize=canonicalize,
        )
        total_removed += count
        if count > 0:
            verb = "found" if check else "removed"
            typer.echo(f"[{path.name}] {count} duplicate(s) {verb}:")
            for d in dups:
                k = get_source_key(d) or ""
                typer.echo(f"  - {d} ({k})")
        else:
            typer.echo(f"[{path.name}] OK (no internal duplicates)")

    # Cross-file check
    nvo_path = Path("sources_no_voiceover.txt")
    vo_path = Path("sources_voiceover.txt")
    if nvo_path.exists() and vo_path.exists():
        usable_vo = _usable_source_lines(vo_path)
        usable_nvo = _usable_source_lines(nvo_path)
        vo_keys = {get_source_key(l): l for l in usable_vo if get_source_key(l)}
        cross = []
        for l in usable_nvo:
            k = get_source_key(l)
            if k and k in vo_keys:
                cross.append((l, vo_keys[k]))
        if cross:
            typer.echo(
                f"\n[Warning] {len(cross)} link(s) exist in BOTH sources_no_voiceover.txt and sources_voiceover.txt:"
            )
            for nvo_l, vo_l in cross:
                typer.echo(f"  - {nvo_l} (in sources_voiceover.txt: {vo_l})")


@app.command("modal", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def modal_cmd(
    ctx: typer.Context,
    shard: Annotated[str | None, typer.Option("--shard", help="Shard: 0/3 (İbrahim Gözlükaya), 1/3 (Damla Kemal), 2/3 (İbrahim Billurcu)")] = None,
    url: Annotated[str | None, typer.Option("--url", help="Process single YouTube URL")] = None,
    voiceover: Annotated[bool, typer.Option("--voiceover", help="Mark single URL as voiceover")] = False,
    no_push: Annotated[bool, typer.Option("--no-push", help="Do not push clips to Hugging Face")] = False,
    limit: Annotated[int, typer.Option("--limit", help="Max videos to process (0 = all)")] = 0,
    gpu: Annotated[str, typer.Option("--gpu", help="GPU type on Modal (T4, A10G, etc.)")] = "T4",
    download_local: Annotated[bool, typer.Option("--download-local", help="Download output clips to local data folder")] = False,
    hf_token: Annotated[str | None, typer.Option("--hf-token", help="Hugging Face write token")] = None,
    contributor: Annotated[str | None, typer.Option("--contributor", help="HF contributor folder name")] = None,
    config: Annotated[str, typer.Option("--config", "-c", help="Config file to use")] = "configs/retina_1080p.yaml",
):
    """Run RetinaFace + 1080p processing in the cloud on your own Modal GPU account."""
    from .cloud import check_hf_login_or_warn

    # 1. Check Modal installation
    if shutil.which("modal") is None:
        typer.secho("\n❌ Modal CLI bulunamadı!", fg=typer.colors.RED, bold=True)
        typer.echo(
            "Ekipteki herkesin kendi Modal hesabında GPU (RetinaFace + 1080p) ile çalışabilmesi için:\n"
            "  1. modal.com adresinde ücretsiz hesap açın (aylık 30$ ücretsiz kredi)\n"
            "  2. Terminalde: pip install modal\n"
            "  3. Terminalde: modal setup (tarayıcıdan hesabınızı bağlayın)\n"
        )
        raise typer.Exit(1)

    # 2. Check Hugging Face authentication
    resolved_tok, detected_user = check_hf_login_or_warn(token=hf_token, required=(not no_push))
    if not resolved_tok and not no_push:
        typer.secho("Modal işlemi Hugging Face girişi yapılmadığı için başlatılmadı.", fg=typer.colors.RED)
        typer.echo("Hugging Face girişi yaptıktan sonra tekrar deneyin veya '--no-push' ekleyin.\n")
        raise typer.Exit(1)

    # 3. Build command
    cmd = ["modal", "run", "modal_app.py"]
    if shard:
        from .sources import parse_shard, get_shard_owner

        st = parse_shard(shard)
        if st:
            canonical_shard = f"{st[0]}/{st[1]}"
            owner = get_shard_owner(st)
            owner_info = f" ({owner})" if owner else ""
            typer.echo(f"Aktif Shard: {canonical_shard}{owner_info}")
            cmd.extend(["--shard", canonical_shard])
        else:
            cmd.extend(["--shard", shard])

    if url:
        cmd.extend(["--url", url])
    if voiceover:
        cmd.append("--voiceover")
    if no_push:
        cmd.append("--no-push")
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    if gpu != "T4":
        cmd.extend(["--gpu", gpu])
    if download_local:
        cmd.append("--download-local")
    if resolved_tok:
        cmd.extend(["--hf-token", resolved_tok])
    if contributor or detected_user:
        cmd.extend(["--contributor", contributor or detected_user or "unknown"])
    if config != "configs/retina_1080p.yaml":
        cmd.extend(["--config", config])

    if ctx.args:
        cmd.extend(ctx.args)

    typer.secho(f"\n🚀 Modal GPU görevi başlatılıyor: {' '.join(cmd)}", fg=typer.colors.CYAN, bold=True)
    res = subprocess.run(cmd)
    raise typer.Exit(res.returncode)


if __name__=="__main__": app()
