from __future__ import annotations
import csv, subprocess, sys, shutil
from pathlib import Path
from typing import Annotated
import typer
from .config import load_config
from .manifest import rebuild
from .pipeline import Pipeline

app=typer.Typer(no_args_is_help=True,help="Prepare permitted videos for Auto-AVSR.")

@app.command()
def process(url:Annotated[str,typer.Argument()],config:Annotated[Path|None,typer.Option("--config","-c")]=None,
            force:Annotated[bool,typer.Option()]=False,
            profile: Annotated[str, typer.Option("--profile", help="no_voiceover or voiceover")] = "no_voiceover"):
    cfg=load_config(config); Pipeline(cfg,force=force,profile=profile).process_url(url)
    typer.echo(f"Done: {cfg.workspace/'manifests'/'accepted.csv'}")

@app.command("process-playlist")
def playlist(url:Annotated[str,typer.Argument()],config:Annotated[Path|None,typer.Option("--config","-c")]=None,
             force:Annotated[bool,typer.Option()]=False,
             profile: Annotated[str, typer.Option("--profile", help="no_voiceover or voiceover")] = "no_voiceover"):
    cfg=load_config(config); Pipeline(cfg,force=force,profile=profile).process_url(url,playlist=True)

@app.command("process-local")
def local(path:Annotated[Path,typer.Argument()],config:Annotated[Path|None,typer.Option("--config","-c")]=None,
          force:Annotated[bool,typer.Option()]=False,
          profile: Annotated[str, typer.Option("--profile", help="no_voiceover or voiceover")] = "no_voiceover"):
    cfg=load_config(config); Pipeline(cfg,force=force,profile=profile).process_local(path)


@app.command("process-sources")
def process_sources(
    sources: Annotated[
        Path,
        typer.Argument(help="Text file containing one YouTube URL per line"),
    ] = Path("sources.txt"),
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
):
    cfg = load_config(config)
    Pipeline(cfg, force=force, profile=profile).process_sources_file(sources)
    typer.echo(f"Done: {cfg.workspace / 'manifests' / 'accepted.csv'}")


@app.command("check-downloader")
def check_downloader(
    url: Annotated[str, typer.Argument(help="A YouTube URL to test")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
):
    """Test YouTube extraction with the same automatic settings used by the pipeline."""
    from yt2avsr.downloader import _ydl_options
    import yt_dlp

    cfg = load_config(config)
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
):
    cfg = load_config(config)
    no_voiceover_path = Path("sources_no_voiceover.txt")
    voiceover_path = Path("sources_voiceover.txt")
    voiceover_urls = {_source_url(line) for line in _usable_source_lines(voiceover_path)}
    no_voiceover_lines = [
        line for line in _usable_source_lines(no_voiceover_path)
        if _source_url(line) not in voiceover_urls
    ]
    skipped = len(_usable_source_lines(no_voiceover_path)) - len(no_voiceover_lines)

    no_voiceover_job_path = (
        _write_filtered_sources(no_voiceover_path, no_voiceover_lines)
        if no_voiceover_lines
        else no_voiceover_path
    )
    jobs = [
        ("no_voiceover", no_voiceover_job_path),
        ("voiceover", voiceover_path),
    ]

    ran = []
    for profile, path in jobs:
        if not _has_usable_sources(path):
            typer.echo(f"Skipping {path.name}: no usable links, moving on.")
            continue
        Pipeline(cfg, force=force, profile=profile).process_sources_file(path)
        ran.append(path.name)

    if skipped:
        typer.echo(
            f"Skipped {skipped} duplicate no_voiceover source(s) because they also "
            "exist in sources_voiceover.txt."
        )

    if not ran:
        raise typer.BadParameter(
            "Neither sources_no_voiceover.txt nor sources_voiceover.txt has any "
            "usable links. Add at least one YouTube URL (one per line, no '#')."
        )
    typer.echo(f"Done ({', '.join(ran)}): {cfg.workspace / 'manifests'}")


@app.command("setup-external")
def setup_external(config:Annotated[Path|None,typer.Option("--config","-c")]=None):
    cfg=load_config(config); repo=cfg.auto_avsr.repo_dir
    repo.parent.mkdir(parents=True,exist_ok=True)
    if not repo.exists():
        subprocess.run(["git","clone","--depth","1",
                        "https://github.com/mpc001/auto_avsr.git",str(repo)],check=True)
    subprocess.run(["python","-m","pip","install","-r",
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
    cfg = load_config(config)
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

    cfg = load_config(config)
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

    cfg = load_config(config)
    repo_id = repo or cfg.cloud.repo_id
    if not repo_id:
        raise typer.BadParameter("Set cloud.repo_id in the config or pass --repo")
    statuses = [s.strip() for s in include.split(",") if s.strip()]
    result = push(cfg.workspace, repo_id, contributor=contributor,
                  statuses=statuses, token=token, private=cfg.cloud.private,
                  include_source=include_source, include_audio=include_audio)
    typer.echo(f"Pushed: {result}")


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

    cfg = load_config(config)
    repo_id = repo or cfg.cloud.repo_id
    if not repo_id:
        raise typer.BadParameter("Set cloud.repo_id in the config or pass --repo")
    local = pull(repo_id, dest, token=token, contributor=contributor)
    typer.echo(f"Pulled into: {local}")


@app.command()
def manifest(config:Annotated[Path|None,typer.Option("--config","-c")]=None):
    cfg=load_config(config); typer.echo(f"Wrote {len(rebuild(cfg.workspace))} records")

@app.command()
def inspect(config:Annotated[Path|None,typer.Option("--config","-c")]=None,
            limit:Annotated[int,typer.Option("--limit","-n")]=20):
    cfg=load_config(config); path=cfg.workspace/"manifests"/"all.csv"
    with path.open(encoding="utf-8") as f: rows=list(csv.DictReader(f))
    for r in rows[:limit]:
        status="ACCEPT" if r["accepted"].lower()=="true" else "REJECT"
        typer.echo(f"[{status}] {r['item_id']}/{r['segment_id']} "
                   f"ASR={r['asr_confidence']} ASD={r['active_speaker_score']} | {r['text'][:70]}")

if __name__=="__main__": app()
