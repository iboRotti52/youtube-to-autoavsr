from __future__ import annotations
import csv, subprocess
from pathlib import Path
from typing import Annotated
import typer
from .config import load_config
from .manifest import rebuild
from .pipeline import Pipeline

app=typer.Typer(no_args_is_help=True,help="Prepare permitted videos for Auto-AVSR.")

@app.command()
def process(url:Annotated[str,typer.Argument()],config:Annotated[Path|None,typer.Option("--config","-c")]=None,
            force:Annotated[bool,typer.Option()]=False):
    cfg=load_config(config); Pipeline(cfg,force=force).process_url(url)
    typer.echo(f"Done: {cfg.workspace/'manifests'/'accepted.csv'}")

@app.command("process-playlist")
def playlist(url:Annotated[str,typer.Argument()],config:Annotated[Path|None,typer.Option("--config","-c")]=None,
             force:Annotated[bool,typer.Option()]=False):
    cfg=load_config(config); Pipeline(cfg,force=force).process_url(url,playlist=True)

@app.command("process-local")
def local(path:Annotated[Path,typer.Argument()],config:Annotated[Path|None,typer.Option("--config","-c")]=None,
          force:Annotated[bool,typer.Option()]=False):
    cfg=load_config(config); Pipeline(cfg,force=force).process_local(path)


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


@app.command("process-both-sources")
def process_both_sources(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    force: Annotated[bool, typer.Option(help="Re-run completed stages")] = False,
):
    cfg = load_config(config)
    jobs = [
        ("no_voiceover", Path("sources_no_voiceover.txt")),
        ("voiceover", Path("sources_voiceover.txt")),
    ]

    ran = []
    for profile, path in jobs:
        if not _has_usable_sources(path):
            typer.echo(f"Skipping {path.name}: no usable links, moving on.")
            continue
        Pipeline(cfg, force=force, profile=profile).process_sources_file(path)
        ran.append(path.name)

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
