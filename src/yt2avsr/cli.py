from __future__ import annotations
import csv, subprocess, shutil
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


@app.command("process-both-sources")
def process_both_sources(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    force: Annotated[bool, typer.Option(help="Re-run completed stages")] = False,
):
    cfg = load_config(config)
    Pipeline(cfg, force=force, profile="no_voiceover").process_sources_file(
        Path("sources_no_voiceover.txt")
    )
    Pipeline(cfg, force=force, profile="voiceover").process_sources_file(
        Path("sources_voiceover.txt")
    )
    typer.echo(f"Done: {cfg.workspace / 'manifests'}")


@app.command("setup-talknet")
def setup_talknet(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
):
    cfg = load_config(config)
    repo = cfg.talknet.repo_dir
    venv_dir = cfg.talknet.python_executable.parent.parent

    repo.parent.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/TaoRuijie/TalkNet-ASD.git", str(repo)],
            check=True,
        )

    chosen = next(
        (name for name in ("python3.10", "python3.11", "python3.9")
         if shutil.which(name)),
        None,
    )
    if chosen is None:
        raise RuntimeError("Python 3.9-3.11 required for TalkNet")

    subprocess.run([chosen, "-m", "venv", str(venv_dir)], check=True)
    python = venv_dir / "bin" / "python"
    subprocess.run([str(python), "-m", "pip", "install", "-U", "pip"], check=True)
    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(repo / "requirement.txt")],
        check=True,
    )
    typer.echo(f"TalkNet ready: {repo}")

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
                  statuses=statuses, token=token, private=cfg.cloud.private)
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
