"""Command line entry point: ``uv run gesturesign <command>``.

Full run, once Kaggle asl-signs is in ``../_datasets/asl-signs`` (or ``$DATA_DIR``)::

    uv run gesturesign prepare
    uv run gesturesign train
    uv run gesturesign evaluate
    uv run gesturesign export
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Annotated

import typer

from gesturesign import config as paths
from gesturesign.config import Config, load_config
from gesturesign.data import read_train_csv
from gesturesign.signs import ASL_SIGNS

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Training pipeline for GestureSign: asl-signs landmarks -> ONNX model for the browser.",
)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="run configuration (TOML)")]
DataDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--data-dir",
        help="asl-signs folder, asl-signs.zip, or a folder holding it "
        "[default: $DATA_DIR or ../_datasets/asl-signs]",
    ),
]
CacheDirOpt = Annotated[
    Path | None,
    typer.Option("--cache-dir", help="landmark cache [default: <data-dir>/gesturesign-cache]"),
]
RunDirOpt = Annotated[
    Path | None,
    typer.Option("--run-dir", help="checkpoints and history [default: pipeline/runs/<name>]"),
]
DeviceOpt = Annotated[str | None, typer.Option("--device", help="auto, mps, cuda or cpu")]


def _data_dir(value: Path | None) -> Path:
    return value if value is not None else paths.default_data_dir()


def _cache_dir(value: Path | None, data_dir: Path) -> Path:
    if value is not None:
        return value
    # Next to the data, whether --data-dir names a folder or the zip itself.
    base = data_dir.parent if data_dir.is_file() else data_dir
    return base / "gesturesign-cache"


def _run_dir(value: Path | None, cfg: Config) -> Path:
    return value if value is not None else paths.RUNS_DIR / cfg.name


def _echo(message: str) -> None:
    typer.echo(message)


@app.command()
def synth(
    out: Annotated[Path, typer.Option("--out", help="folder to write the synthetic dataset to")],
    participants: int = 8,
    signs: int = 12,
    clips: int = 4,
    seed: int = 0,
) -> None:
    """Write a synthetic dataset with the asl-signs file layout and parquet schema."""
    from gesturesign import synthetic

    ids = list(range(101, 101 + participants))
    table = synthetic.write_dataset(
        out,
        participants=ids,
        n_signs=signs,
        clips_per_sign=clips,
        seed=seed,
        left_handed=frozenset(ids[3::4]),
    )
    _echo(f"wrote {len(table)} clips for participants {ids[0]}-{ids[-1]} to {out}")


@app.command()
def participants(data_dir: DataDirOpt = None) -> None:
    """List participant ids and clip counts (to choose the held-out signers)."""
    table = read_train_csv(_data_dir(data_dir))
    for pid, count in table.groupby("participant_id").size().items():
        _echo(f"{pid}\t{count}")


@app.command()
def prepare(
    config: ConfigOpt = paths.DEFAULT_CONFIG,
    data_dir: DataDirOpt = None,
    cache_dir: CacheDirOpt = None,
    workers: Annotated[int, typer.Option(help="parallel processes")] = 8,
    limit: Annotated[int | None, typer.Option(help="clips per participant (for trials)")] = None,
) -> None:
    """Read the parquet files into a sharded landmark cache (resumable)."""
    from gesturesign.prepare import prepare as run

    cfg = load_config(config)
    data = _data_dir(data_dir)
    run(cfg, data, _cache_dir(cache_dir, data), workers=workers, limit=limit, log=_echo)


@app.command()
def train(
    config: ConfigOpt = paths.DEFAULT_CONFIG,
    data_dir: DataDirOpt = None,
    cache_dir: CacheDirOpt = None,
    run_dir: RunDirOpt = None,
    device: DeviceOpt = None,
) -> None:
    """Train on the cache; the best validation checkpoint goes to <run-dir>/best.pt."""
    from gesturesign.train import train as run

    cfg = load_config(config)
    result = run(
        cfg,
        _cache_dir(cache_dir, _data_dir(data_dir)),
        _run_dir(run_dir, cfg),
        device_name=device,
        log=_echo,
    )
    _echo(
        f"best val top-1 {result.best_val_top1:.3f} after {result.epochs} epochs "
        f"({result.steps} steps, {result.seconds / 60:.1f} min) -> {result.checkpoint}"
    )


@app.command()
def evaluate(
    config: ConfigOpt = paths.DEFAULT_CONFIG,
    data_dir: DataDirOpt = None,
    cache_dir: CacheDirOpt = None,
    run_dir: RunDirOpt = None,
    reports_dir: Annotated[
        Path, typer.Option(help="where metrics.json/.md go")
    ] = paths.REPORTS_DIR,
    device: DeviceOpt = None,
) -> None:
    """Held-out-signer metrics, ship set and calibration -> reports/metrics.json and .md."""
    from gesturesign.evaluate import evaluate as run
    from gesturesign.evaluate import metrics_markdown

    cfg = load_config(config)
    metrics = run(
        cfg,
        _cache_dir(cache_dir, _data_dir(data_dir)),
        _run_dir(run_dir, cfg),
        device_name=device,
        log=_echo,
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=1) + "\n", encoding="utf-8"
    )
    table = metrics_markdown(metrics)
    (reports_dir / "metrics.md").write_text(table, encoding="utf-8")
    _echo(table)


@app.command()
def export(
    config: ConfigOpt = paths.DEFAULT_CONFIG,
    run_dir: RunDirOpt = None,
    out: Annotated[Path, typer.Option(help="output folder")] = paths.WEB_MODELS_DIR,
    stub: Annotated[bool, typer.Option("--stub", help="untrained model with the same I/O")] = False,
) -> None:
    """Write model.onnx, model-meta.json and labels.json for the web app."""
    from gesturesign.export import export_stub, export_trained

    cfg = load_config(config)
    if stub:
        written = export_stub(cfg, out, list(ASL_SIGNS))
    else:
        written = export_trained(_run_dir(run_dir, cfg), out)
    for path in written:
        _echo(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


@app.command("export-fixtures")
def export_fixtures(
    config: ConfigOpt = paths.DEFAULT_CONFIG,
    out: Annotated[Path, typer.Option(help="output folder")] = paths.FIXTURES_DIR,
) -> None:
    """Write the Python/TypeScript feature-parity fixture."""
    from gesturesign.fixtures import write_fixture

    path = write_fixture(load_config(config), out)
    _echo(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


@app.command("export-replay")
def export_replay(
    config: ConfigOpt = paths.DEFAULT_CONFIG,
    source: Annotated[str, typer.Option(help="synthetic or asl-signs")] = "synthetic",
    data_dir: DataDirOpt = None,
    signs: Annotated[str | None, typer.Option(help="comma-separated signs (asl-signs)")] = None,
    limit: Annotated[
        int | None, typer.Option(help="read at most this many candidate clips")
    ] = None,
    out: Annotated[Path, typer.Option(help="output file")] = paths.REPLAY_PATH,
) -> None:
    """Write the Replay mode's landmark clips (synthetic placeholders by default)."""
    from gesturesign.replay import asl_signs_clips, shipped_signs, synthetic_clips, write_replay

    cfg = load_config(config)
    if source == "synthetic":
        clips = synthetic_clips(cfg)
    elif source == "asl-signs":
        wanted = [s.strip() for s in signs.split(",")] if signs else None
        if wanted is None:
            wanted = shipped_signs(paths.WEB_MODELS_DIR / "labels.json")
        clips = asl_signs_clips(cfg, _data_dir(data_dir), wanted, limit=limit)
        if not clips:
            raise typer.BadParameter("no held-out clips matched; check the data dir and signs")
    else:
        raise typer.BadParameter("source must be 'synthetic' or 'asl-signs'")
    path = write_replay(cfg, clips, source, out)
    _echo(f"wrote {len(clips)} {source} clips to {path} ({path.stat().st_size / 1024:.0f} KiB)")


@app.command("replay-clips")
def replay_clips(
    config: ConfigOpt = paths.DEFAULT_CONFIG,
    data_dir: DataDirOpt = None,
    signs: Annotated[str | None, typer.Option(help="comma-separated signs to use")] = None,
    limit: Annotated[
        int | None, typer.Option(help="read at most this many candidate clips")
    ] = None,
    out: Annotated[Path, typer.Option(help="output file")] = paths.REPLAY_PATH,
) -> None:
    """Bundle ~6 held-out-signer clips of shipped signs for Replay (run after export)."""
    export_replay(
        config=config, source="asl-signs", data_dir=data_dir, signs=signs, limit=limit, out=out
    )


@app.command()
def report(
    metrics: Annotated[Path, typer.Option(help="metrics.json")] = paths.REPORTS_DIR
    / "metrics.json",
) -> None:
    """Print the README tables from reports/metrics.json."""
    from gesturesign.evaluate import metrics_markdown

    _echo(metrics_markdown(json.loads(metrics.read_text(encoding="utf-8"))))


@app.command()
def smoke(
    keep: Annotated[Path | None, typer.Option(help="keep the outputs in this folder")] = None,
) -> None:
    """Synthetic data -> prepare -> train (2 steps) -> evaluate -> export, in a temp folder."""
    from gesturesign import synthetic
    from gesturesign.evaluate import evaluate as run_evaluate
    from gesturesign.export import export_trained
    from gesturesign.prepare import prepare as run_prepare
    from gesturesign.train import train as run_train

    cfg = load_config(paths.CONFIG_DIR / "smoke.toml")
    start = time.monotonic()
    work = Path(tempfile.mkdtemp(prefix="gesturesign-smoke-")) if keep is None else keep
    try:
        ids = list(range(101, 109))
        data = work / "asl-signs"
        synthetic.write_dataset(
            data, participants=ids, n_signs=6, clips_per_sign=3, seed=1,
            left_handed=frozenset({104}),
        )  # fmt: skip
        _echo(f"[smoke] synthetic dataset: {len(read_train_csv(data))} clips")
        cache = work / "cache"
        run_prepare(cfg, data, cache, workers=2, log=lambda m: _echo(f"[smoke] {m}"))
        run = work / "run"
        result = run_train(cfg, cache, run, log=lambda m: _echo(f"[smoke] {m}"))
        _echo(f"[smoke] trained {result.steps} steps")
        metrics = run_evaluate(cfg, cache, run, log=lambda m: _echo(f"[smoke] {m}"))
        _echo(f"[smoke] test top-1 {metrics['test']['top1']:.3f} (2 steps; not meaningful)")
        written = export_trained(run, work / "models")
        _echo(f"[smoke] exported {', '.join(p.name for p in written)} (ONNX verified)")
    finally:
        if keep is None:
            shutil.rmtree(work, ignore_errors=True)
    _echo(f"[smoke] ok in {time.monotonic() - start:.1f}s")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
