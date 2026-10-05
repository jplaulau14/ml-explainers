import argparse
import time
from collections.abc import Callable
from pathlib import Path

from ce_explainer.config import EXPORTED, FINAL_SEEDS, PRESETS
from ce_explainer.data import load_splits
from ce_explainer.export import (
    LIKELIHOOD_RUN,
    Run,
    field_payload,
    fixed_indices,
    hero_payload,
    likelihood_payload,
    replay_payload,
    write_payload,
)
from ce_explainer.markdown import render_markdown
from ce_explainer.report import build_results
from ce_explainer.runs import load_metrics, load_snapshots, read_json, save_run, write_json
from ce_explainer.sweep import sweep_all
from ce_explainer.train import train

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = ROOT / "out"
TIMING = RUNS / "timing.json"


def load_runs(names: tuple[str, ...]) -> dict[str, Run]:
    return {name: (load_snapshots(RUNS / name), load_metrics(RUNS / name)) for name in names}


def sweep_command() -> None:
    table = sweep_all(load_splits(), PRESETS.values())
    for name, result in table.items():
        print(f"{name} learning rate {result['selected']:g}", flush=True)
    write_json(RUNS / "sweep.json", table)


def train_command() -> None:
    splits = load_splits()
    sweep = read_json(RUNS / "sweep.json")
    for name, config in PRESETS.items():
        rate = sweep[name]["selected"]
        trajectories = [train(config, rate, seed, splits.train) for seed in FINAL_SEEDS]
        save_run(RUNS / name, trajectories, splits.test)
        print(f"{name} trained {len(trajectories)} seeds at rate {rate:g}", flush=True)


def export_command() -> None:
    splits = load_splits()
    runs = load_runs(EXPORTED)
    fixed = fixed_indices(splits.test)
    payloads = {
        "hero.json": hero_payload(splits, runs),
        "field.json": field_payload(splits, runs),
        "replay.json": replay_payload(splits, runs, fixed),
        "likelihood.json": likelihood_payload(splits, runs[LIKELIHOOD_RUN][0]),
    }
    for name, payload in payloads.items():
        print(f"wrote {write_payload(payload, OUT / name).relative_to(ROOT)}")


def report_command() -> None:
    splits = load_splits()
    seconds = read_json(TIMING) if TIMING.exists() else {}
    results = build_results(
        read_json(RUNS / "sweep.json"), load_runs(tuple(PRESETS)), splits, seconds
    )
    write_json(OUT / "results.json", results)
    (OUT / "report.md").write_text(render_markdown(results))
    print(f"wrote {(OUT / 'report.md').relative_to(ROOT)}")


def timed(stage: str, command: Callable[[], None]) -> None:
    began = time.perf_counter()
    command()
    seconds = read_json(TIMING) if TIMING.exists() else {}
    write_json(TIMING, seconds | {stage: time.perf_counter() - began})


COMMANDS = {
    "sweep": sweep_command,
    "train": train_command,
    "export": export_command,
    "report": report_command,
}


def all_command() -> None:
    if TIMING.exists():
        TIMING.unlink()
    for stage in ("sweep", "train", "export"):
        timed(stage, COMMANDS[stage])
    report_command()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ce-explainer")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sweep", help="pick each preset's learning rate on the validation split")
    commands.add_parser("train", help="train every preset with five seeds")
    commands.add_parser("export", help="write hero, field, replay and likelihood JSON to out/")
    commands.add_parser("report", help="write out/results.json and out/report.md")
    commands.add_parser("all", help="sweep, train, export and report")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "all":
        all_command()
    elif args.command == "report":
        report_command()
    else:
        timed(args.command, COMMANDS[args.command])
