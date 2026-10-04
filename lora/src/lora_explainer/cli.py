import argparse
import json
from pathlib import Path
from typing import Any

import torch

from lora_explainer.adapter import Which
from lora_explainer.config import PRESETS
from lora_explainer.evaluate import base_accuracy, validation_accuracy
from lora_explainer.export import build_payload, export_path, write_payload
from lora_explainer.report import build_results, write_report
from lora_explainer.train import save_run, set_determinism, train

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = ROOT / "out"
MATRICES: tuple[Which, ...] = ("q", "v")


def load_metadata(preset: str) -> dict[str, Any]:
    return json.loads((RUNS / preset / "meta.json").read_text())


def train_command(preset: str) -> None:
    config = PRESETS[preset]
    run = train(config)
    revision = run.metadata["datasetRevision"]
    score = validation_accuracy(run.model, run.tokenizer, config, revision)
    save_run(RUNS / preset, run.deltas, run.metadata | {"accuracy": score})
    print(f"{preset} validation accuracy {score:.4f}", flush=True)


def export_command() -> None:
    for preset in PRESETS:
        deltas = torch.load(RUNS / preset / "deltas.pt")
        metadata = load_metadata(preset)
        for matrix in MATRICES:
            payload = build_payload(deltas[matrix], preset, matrix, metadata)
            print(f"wrote {write_payload(payload, OUT).relative_to(ROOT)}")


def report_command() -> None:
    metadata = {preset: load_metadata(preset) for preset in PRESETS}
    config = PRESETS["full"]
    set_determinism(config)
    revisions = metadata["full"]["modelRevision"], metadata["full"]["datasetRevision"]
    base = base_accuracy(config, *revisions)
    paths = [export_path(OUT, preset, matrix) for preset in PRESETS for matrix in MATRICES]
    payloads = [json.loads(path.read_text()) for path in paths]
    write_report(build_results(base, metadata, payloads), OUT)
    print(f"wrote {(OUT / 'report.md').relative_to(ROOT)}")


def all_command() -> None:
    for preset in PRESETS:
        train_command(preset)
    export_command()
    report_command()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lora-explainer")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="fine-tune GPT-2 with one preset")
    train_parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    commands.add_parser("export", help="write singular value JSON files to out/")
    commands.add_parser("report", help="write out/results.json and out/report.md")
    commands.add_parser("all", help="train both presets, export and report")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        train_command(args.preset)
    elif args.command == "export":
        export_command()
    elif args.command == "report":
        report_command()
    else:
        all_command()
