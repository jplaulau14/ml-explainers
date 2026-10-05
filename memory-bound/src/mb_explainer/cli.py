import argparse
import os

from mb_explainer.jsonio import read_json, write_json
from mb_explainer.paths import OUT, ROOT, git_commit
from mb_explainer.report import render
from mb_explainer.settings import settings as make_settings


def prepare() -> None:
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def stamp(payload: dict, quick: bool) -> dict:
    return {"version": 1, "gitCommit": git_commit(), "quick": quick, **payload}


def probe_command(quick: bool) -> None:
    from mb_explainer.machine import probe

    chosen = make_settings(quick)
    payload = probe(chosen.stream_repeats, chosen.peak_n, chosen.peak_repeats, chosen.peak_dtypes)
    path = write_json(OUT / "machine.json", stamp(payload, quick))
    print(f"wrote {path.relative_to(ROOT)}", flush=True)


def roofline_command(quick: bool) -> None:
    from mb_explainer.roofline import sweep

    chosen = make_settings(quick)
    machine = read_json(OUT / "machine.json")
    payload = sweep(machine, chosen.roofline_tokens, chosen.roofline_repeats)
    path = write_json(OUT / "roofline.json", stamp(payload, quick))
    print(f"wrote {path.relative_to(ROOT)}", flush=True)


def decode_command(quick: bool) -> None:
    from mb_explainer.decode import benchmark

    chosen = make_settings(quick)
    payload = benchmark(read_json(OUT / "machine.json"), chosen)
    path = write_json(OUT / "decode.json", stamp(payload, quick))
    print(f"wrote {path.relative_to(ROOT)}", flush=True)


def kv_command(quick: bool) -> None:
    from mb_explainer.machine import triad
    from mb_explainer.tables import kv_payload

    machine = read_json(OUT / "machine.json")
    payload = kv_payload(triad(machine)["bestBytesPerSecond"], git_commit())
    payload["quick"] = quick
    path = write_json(OUT / "kv_cache.json", payload)
    print(f"wrote {path.relative_to(ROOT)}", flush=True)


def specs_command() -> None:
    from mb_explainer.specs import payload as specs_payload

    body = specs_payload()
    body["gitCommit"] = git_commit()
    path = write_json(OUT / "specs.json", body)
    print(f"wrote {path.relative_to(ROOT)}", flush=True)


def report_command() -> None:
    text = render(
        read_json(OUT / "machine.json"),
        read_json(OUT / "roofline.json"),
        read_json(OUT / "decode.json"),
        read_json(OUT / "kv_cache.json"),
        read_json(OUT / "specs.json"),
    )
    path = OUT / "report.md"
    path.write_text(text)
    print(f"wrote {path.relative_to(ROOT)}", flush=True)


def all_command(quick: bool) -> None:
    probe_command(quick)
    roofline_command(quick)
    decode_command(quick)
    kv_command(quick)
    specs_command()
    report_command()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mb-explainer")
    parser.add_argument("--quick", action="store_true", help="fewer repeats and a shorter decode")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe", help="measure bandwidth, peak FLOP/s, and machine metadata")
    commands.add_parser("roofline", help="sweep one real layer matrix from 1 token upward")
    commands.add_parser("decode", help="time prefill and decode on Qwen2.5-0.5B")
    commands.add_parser("kv", help="write the KV cache table from published configs")
    commands.add_parser("specs", help="write the cited hardware spec table")
    commands.add_parser("report", help="write out/report.md from the JSON files")
    commands.add_parser("all", help="probe, roofline, decode, kv, specs, and report")
    return parser


def main(argv: list[str] | None = None) -> None:
    prepare()
    args = build_parser().parse_args(argv)
    if args.command == "probe":
        probe_command(args.quick)
    elif args.command == "roofline":
        roofline_command(args.quick)
    elif args.command == "decode":
        decode_command(args.quick)
    elif args.command == "kv":
        kv_command(args.quick)
    elif args.command == "specs":
        specs_command()
    elif args.command == "report":
        report_command()
    else:
        all_command(args.quick)
