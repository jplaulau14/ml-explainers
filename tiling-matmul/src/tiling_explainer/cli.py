import argparse
import json
import platform
import sys

from tiling_explainer.contract import MIN_RUNS
from tiling_explainer.driver import run_mac
from tiling_explainer.harness import run_process

MAX_CALLS = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tiling-explainer")
    commands = parser.add_subparsers(dest="command", required=True)
    mac = commands.add_parser("mac", help="time every Metal kernel in separate processes")
    mac.add_argument("--n", type=int, choices=(4096, 2048), default=4096)
    mac.add_argument("--processes", type=int, default=MIN_RUNS)
    mac.add_argument(
        "--allow-battery",
        action="store_true",
        help="test run: skip the power checks and write to runs/ instead of out/",
    )
    a10 = commands.add_parser("a10", help="time and profile the CUDA kernels on a Modal A10")
    a10.add_argument("--n", type=int, choices=(4096, 2048, 1024), default=4096)
    a10.add_argument("--calls", type=int, default=MIN_RUNS)
    process = commands.add_parser("process", help="one measurement process, printed as JSON")
    process.add_argument("--n", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "process":
        print(json.dumps(run_process(args.n)))
        return
    if args.command == "a10":
        if not MIN_RUNS <= args.calls <= MAX_CALLS:
            sys.exit(f"--calls must be from {MIN_RUNS} to {MAX_CALLS}")
        from tiling_explainer.a10.run import run_a10

        sys.exit(run_a10(args.n, args.calls))
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        sys.exit("the mac command needs an Apple silicon Mac")
    if args.processes < MIN_RUNS:
        sys.exit(f"--processes must be at least {MIN_RUNS}")
    sys.exit(run_mac(args.n, args.processes, args.allow_battery))


if __name__ == "__main__":
    main()
