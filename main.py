from __future__ import annotations

import argparse
from pathlib import Path

from config import SimulationConfig
from simulation.experiment import (
    build_new_engine,
    build_resumed_engine,
    run_continuous,
    resume,
    start_new,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LifeSim v0.1 artificial-life laboratory")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--new", action="store_true", help="Start a new experiment")
    mode.add_argument("--resume", type=Path, help="Resume brains from a run checkpoint directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, help="Override NUM_TICKS for this run")
    parser.add_argument("--status-every", type=int, help="Console status interval")
    parser.add_argument("--render-every", type=int, default=0, help="Print ASCII world every N ticks")
    parser.add_argument("--debug", action="store_true", help="Print every reward decomposition")
    parser.add_argument("--web", action="store_true", help="Open the local web laboratory mode")
    parser.add_argument(
        "--continuous", action="store_true",
        help="Chain runs using each previous checkpoint until Ctrl+C",
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help="Compact console output without ASCII rendering or PNG generation",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Web server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Web server port (default: 8765)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig()
    if args.ticks is not None:
        if args.ticks <= 0:
            raise SystemExit("--ticks must be positive")
        config.num_ticks = args.ticks
    if args.status_every is not None:
        if args.status_every <= 0:
            raise SystemExit("--status-every must be positive")
        config.status_every = args.status_every
    config.render_every = args.render_every
    config.debug_rewards = args.debug
    if args.text_only:
        config.compact_console = True
        config.generate_plots = False
        config.render_every = 0
    root = Path(__file__).resolve().parent
    if args.web and args.continuous:
        raise SystemExit("--web and --continuous cannot be used together")
    if args.web:
        if not 1 <= args.port <= 65_535:
            raise SystemExit("--port must be between 1 and 65535")
        from web.server import serve_web

        engine = (
            build_new_engine(root, config, args.seed)
            if args.new
            else build_resumed_engine(root, args.resume, config, args.seed)
        )
        serve_web(engine, args.host, args.port, root)
        return
    if args.continuous:
        config.compact_console = True
        run_continuous(
            root,
            config,
            args.seed,
            checkpoint_dir=None if args.new else args.resume,
        )
        return
    if args.new:
        start_new(root, config, args.seed)
    else:
        resume(root, args.resume, config, args.seed)


if __name__ == "__main__":
    main()
