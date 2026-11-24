"""Command line entry point: ``passk generate --config <yaml> [key=value ...]``."""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="passk", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("generate", help="sample N attempts per problem and grade them -> counts.csv")
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*", help="key=value overrides, e.g. n_samples=16 limit=10")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    from .config import GenerateConfig, load_config
    from .generation.run import run_generation

    print(run_generation(load_config(a.config, GenerateConfig, a.overrides)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
