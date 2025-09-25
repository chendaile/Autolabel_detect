"""Backwards compatible wrapper around :mod:`autolabel_toolkit.cli`."""

from __future__ import annotations

import sys

from autolabel_toolkit.cli import main as _cli_main


def main() -> None:
    """Entry point mirroring the historical CLI behaviour."""

    argv = ["detect", *sys.argv[1:]]
    _cli_main(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
