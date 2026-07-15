#!/usr/bin/env python3
"""Run a complete continental biomass workflow with one shared run ID."""

from __future__ import annotations

import argparse
import subprocess

from biomass.continental.workflows import DEFAULT_SCRIPTS, VALID_WORKFLOWS, run_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run continental R biomass scripts with a shared timestamped output "
            "directory."
        )
    )
    parser.add_argument(
        "--workflow",
        choices=VALID_WORKFLOWS,
        default="modified_magnabosco",
        help="Continental workflow to run.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional shared output name under runs/continental/.",
    )
    parser.add_argument(
        "--script",
        action="append",
        choices=sorted({script for scripts in DEFAULT_SCRIPTS.values() for script in scripts}),
        help="Run only this R script; repeat to select multiple scripts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.script:
        invalid_scripts = sorted(set(args.script).difference(DEFAULT_SCRIPTS[args.workflow]))
        if invalid_scripts:
            raise SystemExit(
                f"Selected scripts do not belong to {args.workflow}: "
                f"{', '.join(invalid_scripts)}"
            )
    results = run_workflow(
        workflow=args.workflow,
        scripts=args.script,
        run_id=args.run_id,
    )
    failures = [
        script_name
        for script_name, result in results.items()
        if isinstance(result, subprocess.CalledProcessError)
    ]
    if failures:
        raise SystemExit(f"Continental workflow failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
