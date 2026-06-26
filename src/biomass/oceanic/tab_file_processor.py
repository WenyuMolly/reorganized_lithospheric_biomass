#!/usr/bin/env python3
"""
tab_file_processor.py

Safe processor for PANGAEA TAB files containing oceanic cell-abundance data.

Key design choices:
1. Read only the explicit tabular data section beginning with "Sample label".
2. Do NOT extract numbers from arbitrary metadata/header lines.
3. Parse depth and cell abundance only from named columns such as:
   - Depth sed [m]
   - Bac [#/cm**3]
4. Preserve below-detection / below-quantification rows as cells_cm3 = 0,
   while retaining the original comment for traceability.
5. Write one corrected merged CSV and, optionally, one CSV per input TAB file.

Example:
    python tab_file_processor.py \
        --input-dir data/raw/pangaea_exp357 \
        --output data/processed/pangaea_exp357_cell_abundance_merged_corrected.csv \
        --write-individual

Or with explicit files:
    python tab_file_processor.py \
        357-M0068B_cell_abund.tab 357-M0069A_cell_abund.tab 357-M0072B_cell_abund.tab \
        --output data/processed/pangaea_exp357_cell_abundance_merged_corrected.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Optional


PANGAEA_TABLE_START = "Sample label"

# Strict numeric pattern: accepts values only when the whole field is numeric,
# optionally preceded by <, >, or ~.
STRICT_NUMERIC_RE = re.compile(
    r"^\s*[<>~]?\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$"
)

HOLE_RE = re.compile(r"^357-(M\d{4}[A-Z])-(.+)$")


OUTPUT_COLUMNS = [
    "source_file",
    "sample_label",
    "hole",
    "sample_description",
    "area",
    "depth_m",
    "cells_cm3",
    "original_bac_value",
    "comment",
]


def parse_strict_numeric(value: object) -> Optional[float]:
    """
    Parse a numeric field safely.

    This function intentionally rejects strings such as:
        "357-M0068B"
        "M0068B"
        "12R-1"
        "PANGAEA.867604"

    because those are identifiers or metadata, not numeric measurements.
    """
    if value is None:
        return None

    text = str(value).strip()
    if text == "":
        return None

    if not STRICT_NUMERIC_RE.fullmatch(text):
        return None

    cleaned = text.replace("<", "").replace(">", "").replace("~", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def find_table_header_line(lines: list[str]) -> int:
    """
    Return the zero-based line index of the actual PANGAEA data-table header.

    The expected table header begins with a field named "Sample label".
    Metadata lines above this table are deliberately ignored.
    """
    for i, line in enumerate(lines):
        first_field = line.split("\t", 1)[0].strip()
        if first_field == PANGAEA_TABLE_START:
            return i

    raise ValueError(
        'Could not find a PANGAEA measurement table header beginning with "Sample label".'
    )


def get_column(record: dict[str, str], candidate_names: Iterable[str]) -> str:
    """
    Retrieve a field by exact name first, then by case-insensitive match.

    This keeps parsing column-based and avoids arbitrary free-text extraction.
    """
    for name in candidate_names:
        if name in record:
            return record[name]

    lower_map = {k.lower().strip(): k for k in record}
    for name in candidate_names:
        key = lower_map.get(name.lower().strip())
        if key is not None:
            return record[key]

    return ""


def parse_sample_label(sample_label: str) -> tuple[str, str]:
    """
    Extract hole ID and sample description from labels such as:
        357-M0068B-1R-1

    Returns:
        ("M0068B", "1R-1")
    """
    sample_label = sample_label.strip()
    match = HOLE_RE.match(sample_label)
    if match:
        return match.group(1), match.group(2)

    # Fallback for non-Expedition-357 labels: keep the full label as hole.
    # This fallback does not extract numeric measurements.
    return sample_label, ""


def cell_value_from_bac_and_comment(bac_raw: str, comment: str) -> Optional[float]:
    """
    Convert PANGAEA Bac [#/cm**3] field to cells_cm3.

    If the Bac field is blank but the comment indicates below detection or
    quantification limit, retain the row and set cells_cm3 = 0. This matches
    the convention used in the curated dataset, while preserving the comment.
    """
    bac_raw = (bac_raw or "").strip()
    comment = (comment or "").strip()
    comment_lower = comment.lower()

    if bac_raw:
        return parse_strict_numeric(bac_raw)

    if (
        "below method detection limit" in comment_lower
        or "below method quantification limit" in comment_lower
        or "below detection limit" in comment_lower
        or "below quantification limit" in comment_lower
    ):
        return 0.0

    return None


def parse_pangaea_tab(path: Path) -> list[dict[str, object]]:
    """
    Parse one PANGAEA TAB file and return validated measurement rows.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx = find_table_header_line(lines)

    header = lines[header_idx].split("\t")
    rows: list[dict[str, object]] = []

    for line in lines[header_idx + 1 :]:
        if not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        elif len(parts) > len(header):
            # Preserve the main fields and merge unexpected extra fields into Comment-like text.
            parts = parts[: len(header)]

        record = {header[i].strip(): parts[i].strip() for i in range(len(header))}

        sample_label = get_column(record, ["Sample label"]).strip()
        if not sample_label:
            continue

        depth_raw = get_column(record, ["Depth sed [m]", "Depth [m]", "Depth m", "Depth"])
        bac_raw = get_column(record, ["Bac [#/cm**3]", "Bac", "Cells [#/cm**3]", "Cells cm-3"])
        comment = get_column(record, ["Comment", "Comments"])
        area = get_column(record, ["Area"])

        depth_m = parse_strict_numeric(depth_raw)
        cells_cm3 = cell_value_from_bac_and_comment(bac_raw, comment)

        # Require a real numeric depth and a valid cell abundance value.
        # This prevents sample IDs, citations, DOI values, and other metadata
        # from being converted into measurement rows.
        if depth_m is None or cells_cm3 is None:
            continue

        hole, sample_description = parse_sample_label(sample_label)

        rows.append(
            {
                "source_file": path.name,
                "sample_label": sample_label,
                "hole": hole,
                "sample_description": sample_description,
                "area": area.strip(),
                "depth_m": depth_m,
                "cells_cm3": cells_cm3,
                "original_bac_value": bac_raw.strip(),
                "comment": comment.strip(),
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """
    Write rows to CSV and create the parent directory if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def collect_input_files(args: argparse.Namespace) -> list[Path]:
    """
    Collect TAB files from explicit positional arguments and/or --input-dir.
    """
    paths: list[Path] = []

    if args.files:
        paths.extend(Path(p) for p in args.files)

    if args.input_dir:
        input_dir = Path(args.input_dir)
        paths.extend(sorted(input_dir.glob("*.tab")))

    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(p)

    if not unique_paths:
        raise SystemExit("No input TAB files provided. Use positional files or --input-dir.")

    missing = [str(p) for p in unique_paths if not p.exists()]
    if missing:
        raise SystemExit("Input file(s) not found:\n" + "\n".join(missing))

    return unique_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely convert PANGAEA cell-abundance TAB files to corrected CSV."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Input PANGAEA .tab files. Can be used together with --input-dir.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing PANGAEA .tab files.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/pangaea_cell_abundance_merged_corrected.csv",
        help="Output path for merged corrected CSV.",
    )
    parser.add_argument(
        "--write-individual",
        action="store_true",
        help="Also write one corrected CSV per input TAB file.",
    )
    parser.add_argument(
        "--individual-output-dir",
        default="data/processed/pangaea_converted_csv",
        help="Directory for individual corrected CSV files when --write-individual is used.",
    )
    args = parser.parse_args()

    input_files = collect_input_files(args)

    merged_rows: list[dict[str, object]] = []
    per_file_counts: dict[str, int] = {}

    for path in input_files:
        rows = parse_pangaea_tab(path)
        merged_rows.extend(rows)
        per_file_counts[path.name] = len(rows)

        if args.write_individual:
            individual_path = Path(args.individual_output_dir) / f"{path.stem}.csv"
            write_csv(individual_path, rows)

    output_path = Path(args.output)
    write_csv(output_path, merged_rows)

    print("Processed PANGAEA TAB files safely.")
    print(f"Merged rows written: {len(merged_rows)}")
    print(f"Merged output: {output_path}")

    for name, count in per_file_counts.items():
        print(f"  {name}: {count} rows")


if __name__ == "__main__":
    main()
