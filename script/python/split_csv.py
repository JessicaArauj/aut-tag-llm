"""Utility script to split large CSV/XLSX files into fixed-size chunks.

Example:
    python -m script.python.split_csv inputs/vigilancia.xlsx --output-dir inputs/chunks --chunk-size 500

CSV inputs generate CSV chunks, while XLS/XLSX inputs generate Excel chunks named
`<original>_part_XXX.xlsx`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def chunk_rows(reader: Iterable[List[str]], chunk_size: int):
    """Yield successive chunks from the CSV reader."""
    chunk: List[List[str]] = []
    for row in reader:
        chunk.append(row)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def split_csv_file(
    input_file: Path,
    output_dir: Path,
    chunk_size: int = 500,
    encoding: str = 'utf-8',
) -> None:
    """Split the CSV into multiple files with the desired chunk size."""
    _ensure_positive_chunk(chunk_size)
    _ensure_input_exists(input_file)
    _ensure_output_dir(output_dir)

    with input_file.open('r', encoding=encoding, newline='') as csv_in:
        reader = csv.reader(csv_in)
        try:
            header = next(reader)
        except StopIteration:
            print('Input CSV is empty; no files were generated.')
            return

        for index, chunk in enumerate(chunk_rows(reader, chunk_size), start=1):
            output_path = output_dir / f'{input_file.stem}_part_{index:03d}.csv'
            with output_path.open('w', encoding=encoding, newline='') as csv_out:
                writer = csv.writer(csv_out)
                writer.writerow(header)
                writer.writerows(chunk)
            print(f'Created {output_path} with {len(chunk)} rows.')


def split_excel_file(
    input_file: Path,
    output_dir: Path,
    chunk_size: int = 500,
) -> None:
    """Split an Excel file into XLSX chunks."""
    _ensure_positive_chunk(chunk_size)
    _ensure_input_exists(input_file)
    _ensure_output_dir(output_dir)

    df = pd.read_excel(input_file)
    if df.empty:
        print('Input spreadsheet is empty; no files were generated.')
        return

    total_rows = len(df)
    chunk_index = 1
    for start in range(0, total_rows, chunk_size):
        end = start + chunk_size
        chunk = df.iloc[start:end]
        output_path = output_dir / f'{input_file.stem}_part_{chunk_index:03d}.xlsx'
        chunk.to_excel(output_path, index=False)
        print(f'Created {output_path} with {len(chunk)} rows.')
        chunk_index += 1


def _ensure_positive_chunk(chunk_size: int) -> None:
    if chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer.')


def _ensure_input_exists(input_file: Path) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f'Input file not found: {input_file}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Split a CSV file into CSV chunks or an XLSX file into XLSX chunks.',
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help='Path to the original CSV/XLSX file.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('inputs'),
        help='Directory where the chunked files will be stored.',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Number of rows (excluding header) per file. Default: 500.',
    )
    parser.add_argument(
        '--encoding',
        default='utf-8',
        help='Encoding used to read/write the CSV. Default: utf-8.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = args.input_file.suffix.lower()
    if suffix == '.csv':
        split_csv_file(
            input_file=args.input_file,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            encoding=args.encoding,
        )
    elif suffix in {'.xlsx', '.xls'}:
        split_excel_file(
            input_file=args.input_file,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
        )
    else:
        raise ValueError(
            f'Unsupported file type "{suffix}". Provide a CSV or XLS/XLSX file.'
        )


if __name__ == '__main__':
    main()
