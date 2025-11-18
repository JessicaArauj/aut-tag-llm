from pathlib import Path

import pandas as pd
import pytest

from script.python import split_csv


def test_chunk_rows_yields_equal_sized_chunks():
    rows = [[str(i)] for i in range(5)]
    chunks = list(split_csv.chunk_rows(rows, chunk_size=2))

    assert chunks == [[['0'], ['1']], [['2'], ['3']], [['4']]]


def test_split_csv_file_creates_chunks(tmp_path):
    input_path = tmp_path / 'dataset.csv'
    input_path.write_text('a,b\n1,2\n3,4\n5,6\n', encoding='utf-8')
    output_dir = tmp_path / 'chunks'

    split_csv.split_csv_file(input_path, output_dir, chunk_size=2)

    created = sorted(output_dir.glob('dataset_part_*.csv'))
    assert len(created) == 2
    assert created[0].read_text(encoding='utf-8').startswith('a,b')


def test_split_excel_file_creates_expected_chunks(tmp_path):
    input_path = tmp_path / 'dataset.xlsx'
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.to_excel(input_path, index=False)
    output_dir = tmp_path / 'chunks'

    split_csv.split_excel_file(input_path, output_dir, chunk_size=2)

    created = sorted(output_dir.glob('dataset_part_*.xlsx'))
    assert len(created) == 2


def test_ensure_positive_chunk_raises_for_invalid():
    with pytest.raises(ValueError):
        split_csv._ensure_positive_chunk(0)
