"""Extract the published Phase H abundance sheet to the historical TSV schema."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table


import argparse
import csv
import gzip
import hashlib
from itertools import zip_longest
from pathlib import Path
from tempfile import NamedTemporaryFile

from openpyxl import load_workbook


EXPECTED_SHA256 = "e289aae3279394a326d35ea31db5c5eaa945fdbd68b8e30d187d70a18b39aeac"
SHEET = "b. 9,300 proteins"
HEADER = ("UniProt ID", "Gene symbol") + tuple(
    f"{patient}{tissue}" for patient in range(1, 125) for tissue in ("N", "T")
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_tables(left, right):
    with (
        gzip.open(left, "rt", encoding="utf-8-sig", newline="") as left_file,
        gzip.open(right, "rt", encoding="utf-8-sig", newline="") as right_file,
    ):
        left_rows = csv.reader(left_file, delimiter="\t")
        right_rows = csv.reader(right_file, delimiter="\t")
        for row_number, (actual, expected) in enumerate(zip_longest(left_rows, right_rows), 1):
            if actual is None or expected is None or len(actual) != len(expected):
                raise ValueError(f"Proteomics reference dimensions differ on TSV line {row_number}")
            if actual[:2] != expected[:2] or (row_number == 1 and actual != expected):
                raise ValueError(f"Proteomics reference identifiers differ on TSV line {row_number}")
            if row_number == 1:
                continue
            for column, (x, y) in enumerate(zip(actual[2:], expected[2:]), 3):
                if x == y:
                    continue
                if not x or not y or float(x) != float(y):
                    raise ValueError(f"Proteomics reference value differs on TSV line {row_number}, column {column}")


def extract(source, output, reference):
    if sha256(source) != EXPECTED_SHA256:
        raise ValueError("Workbook checksum differs from the analysis input; verify the supplement version")
    output.parent.mkdir(parents=True, exist_ok=True)
    book = load_workbook(source, read_only=True, data_only=True)
    with NamedTemporaryFile(prefix=output.stem, suffix=".tsv", dir=output.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        rows = book[SHEET].values
        if next(rows) != HEADER:
            raise ValueError("Unexpected protein, gene symbol, or paired-sample columns")
        n = 0
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(HEADER)
            for row in rows:
                writer.writerow(row)
                n += 1
        if n != 9300:
            raise ValueError(f"Expected 9,300 protein rows, observed {n}")
        if reference is not None:
            compare_tables(temporary, reference)
        temporary.replace(output)
        print(f"Extracted {n} proteins across 248 samples in 124 complete pairs.")
        if reference is not None:
            print("All identifiers and numeric cells agree with the retained reference.")
    finally:
        book.close()
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare-reference", type=Path, help="Optional retained Phase H TSV.GZ to verify cellwise")
    args = parser.parse_args()
    extract(args.workbook, args.output, args.compare_reference)


if __name__ == "__main__":
    main()
