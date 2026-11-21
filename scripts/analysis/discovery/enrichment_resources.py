#!/usr/bin/env python
"""Download and freeze the exact Phase D gene-set resources."""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table


import hashlib
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = INPUT_ROOT
OUT = PROJECT_ROOT / "signatures" / "enrichment_gene_sets"
LIBRARIES = (
    "MSigDB_Hallmark_2020",
    "Reactome_Pathways_2024",
    "GO_Biological_Process_2026",
    "KEGG_2026",
)
URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={library}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for library in LIBRARIES:
        url = URL.format(library=library)
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        raw_path = OUT / f"{library}.raw.txt"
        raw_path.write_bytes(response.content)
        parsed = []
        for line in response.text.splitlines():
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 3:
                continue
            term = fields[0].strip()
            genes = list(dict.fromkeys(g.strip().upper() for g in fields[2:] if g.strip()))
            if term and genes:
                parsed.append((term, genes))
        if not parsed:
            raise RuntimeError(f"No gene sets parsed from {library}")
        gmt_path = OUT / f"{library}.gmt"
        with gmt_path.open("w", encoding="utf-8", newline="\n") as handle:
            for term, genes in parsed:
                handle.write("\t".join([term, url, *genes]) + "\n")
        sizes = [len(genes) for _, genes in parsed]
        rows.append({
            "library": library, "source": "Enrichr geneSetLibrary text API", "source_url": url,
            "access_date": "2026-08-26", "raw_file": str(raw_path.resolve()),
            "raw_sha256": sha256(raw_path), "gmt_file": str(gmt_path.resolve()),
            "gmt_sha256": sha256(gmt_path), "gene_sets": len(parsed),
            "min_genes": min(sizes), "median_genes": float(pd.Series(sizes).median()), "max_genes": max(sizes),
        })
    pd.DataFrame(rows).to_csv(OUT / "gene_set_resource_manifest.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
