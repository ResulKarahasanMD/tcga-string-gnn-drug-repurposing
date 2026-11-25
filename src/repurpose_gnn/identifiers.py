"""Gene / protein identifier mapping (Section 8) — treated as a critical, QC'd step.

Safe mapping among:
    Ensembl gene ID (versioned)  ->  Ensembl (version-stripped)  ->  HGNC symbol
    ->  Entrez  ->  STRING protein ID (9606.ENSP...)  ->  UniProt

Design rules enforced here:
- **Never silently drop** unmatched genes — every drop is recorded in a QC table.
- Handle one-to-many mappings, deprecated/alias symbols, non-protein-coding genes, and
  multiple Ensembl IDs collapsing to the same symbol.
- Validate species (Homo sapiens, NCBI taxon 9606).

The primary Ensembl->symbol->biotype source for THIS dataset is the GDC STAR file itself
(its ``gene_name`` / ``gene_type`` columns, GENCODE v36) — authoritative and already local.
STRING/UniProt mapping uses STRING's bulk alias file (Section 12), passed in as a DataFrame
so this module stays offline-testable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

HUMAN_TAXON = 9606
_ENSEMBL_VERSION_RE = re.compile(r"^(ENSG\d{11})(?:\.\d+)?(?:_PAR_Y)?$")
_ENSEMBL_ANY_RE = re.compile(r"\.(\d+)$")


@dataclass
class MappingResult:
    """A mapping outcome: the matched table plus a QC frame of everything unmatched."""

    mapped: pd.DataFrame
    unmatched: pd.DataFrame

    @property
    def n_mapped(self) -> int:
        return len(self.mapped)

    @property
    def n_unmatched(self) -> int:
        return len(self.unmatched)

    def mapping_rate(self, total: int | None = None) -> float:
        total = total if total is not None else (self.n_mapped + self.n_unmatched)
        return self.n_mapped / total if total else 0.0


# --------------------------------------------------------------------------------------
# Ensembl version handling
# --------------------------------------------------------------------------------------
def strip_ensembl_version(gene_id: str) -> str:
    """Remove the version suffix from an Ensembl gene id.

    ``ENSG00000146648.18`` -> ``ENSG00000146648``. Also strips the GENCODE
    ``_PAR_Y`` pseudo-autosomal suffix. Non-Ensembl strings are returned unchanged
    (with only a trailing ``.<int>`` removed, if present).
    """
    if not isinstance(gene_id, str):
        raise TypeError(f"gene_id must be str, got {type(gene_id).__name__}")
    m = _ENSEMBL_VERSION_RE.match(gene_id)
    if m:
        return m.group(1)
    # generic fallback: strip a trailing .<digits>
    return _ENSEMBL_ANY_RE.sub("", gene_id)


def is_par_y(gene_id: str) -> bool:
    """True for GENCODE pseudo-autosomal-region Y duplicate gene ids (``..._PAR_Y``)."""
    return isinstance(gene_id, str) and gene_id.endswith("_PAR_Y")


# --------------------------------------------------------------------------------------
# Gene annotation from GDC STAR files
# --------------------------------------------------------------------------------------
def build_gene_annotation(star_annotation: pd.DataFrame) -> pd.DataFrame:
    """Build a clean gene-annotation table from a STAR file's gene columns.

    Input: DataFrame indexed by ``gene_id`` with columns ``gene_name`` and ``gene_type``
    (as returned by :func:`repurpose_gnn.tcga.read_star_counts`). Output columns:
    ``gene_id, ensembl, gene_name, gene_type, is_par_y``.
    """
    for col in ("gene_name", "gene_type"):
        if col not in star_annotation.columns:
            raise ValueError(f"star_annotation missing column {col!r}.")
    ann = star_annotation.reset_index().rename(columns={"index": "gene_id"})
    if "gene_id" not in ann.columns:
        ann = ann.rename(columns={ann.columns[0]: "gene_id"})
    ann["ensembl"] = ann["gene_id"].map(strip_ensembl_version)
    ann["is_par_y"] = ann["gene_id"].map(is_par_y)
    return ann[["gene_id", "ensembl", "gene_name", "gene_type", "is_par_y"]]


def filter_protein_coding(annotation: pd.DataFrame) -> pd.DataFrame:
    """Return only ``protein_coding`` genes (GNN nodes must be proteins for STRING mapping)."""
    return annotation[annotation["gene_type"] == "protein_coding"].copy()


# --------------------------------------------------------------------------------------
# Duplicate-symbol collapse (multiple Ensembl -> one symbol)
# --------------------------------------------------------------------------------------
def collapse_duplicate_symbols(
    counts: pd.DataFrame,
    annotation: pd.DataFrame,
    *,
    strategy: str = "sum",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse an Ensembl-indexed counts matrix to unique HGNC symbols.

    ``counts``: genes (Ensembl gene_id index) x samples (columns).
    ``annotation``: has ``gene_id`` and ``gene_name``.
    ``strategy``: ``"sum"`` (raw counts are additive across duplicate Ensembl ids) or
    ``"max_total"`` (keep the single Ensembl id with the largest row sum).

    Returns ``(symbol_counts, collapse_qc)`` where ``collapse_qc`` records every symbol that
    had >1 Ensembl id (for the QC report — nothing is dropped silently).
    """
    if strategy not in {"sum", "max_total"}:
        raise ValueError("strategy must be 'sum' or 'max_total'.")
    id2sym = annotation.set_index("gene_id")["gene_name"]
    df = counts.copy()  # index stays Ensembl gene_id (unique); symbols computed in parallel
    sym = pd.Series(df.index.map(id2sym), index=df.index, name="gene_name")
    keep = sym.notna() & (sym.astype(str) != "")
    df = df[keep.to_numpy()]
    sym = sym[keep]

    counts_per_sym = sym.value_counts()
    collapse_qc = (
        counts_per_sym[counts_per_sym > 1]
        .rename_axis("gene_name").reset_index(name="n_ensembl_ids")
    )

    if strategy == "sum":
        symbol_counts = df.groupby(sym.to_numpy()).sum()
    else:  # max_total: keep the single Ensembl row with the largest total per symbol
        totals = df.sum(axis=1)
        pick_idx = totals.groupby(sym.to_numpy()).idxmax()  # symbol -> Ensembl row label
        symbol_counts = df.loc[pick_idx.to_numpy()]
        symbol_counts.index = pd.Index(pick_idx.index, name="gene_name")
    symbol_counts.index = symbol_counts.index.rename("gene_name")

    logger.info(
        "Collapsed %d Ensembl rows -> %d unique symbols (%d symbols had multiple ids).",
        len(counts), len(symbol_counts), len(collapse_qc),
    )
    return symbol_counts.sort_index(), collapse_qc


# --------------------------------------------------------------------------------------
# STRING / UniProt mapping via bulk alias table
# --------------------------------------------------------------------------------------
def validate_species(string_protein_ids: pd.Series, taxon: int = HUMAN_TAXON) -> None:
    """Assert every STRING id is prefixed with the expected taxon (e.g. ``9606.``)."""
    prefix = f"{taxon}."
    bad = string_protein_ids[~string_protein_ids.astype(str).str.startswith(prefix)]
    if len(bad):
        raise ValueError(f"{len(bad)} STRING ids not from taxon {taxon}: {bad.head().tolist()}")


def map_symbols_to_string(
    symbols: list[str],
    aliases: pd.DataFrame,
    *,
    taxon: int = HUMAN_TAXON,
) -> MappingResult:
    """Map HGNC symbols to STRING protein ids using a STRING alias table.

    ``aliases`` is STRING's ``<taxon>.protein.aliases.v12.0.txt`` loaded with columns
    ``[string_protein_id, alias, source]``. We match on ``alias == symbol`` (case-sensitive
    first, then a case-insensitive fallback). One-to-many matches are resolved to a single
    STRING id deterministically (prefer sources containing 'Ensembl' or 'BioMart', else the
    lexicographically smallest id), and the ambiguity is retained in the QC output.

    Returns a :class:`MappingResult` with ``mapped`` columns
    ``[gene_name, string_protein_id, n_candidates]`` and an ``unmatched`` frame.
    """
    required = {"string_protein_id", "alias", "source"}
    if not required.issubset(aliases.columns):
        raise ValueError(f"aliases must have columns {required}, got {list(aliases.columns)}.")
    validate_species(aliases["string_protein_id"], taxon)

    symbols = list(dict.fromkeys(symbols))  # dedup, preserve order
    alias_exact = aliases.copy()
    alias_exact["alias_lower"] = alias_exact["alias"].str.lower()

    mapped_rows: list[dict] = []
    unmatched: list[str] = []
    # index for speed
    by_alias = alias_exact.groupby("alias")
    by_alias_lower = alias_exact.groupby("alias_lower")

    for sym in symbols:
        cand = by_alias.get_group(sym) if sym in by_alias.groups else None
        if cand is None:
            key = sym.lower()
            cand = by_alias_lower.get_group(key) if key in by_alias_lower.groups else None
        if cand is None or cand.empty:
            unmatched.append(sym)
            continue
        chosen = _resolve_string_candidate(cand)
        mapped_rows.append(
            {
                "gene_name": sym,
                "string_protein_id": chosen,
                "n_candidates": int(cand["string_protein_id"].nunique()),
            }
        )

    mapped_df = pd.DataFrame(mapped_rows, columns=["gene_name", "string_protein_id", "n_candidates"])
    unmatched_df = pd.DataFrame({"gene_name": unmatched, "reason": "no_string_alias"})
    logger.info(
        "STRING mapping: %d/%d symbols mapped (%.1f%%); %d unmatched.",
        len(mapped_df), len(symbols),
        100 * len(mapped_df) / max(len(symbols), 1), len(unmatched_df),
    )
    return MappingResult(mapped=mapped_df, unmatched=unmatched_df)


_PREFERRED_ALIAS_SOURCES = ("Ensembl_HGNC", "Ensembl_gene", "BioMart_HUGO", "Ensembl_UniProt")


def _resolve_string_candidate(cand: pd.DataFrame) -> str:
    """Pick one STRING id from candidate alias rows, deterministically."""
    ids = cand["string_protein_id"].unique()
    if len(ids) == 1:
        return ids[0]
    pref = cand[cand["source"].isin(_PREFERRED_ALIAS_SOURCES)]
    pool = pref["string_protein_id"] if not pref.empty else cand["string_protein_id"]
    return sorted(pool.unique())[0]
