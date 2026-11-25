"""Pathway / gene-set enrichment (Section 22).

Over-representation analysis via gseapy/Enrichr. Critically, the background is the **analysis
gene universe** (genes that were expressed and mapped into STRING), NOT the whole genome —
using the genome as background inflates significance.

Libraries default to Reactome / GO-BP / KEGG (respecting each source's usage terms). Requires
network access to Enrichr; on failure it reports the error rather than fabricating results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def over_representation(
    gene_list: list[str],
    background: list[str],
    *,
    libraries: list[str] | None = None,
    out_dir: Path | None = None,
    tag: str = "enrichment",
) -> pd.DataFrame:
    """Run Enrichr ORA of ``gene_list`` against a custom ``background``.

    Returns a tidy results frame (Term, library, Overlap, P-value, Adjusted P-value, Genes).
    gseapy runs a local Fisher exact test when a ``background`` list is provided, which is what
    lets us use the analysis-gene background instead of Enrichr's genome default.
    """
    import gseapy as gp

    libraries = libraries or ["Reactome_2022", "GO_Biological_Process_2023", "KEGG_2021_Human"]
    gene_list = sorted({g for g in gene_list if isinstance(g, str) and g})
    background = sorted({g for g in background if isinstance(g, str) and g})
    if len(gene_list) < 3:
        logger.warning("Enrichment skipped: gene_list has <3 valid symbols.")
        return pd.DataFrame()

    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=libraries,
            background=background,
            outdir=None,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - network/library errors reported, not fabricated
        logger.error("Enrichr ORA failed (%s). Check network access / library names.", exc)
        return pd.DataFrame()

    res = enr.results.copy()
    res = res.rename(columns={"Gene_set": "library"})
    res = res.sort_values("Adjusted P-value")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_dir / f"{tag}.tsv", sep="\t", index=False)
        logger.info("Wrote enrichment: %s (%d terms)", out_dir / f"{tag}.tsv", len(res))
    return res


def run_enrichment_stage(cfg) -> dict[str, Any]:  # noqa: ANN001
    """Enrich the up/down DE gene sets and the disease module against the analysis background."""
    cohort = cfg.cohort
    processed = cfg.path("paths.processed") / cohort
    tables = cfg.path("paths.tables") / cohort
    graph_dir = processed / "graph"

    node_table = pd.read_parquet(graph_dir / "node_table.parquet")
    background = node_table["gene_name"].dropna().astype(str).tolist()  # analysis universe

    de = pd.read_csv(processed / "differential_expression_significant.tsv", sep="\t")
    ecfg = cfg.section("enrichment")
    libs = ecfg.get("libraries")

    results: dict[str, int] = {}
    up = de[(de["direction"] == "up")]["gene_name"].dropna().tolist()
    down = de[(de["direction"] == "down")]["gene_name"].dropna().tolist()
    for tag, genes in [("enrichment_DE_up", up), ("enrichment_DE_down", down)]:
        r = over_representation(genes, background, libraries=libs, out_dir=tables, tag=tag)
        results[tag] = int((r["Adjusted P-value"] < 0.05).sum()) if len(r) else 0

    # disease module (seeds + propagated), if the drug stage built one; else use positives
    module_genes = node_table.loc[node_table["label"] == 1, "gene_name"].dropna().tolist()
    r = over_representation(module_genes, background, libraries=libs, out_dir=tables,
                            tag="enrichment_disease_seeds")
    results["enrichment_disease_seeds"] = int((r["Adjusted P-value"] < 0.05).sum()) if len(r) else 0

    logger.info("Enrichment significant-term counts (FDR<0.05): %s", results)
    return {"cohort": cohort, "significant_terms": results,
            "background_size": len(set(background)), "outputs_dir": str(tables)}
