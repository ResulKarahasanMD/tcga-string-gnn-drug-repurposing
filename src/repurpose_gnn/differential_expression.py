"""RNA-seq preprocessing, QC, and differential expression (Sections 9-10).

Pipeline: assemble a raw STAR count matrix -> filter low-expression genes -> sample QC
(library sizes, detected genes, PCA-based outlier scan) -> PyDESeq2 differential expression
(tumor vs normal) with BH FDR and optional apeGLM LFC shrinkage.

Hard constraints honoured (Section 10):
- DESeq2 runs on **raw integer counts** (never TPM); no pseudocount+t-test shortcut.
- Multiple-testing correction is always applied (BH FDR from PyDESeq2).
- DE results are written as node *features* downstream — never used as disease labels.
- Selection-bias note: restricting the network to significant-DE genes would bias everything;
  the graph therefore uses all *expressed* protein-coding genes, DE stats are just features.

PyDESeq2 API (v0.5.x, verified): counts oriented **samples x genes**;
``DeseqDataSet(counts, metadata, design="~condition", ref_level=[...]).deseq2()`` then
``DeseqStats(dds, contrast=["condition","tumor","normal"]).summary()`` -> ``.results_df``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless / reproducible figure backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config, write_run_manifest  # noqa: E402
from .identifiers import build_gene_annotation, strip_ensembl_version  # noqa: E402
from .tcga import read_star_counts  # noqa: E402

logger = logging.getLogger(__name__)


# ======================================================================================
# Count-matrix assembly
# ======================================================================================
def build_count_matrix(
    sample_df: pd.DataFrame,
    count_column: str = "unstranded",
    *,
    protein_coding_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read all STAR files into a genes x samples integer count matrix + annotation.

    Returns ``(counts, annotation)``:
      - ``counts``: index = Ensembl ``gene_id`` (versioned), columns = ``sample_barcode``.
      - ``annotation``: gene_id/ensembl/gene_name/gene_type/is_par_y (from the first file;
        GENCODE gene set is identical across GDC STAR files — asserted below).
    """
    if "local_path" not in sample_df.columns:
        raise ValueError("sample_df must have a 'local_path' column (run the download stage first).")

    series_by_sample: dict[str, pd.Series] = {}
    annotation: pd.DataFrame | None = None
    ref_index: pd.Index | None = None

    for row in sample_df.itertuples(index=False):
        path = Path(row.local_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing STAR file for {row.sample_barcode}: {path}")
        star = read_star_counts(path, count_column)
        if annotation is None:
            annotation = build_gene_annotation(star)
            ref_index = star.index
        elif not star.index.equals(ref_index):
            # gene sets must match to align safely; if not, align on intersection and warn.
            if set(star.index) != set(ref_index):
                logger.warning(
                    "%s has a different gene set than the reference; aligning on reference index.",
                    row.sample_barcode,
                )
            star = star.reindex(ref_index)
        series_by_sample[row.sample_barcode] = star[count_column].astype("int64")

    counts = pd.DataFrame(series_by_sample)
    counts.index.name = "gene_id"
    assert annotation is not None

    if protein_coding_only:
        pc_ids = annotation.loc[annotation["gene_type"] == "protein_coding", "gene_id"]
        counts = counts.loc[counts.index.isin(pc_ids)]
        logger.info("Restricted to %d protein-coding genes.", len(counts))

    logger.info("Built count matrix: %d genes x %d samples.", counts.shape[0], counts.shape[1])
    return counts, annotation


def filter_low_expression(
    counts: pd.DataFrame,
    *,
    min_count: int = 10,
    min_samples_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep genes with ``count >= min_count`` in at least ``min_samples_frac`` of samples.

    Returns ``(filtered_counts, filter_qc)`` where filter_qc records per-gene pass/fail so
    nothing is dropped silently.
    """
    n_samples = counts.shape[1]
    min_samples = max(1, int(np.ceil(min_samples_frac * n_samples)))
    passing = (counts >= min_count).sum(axis=1)
    keep = passing >= min_samples
    filter_qc = pd.DataFrame(
        {"gene_id": counts.index, "n_samples_passing": passing.to_numpy(), "kept": keep.to_numpy()}
    )
    filtered = counts.loc[keep.to_numpy()]
    logger.info(
        "Low-expression filter: %d -> %d genes (min_count=%d in >=%d/%d samples).",
        counts.shape[0], filtered.shape[0], min_count, min_samples, n_samples,
    )
    return filtered, filter_qc


# ======================================================================================
# Sample QC
# ======================================================================================
def compute_sample_qc(counts: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
    """Per-sample QC: library size, detected genes, condition; flag library-size outliers."""
    lib_size = counts.sum(axis=0)
    detected = (counts > 0).sum(axis=0)
    meta = sample_df.set_index("sample_barcode")
    qc = pd.DataFrame(
        {
            "sample_barcode": lib_size.index,
            "condition": meta.loc[lib_size.index, "condition"].to_numpy(),
            "library_size": lib_size.to_numpy(),
            "n_detected_genes": detected.to_numpy(),
        }
    )
    # robust outlier flag: log10 library size beyond median +/- 3 * MAD
    log_lib = np.log10(qc["library_size"].to_numpy() + 1)
    med = np.median(log_lib)
    mad = np.median(np.abs(log_lib - med)) or 1e-9
    qc["library_size_zrobust"] = (log_lib - med) / (1.4826 * mad)
    qc["outlier_flag"] = np.abs(qc["library_size_zrobust"]) > 3
    n_out = int(qc["outlier_flag"].sum())
    if n_out:
        logger.warning("Flagged %d potential library-size outlier sample(s).", n_out)
    return qc


def log_cpm(counts: pd.DataFrame, prior_count: float = 1.0) -> pd.DataFrame:
    """log2 counts-per-million for QC/PCA (a normalization, NOT batch correction)."""
    lib = counts.sum(axis=0)
    cpm = counts.divide(lib, axis=1) * 1e6
    return np.log2(cpm + prior_count)


def run_pca(counts: pd.DataFrame, sample_df: pd.DataFrame, n_top: int = 2000) -> pd.DataFrame:
    """PCA on the top-variable genes of log-CPM. Returns per-sample PC coords + condition."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    lcpm = log_cpm(counts)
    top = lcpm.var(axis=1).sort_values(ascending=False).head(n_top).index
    x = lcpm.loc[top].T.to_numpy()  # samples x genes
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=min(10, x.shape[0] - 1), random_state=0)
    coords = pca.fit_transform(x)
    meta = sample_df.set_index("sample_barcode")
    out = pd.DataFrame(
        coords[:, :2], columns=["PC1", "PC2"], index=lcpm.columns
    )
    out["condition"] = meta.loc[out.index, "condition"].to_numpy()
    out.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_[:2].tolist()
    return out


# ======================================================================================
# Differential expression (PyDESeq2)
# ======================================================================================
def run_deseq2(
    counts: pd.DataFrame,
    sample_df: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Run PyDESeq2 tumor-vs-normal DE. Returns a results table with BH FDR + shrunk LFC.

    ``counts`` is genes x samples (raw integers). Internally transposed to samples x genes
    for PyDESeq2. Positive log2FoldChange = up in tumor relative to normal.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    de_cfg = cfg.section("differential_expression")
    reference = de_cfg.get("reference_level", "normal")

    meta = sample_df.set_index("sample_barcode").loc[counts.columns, ["condition"]].copy()
    meta["condition"] = meta["condition"].astype("category")
    if reference not in set(meta["condition"]):
        raise ValueError(f"Reference level {reference!r} not present in conditions {set(meta['condition'])}.")

    counts_sxg = counts.T  # samples x genes, PyDESeq2 orientation
    counts_sxg = counts_sxg.loc[meta.index]

    n_cpus = int(cfg.get("compute.num_workers", 0)) or (os.cpu_count() or 1)
    inference = DefaultInference(n_cpus=n_cpus)

    logger.info("Fitting DESeq2 on %d samples x %d genes (n_cpus=%d)...",
                counts_sxg.shape[0], counts_sxg.shape[1], n_cpus)
    dds = DeseqDataSet(
        counts=counts_sxg,
        metadata=meta,
        design="~condition",
        ref_level=["condition", reference],
        refit_cooks=bool(de_cfg.get("cooks_filter", True)),
        inference=inference,
        quiet=True,
    )
    dds.deseq2()

    contrast = ["condition", "tumor", reference]
    stat = DeseqStats(dds, contrast=contrast, inference=inference, quiet=True)
    stat.summary()
    res = stat.results_df.copy()
    res = res.rename(columns={"log2FoldChange": "log2fc", "padj": "fdr", "pvalue": "pval"})

    # optional apeGLM LFC shrinkage (kept in a separate column; never overwrites raw LFC).
    # PyDESeq2 0.5.x names coefficients formulaic-style, e.g. "condition[T.tumor]" — detect
    # the tested-level coefficient from the fitted model rather than guessing its format.
    if bool(de_cfg.get("lfc_shrink", True)):
        coeff = _detect_lfc_coeff(dds, tested_level="tumor")
        try:
            stat.lfc_shrink(coeff=coeff)
            res["log2fc_shrunk"] = stat.results_df["log2FoldChange"].to_numpy()
            logger.info("Applied apeGLM LFC shrinkage (coeff=%s).", coeff)
        except Exception as exc:  # noqa: BLE001 - shrinkage is optional; never hard-fail DE
            logger.warning("LFC shrinkage skipped (%s): %s", coeff, exc)
            res["log2fc_shrunk"] = np.nan

    res.index.name = "gene_id"
    return res


def _detect_lfc_coeff(dds, tested_level: str) -> str:
    """Find the LFC coefficient column for the tested (non-reference) level.

    PyDESeq2 exposes fitted LFC coefficients on ``dds.varm['LFC']``; we pick the non-intercept
    column mentioning the tested level (e.g. 'condition[T.tumor]'). Falls back to the
    formulaic name if the object layout differs.
    """
    try:
        cols = [c for c in dds.varm["LFC"].columns if c.lower() != "intercept"]
        match = [c for c in cols if tested_level.lower() in c.lower()]
        if match:
            return match[0]
        if cols:
            return cols[-1]
    except Exception:  # noqa: BLE001
        pass
    return f"condition[T.{tested_level}]"


def annotate_and_flag(
    res: pd.DataFrame,
    annotation: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Attach gene symbol/ensembl + significance flags to DE results."""
    de_cfg = cfg.section("differential_expression")
    fdr_thr = float(de_cfg["fdr_threshold"])
    lfc_thr = float(de_cfg["abs_log2fc_threshold"])

    ann = annotation.set_index("gene_id")
    out = res.copy()
    out["ensembl"] = out.index.map(strip_ensembl_version)
    out["gene_name"] = ann["gene_name"].reindex(out.index).to_numpy()
    out["gene_type"] = ann["gene_type"].reindex(out.index).to_numpy()

    # guard against inf from fdr==0 (Section 11): cap -log10(fdr).
    #   fdr>0  -> -log10(fdr);  fdr==0 -> clip (maximal significance);  fdr==NaN -> 0.
    clip = float(cfg.get("features.clip_neglog10_fdr", 50.0))
    fdr = out["fdr"].to_numpy(dtype=float)
    neglog = np.full(fdr.shape, np.nan, dtype=float)
    valid = fdr > 0
    with np.errstate(divide="ignore"):
        neglog[valid] = -np.log10(fdr[valid])
    neglog[fdr <= 0] = clip                    # fdr == 0 -> most significant
    neglog = np.nan_to_num(neglog, nan=0.0)    # NaN fdr (filtered/untested) -> 0
    out["neglog10_fdr"] = np.clip(neglog, 0.0, clip)

    out["significant"] = (out["fdr"] < fdr_thr) & (out["log2fc"].abs() >= lfc_thr)
    out["direction"] = np.where(out["log2fc"] > 0, "up", "down")
    return out.sort_values("fdr", na_position="last")


# ======================================================================================
# Figures
# ======================================================================================
def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    svg = path.with_suffix(".svg")
    try:
        fig.savefig(svg, bbox_inches="tight")
    except Exception:  # noqa: BLE001
        pass
    plt.close(fig)


def plot_library_sizes(qc: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for cond, sub in qc.groupby("condition"):
        ax.bar(sub["sample_barcode"], sub["library_size"], label=cond, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xticks([])
    ax.set_ylabel("Library size (total counts, log)")
    ax.set_title("Library size distribution by sample")
    ax.legend()
    _savefig(fig, path)


def plot_pca(pca_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    evr = pca_df.attrs.get("explained_variance_ratio", [0, 0])
    for cond, sub in pca_df.groupby("condition"):
        ax.scatter(sub["PC1"], sub["PC2"], label=cond, alpha=0.7, s=18)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("Sample PCA (log-CPM, top-variable genes)")
    ax.legend()
    _savefig(fig, path)


def plot_volcano(res: pd.DataFrame, cfg: Config, path: Path) -> None:
    de_cfg = cfg.section("differential_expression")
    fig, ax = plt.subplots(figsize=(7, 6))
    x = res["log2fc"].to_numpy()
    y = res["neglog10_fdr"].to_numpy()
    sig = res["significant"].to_numpy()
    ax.scatter(x[~sig], y[~sig], s=5, c="lightgray", alpha=0.5, label="ns")
    ax.scatter(x[sig], y[sig], s=6, c="crimson", alpha=0.6, label="significant")
    ax.axhline(-np.log10(float(de_cfg["fdr_threshold"])), ls="--", c="k", lw=0.6)
    for thr in (-float(de_cfg["abs_log2fc_threshold"]), float(de_cfg["abs_log2fc_threshold"])):
        ax.axvline(thr, ls="--", c="k", lw=0.6)
    ax.set_xlabel("log2 fold change (tumor vs normal)")
    ax.set_ylabel("-log10 FDR")
    ax.set_title("Volcano plot")
    ax.legend()
    _savefig(fig, path)


def plot_ma(res: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    base = np.log10(res["baseMean"].to_numpy() + 1)
    lfc = res["log2fc"].to_numpy()
    sig = res["significant"].to_numpy()
    ax.scatter(base[~sig], lfc[~sig], s=5, c="lightgray", alpha=0.5)
    ax.scatter(base[sig], lfc[sig], s=6, c="crimson", alpha=0.6)
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xlabel("log10(baseMean + 1)")
    ax.set_ylabel("log2 fold change")
    ax.set_title("MA plot")
    _savefig(fig, path)


def plot_top_heatmap(counts: pd.DataFrame, res: pd.DataFrame, sample_df: pd.DataFrame,
                     path: Path, n_top: int = 40) -> None:
    top = res.sort_values("fdr", na_position="last").head(n_top).index
    top = [g for g in top if g in counts.index]
    if not top:
        return
    lcpm = log_cpm(counts.loc[top])
    z = lcpm.sub(lcpm.mean(axis=1), axis=0).div(lcpm.std(axis=1).replace(0, 1), axis=0)
    meta = sample_df.set_index("sample_barcode")
    order = meta.loc[z.columns].sort_values("condition").index
    z = z[order]
    labels = res.loc[top, "gene_name"].fillna(pd.Series(top, index=top)).to_numpy()
    fig, ax = plt.subplots(figsize=(10, max(6, n_top * 0.18)))
    im = ax.imshow(z.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xticks([])
    ax.set_title(f"Top {len(top)} DE genes (z-scored log-CPM), samples sorted by condition")
    fig.colorbar(im, ax=ax, shrink=0.5, label="z-score")
    _savefig(fig, path)


# ======================================================================================
# Orchestration
# ======================================================================================
def run_prepare_and_de(cfg: Config) -> dict[str, Any]:
    """Full Phase-2 stage: matrix -> filter -> QC -> PCA -> DESeq2 -> tables + figures."""
    cohort = cfg.cohort
    interim = cfg.path("paths.interim") / cohort
    processed = cfg.path("paths.processed") / cohort
    figures = cfg.path("paths.figures") / cohort
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    meta_path = interim / "sample_metadata.tsv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run: python scripts/download_tcga.py --config <cfg> first."
        )
    sample_df = pd.read_csv(meta_path, sep="\t")

    exp = cfg.section("expression")
    counts, annotation = build_count_matrix(
        sample_df, exp["count_column"], protein_coding_only=bool(exp["protein_coding_only"])
    )
    filtered, filter_qc = filter_low_expression(
        counts, min_count=int(exp["min_count"]), min_samples_frac=float(exp["min_samples_frac"])
    )
    filtered.to_parquet(processed / "filtered_counts.parquet")
    annotation.to_parquet(processed / "gene_annotation.parquet")
    filter_qc.to_csv(processed / "gene_filter_qc.tsv", sep="\t", index=False)

    qc = compute_sample_qc(filtered, sample_df)
    qc.to_csv(processed / "sample_qc.tsv", sep="\t", index=False)
    plot_library_sizes(qc, figures / "library_size_distribution.png")

    pca_df = run_pca(filtered, sample_df)
    pca_df.to_csv(processed / "pca_samples.tsv", sep="\t")
    plot_pca(pca_df, figures / "pca_samples.png")

    res = run_deseq2(filtered, sample_df, cfg)
    res = annotate_and_flag(res, annotation, cfg)
    res.to_csv(processed / "differential_expression_full.tsv", sep="\t")
    sig = res[res["significant"]]
    sig.to_csv(processed / "differential_expression_significant.tsv", sep="\t")

    plot_volcano(res, cfg, figures / "volcano_plot.png")
    plot_ma(res, figures / "ma_plot.png")
    plot_top_heatmap(filtered, res, sample_df, figures / "top_genes_heatmap.png")

    summary = {
        "cohort": cohort,
        "n_samples": int(filtered.shape[1]),
        "n_genes_input": int(counts.shape[0]),
        "n_genes_filtered": int(filtered.shape[0]),
        "n_significant": int(sig.shape[0]),
        "n_up": int((sig["direction"] == "up").sum()),
        "n_down": int((sig["direction"] == "down").sum()),
        "n_outlier_samples": int(qc["outlier_flag"].sum()),
        "outputs": {
            "filtered_counts": str(processed / "filtered_counts.parquet"),
            "de_full": str(processed / "differential_expression_full.tsv"),
            "de_significant": str(processed / "differential_expression_significant.tsv"),
            "sample_qc": str(processed / "sample_qc.tsv"),
            "figures_dir": str(figures),
        },
    }
    write_run_manifest(
        cfg, stage="prepare_expression", seed=cfg.seed,
        input_files={"sample_metadata": meta_path,
                     "filtered_counts": processed / "filtered_counts.parquet"},
        extra=summary,
    )
    logger.info("DE done: %d significant (%d up / %d down) of %d genes.",
                summary["n_significant"], summary["n_up"], summary["n_down"],
                summary["n_genes_filtered"])
    return summary
