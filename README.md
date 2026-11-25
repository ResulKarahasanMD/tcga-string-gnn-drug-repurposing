# tcga-string-gnn-drug-repurposing

Prioritize disease-associated genes by combining multi-layer TCGA-derived gene features with
STRING PPI topology using a Graph Neural Network, then identify existing drugs whose targets are
**unusually close to the resulting disease module** in the molecular interaction network.

**Default cohort:** TCGA-LUAD · **Labels:** Open Targets (open) · **Drugs:** DGIdb (open; DrugBank if you supply the licensed XML)

> ⚠️ **Research / hypothesis-generation only — NOT medical advice.** Network proximity does not
> demonstrate target binding, adequate exposure, cellular activity, or clinical efficacy. Nothing
> here is a treatment recommendation.

---

## 1. Scientific objective & the 8 questions

| Question | Answer (as implemented) |
|---|---|
| What does the GNN solve? | Semi-supervised **node classification / ranking** — score each gene for disease-gene association. |
| Nodes | Human protein-coding genes that are **expressed in the cohort AND STRING-mappable** (not "significant-DE-only" — avoids selection bias). |
| Edges | STRING PPIs ≥ threshold; undirected → bidirectional `edge_index`, `edge_attr = combined_score/1000`. |
| Node features | DE `log2fc`, `-log10(FDR)`, mean/variance log-CPM, Wald effect size (+ optional mutation/CNV). |
| Labels | **Open Targets** curated disease genes (default) — **never** DE results (that would be circular). |
| Output | Per-gene disease-association probability → ranked genes → disease module. |
| When do drugs enter? | **After** the disease module exists — via drug-target → network proximity. |
| A top drug hypothesis | "This drug's targets sit unusually close to the disease module" — a hypothesis, not efficacy. |

Two prioritization paradigms are provided: **(A) supervised/PU** (labels above) and
**(B) unsupervised** Graph-Autoencoder + network propagation (needs no labels).

## 2. Workflow

```
TCGA RNA-seq (GDC) ─▶ differential expression (PyDESeq2)
        │
        ▼
STRING v12 PPI ─▶ PyG graph (features + Open-Targets labels + splits)
        │
        ▼
baselines + GCN/GraphSAGE/GATv2 ─▶ gene ranking + disease module
        │
        ▼
DGIdb / DrugBank targets ─▶ network proximity (degree-matched null) ─▶ mechanism score ─▶ ranked drugs
        │
        ▼
enrichment · controls · ablations · explainability · validation
```

## 3. Data sources & licensing

| Source | Used for | Access / license |
|---|---|---|
| **NCI GDC** (`api.gdc.cancer.gov`) | TCGA RNA-seq + metadata | Open (STAR-Counts, open tier). |
| **STRING v12** (bulk flat files) | PPI network | Free (CC BY 4.0); we download species-9606 bulk files. |
| **Open Targets** (GraphQL) | disease-gene labels (default) | Open (CC0). |
| **DGIdb v5** (GraphQL) | drug-target interactions (default) | Open. |
| **DrugBank** (optional) | drug-target (alternative) | **License-restricted**; free academic account + approval at go.drugbank.com. Never committed. |
| COSMIC CGC / OncoKB / DisGeNET | alternative labels | Registration / token required. |
| Reactome / GO / KEGG (via gseapy/Enrichr) | enrichment | Respect each library's terms (KEGG academic-only). |

## 4. Installation

Python 3.11–3.13 (torch/PyG have **no 3.14 wheels**). `uv` recommended:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"      # torch, torch-geometric, pydeseq2, gseapy, ...
pytest -q                        # 57 tests
```
Conda alternative: `conda env create -f environment.yml`. PyG's compiled extensions
(`torch-scatter`/`torch-sparse`) are **not required** for GCN/GraphSAGE/GATv2.

## 5. CPU / GPU / MPS

Device auto-selects **cuda → mps → cpu**; CUDA is never required. On Apple Silicon it runs on
MPS with an automatic CPU fallback for unimplemented ops (`PYTORCH_ENABLE_MPS_FALLBACK=1`).
Force a device with `--device cpu|cuda|mps`. Reproducibility: seeds set for
python/numpy/torch/cuda; MPS lacks deterministic scatter, so **multi-seed mean ± sd**
(`reproducibility.seeds`) is the reproducibility unit, not a single run.

## 6. Running the pipeline

```bash
python scripts/download_tcga.py       --config configs/luad.yaml   # ~576 samples, md5-verified
python scripts/prepare_expression.py  --config configs/luad.yaml   # DESeq2 tumor vs normal
python scripts/build_string_graph.py  --config configs/luad.yaml   # PyG Data object
python scripts/train_gnn.py           --config configs/luad.yaml   # baselines + GNNs + controls
python scripts/rank_genes.py          --config configs/luad.yaml
python scripts/map_drug_targets.py    --config configs/luad.yaml   # DGIdb (or DrugBank if XML present)
python scripts/rank_drugs.py          --config configs/luad.yaml   # network-proximity ranking
# or the whole chain:
python -m repurpose_gnn.pipeline      --config configs/luad.yaml
```
Fast smoke run: add `--set gdc.max_samples_per_group=8`. Override any key with `--set a.b=value`.

**DrugBank:** place `full_database.xml` at `data/raw/drugbank/` and set `DRUGBANK_XML` (or
`drugs.drugbank_xml`); the map-drugs stage then uses DrugBank automatically. If absent, DGIdb is
used and a clear message is logged — the pipeline never fabricates drug data.

## 7. Outputs

`data/processed/<cohort>/`: `filtered_counts.parquet`, `differential_expression_{full,significant}.tsv`,
`sample_qc.tsv`, `graph/{pyg_data.pt, node_table.parquet, string_edges.tsv, graph.graphml, mappings.json}`.
`results/tables/<cohort>/`: `model_metrics{,_summary}.tsv`, `graph_information_controls.tsv`,
`gene_ranking.tsv`, `drug_ranking.tsv` (with `mechanism_score`, `direction_consistency`,
`mechanistic_summary`), `enrichment_*.tsv`, `gene_explanations.tsv`.
`results/figures/<cohort>/`: volcano, MA, PCA, heatmap, degree distribution, model comparison,
ROC/PR, top-20 drugs, three-layer drug-target network.
`results/logs/`: per-stage logs + `manifest_*.json` (config + package versions + input sha256 + seed).

## 8. Tests

`pytest -q` — 57 tests covering Ensembl version stripping, STRING mapping, bidirectional edges,
edge-index bounds, DrugBank XML namespaces, missing/no-target drugs, disconnected shortest paths,
degree-bin matching, zero-std nulls, BH FDR, NaN/inf handling, and reproducible seeds.

## 9. Scientific limitations (read before interpreting)

- PPI networks are incomplete and **literature-biased**; STRING **text-mining** evidence inflates
  scores (sensitivity: `--set string.drop_textmining_channel=true`).
- **Hub bias**: high-degree proteins (e.g. proteasome subunits) look "close" to everything; the
  degree-matched null mitigates but does not eliminate this — see proteasome inhibitors ranking high.
- **Bulk RNA-seq** mixes tumor/stroma/immune cells; TCGA has **few normal samples** (LUAD ≈ 59).
- **Gene expression ≠ protein activity** — e.g. EGFR is oncogenic via mutation/amplification, not
  strong mRNA overexpression, so mRNA-based direction-consistency underrates it. No phosphoproteomics.
- GNNs learn **correlation, not causation**; explanations are attributions, not mechanisms.
- Labels are **incomplete/biased** (few positives; PU problem); random splits leak through the graph
  (transductive) — we report degree-aware splits and graph-shuffle controls.
- **Network proximity ≠ pharmacology**: no dose, exposure, PK, tissue penetration (e.g. BBB),
  off-target toxicity, or cell-type-specific effect is modeled. Signature reversal ≠ clinical benefit.

## 10. Clinical-use warning & data privacy

This is a **research hypothesis-generation tool**. It must not be used for diagnosis or treatment.
Ranked drugs require full preclinical/clinical validation. TCGA open-tier data are de-identified;
controlled-access data (some mutation/clinical) require dbGaP authorization — do not commit raw
patient-level data (`.gitignore` excludes `data/`).

## 11. RMS / bosutinib adaptation

See `src/repurpose_gnn/rms_adaptation.py` and `configs/rms_template.yaml`. **TCGA has no pediatric
RMS cohort and TCGA-SARC is not RMS** — use TARGET-RT / St. Jude Cloud / GEO / SRA, stratify by
subtype (embryonal vs alveolar) and fusion (PAX3::FOXO1 / PAX7::FOXO1), and verify bosutinib targets
per-target (ABL1 ≠ ABL2 ≠ SRC-family). No RMS/bosutinib result is produced without real RMS data.

## 12. References (verified identifiers)

- Love, Huber, Anders. *DESeq2*. Genome Biology 2014;15:550. doi:10.1186/s13059-014-0550-8
- Muzellec et al. *PyDESeq2*. Bioinformatics 2023;btad547. doi:10.1093/bioinformatics/btad547
- Zhu, Ibrahim, Love. *apeGLM*. Bioinformatics 2019;35:2084. doi:10.1093/bioinformatics/bty895
- Szklarczyk et al. *STRING v12*. Nucleic Acids Res 2023;51:D638. doi:10.1093/nar/gkac1000
- Kipf & Welling. *GCN*. ICLR 2017. arXiv:1609.02907
- Hamilton, Ying, Leskovec. *GraphSAGE*. NeurIPS 2017. arXiv:1706.02216
- Veličković et al. *GAT*. ICLR 2018. arXiv:1710.10903
- Brody, Alon, Yahav. *GATv2*. ICLR 2022. arXiv:2105.14491
- Ying et al. *GNNExplainer*. NeurIPS 2019. arXiv:1903.03894
- Fey & Lenssen. *PyTorch Geometric*. ICLR-W 2019. arXiv:1903.02428
- Guney, Menche, Vidal, Barabási. *Network-based drug efficacy / proximity*. Nat Commun 2016;7:10331. doi:10.1038/ncomms10331
- Menche et al. *Disease module / interactome*. Science 2015;347:1257601. doi:10.1126/science.1257601
- Grossman et al. *NCI Genomic Data Commons*. N Engl J Med 2016;375:1109. doi:10.1056/NEJMp1607591
- TCGA Research Network. *Lung adenocarcinoma*. Nature 2014;511:543. doi:10.1038/nature13385
- Traag, Waltman, van Eck. *Leiden*. Sci Rep 2019;9:5233. doi:10.1038/s41598-019-41695-z
- Subramanian et al. *L1000 / CMap*. Cell 2017;171:1437. doi:10.1016/j.cell.2017.10.049

(Additional live-API sources: Open Targets Platform, DGIdb, gseapy/Enrichr — see their sites for current citation.)

## Tested versions

The test suite (57 tests) last passed on 2026-11-25 in a fresh environment with Python 3.12, torch 2.15.0, PyTorch Geometric 2.8.1 and pydeseq2 0.7.0. Dependencies are not pinned, so a new install may resolve newer versions.

An end-to-end run on real TCGA and STRING data has **not** been verified with these versions. The report under `results/` comes from an earlier run.

## License

[MIT](LICENSE).
