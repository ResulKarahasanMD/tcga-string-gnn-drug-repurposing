"""TCGA / GDC download and sample-metadata module (Section 7).

Retrieves STAR-Counts "Gene Expression Quantification" RNA-seq files for a TCGA project
from the NCI Genomic Data Commons (GDC) REST API, verifies download integrity by md5,
parses TCGA barcodes, assembles a sample-metadata table, and prevents participant-level
data leakage by de-duplicating multiple samples from the same participant/condition.

Primary sources (verified 2026-07):
- GDC API base ``https://api.gdc.cancer.gov`` with ``/files`` (search) and ``/data`` (download).
- Filters are JSON ``{"op","content":{"field","value"}}`` trees; POST is used for large queries.
- RNA-seq counts: ``data_type="Gene Expression Quantification"`` +
  ``analysis.workflow_type="STAR - Counts"`` (the exact enum is validated at runtime via
  the ``/files`` facet buckets rather than trusted blindly).

Docs: https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Config, write_run_manifest

logger = logging.getLogger(__name__)

# GDC endpoints (do not hardcode-guess deprecated paths; these are the current documented ones).
FILES_ENDPOINT = "/files"
DATA_ENDPOINT = "/data"

# TCGA sample-type code -> condition. 01-09 = tumor; 10-19 = normal; else other/control.
# https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/sample-type-codes
_TUMOR_CODES = {f"{i:02d}" for i in range(1, 10)}
_NORMAL_CODES = {f"{i:02d}" for i in range(10, 20)}


# ======================================================================================
# TCGA barcode parsing (Section 7)
# ======================================================================================
@dataclass(frozen=True)
class TCGABarcode:
    """Parsed TCGA barcode.

    A *sample* barcode has 4 hyphen fields (``TCGA-05-4384-01A``); a full *aliquot* barcode
    has up to 7 (``TCGA-05-4384-01A-01D-0182-01``). We reliably parse the first four fields
    (project / tissue-source-site / participant / sample+vial) — enough for tumor-vs-normal
    labelling and participant-level de-duplication — and best-effort the rest.
    """

    barcode: str
    project: str
    tss: str
    participant_id: str
    sample_type_code: str
    vial: str | None = None
    portion: str | None = None
    analyte: str | None = None
    plate: str | None = None
    center: str | None = None

    @property
    def participant_barcode(self) -> str:
        """``TCGA-05-4384`` — identifies the individual; use for leakage-preventing dedup."""
        return f"{self.project}-{self.tss}-{self.participant_id}"

    @property
    def sample_barcode(self) -> str:
        """``TCGA-05-4384-01A`` — participant + sample-type + vial."""
        v = self.vial or ""
        return f"{self.participant_barcode}-{self.sample_type_code}{v}"

    @property
    def is_tumor(self) -> bool:
        return self.sample_type_code in _TUMOR_CODES

    @property
    def is_normal(self) -> bool:
        return self.sample_type_code in _NORMAL_CODES

    @property
    def condition(self) -> str:
        if self.is_tumor:
            return "tumor"
        if self.is_normal:
            return "normal"
        return "other"


def parse_tcga_barcode(barcode: str) -> TCGABarcode:
    """Parse a TCGA sample or aliquot barcode.

    Raises :class:`ValueError` for anything that is not at least a 4-field TCGA barcode
    (``TCGA-<TSS>-<participant>-<sampletype+vial>``).
    """
    if not isinstance(barcode, str) or not barcode:
        raise ValueError(f"Empty/invalid barcode: {barcode!r}")
    parts = barcode.strip().split("-")
    if len(parts) < 4:
        raise ValueError(
            f"Barcode {barcode!r} has {len(parts)} fields; need >=4 (TCGA-TSS-participant-sampleVial)."
        )
    if parts[0].upper() != "TCGA":
        raise ValueError(f"Barcode {barcode!r} does not start with 'TCGA'.")

    sample_vial = parts[3]
    if len(sample_vial) < 2 or not sample_vial[:2].isdigit():
        raise ValueError(f"Barcode {barcode!r}: cannot parse sample-type code from {sample_vial!r}.")
    sample_type_code = sample_vial[:2]
    vial = sample_vial[2:] or None

    portion = analyte = plate = center = None
    if len(parts) >= 5 and parts[4]:
        portion = parts[4][:2]
        analyte = parts[4][2:] or None
    if len(parts) >= 6:
        plate = parts[5]
    if len(parts) >= 7:
        center = parts[6]

    return TCGABarcode(
        barcode=barcode,
        project=parts[0].upper(),
        tss=parts[1],
        participant_id=parts[2],
        sample_type_code=sample_type_code,
        vial=vial,
        portion=portion,
        analyte=analyte,
        plate=plate,
        center=center,
    )


# ======================================================================================
# GDC HTTP client
# ======================================================================================
def _make_session(retries: int, backoff: float) -> requests.Session:
    """Session with urllib3 retry/backoff for transient HTTP + connection errors."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "repurpose-gnn/0.1 (academic research)"})
    return session


def build_files_filter(cfg: Config) -> dict[str, Any]:
    """Build the GDC ``/files`` JSON filter for the configured cohort + RNA-seq STAR counts."""
    g = cfg.section("gdc")
    sample_types = g["sample_types"]
    wanted_sample_types = list(sample_types.get("tumor", [])) + list(sample_types.get("normal", []))
    return {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": cfg.cohort}},
            {"op": "=", "content": {"field": "data_category", "value": g["data_category"]}},
            {"op": "=", "content": {"field": "data_type", "value": g["data_type"]}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": g["workflow_type"]}},
            {"op": "=", "content": {"field": "experimental_strategy", "value": g["experimental_strategy"]}},
            {"op": "in", "content": {"field": "cases.samples.sample_type", "value": wanted_sample_types}},
        ],
    }


def _post_files(
    session: requests.Session,
    api_base: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + FILES_ENDPOINT
    resp = session.post(url, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"GDC /files POST failed [{resp.status_code}]: {resp.text[:500]}"
        )
    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GDC /files returned non-JSON: {resp.text[:500]}") from exc


def validate_workflow_type(cfg: Config, session: requests.Session | None = None) -> list[str]:
    """Confirm the configured ``workflow_type`` exists in GDC facet buckets for this cohort.

    Returns the list of available workflow_type values. Raises :class:`ValueError` if the
    configured value is absent (so we never silently query a non-existent enum).
    """
    g = cfg.section("gdc")
    api_base = g["api_base"]
    timeout = int(g["timeout_seconds"])
    own_session = session is None
    session = session or _make_session(int(g["retries"]), float(g["backoff_seconds"]))
    try:
        # Facet on workflow_type, filtered to the cohort's Gene Expression Quantification files.
        payload = {
            "filters": {
                "op": "and",
                "content": [
                    {"op": "=", "content": {"field": "cases.project.project_id", "value": cfg.cohort}},
                    {"op": "=", "content": {"field": "data_type", "value": g["data_type"]}},
                ],
            },
            "facets": "analysis.workflow_type",
            "size": 0,
        }
        data = _post_files(session, api_base, payload, timeout)
        buckets = (
            data.get("data", {})
            .get("aggregations", {})
            .get("analysis.workflow_type", {})
            .get("buckets", [])
        )
        available = [b["key"] for b in buckets]
        if g["workflow_type"] not in available:
            raise ValueError(
                f"Configured gdc.workflow_type={g['workflow_type']!r} not found for {cfg.cohort}. "
                f"Available: {available}. Update configs/*.yaml to a valid value."
            )
        logger.info("Validated workflow_type=%r (available: %s)", g["workflow_type"], available)
        return available
    finally:
        if own_session:
            session.close()


def query_gdc_files(cfg: Config, session: requests.Session | None = None) -> pd.DataFrame:
    """Query GDC ``/files`` and return one row per file with flattened sample metadata."""
    g = cfg.section("gdc")
    api_base = g["api_base"]
    timeout = int(g["timeout_seconds"])
    own_session = session is None
    session = session or _make_session(int(g["retries"]), float(g["backoff_seconds"]))

    fields = [
        "file_id", "file_name", "md5sum", "file_size", "data_type",
        "experimental_strategy", "analysis.workflow_type",
        "cases.project.project_id", "cases.submitter_id",
        "cases.samples.sample_type", "cases.samples.submitter_id",
    ]
    records: list[dict[str, Any]] = []
    try:
        page_from, size = 0, 500
        while True:
            payload = {
                "filters": build_files_filter(cfg),
                "fields": ",".join(fields),
                "format": "JSON",
                "size": size,
                "from": page_from,
            }
            data = _post_files(session, api_base, payload, timeout)
            hits = data.get("data", {}).get("hits", [])
            for hit in hits:
                records.append(_flatten_file_hit(hit))
            pagination = data.get("data", {}).get("pagination", {})
            total = int(pagination.get("total", len(records)))
            page_from += size
            if page_from >= total or not hits:
                break
    finally:
        if own_session:
            session.close()

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError(
            f"GDC returned 0 files for {cfg.cohort} with the configured filters. "
            "Check cohort name and gdc.* settings."
        )
    logger.info("GDC query returned %d files for %s", len(df), cfg.cohort)
    return df


def _flatten_file_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Flatten a GDC file hit's nested case/sample structure into a flat record."""
    cases = hit.get("cases", []) or [{}]
    case0 = cases[0]
    samples = case0.get("samples", []) or [{}]
    sample0 = samples[0]
    sample_barcode = sample0.get("submitter_id")
    condition = None
    participant = case0.get("submitter_id")
    if sample_barcode:
        try:
            bc = parse_tcga_barcode(sample_barcode)
            condition = bc.condition
            participant = participant or bc.participant_barcode
        except ValueError:
            pass
    return {
        "file_id": hit.get("file_id"),
        "file_name": hit.get("file_name"),
        "md5sum": hit.get("md5sum"),
        "file_size": hit.get("file_size"),
        "data_type": hit.get("data_type"),
        "workflow_type": (hit.get("analysis") or {}).get("workflow_type"),
        "project_id": (case0.get("project") or {}).get("project_id"),
        "case_barcode": participant,
        "sample_barcode": sample_barcode,
        "gdc_sample_type": sample0.get("sample_type"),
        "condition": condition,
    }


# ======================================================================================
# Manifest + download
# ======================================================================================
def write_file_manifest(files_df: pd.DataFrame, out_path: Path) -> Path:
    """Write a GDC-style file manifest (id, filename, md5, size) + our metadata columns."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["file_id", "file_name", "md5sum", "file_size",
            "case_barcode", "sample_barcode", "gdc_sample_type", "condition"]
    files_df.loc[:, cols].to_csv(out_path, sep="\t", index=False)
    logger.info("Wrote file manifest: %s", out_path)
    return out_path


def _md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()  # noqa: S324 - md5 is GDC's integrity checksum, not for security
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    session: requests.Session,
    api_base: str,
    file_id: str,
    dest: Path,
    *,
    expected_md5: str | None,
    timeout: int,
    max_verify_retries: int = 3,
) -> tuple[bool, str]:
    """Download one file from GDC ``/data/<id>`` with streaming + md5 integrity verification.

    Returns ``(downloaded, status)`` where status is one of
    ``{"cached", "downloaded", "md5_mismatch", "error"}``. Skips re-download if a valid
    (md5-matching) copy already exists (resume support).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_md5 and _md5_file(dest) == expected_md5:
        return (False, "cached")

    url = api_base.rstrip("/") + f"{DATA_ENDPOINT}/{file_id}"
    for attempt in range(1, max_verify_retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout) as resp:
                if resp.status_code != 200:
                    logger.warning("Download %s failed [%s] (attempt %d)", file_id, resp.status_code, attempt)
                    continue
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                tmp.replace(dest)
        except (requests.RequestException, OSError) as exc:
            logger.warning("Download %s error (attempt %d): %s", file_id, attempt, exc)
            time.sleep(min(2 ** attempt, 30))
            continue

        if expected_md5 is None:
            return (True, "downloaded")
        actual = _md5_file(dest)
        if actual == expected_md5:
            return (True, "downloaded")
        logger.warning("md5 mismatch for %s (attempt %d): %s != %s", file_id, attempt, actual, expected_md5)
        time.sleep(min(2 ** attempt, 30))

    return (False, "md5_mismatch" if dest.exists() else "error")


# ======================================================================================
# STAR-Counts reader (GDC augmented star gene counts format)
# ======================================================================================
_STAR_SUMMARY_ROWS = {"N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"}
_VALID_COUNT_COLUMNS = {"unstranded", "stranded_first", "stranded_second"}


def read_star_counts(path: Path, count_column: str = "unstranded") -> pd.DataFrame:
    """Read one GDC STAR-Counts TSV.

    Returns a DataFrame indexed by ``gene_id`` (Ensembl w/ version) with columns
    ``[gene_name, gene_type, <count_column>]``. The four STAR summary rows
    (N_unmapped, N_multimapping, N_noFeature, N_ambiguous) are dropped.
    """
    if count_column not in _VALID_COUNT_COLUMNS:
        raise ValueError(f"count_column must be one of {_VALID_COUNT_COLUMNS}, got {count_column!r}.")
    # GDC augmented STAR files begin with a comment line, e.g. "# gene-model: GENCODE v36",
    # before the real header. comment="#" skips it (and is a no-op when absent).
    df = pd.read_csv(path, sep="\t", dtype={"gene_id": str}, comment="#")
    required = {"gene_id", "gene_name", "gene_type", count_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing expected STAR columns {missing}. Got {list(df.columns)}.")
    df = df[~df["gene_id"].isin(_STAR_SUMMARY_ROWS)].copy()
    df = df.set_index("gene_id")
    return df[["gene_name", "gene_type", count_column]]


# ======================================================================================
# Sample metadata assembly + de-duplication
# ======================================================================================
def build_sample_metadata(files_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Assemble the sample-metadata table and de-duplicate to prevent participant leakage.

    Keeps only tumor/normal samples, adds parsed barcode fields, and (if configured)
    retains a single sample per (participant, condition). Also optionally caps the number
    of samples per group for fast smoke runs (``gdc.max_samples_per_group``).
    """
    g = cfg.section("gdc")
    df = files_df.copy()
    df = df[df["condition"].isin(["tumor", "normal"])].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No tumor/normal samples after filtering — check sample_type config.")

    # enrich with parsed barcode fields
    parsed = df["sample_barcode"].map(parse_tcga_barcode)
    df["participant"] = parsed.map(lambda b: b.participant_barcode)
    df["sample_type_code"] = parsed.map(lambda b: b.sample_type_code)
    df["tss"] = parsed.map(lambda b: b.tss)

    n_before = len(df)
    if g.get("dedup_multiple_per_participant", True):
        # Deterministic choice: sort by sample_barcode then keep first per (participant, condition).
        df = (
            df.sort_values(["participant", "condition", "sample_barcode"])
            .drop_duplicates(subset=["participant", "condition"], keep="first")
            .reset_index(drop=True)
        )
    n_dedup = len(df)
    if n_dedup < n_before:
        logger.info("De-dup: %d -> %d samples (one per participant/condition).", n_before, n_dedup)

    cap = g.get("max_samples_per_group")
    if cap is not None:
        df = (
            df.sort_values("sample_barcode")
            .groupby("condition", group_keys=False)
            .head(int(cap))
            .reset_index(drop=True)
        )
        logger.info("Capped to <=%d samples per group -> %d total (smoke run).", cap, len(df))

    return df


def report_counts(sample_df: pd.DataFrame) -> dict[str, int]:
    """Return and log tumor/normal/participant counts."""
    counts = {
        "n_samples": int(len(sample_df)),
        "n_tumor": int((sample_df["condition"] == "tumor").sum()),
        "n_normal": int((sample_df["condition"] == "normal").sum()),
        "n_participants": int(sample_df["participant"].nunique()),
    }
    logger.info(
        "Cohort counts: %d samples (%d tumor / %d normal) across %d participants.",
        counts["n_samples"], counts["n_tumor"], counts["n_normal"], counts["n_participants"],
    )
    return counts


# ======================================================================================
# Orchestration (called by scripts/download_tcga.py and the CLI)
# ======================================================================================
def run_download(cfg: Config) -> dict[str, Any]:
    """Full Phase-1 download stage: query -> manifest -> download+verify -> metadata -> report.

    Writes:
      - <interim>/<cohort>/file_manifest.tsv
      - <interim>/<cohort>/sample_metadata.tsv
      - raw STAR-Counts files under <raw_tcga>/<cohort>/
      - a run manifest under results/logs/
    Returns a summary dict.
    """
    g = cfg.section("gdc")
    api_base = g["api_base"]
    timeout = int(g["timeout_seconds"])
    cohort = cfg.cohort

    raw_dir = cfg.path("paths.raw_tcga") / cohort
    interim_dir = cfg.path("paths.interim") / cohort
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    session = _make_session(int(g["retries"]), float(g["backoff_seconds"]))
    try:
        validate_workflow_type(cfg, session=session)
        files_df = query_gdc_files(cfg, session=session)
        sample_df = build_sample_metadata(files_df, cfg)

        manifest_path = interim_dir / "file_manifest.tsv"
        write_file_manifest(sample_df, manifest_path)

        # Download (only the de-duplicated / capped set we will actually use).
        statuses: dict[str, int] = {}
        n = len(sample_df)
        for i, row in enumerate(sample_df.itertuples(index=False), start=1):
            dest = raw_dir / f"{row.file_id}__{row.file_name}"
            _, status = download_file(
                session, api_base, row.file_id, dest,
                expected_md5=row.md5sum, timeout=timeout,
            )
            statuses[status] = statuses.get(status, 0) + 1
            if i % 25 == 0 or i == n:
                logger.info("Downloaded %d/%d files (%s)", i, n, dict(statuses))

        # record local path for each sample
        sample_df = sample_df.copy()
        sample_df["local_path"] = [
            str((raw_dir / f"{r.file_id}__{r.file_name}").resolve())
            for r in sample_df.itertuples(index=False)
        ]
        meta_path = interim_dir / "sample_metadata.tsv"
        sample_df.to_csv(meta_path, sep="\t", index=False)
        logger.info("Wrote sample metadata: %s", meta_path)

        counts = report_counts(sample_df)
        failed = statuses.get("md5_mismatch", 0) + statuses.get("error", 0)
        if failed:
            logger.error("%d files failed integrity/download — see logs above.", failed)

        summary = {
            "cohort": cohort,
            "counts": counts,
            "download_status": statuses,
            "file_manifest": str(manifest_path),
            "sample_metadata": str(meta_path),
            "raw_dir": str(raw_dir),
            "failed_files": failed,
        }
        write_run_manifest(
            cfg, stage="download_tcga", seed=cfg.seed,
            input_files={"file_manifest": manifest_path, "sample_metadata": meta_path},
            extra=summary,
        )
        return summary
    finally:
        session.close()


def _print_summary(summary: dict[str, Any]) -> None:
    logger.info("Download summary:\n%s", json.dumps(summary, indent=2))
