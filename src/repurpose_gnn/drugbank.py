"""DrugBank XML parser (Sections 4, 23) — license-restricted; user must supply the file.

DrugBank's full database is NOT open: it requires a (free-for-academics, non-commercial)
account and approval at https://go.drugbank.com/. This module parses a locally-supplied
``full_database.xml``; it never invents credentials or a download URL. If the file is
missing it raises a clear error and the pipeline falls back to the open DGIdb default.

Memory-safe streaming via ``lxml.etree.iterparse`` (the file is ~1.4 GB). The DrugBank XML
uses the default namespace ``http://www.drugbank.ca`` — handled explicitly.

Target/enzyme/carrier/transporter relationships are kept SEPARATE; the primary repurposing
analysis uses true targets, the others are available for sensitivity analyses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

DRUGBANK_NS = "http://www.drugbank.ca"
RELATIONSHIP_KINDS = ("targets", "enzymes", "carriers", "transporters")


def _q(tag: str, ns: str = DRUGBANK_NS) -> str:
    """Namespace-qualified tag name for ElementTree lookups."""
    return f"{{{ns}}}{tag}"


def _detect_namespace(path: Path) -> str:
    """Read the root element to detect the XML default namespace (robust to version changes)."""
    import lxml.etree as ET

    for _event, elem in ET.iterparse(str(path), events=("start",)):
        tag = elem.tag
        if isinstance(tag, str) and tag.endswith("drugbank"):
            return tag[1:].split("}")[0] if tag.startswith("{") else DRUGBANK_NS
        break
    return DRUGBANK_NS


def iter_drugs(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one dict per <drug>, streaming with iterparse + element clearing.

    Each dict: drugbank_id, name, type, groups (list), and per-relationship lists of
    {gene_name, uniprot, actions, known_action, organism} under keys in RELATIONSHIP_KINDS.
    """
    import lxml.etree as ET

    if not path.exists():
        raise FileNotFoundError(
            f"DrugBank XML not found: {path}. DrugBank is license-restricted — obtain it via a "
            "free academic account at https://go.drugbank.com/ and set drugs.drugbank_xml or the "
            "DRUGBANK_XML env var. To run without it, use drugs.source=dgidb (the open default)."
        )
    ns = _detect_namespace(path)
    drug_tag = _q("drug", ns)

    context = ET.iterparse(str(path), events=("end",), tag=drug_tag)
    for _event, elem in context:
        # only top-level <drug> (skip nested <drug> inside drug-interactions etc.)
        parent = elem.getparent()
        if parent is None or not parent.tag.endswith("drugbank"):
            _clear(elem)
            continue
        yield _parse_drug(elem, ns)
        _clear(elem)


def _clear(elem) -> None:
    """Free memory: clear the element and remove preceding siblings."""
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]


def _parse_drug(elem, ns: str) -> dict[str, Any]:
    def findtext(node, tag):
        child = node.find(_q(tag, ns))
        return child.text if child is not None else None

    # primary DrugBank id
    db_id = None
    for idnode in elem.findall(_q("drugbank-id", ns)):
        if idnode.get("primary") == "true":
            db_id = idnode.text
            break
    if db_id is None:
        first = elem.find(_q("drugbank-id", ns))
        db_id = first.text if first is not None else None

    groups_node = elem.find(_q("groups", ns))
    groups = [g.text for g in groups_node.findall(_q("group", ns))] if groups_node is not None else []

    drug: dict[str, Any] = {
        "drugbank_id": db_id,
        "name": findtext(elem, "name"),
        "type": elem.get("type"),
        "groups": groups,
    }
    for kind in RELATIONSHIP_KINDS:
        drug[kind] = _parse_relationships(elem, kind, ns)
    return drug


def _parse_relationships(elem, kind: str, ns: str) -> list[dict[str, Any]]:
    container = elem.find(_q(kind, ns))
    if container is None:
        return []
    singular = kind[:-1]  # targets -> target
    out = []
    for rel in container.findall(_q(singular, ns)):
        actions_node = rel.find(_q("actions", ns))
        actions = [a.text for a in actions_node.findall(_q("action", ns))] if actions_node is not None else []
        known = rel.find(_q("known-action", ns))
        known_action = known.text if known is not None else None
        gene_name = uniprot = organism = None
        poly = rel.find(_q("polypeptide", ns))
        if poly is not None:
            gn = poly.find(_q("gene-name", ns))
            gene_name = gn.text if gn is not None else None
            org = poly.find(_q("organism", ns))
            organism = org.text if org is not None else None
            uniprot = poly.get("id")  # source usually UniProtKB
            ext = poly.find(_q("external-identifiers", ns))
            if ext is not None:
                for e in ext.findall(_q("external-identifier", ns)):
                    res = e.find(_q("resource", ns))
                    if res is not None and res.text == "UniProtKB":
                        val = e.find(_q("identifier", ns))
                        if val is not None:
                            uniprot = val.text
        out.append({"gene_name": gene_name, "uniprot": uniprot, "actions": actions,
                    "known_action": known_action, "organism": organism})
    return out


def load_drugbank_targets(cfg: Config, gene_symbols: list[str] | None = None):
    """Parse DrugBank -> (interactions_df, drug_targets, drug_meta) matching drug_sources' API."""
    dcfg = cfg.section("drugs")
    import os
    xml = os.environ.get("DRUGBANK_XML") or dcfg.get("drugbank_xml")
    if not xml:
        raise FileNotFoundError("No DrugBank XML configured (drugs.drugbank_xml / DRUGBANK_XML).")
    path = Path(xml)
    rel_kinds = set(dcfg.get("relationship_types", ["target"]))
    kinds = tuple(k + "s" for k in rel_kinds) if rel_kinds else ("targets",)
    approved_only = bool(dcfg.get("approved_only", True))
    known_action_only = bool(dcfg.get("known_action_only", True))
    human_only = True

    rows: list[dict[str, Any]] = []
    n_drugs = 0
    for drug in iter_drugs(path):
        n_drugs += 1
        approved = "approved" in (drug["groups"] or [])
        if approved_only and not approved:
            continue
        for kind in kinds:
            for rel in drug.get(kind, []):
                if human_only and rel.get("organism") not in (None, "Humans", "Human"):
                    continue
                if known_action_only and (rel.get("known_action") not in ("yes", "unknown", None)):
                    # keep 'yes' and unknown; drop 'no' only when strict
                    if rel.get("known_action") == "no":
                        continue
                if not rel.get("gene_name"):
                    continue
                rows.append({
                    "gene_name": rel["gene_name"],
                    "drug_name": drug["name"],
                    "drug_id": drug["drugbank_id"],
                    "approved": approved,
                    "action": ";".join(sorted({a for a in rel.get("actions", []) if a})),
                    "relationship": kind[:-1],
                    "uniprot": rel.get("uniprot"),
                    "source": "drugbank",
                })
    inter = pd.DataFrame(rows).drop_duplicates(subset=["drug_id", "gene_name", "relationship"])
    logger.info("DrugBank: parsed %d drugs -> %d target interactions.", n_drugs, len(inter))

    from .drug_sources import interactions_to_drug_targets
    drug_targets, drug_meta = interactions_to_drug_targets(
        inter, approved_only=False, known_action_only=False  # already filtered above
    )
    return inter, drug_targets, drug_meta
