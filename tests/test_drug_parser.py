"""DrugBank XML parsing (namespace, missing targets, no-mapped-targets) — Section 32."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from repurpose_gnn.drug_sources import interactions_to_drug_targets
from repurpose_gnn.drugbank import iter_drugs

DRUGBANK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<drugbank xmlns="http://www.drugbank.ca" version="5.1">
  <drug type="small molecule">
    <drugbank-id primary="true">DB0001</drugbank-id>
    <drugbank-id>DB9999</drugbank-id>
    <name>TestDrugA</name>
    <groups><group>approved</group></groups>
    <targets>
      <target>
        <actions><action>inhibitor</action></actions>
        <known-action>yes</known-action>
        <polypeptide id="P00533" source="Swiss-Prot">
          <gene-name>EGFR</gene-name>
          <organism>Humans</organism>
          <external-identifiers>
            <external-identifier><resource>UniProtKB</resource><identifier>P00533</identifier></external-identifier>
          </external-identifiers>
        </polypeptide>
      </target>
    </targets>
    <enzymes></enzymes>
  </drug>
  <drug type="biotech">
    <drugbank-id primary="true">DB0002</drugbank-id>
    <name>TestDrugB</name>
    <groups><group>investigational</group></groups>
    <targets>
      <target>
        <actions></actions>
        <polypeptide id="X" source="Swiss-Prot"><organism>Humans</organism></polypeptide>
      </target>
    </targets>
  </drug>
</drugbank>
"""


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "mini_drugbank.xml"
    p.write_text(DRUGBANK_XML)
    return p


def test_iter_drugs_namespace_and_fields(tmp_path):
    drugs = list(iter_drugs(_write(tmp_path)))
    assert len(drugs) == 2
    a = drugs[0]
    assert a["drugbank_id"] == "DB0001"  # primary id, not DB9999
    assert a["name"] == "TestDrugA"
    assert "approved" in a["groups"]
    tgt = a["targets"][0]
    assert tgt["gene_name"] == "EGFR"
    assert tgt["uniprot"] == "P00533"
    assert tgt["actions"] == ["inhibitor"]
    assert tgt["known_action"] == "yes"


def test_iter_drugs_missing_gene_name(tmp_path):
    drugs = list(iter_drugs(_write(tmp_path)))
    b = drugs[1]
    # target with a polypeptide but no gene-name -> gene_name is None (handled, not crashing)
    assert b["targets"][0]["gene_name"] is None


def test_missing_file_raises_clear_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError) as exc:
        list(iter_drugs(tmp_path / "nope.xml"))
    assert "DrugBank" in str(exc.value)


def test_interactions_to_drug_targets_and_no_target_drug():
    inter = pd.DataFrame([
        {"drug_id": "DB1", "drug_name": "D1", "gene_name": "EGFR", "approved": True,
         "action": "inhibitor", "source": "drugbank"},
        {"drug_id": "DB1", "drug_name": "D1", "gene_name": "ERBB2", "approved": True,
         "action": "inhibitor", "source": "drugbank"},
        {"drug_id": "DB2", "drug_name": "D2", "gene_name": "KRAS", "approved": False,
         "action": "", "source": "drugbank"},
    ])
    targets, meta = interactions_to_drug_targets(inter, approved_only=True, known_action_only=True)
    # DB2 filtered out (not approved, no action)
    assert set(targets.keys()) == {"DB1"}
    assert targets["DB1"] == {"EGFR", "ERBB2"}
    assert int(meta.loc[meta.drug_id == "DB1", "n_targets"].iloc[0]) == 2
