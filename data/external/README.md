# External / user-supplied data

Files here are **not** downloaded automatically because they are license-restricted or
require registration. Nothing in the default (open) pipeline depends on them.

| Source | Path to place file | How to obtain | License note |
|---|---|---|---|
| **DrugBank** (optional) | `../raw/drugbank/full_database.xml` | Academic account + approval at https://go.drugbank.com/ | Free for **non-commercial academic** use; **never commit** the file. |
| **COSMIC Cancer Gene Census** (optional labels) | `cosmic/cancer_gene_census.csv` | Free academic registration at https://cancer.sanger.ac.uk/cosmic | Registration required; redistribution restricted. |
| **Open Targets** (default labels) | auto-fetched to `opentargets/` | https://platform.opentargets.org/ (open API/downloads) | CC0 / open. |
| **DGIdb** (default drugs) | auto-fetched | https://dgidb.org/ (open GraphQL API) | Open (MIT-style). |

Set paths/credentials via `.env` (copy from `.env.example`). If an optional file is
missing, the relevant stage reports a clear error and, where possible, falls back to an
open source — it does not fabricate data.
