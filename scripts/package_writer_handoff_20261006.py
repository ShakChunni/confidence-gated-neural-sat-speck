"""Package audited drafting materials; this is not an executable attack bundle."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "output/publication_figures_20261005_final"
OUTPUT = (
    Path(sys.argv[1])
    if len(sys.argv) == 2
    else ROOT / "output/writing_handoff_20261006"
)
if len(sys.argv) > 2:
    raise ValueError("Supply at most one new output directory")
OUTPUT.mkdir(parents=True, exist_ok=True)
archive = OUTPUT / "six_page_writer_bundle_v3.zip"
external_manifest = OUTPUT / "handoff_manifest_v3.json"
if (
    archive.exists()
    or archive.with_name(archive.name + ".sha256").exists()
    or external_manifest.exists()
):
    raise FileExistsError("Never overwrite a writer bundle; choose a new directory")

paths = {
    "READ_FIRST.md": "docs/writing_handoff_20261006/UPLOAD_FILES.md",
    "markdown/00_SIX_PAGE_WRITING_PROMPT.md": "docs/writing_handoff_20261006/00_SIX_PAGE_WRITING_PROMPT.md",
    "markdown/01_CURRENT_STATUS_AND_NEW_RESULTS.md": "docs/writing_handoff_20261006/01_CURRENT_STATUS_AND_NEW_RESULTS.md",
    "markdown/02_RELATED_WORK_ADDENDUM.md": "docs/writing_handoff_20261006/02_RELATED_WORK_ADDENDUM.md",
    "markdown/WRITER_UPLOAD_PACKET.md": "docs/writing_handoff_20261005/WRITER_UPLOAD_PACKET.md",
    "markdown/QUEUE_ORDER_ADDENDUM_20261005.md": "docs/writing_handoff_20261005/QUEUE_ORDER_ADDENDUM_20261005.md",
    "markdown/SIMON_MATCHED_STAGE2_CLOSURE_20261006.md": "docs/SIMON_MATCHED_STAGE2_CLOSURE_20261006.md",
    "markdown/PUBLICATION_PRIORITIES.md": "docs/PUBLICATION_STRENGTHENING_PRIORITIES_20261005.md",
    "tables/queue_order_analysis.csv": "docs/queue_order_opened_evidence_20261005/analysis_v1.csv",
    "tables/simon_stage2_cells.csv": "docs/stage2_opened_evidence_20261006/cells_v1.csv",
    "tables/simon_stage2_summary.csv": "docs/stage2_opened_evidence_20261006/summary_v1.csv",
    "evidence/simon_stage2_analysis.json": "docs/stage2_opened_evidence_20261006/analysis_v1.json",
    "evidence/simon_stage2_independent_audit.json": "docs/stage2_opened_evidence_20261006/independent_audit_v1.json",
    "evidence/simon_stage2_hostile_review.json": "docs/stage2_opened_evidence_20261006/hostile_review_v1.json",
    "evidence/queue_order_analysis.json": "docs/queue_order_opened_evidence_20261005/analysis_v1.json",
    "evidence/queue_order_independent_audit.json": "docs/queue_order_opened_evidence_20261005/independent_audit_v1.json",
    "code/recompute_simon_stage2_independent.py": "docs/stage2_opened_evidence_20261006/recompute_independent.py",
    "code/package_writer_handoff_20261006.py": "scripts/package_writer_handoff_20261006.py",
}
figure_manifest = json.loads((FIGURES / "metadata/figure_manifest.json").read_text())
expected_hashes = {}
for category, directory in (
    ("figure_artifacts", "figures"),
    ("table_artifacts", "tables"),
    ("metadata_artifacts", "metadata"),
    ("code_artifacts", ""),
):
    for name, expected in figure_manifest[category].items():
        if category == "figure_artifacts" and not name.endswith(".pdf"):
            continue
        relative = Path(directory) / name
        entry = relative.as_posix()
        paths[entry] = (FIGURES / relative).relative_to(ROOT).as_posix()
        expected_hashes[entry] = expected
paths["metadata/figure_manifest.json"] = (
    (FIGURES / "metadata/figure_manifest.json").relative_to(ROOT).as_posix()
)

data = {}
records = []
for entry, relative in sorted(paths.items()):
    source = ROOT / relative
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if entry in expected_hashes and digest != expected_hashes[entry]:
        raise ValueError(f"Frozen figure-release hash mismatch: {relative}")
    sidecar = source.with_name(source.name + ".sha256")
    if sidecar.exists() and sidecar.read_text().split()[0] != digest:
        raise ValueError(f"Detached source hash mismatch: {relative}")
    data[entry] = content
    records.append(
        {"entry": entry, "source": relative, "sha256": digest, "bytes": len(content)}
    )
manifest = {
    "schema": "six_page_writer_handoff_v3",
    "release_date": "2026-10-06",
    "purpose": "Local drafting evidence; not a complete executable or publicly licensed artifact",
    "page_budget": "Six pages including references as a conservative author planning assumption; venue rules unassigned",
    "literature_review": {
        "source": "Literature Review (Cryptanalysis).pdf",
        "included": False,
        "sha256": "b710086d533f533c17fa6a5ca8a75e9d61602a7b7ced40ea83cea676da3d81d2",
        "scope": "Reference-list presence checked; selected missing/source citations checked in related-work addendum; remaining PDF claims not revalidated",
    },
    "pending_experiments": [
        "permanent clauses versus temporary assumptions",
        "conditional counterbalanced observation-count follow-up",
    ],
    "files": records,
}
literature_pdf = ROOT / manifest["literature_review"]["source"]
if (
    hashlib.sha256(literature_pdf.read_bytes()).hexdigest()
    != manifest["literature_review"]["sha256"]
):
    raise ValueError("The literature review differs from the addendum's checked PDF")
data["metadata/HANDOFF_MANIFEST.json"] = (
    json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
).encode()
with zipfile.ZipFile(
    archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
) as handle:
    for entry, content in sorted(data.items()):
        info = zipfile.ZipInfo(entry, date_time=(2026, 10, 6, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        handle.writestr(info, content, compresslevel=9)
with zipfile.ZipFile(archive) as handle:
    if set(handle.namelist()) != set(data) or handle.testzip() is not None:
        raise ValueError("ZIP membership or CRC check failed")
    for entry, content in data.items():
        if handle.read(entry) != content:
            raise ValueError(f"Copied content differs: {entry}")
archive.with_name(archive.name + ".sha256").write_text(
    hashlib.sha256(archive.read_bytes()).hexdigest() + "  " + archive.name + "\n"
)
external_manifest.write_bytes(data["metadata/HANDOFF_MANIFEST.json"])
print(
    json.dumps(
        {
            "archive": str(archive),
            "files": len(data),
            "bytes": archive.stat().st_size,
            "csv_count": sum(name.endswith(".csv") for name in data),
            "pdf_count": sum(name.endswith(".pdf") for name in data),
        }
    )
)
