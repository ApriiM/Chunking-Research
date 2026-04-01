"""Build reproducible baseline caches for PIRB export regression checks."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence


@dataclass(frozen=True)
class TimingRecord:
    """Timing/status row parsed from timeout batch TSV report."""

    run_name: str
    status: str
    duration_sec: int
    log_path: str


@dataclass(frozen=True)
class RunSignature:
    """Deterministic signatures describing one exported run output."""

    run_name: str
    dataset: str
    queries_path: str
    passages_path: str
    query_count: int
    passage_count: int
    minus_one_query_count: int
    query_id_order_sha256: str
    query_relevance_sha256: str
    passage_parent_mapping_sha256: str
    queries_file_sha256: str
    passages_file_sha256: str


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            yield json.loads(raw)


def _to_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _query_signatures(queries_path: Path) -> tuple[int, int, str, str]:
    query_count = 0
    minus_one_query_count = 0
    id_hasher = hashlib.sha256()
    relevance_hasher = hashlib.sha256()

    for row in _iter_jsonl(queries_path):
        query_count += 1
        qid = str(row.get("id") or "")
        relevant = _to_str_list(row.get("relevant"))
        scores = _to_str_list(row.get("relevant_scores"))
        if "-1" in relevant:
            minus_one_query_count += 1

        id_hasher.update((qid + "\n").encode("utf-8"))
        normalized = {
            "id": qid,
            "relevant": relevant,
            "relevant_scores": scores,
        }
        relevance_hasher.update(
            (json.dumps(normalized, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )

    return (
        query_count,
        minus_one_query_count,
        id_hasher.hexdigest(),
        relevance_hasher.hexdigest(),
    )


def _passage_signatures(passages_path: Path) -> tuple[int, str]:
    passage_count = 0
    mapping_hasher = hashlib.sha256()

    for row in _iter_jsonl(passages_path):
        passage_count += 1
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        normalized = {
            "id": str(row.get("id") or ""),
            "parentId": str(metadata.get("parentId") or row.get("parentId") or ""),
            "original_id": str(metadata.get("original_id") or ""),
        }
        mapping_hasher.update(
            (json.dumps(normalized, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )

    return passage_count, mapping_hasher.hexdigest()


def parse_timing_report(report_tsv_path: Path) -> list[TimingRecord]:
    """Parse timeout batch TSV and return rows preserving file order."""

    rows: list[TimingRecord] = []
    with report_tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            duration_raw = row.get("duration_sec") or "0"
            duration_sec = int(duration_raw)
            rows.append(
                TimingRecord(
                    run_name=str(row.get("run") or ""),
                    status=str(row.get("status") or ""),
                    duration_sec=duration_sec,
                    log_path=str(row.get("log_path") or ""),
                )
            )
    return rows


def _load_dataset_slug(run_dir: Path) -> str:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return run_dir.name
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_slug = payload.get("dataset_slug")
    if dataset_slug:
        return str(dataset_slug)
    return run_dir.name


def collect_run_signature(run_dir: Path) -> RunSignature:
    """Compute deterministic regression signatures for one exported run."""

    queries_path = run_dir / "queries" / "queries.jsonl"
    passages_path = run_dir / "passages" / "passages.jsonl"

    (
        query_count,
        minus_one_query_count,
        query_id_order_sha256,
        query_relevance_sha256,
    ) = _query_signatures(queries_path)
    passage_count, passage_parent_mapping_sha256 = _passage_signatures(passages_path)

    return RunSignature(
        run_name=run_dir.name,
        dataset=_load_dataset_slug(run_dir),
        queries_path=str(queries_path),
        passages_path=str(passages_path),
        query_count=query_count,
        passage_count=passage_count,
        minus_one_query_count=minus_one_query_count,
        query_id_order_sha256=query_id_order_sha256,
        query_relevance_sha256=query_relevance_sha256,
        passage_parent_mapping_sha256=passage_parent_mapping_sha256,
        queries_file_sha256=_sha256_file(queries_path),
        passages_file_sha256=_sha256_file(passages_path),
    )


def build_baseline_cache(
    report_tsv_path: Path,
    export_session_path: Path,
    output_root: Path,
    *,
    baseline_name: str,
) -> Path:
    """Create a baseline cache folder for finished runs from a batch TSV report."""

    baseline_dir = output_root / baseline_name
    baseline_dir.mkdir(parents=True, exist_ok=True)

    timing_rows = parse_timing_report(report_tsv_path)
    finished_rows = [row for row in timing_rows if row.status == "finished"]

    run_signatures: list[RunSignature] = []
    for row in finished_rows:
        run_dir = export_session_path / row.run_name
        run_signatures.append(collect_run_signature(run_dir))

    timing_out = baseline_dir / "timing_snapshot.tsv"
    with timing_out.open("w", encoding="utf-8", newline="") as handle:
        handle.write("run\tdataset\tstatus\tduration_sec\tlog_path\n")
        by_name = {signature.run_name: signature for signature in run_signatures}
        for row in finished_rows:
            signature = by_name[row.run_name]
            handle.write(
                "\t".join(
                    [
                        row.run_name,
                        signature.dataset,
                        row.status,
                        str(row.duration_sec),
                        row.log_path,
                    ]
                )
                + "\n"
            )

    signatures_out = baseline_dir / "run_signatures.jsonl"
    with signatures_out.open("w", encoding="utf-8") as handle:
        for signature in run_signatures:
            handle.write(
                json.dumps(
                    {
                        "run_name": signature.run_name,
                        "dataset": signature.dataset,
                        "queries_path": signature.queries_path,
                        "passages_path": signature.passages_path,
                        "query_count": signature.query_count,
                        "passage_count": signature.passage_count,
                        "minus_one_query_count": signature.minus_one_query_count,
                        "query_id_order_sha256": signature.query_id_order_sha256,
                        "query_relevance_sha256": signature.query_relevance_sha256,
                        "passage_parent_mapping_sha256": signature.passage_parent_mapping_sha256,
                        "queries_file_sha256": signature.queries_file_sha256,
                        "passages_file_sha256": signature.passages_file_sha256,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    manifest_out = baseline_dir / "manifest.json"
    manifest_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_tsv_path": str(report_tsv_path),
        "export_session_path": str(export_session_path),
        "baseline_name": baseline_name,
        "included_status": ["finished"],
        "run_count": len(run_signatures),
        "runs": [signature.run_name for signature in run_signatures],
    }
    manifest_out.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    readme_out = baseline_dir / "README.md"
    readme_lines: Sequence[str] = (
        "# PIRB Export Baseline Cache",
        "",
        "This folder stores baseline signatures and timing snapshots for regression checks.",
        "",
        "## Files",
        "",
        "- `manifest.json`: baseline metadata and run list",
        "- `timing_snapshot.tsv`: per-run baseline durations",
        "- `run_signatures.jsonl`: deterministic output signatures per run",
        "",
        "## Reuse",
        "",
        "Use `run_signatures.jsonl` and `timing_snapshot.tsv` as before-state references for post-change comparison.",
    )
    readme_out.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    return baseline_dir
