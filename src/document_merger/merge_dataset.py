import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object JSON in {path} at line {line_no}")
            rows.append(payload)
    return rows


def _save_jsonl(rows: List[Dict[str, Any]], path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _doc_id(doc: Dict[str, Any]) -> str | None:
    value = doc.get("id")
    if value is None:
        value = doc.get("doc_id")
    if value is None:
        return None
    return str(value)


def _doc_text(doc: Dict[str, Any]) -> str:
    value = doc.get("contents")
    if value is None:
        value = doc.get("text", "")
    if value is None:
        return ""
    return str(value)


def _build_super_document(
    documents: List[Dict[str, Any]],
    separator: str,
    merged_doc_id: str,
) -> Dict[str, Any]:
    merged_text = separator.join(_doc_text(doc) for doc in documents)
    return {
        "id": merged_doc_id,
        "contents": merged_text,
        "metadata": {
            "source": "document_merger",
            "strategy": "concatenate_no_shuffle",
            "original_doc_count": len(documents),
        },
    }


def _query_has_extractive_answer(query: Dict[str, Any]) -> bool:
    if "extractive_span_text_answer" in query:
        return True
    metadata = query.get("metadata")
    return isinstance(metadata, dict) and "extractive_span_text_answer" in metadata


def _relevant_texts_for_ids(
    relevant_doc_ids: List[str],
    doc_text_by_id: Dict[str, str],
) -> List[str]:
    if not isinstance(relevant_doc_ids, list):
        return []

    out: List[str] = []
    for doc_id in relevant_doc_ids:
        if not isinstance(doc_id, str):
            continue
        text = doc_text_by_id.get(doc_id)
        if text is not None:
            out.append(text)
    return out


def _prepare_queries(
    queries: List[Dict[str, Any]],
    doc_text_by_id: Dict[str, str],
    merged_doc_id: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    output: List[Dict[str, Any]] = []
    kept_unchanged = 0
    filled_from_relevant = 0

    for query in queries:
        new_query = copy.deepcopy(query)
        original_relevant = new_query.get("relevant")
        original_relevant_doc_ids = (
            [str(x) for x in original_relevant if isinstance(x, str)]
            if isinstance(original_relevant, list)
            else []
        )
        # In merged dataset all relevant links should point to the single super document.
        new_query["relevant"] = [merged_doc_id] if original_relevant_doc_ids else []

        if _query_has_extractive_answer(new_query):
            kept_unchanged += 1
            output.append(new_query)
            continue

        metadata = new_query.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            new_query["metadata"] = metadata

        metadata["extractive_span_text_answer"] = _relevant_texts_for_ids(
            original_relevant_doc_ids,
            doc_text_by_id,
        )
        filled_from_relevant += 1
        output.append(new_query)

    return output, kept_unchanged, filled_from_relevant


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a merged dataset folder from an input dataset folder. "
            "Input is expected at <dataset>/documents/documents.jsonl and "
            "<dataset>/queries/queries.jsonl. Output is written to "
            "<dataset_parent>/<dataset_name>_merged."
        )
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Input dataset folder path, e.g. data/processed/squad/train",
    )
    parser.add_argument(
        "--separator",
        default="\n\n",
        help="Separator inserted between source documents in the super document.",
    )
    parser.add_argument(
        "--merged-doc-id",
        default="merged_doc_001",
        help="ID assigned to the generated super document.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting output files in the merged dataset folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    input_documents_path = dataset_path / "documents" / "documents.jsonl"
    input_queries_path = dataset_path / "queries" / "queries.jsonl"

    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    output_dataset_path = dataset_path.parent / f"{dataset_path.name}_merged"
    output_documents_path = output_dataset_path / "documents" / "documents.jsonl"
    output_queries_path = output_dataset_path / "queries" / "queries.jsonl"

    # Ensure output dataset structure exists.
    (output_dataset_path / "documents").mkdir(parents=True, exist_ok=True)
    (output_dataset_path / "queries").mkdir(parents=True, exist_ok=True)

    documents = _load_jsonl(input_documents_path)
    queries = _load_jsonl(input_queries_path)

    doc_text_by_id: Dict[str, str] = {}
    missing_doc_ids = 0
    for doc in documents:
        doc_id = _doc_id(doc)
        if doc_id is None:
            missing_doc_ids += 1
            continue
        doc_text_by_id[doc_id] = _doc_text(doc)

    merged_document = _build_super_document(
        documents=documents,
        separator=args.separator,
        merged_doc_id=str(args.merged_doc_id),
    )
    merged_queries, kept_unchanged, filled_from_relevant = _prepare_queries(
        queries=queries,
        doc_text_by_id=doc_text_by_id,
        merged_doc_id=str(args.merged_doc_id),
    )

    _save_jsonl([merged_document], output_documents_path, overwrite=args.overwrite)
    _save_jsonl(merged_queries, output_queries_path, overwrite=args.overwrite)

    print(f"[DocumentMerger] Input dataset: {dataset_path}")
    print(f"[DocumentMerger] Output dataset: {output_dataset_path}")
    print(f"[DocumentMerger] Loaded {len(documents)} documents and {len(queries)} queries.")
    if missing_doc_ids:
        print(f"[DocumentMerger] Warning: {missing_doc_ids} source documents were missing IDs.")
    print(f"[DocumentMerger] Wrote 1 super document to {output_documents_path}")
    print(
        "[DocumentMerger] Queries unchanged with extractive answers: "
        f"{kept_unchanged}, filled from relevant docs: {filled_from_relevant}"
    )
    print(f"[DocumentMerger] Wrote {len(merged_queries)} queries to {output_queries_path}")


if __name__ == "__main__":
    main()
