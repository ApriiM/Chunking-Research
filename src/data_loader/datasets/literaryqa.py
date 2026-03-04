from __future__ import annotations

import sys
from importlib.util import find_spec
from typing import Dict, List, Optional, Tuple

import datasets as hf_datasets
from datasets import load_dataset

from src.data_loader.core.registry import dataset
from src.data_loader.core.schemas import DocumentRecord, QueryRecord


_LITERARYQA_DATASET_NAME = "sapienzanlp/LiteraryQA"
_LITERARYQA_EXPECTED_DATASETS_MAJOR = 3
_LITERARYQA_EXPECTED_DATASETS_MINOR = 6


def _parse_datasets_version(version: str) -> Tuple[int, int]:
    parts = version.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else -1
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
    return (major, minor)


def _validate_literaryqa_runtime() -> None:
    """Validate the runtime expected by the upstream LiteraryQA dataset script."""
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "LiteraryQA requires Python >= 3.12 according to the upstream dataset card.\n"
            "Use a Python 3.12+ environment to load this dataset."
        )

    version = getattr(hf_datasets, "__version__", "")
    major, minor = _parse_datasets_version(version)
    if (major, minor) != (
        _LITERARYQA_EXPECTED_DATASETS_MAJOR,
        _LITERARYQA_EXPECTED_DATASETS_MINOR,
    ):
        raise RuntimeError(
            "LiteraryQA requires the Hugging Face datasets loader with remote dataset "
            "script support. The upstream dataset card recommends `datasets==3.6.0`.\n"
            f"Current datasets version: {version or 'unknown'}."
        )

    missing_modules = [
        module_name
        for module_name in ("chardet", "bs4", "ftfy")
        if find_spec(module_name) is None
    ]
    if missing_modules:
        raise RuntimeError(
            "LiteraryQA requires additional dependencies that are not installed: "
            f"{', '.join(missing_modules)}.\n"
            "Install the upstream recommended environment:\n"
            'pip install "datasets==3.6.0" "chardet==5.2.0" '
            '"beautifulsoup4[html5lib]==4.14.2" "ftfy==6.3.1"'
        )


@dataset("literaryqa")
@dataset("literary_qa")
def load_literaryqa(
    split: str = "train",
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    dataset_name: str = _LITERARYQA_DATASET_NAME,
) -> Tuple[List[DocumentRecord], List[QueryRecord]]:
    """Load LiteraryQA from Hugging Face Hub into full-book documents + queries.

    Upstream dataset:
    - Dataset repo: `sapienzanlp/LiteraryQA`
    - Splits: train, validation, test
    - Rows are document-level, with nested `qas`
    - Features include: `document_id`, `gutenberg_id`, `title`, `text`, `summary`,
      `qas`, `metadata`

    This loader creates:
    - one `DocumentRecord` per dataset row/book (`text` as contents)
    - one `QueryRecord` per nested QA entry, linked to that book

    Query metadata preserves:
    - `answers`: list of gold answer strings
    - `gutenberg_id`, `title`
    - selected top-level metadata and QA flags
    """
    _validate_literaryqa_runtime()

    try:
        ds = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load LiteraryQA from Hugging Face.\n"
            "This dataset uses an upstream loading script that downloads and preprocesses "
            "Project Gutenberg books.\n"
            "Use Python >= 3.12 and install:\n"
            'pip install "datasets==3.6.0" "chardet==5.2.0" '
            '"beautifulsoup4[html5lib]==4.14.2" "ftfy==6.3.1"'
        ) from exc

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    documents: List[DocumentRecord] = []
    queries: List[QueryRecord] = []

    for row in ds:
        document_id = str(row.get("document_id", "") or "").strip()
        gutenberg_id = str(row.get("gutenberg_id", "") or "").strip()
        title = str(row.get("title", "") or "").strip()
        book_text = str(row.get("text", "") or "")
        summary = str(row.get("summary", "") or "").strip()
        metadata = row.get("metadata") or {}
        qas = row.get("qas") or []

        if not document_id or not book_text:
            continue

        doc_id = f"literaryqa-{document_id}"
        doc_metadata: Dict[str, object] = {
            "dataset": "literaryqa",
            "document_id": document_id,
        }
        if gutenberg_id:
            doc_metadata["gutenberg_id"] = gutenberg_id
        if title:
            doc_metadata["title"] = title
        if summary:
            doc_metadata["summary"] = summary
        if isinstance(metadata, dict):
            for key in (
                "author",
                "publication_date",
                "genre_tags",
                "text_url",
                "summary_url",
            ):
                if key in metadata and metadata[key] not in (None, "", []):
                    doc_metadata[key] = metadata[key]

        documents.append(
            DocumentRecord(
                doc_id=doc_id,
                contents=book_text,
                metadata=doc_metadata,
            )
        )

        for qa_idx, qa in enumerate(qas):
            if not isinstance(qa, dict):
                continue

            question = str(qa.get("question", "") or "").strip()
            answers = [
                str(answer).strip()
                for answer in (qa.get("answers") or [])
                if str(answer).strip()
            ]
            if not question:
                continue

            query_metadata: Dict[str, object] = {
                "dataset": "literaryqa",
                "document_id": document_id,
            }
            if gutenberg_id:
                query_metadata["gutenberg_id"] = gutenberg_id
            if title:
                query_metadata["title"] = title
            if answers:
                query_metadata["answers"] = answers
            if "is_question_modified" in qa:
                query_metadata["is_question_modified"] = qa.get("is_question_modified")
            if "is_answer_modified" in qa:
                query_metadata["is_answer_modified"] = qa.get("is_answer_modified")

            queries.append(
                QueryRecord(
                    query_id=f"q.literaryqa.{split}.{document_id}.{qa_idx}",
                    contents=question,
                    relevant=[doc_id],
                    metadata=query_metadata,
                )
            )

            if limit is not None and len(queries) >= limit:
                return documents, queries

    return documents, queries
