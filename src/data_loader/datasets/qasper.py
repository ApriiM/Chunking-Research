from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from src.data_loader.datasets._answer_utils import (
    build_unified_answer_metadata,
    normalize_unanswerable_answers,
    normalize_yes_no_answers,
    split_text_answers,
)
from src.data_loader.core.registry import dataset
from src.data_loader.core.schemas import DocumentRecord, QueryRecord


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _get_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _render_full_text(full_text: Any) -> Tuple[str, int]:
    """Render Qasper full_text into a single document string."""
    if not isinstance(full_text, dict):
        return ("", 0)

    section_names = _get_list(full_text.get("section_name"))
    section_paragraphs = _get_list(full_text.get("paragraphs"))

    parts: List[str] = []
    section_count = 0

    for idx, paragraphs in enumerate(section_paragraphs):
        section_name = _clean_text(section_names[idx] if idx < len(section_names) else "")
        para_list = _get_list(paragraphs)
        cleaned_paragraphs = [_clean_text(paragraph) for paragraph in para_list if _clean_text(paragraph)]

        if not section_name and not cleaned_paragraphs:
            continue

        section_count += 1
        if section_name:
            parts.append(section_name)
        parts.extend(cleaned_paragraphs)

    return ("\n\n".join(parts).strip(), section_count)


def _render_figures_and_tables(figures_and_tables: Any) -> Tuple[List[str], int]:
    if not isinstance(figures_and_tables, dict):
        return ([], 0)

    captions = _get_list(figures_and_tables.get("caption"))
    cleaned = [_clean_text(caption) for caption in captions if _clean_text(caption)]
    return (cleaned, len(cleaned))


def _normalize_answer_annotations(raw_answers: Any) -> List[Dict[str, Any]]:
    """Normalize one question's answer annotations into a list of dicts."""
    if isinstance(raw_answers, list):
        return [item for item in raw_answers if isinstance(item, dict)]

    if isinstance(raw_answers, dict):
        answers = _get_list(raw_answers.get("answer"))
        annotation_ids = _get_list(raw_answers.get("annotation_id"))
        worker_ids = _get_list(raw_answers.get("worker_id"))
        max_len = max(len(answers), len(annotation_ids), len(worker_ids), 0)

        normalized: List[Dict[str, Any]] = []
        for idx in range(max_len):
            payload = answers[idx] if idx < len(answers) else {}
            normalized.append(
                {
                    "answer": payload if isinstance(payload, dict) else {},
                    "annotation_id": annotation_ids[idx] if idx < len(annotation_ids) else None,
                    "worker_id": worker_ids[idx] if idx < len(worker_ids) else None,
                }
            )
        return normalized

    return []


def _build_query_metadata(
    *,
    document_text: str,
    paper_id: str,
    title: str,
    question_id: str,
    question_writer: str,
    nlp_background: str,
    topic_background: str,
    paper_read: str,
    search_query: str,
    raw_answers: Any,
) -> Dict[str, Any]:
    annotations = _normalize_answer_annotations(raw_answers)

    free_form_answers: List[str] = []
    extractive_spans: List[str] = []
    evidence: List[str] = []
    highlighted_evidence: List[str] = []
    yes_no_values: List[Any] = []
    unanswerable_values: List[Any] = []
    annotation_ids: List[str] = []
    worker_ids: List[str] = []

    normalized_annotations: List[Dict[str, Any]] = []
    for annotation in annotations:
        answer_payload = annotation.get("answer") if isinstance(annotation, dict) else {}
        answer_payload = answer_payload if isinstance(answer_payload, dict) else {}

        free_form = _clean_text(answer_payload.get("free_form_answer"))
        spans = [_clean_text(item) for item in _get_list(answer_payload.get("extractive_spans")) if _clean_text(item)]
        ev = [_clean_text(item) for item in _get_list(answer_payload.get("evidence")) if _clean_text(item)]
        hev = [
            _clean_text(item)
            for item in _get_list(answer_payload.get("highlighted_evidence"))
            if _clean_text(item)
        ]

        if free_form:
            free_form_answers.append(free_form)
        extractive_spans.extend(spans)
        evidence.extend(ev)
        highlighted_evidence.extend(hev)

        yes_no_values.append(answer_payload.get("yes_no"))
        unanswerable_values.append(answer_payload.get("unanswerable"))

        annotation_id = _clean_text(annotation.get("annotation_id"))
        worker_id = _clean_text(annotation.get("worker_id"))
        if annotation_id:
            annotation_ids.append(annotation_id)
        if worker_id:
            worker_ids.append(worker_id)

        normalized_annotations.append(
            {
                "annotation_id": annotation.get("annotation_id"),
                "worker_id": annotation.get("worker_id"),
                "answer": {
                    "unanswerable": answer_payload.get("unanswerable"),
                    "extractive_spans": spans,
                    "yes_no": answer_payload.get("yes_no"),
                    "free_form_answer": free_form,
                    "evidence": ev,
                    "highlighted_evidence": hev,
                },
            }
        )

    span_extractive, _ = split_text_answers(document_text, _dedupe_preserve_order(extractive_spans))

    free_candidates = list(free_form_answers)
    free_candidates.extend(normalize_yes_no_answers(yes_no_values))
    free_candidates.extend(normalize_unanswerable_answers(unanswerable_values))
    free_extractive, free_non_extractive = split_text_answers(document_text, free_candidates)

    metadata_base: Dict[str, Any] = {
        "dataset": "qasper",
        "paper_id": paper_id,
        "question_id": question_id,
        "answers": normalized_annotations,
        "answer_annotation_ids": _dedupe_preserve_order(annotation_ids),
        "answer_worker_ids": _dedupe_preserve_order(worker_ids),
        "free_form_answers": _dedupe_preserve_order(free_form_answers),
        "extractive_spans": _dedupe_preserve_order(extractive_spans),
        "evidence": _dedupe_preserve_order(evidence),
        "highlighted_evidence": _dedupe_preserve_order(highlighted_evidence),
        "yes_no_values": yes_no_values,
        "unanswerable_values": unanswerable_values,
    }
    if title:
        metadata_base["title"] = title
    if question_writer:
        metadata_base["question_writer"] = question_writer
    if nlp_background:
        metadata_base["nlp_background"] = nlp_background
    if topic_background:
        metadata_base["topic_background"] = topic_background
    if paper_read:
        metadata_base["paper_read"] = paper_read
    if search_query:
        metadata_base["search_query"] = search_query

    return build_unified_answer_metadata(
        base_metadata=metadata_base,
        extractive_answers=span_extractive + free_extractive,
        free_text_answers=free_non_extractive,
    )


@dataset("qasper")
@dataset("QASPER")
def load_qasper(
    split: str = "train",
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    dataset_name: str = "allenai/qasper",
) -> Tuple[List[DocumentRecord], List[QueryRecord]]:
    """Load QASPER into full-paper documents and flattened QA queries.

    Upstream dataset rows are paper-level. Each row contains:
    - paper metadata (`id`, `title`, `abstract`)
    - `full_text` as section names + per-section paragraphs
    - `qas` as a nested question collection with per-question answer annotations

    This loader creates:
    - one `DocumentRecord` per paper
    - one `QueryRecord` per question inside that paper
    """
    ds = load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir,
    )

    documents: List[DocumentRecord] = []
    queries: List[QueryRecord] = []

    for row in ds:
        paper_id = _clean_text(row.get("id"))
        title = _clean_text(row.get("title"))
        abstract = _clean_text(row.get("abstract"))
        full_text = row.get("full_text")
        qas = row.get("qas")
        figures_and_tables = row.get("figures_and_tables")

        if not paper_id:
            continue

        full_text_rendered, section_count = _render_full_text(full_text)
        figure_captions, figure_count = _render_figures_and_tables(figures_and_tables)

        doc_parts: List[str] = []
        if title:
            doc_parts.append(title)
        if abstract:
            doc_parts.extend(["Abstract", abstract])
        if full_text_rendered:
            doc_parts.append(full_text_rendered)
        if figure_captions:
            doc_parts.extend(["Figures and Tables", "\n\n".join(figure_captions)])

        contents = "\n\n".join(part for part in doc_parts if part).strip()
        if not contents:
            continue

        qas_dict = qas if isinstance(qas, dict) else {}
        questions = _get_list(qas_dict.get("question"))
        question_ids = _get_list(qas_dict.get("question_id"))
        question_writers = _get_list(qas_dict.get("question_writer"))
        nlp_backgrounds = _get_list(qas_dict.get("nlp_background"))
        topic_backgrounds = _get_list(qas_dict.get("topic_background"))
        paper_read_values = _get_list(qas_dict.get("paper_read"))
        search_queries = _get_list(qas_dict.get("search_query"))
        answers_by_question = _get_list(qas_dict.get("answers"))

        documents.append(
            DocumentRecord(
                doc_id=f"qasper-{paper_id}",
                contents=contents,
                metadata={
                    "dataset": "qasper",
                    "paper_id": paper_id,
                    "title": title,
                    "section_count": section_count,
                    "figure_and_table_count": figure_count,
                    "question_count": len(questions),
                },
            )
        )

        for idx, question_value in enumerate(questions):
            question = _clean_text(question_value)
            if not question:
                continue

            question_id = _clean_text(question_ids[idx] if idx < len(question_ids) else "")
            if not question_id:
                question_id = f"{paper_id}-{idx}"

            queries.append(
                QueryRecord(
                    query_id=f"q.qasper.{question_id}",
                    contents=question,
                    relevant=[f"qasper-{paper_id}"],
                    metadata=_build_query_metadata(
                        document_text=contents,
                        paper_id=paper_id,
                        title=title,
                        question_id=question_id,
                        question_writer=_clean_text(
                            question_writers[idx] if idx < len(question_writers) else ""
                        ),
                        nlp_background=_clean_text(
                            nlp_backgrounds[idx] if idx < len(nlp_backgrounds) else ""
                        ),
                        topic_background=_clean_text(
                            topic_backgrounds[idx] if idx < len(topic_backgrounds) else ""
                        ),
                        paper_read=_clean_text(
                            paper_read_values[idx] if idx < len(paper_read_values) else ""
                        ),
                        search_query=_clean_text(
                            search_queries[idx] if idx < len(search_queries) else ""
                        ),
                        raw_answers=answers_by_question[idx]
                        if idx < len(answers_by_question)
                        else [],
                    ),
                )
            )

            if limit is not None and len(queries) >= limit:
                return documents, queries

    return documents, queries
