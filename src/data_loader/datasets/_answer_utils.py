from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def split_text_answers(
    document_text: str,
    answers: Iterable[Any],
) -> tuple[List[str], List[str]]:
    """Split answers into exact document substrings vs free-text answers."""
    extractive: List[str] = []
    free_text: List[str] = []
    haystack = document_text or ""

    for raw_answer in answers:
        answer = clean_text(raw_answer)
        if not answer:
            continue
        if answer in haystack:
            extractive.append(answer)
        else:
            free_text.append(answer)

    return (
        dedupe_preserve_order(extractive),
        dedupe_preserve_order(free_text),
    )


def extract_span_texts(
    document_text: str,
    span_offsets: Iterable[Sequence[Any]],
) -> List[str]:
    """Extract text substrings from [start, end] offsets when valid."""
    if not isinstance(document_text, str) or not document_text:
        return []

    extracted: List[str] = []
    for item in span_offsets:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start, end = item
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or end > len(document_text):
            continue
        text = clean_text(document_text[start:end])
        if text:
            extracted.append(text)

    return dedupe_preserve_order(extracted)


def normalize_yes_no_answers(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value is True:
            out.append("yes")
            continue
        if value is False:
            out.append("no")
            continue
        text = clean_text(value).lower()
        if text in {"yes", "no"}:
            out.append(text)
    return dedupe_preserve_order(out)


def normalize_unanswerable_answers(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value is True:
            out.append("unanswerable")
            continue
        text = clean_text(value).lower()
        if text == "unanswerable":
            out.append(text)
    return dedupe_preserve_order(out)


def build_unified_answer_metadata(
    *,
    base_metadata: Mapping[str, Any],
    extractive_answers: Iterable[Any] = (),
    free_text_answers: Iterable[Any] = (),
) -> Dict[str, Any]:
    """Prepend unified answer fields, then preserve the rest of metadata."""
    metadata: Dict[str, Any] = {}

    extractive = dedupe_preserve_order(
        [clean_text(value) for value in extractive_answers if clean_text(value)]
    )
    free_text = dedupe_preserve_order(
        [clean_text(value) for value in free_text_answers if clean_text(value)]
    )

    if extractive:
        metadata["extractive_span_text_answer"] = extractive
    if free_text:
        metadata["free_text_answer"] = free_text

    metadata.update(dict(base_metadata))
    return metadata
