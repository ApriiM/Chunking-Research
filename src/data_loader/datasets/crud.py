from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.request
import zipfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.data_loader.core.registry import dataset
from src.data_loader.core.schemas import DocumentRecord, QueryRecord


_DEFAULT_QA_TASK_KEYS = (
    "questanswer_1doc",
    "questanswer_2docs",
    "questanswer_3docs",
)

_DOC_FIELDS_BY_TASK: Dict[str, Tuple[str, ...]] = {
    "questanswer_1doc": ("news1",),
    "questanswer_2docs": ("news1", "news2"),
    "questanswer_3docs": ("news1", "news2", "news3"),
}

_TASK_NAMES: Dict[str, str] = {
    "questanswer_1doc": "question answering 1-doc",
    "questanswer_2docs": "question answering 2-docs",
    "questanswer_3docs": "question answering 3-docs",
}

_SPLIT_ALIASES: Dict[str, str] = {
    "1doc": "questanswer_1doc",
    "2docs": "questanswer_2docs",
    "3docs": "questanswer_3docs",
}

_DEFAULT_DOWNLOAD_URL = (
    "https://github.com/IAAR-Shanghai/CRUD_RAG/raw/main/data/crud/merged.zip"
)


def _build_canonical_doc_id(contents: str) -> str:
    """Build a stable document id from normalized document text."""

    digest = hashlib.md5(contents.encode("utf-8")).hexdigest()[:16]
    return f"crud-doc-{digest}"


def _split_base_and_slice(split_expr: Optional[str]) -> Tuple[str, Optional[slice]]:
    """Parse split expressions like 'all[:200]' or 'questanswer_2docs[0:100]'."""

    if not split_expr:
        return "all", None

    split_expr = split_expr.strip()
    if "[" not in split_expr or not split_expr.endswith("]"):
        return split_expr, None

    base, bracket = split_expr.split("[", 1)
    base = base.strip()
    inner = bracket[:-1].strip()

    if ":" not in inner:
        return base, None

    left, right = inner.split(":", 1)
    start = int(left.strip()) if left.strip() else None
    stop = int(right.strip()) if right.strip() else None
    return base, slice(start, stop)


def _resolve_data_paths(
    merged_zip_path: Optional[str],
    merged_json_path: Optional[str],
    base_raw: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve CRUD zip/json paths from explicit path, BASE_RAW, or local defaults."""

    if merged_zip_path:
        return merged_zip_path, merged_json_path
    if merged_json_path:
        return None, merged_json_path

    base = base_raw or os.getenv("BASE_RAW")
    if base:
        return (
            os.path.join(base, "data", "crud", "merged.zip"),
            os.path.join(base, "data", "crud", "merged.json"),
        )

    return (
        os.path.join("data", "crud", "merged.zip"),
        os.path.join("data", "crud", "merged.json"),
    )


def _load_merged_payload(
    merged_zip_path: Optional[str],
    merged_json_path: Optional[str],
    json_member: str,
    download_if_missing: bool,
    download_url: str,
    download_timeout_seconds: int,
) -> Mapping[str, Any]:
    """Load CRUD payload from local zip/json, optionally auto-download from GitHub."""

    payload: Any

    if merged_zip_path and os.path.exists(merged_zip_path):
        with zipfile.ZipFile(merged_zip_path, "r") as zf:
            try:
                with zf.open(json_member) as fp:
                    payload = json.load(fp)
            except KeyError as exc:
                available = sorted(zf.namelist())
                raise KeyError(
                    f"JSON member '{json_member}' was not found in '{merged_zip_path}'. "
                    f"Available members: {available}"
                ) from exc
    elif merged_json_path and os.path.exists(merged_json_path):
        with open(merged_json_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    else:
        if not download_if_missing:
            raise FileNotFoundError(
                "CRUD data file not found.\n"
                f"Tried zip: '{merged_zip_path}'\n"
                f"Tried json: '{merged_json_path}'\n"
                "Pass loader kwargs merged_zip_path / merged_json_path / base_raw, "
                "or enable download_if_missing."
            )

        try:
            with urllib.request.urlopen(download_url, timeout=download_timeout_seconds) as resp:
                raw_download = resp.read()
        except Exception as exc:
            raise RuntimeError(
                "CRUD dataset is missing locally and automatic download failed.\n"
                f"URL: {download_url}\n"
                "Please download merged.json from GitHub and pass merged_json_path/merged_zip_path."
            ) from exc

        raw_json: Optional[bytes] = None
        try:
            maybe_payload = json.loads(raw_download.decode("utf-8"))
            if isinstance(maybe_payload, dict):
                payload = maybe_payload
                raw_json = raw_download
            else:
                raise ValueError("Downloaded JSON payload is not a dict.")
        except Exception:
            # If a zip was downloaded (e.g. merged.zip), extract merged.json from it.
            try:
                with zipfile.ZipFile(io.BytesIO(raw_download), "r") as zf:
                    member = None
                    if json_member in zf.namelist():
                        member = json_member
                    elif "merged.json" in zf.namelist():
                        member = "merged.json"
                    else:
                        for name in zf.namelist():
                            if name.lower().endswith("/merged.json") or name.lower() == "merged.json":
                                member = name
                                break
                    if member is None:
                        raise KeyError("merged.json not found in downloaded zip")

                    with zf.open(member) as fp:
                        raw_json = fp.read()
                payload = json.loads(raw_json.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Downloaded merged.json is not a dict.")
            except Exception as exc:
                raise RuntimeError(
                    "Downloaded CRUD artifact is neither valid JSON nor a ZIP containing merged.json.\n"
                    f"URL: {download_url}"
                ) from exc

        # Cache to whichever path is available so subsequent runs are local-only.
        if merged_json_path:
            os.makedirs(os.path.dirname(merged_json_path) or ".", exist_ok=True)
            with open(merged_json_path, "wb") as f:
                f.write(raw_json if raw_json is not None else raw_download)
        if merged_zip_path:
            os.makedirs(os.path.dirname(merged_zip_path) or ".", exist_ok=True)
            with zipfile.ZipFile(merged_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(json_member, raw_json if raw_json is not None else raw_download)
        print(
            f"Downloaded CRUD payload from {download_url} and cached to "
            f"zip='{merged_zip_path}' json='{merged_json_path}'"
        )

    if not isinstance(payload, dict):
        source = merged_zip_path if merged_zip_path and os.path.exists(merged_zip_path) else merged_json_path
        raise ValueError(f"Expected top-level dict in CRUD payload from '{source}', got {type(payload)}.")

    return payload


def _normalize_task_keys(task_keys: Optional[Sequence[str]]) -> List[str]:
    if task_keys is None:
        return list(_DEFAULT_QA_TASK_KEYS)

    if isinstance(task_keys, str):
        items = [x.strip() for x in task_keys.split(",")]
        return [x for x in items if x]

    return [str(x).strip() for x in task_keys if str(x).strip()]


def _resolve_selected_tasks(split_base: str, task_keys: Sequence[str]) -> List[str]:
    if split_base in ("", "all", "train", "test", "validation", "dev"):
        return list(task_keys)

    if split_base in _SPLIT_ALIASES:
        return [_SPLIT_ALIASES[split_base]]

    return [split_base]


@dataset("crud")
def load_crud(
    split: str = "all",
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    merged_zip_path: Optional[str] = None,
    merged_json_path: Optional[str] = None,
    base_raw: Optional[str] = None,
    json_member: str = "merged.json",
    task_keys: Optional[Sequence[str]] = None,
    download_if_missing: bool = True,
    download_url: str = _DEFAULT_DOWNLOAD_URL,
    download_timeout_seconds: int = 120,
) -> Tuple[List[DocumentRecord], List[QueryRecord]]:
    """Load CRUD-RAG QA tasks from local merged.zip into (documents, queries).

    Expected source shape:
    - merged.zip containing merged.json, or plain merged.json
    - merged.json has task lists keyed by e.g. questanswer_1doc/2docs/3docs

    If source files are missing locally and `download_if_missing=True` (default),
    loader downloads merged.json from `download_url`, then caches it to local json/zip paths.

    Split handling:
    - all/train/test/validation/dev -> all selected QA task keys
    - questanswer_1doc / questanswer_2docs / questanswer_3docs
    - aliases: 1doc, 2docs, 3docs
    - optional slicing: split='all[:200]' or split='questanswer_2docs[100:200]'
    """

    # Required by the shared loader signature; not used for local file-based loading.
    _ = cache_dir

    zip_path, json_path = _resolve_data_paths(
        merged_zip_path=merged_zip_path,
        merged_json_path=merged_json_path,
        base_raw=base_raw,
    )
    payload = _load_merged_payload(
        merged_zip_path=zip_path,
        merged_json_path=json_path,
        json_member=json_member,
        download_if_missing=download_if_missing,
        download_url=download_url,
        download_timeout_seconds=download_timeout_seconds,
    )

    split_base, split_slice = _split_base_and_slice(split)
    selected_task_keys = _resolve_selected_tasks(
        split_base=split_base, task_keys=_normalize_task_keys(task_keys)
    )

    for task_key in selected_task_keys:
        if task_key not in payload:
            available = sorted(payload.keys())
            raise ValueError(
                f"Task key '{task_key}' was not found in {json_member}. Available keys: {available}"
            )
        if task_key not in _DOC_FIELDS_BY_TASK:
            raise ValueError(
                f"Task key '{task_key}' is not a QA task with news documents. "
                f"Supported QA task keys: {sorted(_DOC_FIELDS_BY_TASK.keys())}"
            )

    samples: List[Tuple[str, int, Mapping[str, Any]]] = []
    for task_key in selected_task_keys:
        rows = payload.get(task_key)
        if not isinstance(rows, list):
            raise ValueError(f"Expected list payload for task '{task_key}', got {type(rows)}.")
        for row_idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            if "questions" not in row or "answers" not in row:
                continue
            samples.append((task_key, row_idx, row))

    if split_slice is not None:
        samples = samples[split_slice]
    if limit is not None:
        samples = samples[: min(limit, len(samples))]

    documents: List[DocumentRecord] = []
    queries: List[QueryRecord] = []
    content_to_doc_id: Dict[str, str] = {}
    doc_index_by_id: Dict[str, int] = {}
    doc_id_to_contents: Dict[str, str] = {}

    for task_key, row_idx, row in samples:
        row_id = str(row.get("ID", row_idx))
        task_name = _TASK_NAMES.get(task_key, task_key)
        question = str(row.get("questions", "") or "").strip()
        if not question:
            continue

        relevant_doc_ids: List[str] = []
        for news_index, news_field in enumerate(_DOC_FIELDS_BY_TASK[task_key], start=1):
            contents = str(row.get(news_field, "") or "").strip()
            if not contents:
                continue

            doc_id = content_to_doc_id.get(contents)
            if doc_id is None:
                doc_id = _build_canonical_doc_id(contents)
                # Guard against rare hash collisions by adding a deterministic suffix.
                if doc_id in doc_id_to_contents and doc_id_to_contents[doc_id] != contents:
                    suffix = 1
                    candidate = f"{doc_id}-{suffix}"
                    while candidate in doc_id_to_contents and doc_id_to_contents[candidate] != contents:
                        suffix += 1
                        candidate = f"{doc_id}-{suffix}"
                    doc_id = candidate

                content_to_doc_id[contents] = doc_id
                doc_id_to_contents[doc_id] = contents

                documents.append(
                    DocumentRecord(
                        doc_id=doc_id,
                        contents=contents,
                        metadata={
                            "dataset": "crud",
                            "original_task_key": task_key,
                            "original_task_name": task_name,
                            "source_id": row_id,
                            "source_index": row_idx,
                            "source_task_key": task_key,
                            "source_task_name": task_name,
                            "source_field": news_field,
                            "event": row.get("event"),
                            "occurrences": 1,
                        },
                    )
                )
                doc_index_by_id[doc_id] = len(documents) - 1
            else:
                # Keep provenance lightweight while preserving uniqueness of documents.
                doc_idx = doc_index_by_id[doc_id]
                meta = documents[doc_idx].metadata
                meta["occurrences"] = int(meta.get("occurrences", 1)) + 1

            if doc_id not in relevant_doc_ids:
                relevant_doc_ids.append(doc_id)

        if not relevant_doc_ids:
            continue

        answer = row.get("answers")
        raw_metadata = {k: v for k, v in row.items() if k not in {"questions", "answers"}}

        queries.append(
            QueryRecord(
                query_id=f"q.crud.{task_key}.{row_id}.r{row_idx}",
                contents=question,
                relevant=relevant_doc_ids,
                metadata={
                    "dataset": "crud",
                    "answer": answer,
                    "original_task_key": task_key,
                    "original_task_name": task_name,
                    "source_task_key": task_key,
                    "source_task_name": task_name,
                    "source_id": row_id,
                    "source_index": row_idx,
                    "raw_metadata": raw_metadata,
                },
            )
        )

    return documents, queries
