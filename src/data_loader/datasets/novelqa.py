from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.data_loader.core.registry import dataset
from src.data_loader.core.schemas import DocumentRecord, QueryRecord


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Subdirectory names that may hold QA JSON files inside the cloned repo.
_QA_SUBDIRS: Tuple[str, ...] = (
    os.path.join("Data", "PublicDomain"),
    os.path.join("Data", "CopyrightProtected"),
    "Data",
    "Demonstration",
)

#: Subdirectory names that may hold novel TXT files.
_BOOK_SUBDIRS: Tuple[str, ...] = (
    os.path.join("Books", "PublicDomain"),
    os.path.join("Books", "CopyrightProtected"),
    "Books",
)

#: Default Hugging Face repository path.
_DEFAULT_HF_REPO = "NovelQA/NovelQA"

#: Split aliases – mirror the CRUD pattern.
_SPLIT_ALIASES: Dict[str, str] = {
    "public": "PublicDomain",
    "copyright": "CopyrightProtected",
    "all": "all",
}

#: Field names used in each QA record.
_QA_FIELD_QID = "QID"
_QA_FIELD_QUESTION = "Question"
_QA_FIELD_OPTIONS = "Options"
_QA_FIELD_ASPECT = "Aspect"
_QA_FIELD_COMPLEXITY = "Complexity"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_canonical_doc_id(contents: str) -> str:
    """Build a stable document id from a hash of the novel text."""
    digest = hashlib.md5(contents.encode("utf-8")).hexdigest()[:16]
    return f"novelqa-doc-{digest}"


def _split_base_and_slice(split_expr: Optional[str]) -> Tuple[str, Optional[slice]]:
    """Parse split expressions such as 'all[:100]' or 'public[0:50]'.

    Mirrors the same helper in the CRUD loader so callers have a consistent
    interface across datasets.
    """
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


def _resolve_repo_path(
    repo_path: Optional[str],
    base_raw: Optional[str],
) -> Path:
    """Determine where the cloned NovelQA repository lives on disk."""
    if repo_path:
        return Path(repo_path)

    base = base_raw or os.getenv("BASE_RAW")
    if base:
        return Path(base) / "data" / "novelqa" / "NovelQA"

    downloads_dir = os.getenv("DOWNLOADS_DIR", "downloads")
    return Path(downloads_dir) / "NovelQA"


def _clone_repository(
    target_path: Path,
    hf_token: Optional[str],
    hf_repo: str,
) -> None:
    """Clone the NovelQA HF dataset repository into *target_path*.

    Removes an existing clone first (same behaviour as the original
    ``download_novelQA`` utility).
    """
    if target_path.exists():
        print(f"[novelqa] Removing existing repository at: {target_path}")
        shutil.rmtree(target_path)

    token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "A Hugging Face token is required to clone NovelQA.\n"
            "Set the HF_TOKEN environment variable or pass hf_token= to the loader."
        )

    # Build authenticated git URL.
    repo_url = f"https://user:{token}@huggingface.co/datasets/{hf_repo}"
    print(f"[novelqa] Cloning {hf_repo} → {target_path} …")
    subprocess.run(
        ["git", "clone", repo_url, str(target_path)],
        check=True,
    )
    print("[novelqa] Clone finished.")


def _find_files_in_subdirs(
    base: Path,
    subdirs: Sequence[str],
    extension: str,
    domain_filter: Optional[str] = None,
) -> List[Path]:
    """Collect files with *extension* from candidate subdirectories under *base*.

    *domain_filter* is an optional substring the parent directory name must
    contain (e.g. ``'PublicDomain'`` or ``'CopyrightProtected'``).
    """
    found: List[Path] = []
    for subdir in subdirs:
        candidate = base / subdir
        if not candidate.is_dir():
            continue
        if domain_filter and domain_filter not in str(candidate):
            continue
        for fp in sorted(candidate.iterdir()):
            if fp.suffix.lower() == extension:
                found.append(fp)
    return found


def _load_novelqa_data(
    resolved_repo_path: Path,
    download_if_missing: bool,
    hf_token: Optional[str],
    hf_repo: str,
    domain_filter: Optional[str],
) -> Tuple[Dict[str, str], Dict[str, List[dict]]]:
    """Load novel texts and QA pairs from the cloned repository.

    Returns:
        book_texts  – mapping of ``book_id -> full novel text``
        book_qa     – mapping of ``book_id -> list of QA dicts``
    """
    if not resolved_repo_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"NovelQA repository not found at '{resolved_repo_path}'.\n"
                "Pass repo_path=, base_raw=, or enable download_if_missing=True."
            )
        _clone_repository(resolved_repo_path, hf_token=hf_token, hf_repo=hf_repo)

    # --- QA files -----------------------------------------------------------
    qa_files = _find_files_in_subdirs(
        resolved_repo_path, _QA_SUBDIRS, ".json", domain_filter
    )
    if not qa_files:
        raise FileNotFoundError(
            f"No QA JSON files found in '{resolved_repo_path}' under {list(_QA_SUBDIRS)}.\n"
            "Is the repository properly cloned?"
        )

    book_qa: Dict[str, List[dict]] = {}
    for qa_path in qa_files:
        book_id = qa_path.stem  # filename without extension = book id
        try:
            with open(qa_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"[novelqa] WARNING – could not parse {qa_path}: {exc}")
            continue
        if isinstance(raw, list):
            book_qa[book_id] = raw
        elif isinstance(raw, dict):
            # Some files wrap the list: {"data": [...]}
            for v in raw.values():
                if isinstance(v, list):
                    book_qa[book_id] = v
                    break
                elif isinstance(v, dict):
                    book_qa[book_id] = raw
        else:
            print(f"[novelqa] WARNING – unexpected QA format in {qa_path}, skipping.")

    # --- Book text files ----------------------------------------------------
    book_files = _find_files_in_subdirs(
        resolved_repo_path, _BOOK_SUBDIRS, ".txt", domain_filter
    )

    book_texts: Dict[str, str] = {}
    for book_path in book_files:
        book_id = book_path.stem
        if book_id not in book_qa:
            # Skip novels that have no QA file (would never become a document).
            continue
        try:
            text = book_path.read_text(encoding="utf-8")
            book_texts[book_id] = text.strip()
        except OSError as exc:
            print(f"[novelqa] WARNING – could not read {book_path}: {exc}")

    return book_texts, book_qa


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


@dataset("novelqa")
def load_novelqa(
    split: str = "all",
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    # --- path resolution (mirrors CRUD kwargs) ---
    repo_path: Optional[str] = None,
    base_raw: Optional[str] = None,
    # --- download control ---
    download_if_missing: bool = True,
    hf_token: Optional[str] = None,
    hf_repo: str = _DEFAULT_HF_REPO,
    # --- filtering ---
    book_ids: Optional[Sequence[str]] = None,
    revision: Optional[str] = "",
) -> Tuple[List[DocumentRecord], List[QueryRecord]]:
    """Load the NovelQA benchmark into ``(documents, queries)``.

    Repository structure assumed on disk::

        NovelQA/
        ├── Books/
        │   └── PublicDomain/
        │       └── <book_id>.txt
        ├── Data/
        │   ├── PublicDomain/
        │   │   └── <book_id>.json
        │   └── CopyrightProtected/
        │       └── <book_id>.json
        └── bookmeta.json

    Each QA JSON is a list of dicts with at minimum::

        {
            "QID": "Q0148",
            "Question": "How many times …",
            "Options": {"A": "11", "B": "9", "C": "12", "D": "10"},
            "Aspect": "times",
            "Complexity": "mh"
        }

    One ``DocumentRecord`` is created **per unique novel text**.  Questions
    about that novel become ``QueryRecord`` instances with ``relevant`` pointing
    to the novel's ``doc_id``.

    Args:
        split:
            Controls which books to load.  Recognised values:

            * ``"all"`` / ``"train"`` / ``"test"`` / ``"validation"`` –
              load all available books.
            * ``"public"`` – only ``PublicDomain`` books.
            * ``"copyright"`` – only ``CopyrightProtected`` books.
            * Slicing suffix supported: ``"all[:200]"``, ``"public[0:50]"``.

        cache_dir:
            Unused; kept for API compatibility with the shared loader signature.
        limit:
            Hard cap on the total number of queries returned.
        repo_path:
            Explicit path to the cloned NovelQA directory.  When absent the
            loader falls back to ``BASE_RAW`` env-var and then
            ``DOWNLOADS_DIR/NovelQA``.
        base_raw:
            Root for raw datasets (mirrors CRUD's ``base_raw`` kwarg).
        download_if_missing:
            When ``True`` (default) automatically git-clones the repository if
            the local path does not exist.
        hf_token:
            Hugging Face access token.  Falls back to ``HF_TOKEN`` env-var.
        hf_repo:
            HF dataset repository slug, default ``"NovelQA/NovelQA"``.
        book_ids:
            Optional allowlist of book IDs (filename stems).  When provided
            only those books are loaded.

    Returns:
        ``(documents, queries)`` – lists of ``DocumentRecord`` /
        ``QueryRecord`` instances.
    """
    _ = cache_dir  # unused – kept for shared loader API compatibility

    # --- resolve split ------------------------------------------------------
    split_base, split_slice = _split_base_and_slice(split)

    domain_filter: Optional[str] = None
    normalised = split_base.lower()
    if normalised in ("public",):
        domain_filter = "PublicDomain"
    elif normalised in ("copyright",):
        domain_filter = "CopyrightProtected"
    # else: "all" / "train" / "test" / "validation" → no domain filter

    # --- load raw data ------------------------------------------------------
    resolved = _resolve_repo_path(repo_path=repo_path, base_raw=base_raw)
    book_texts, book_qa = _load_novelqa_data(
        resolved_repo_path=resolved,
        download_if_missing=download_if_missing,
        hf_token=hf_token,
        hf_repo=hf_repo,
        domain_filter=domain_filter,
    )

    # --- optional book allowlist --------------------------------------------
    if book_ids is not None:
        allowed = set(book_ids)
        book_qa = {k: v for k, v in book_qa.items() if k in allowed}
        book_texts = {k: v for k, v in book_texts.items() if k in allowed}

    # --- build flat list of (book_id, qa_dict) samples ---------------------
    samples: List[Tuple[str, dict]] = []
    for book_id in sorted(book_qa.keys()):
        for qid, qa in book_qa[book_id].items():
            if not isinstance(qa, dict):
                continue
            if not qa.get(_QA_FIELD_QUESTION):
                continue
            qa[_QA_FIELD_QID] = qid
            samples.append((book_id, qa))

    # Apply slice then limit (mirrors CRUD pattern exactly).
    if split_slice is not None:
        samples = samples[split_slice]
    if limit is not None:
        samples = samples[: min(limit, len(samples))]

    # --- deduplication tracking --------------------------------------------
    content_to_doc_id: Dict[str, str] = {}
    doc_id_to_contents: Dict[str, str] = {}
    doc_index_by_id: Dict[str, int] = {}

    documents: List[DocumentRecord] = []
    queries: List[QueryRecord] = []

    for book_id, qa in samples:
        # ---- document (novel text) ----------------------------------------
        contents = book_texts.get(book_id, "")
        # Books in the CopyrightProtected split have no publicly distributed
        # text; we still create a stub document so the query remains linkable.
        if not contents:
            contents = f"[NovelQA] Text not publicly available for book '{book_id}'."

        doc_id = content_to_doc_id.get(contents)
        if doc_id is None:
            doc_id = _build_canonical_doc_id(contents)

            # Collision guard (mirrors CRUD).
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
                        "dataset": "novelqa",
                        "book_id": book_id,
                        "has_full_text": bool(book_texts.get(book_id)),
                        "occurrences": 1,
                    },
                )
            )
            doc_index_by_id[doc_id] = len(documents) - 1
        else:
            doc_idx = doc_index_by_id[doc_id]
            meta = documents[doc_idx].metadata
            meta["occurrences"] = int(meta.get("occurrences", 1)) + 1

        # ---- query ---------------------------------------------------------
        qid = str(qa.get(_QA_FIELD_QID, ""))
        question = str(qa.get(_QA_FIELD_QUESTION, "")).strip()
        options: Optional[Dict[str, str]] = qa.get(_QA_FIELD_OPTIONS)
        aspect = qa.get(_QA_FIELD_ASPECT)
        complexity = qa.get(_QA_FIELD_COMPLEXITY) or qa.get("Complex")

        queries.append(
            QueryRecord(
                query_id=f"q.novelqa.{book_id}.{qid}",
                contents=question,
                relevant=[doc_id],
                metadata={
                    "dataset": "novelqa",
                    "qid": qid,
                    "book_id": book_id,
                    "options": options,
                    "aspect": aspect,
                    "complexity": complexity,
                    # Store the raw QA record for downstream use.
                    "raw_metadata": {k: v for k, v in qa.items()},
                },
            )
        )

    return documents, queries