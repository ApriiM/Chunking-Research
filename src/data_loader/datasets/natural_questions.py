from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

from src.data_loader.core.registry import dataset
from src.data_loader.core.schemas import DocumentRecord, QueryRecord


# ============================================================================
# Split helpers
# ============================================================================
def _split_base_and_slice(split_expr: str) -> Tuple[str, Optional[slice]]:
    """Parse strings like 'test[:100]' or 'all[100:200]'."""
    if "[" not in split_expr or not split_expr.endswith("]"):
        return split_expr, None
    base, bracket = split_expr.split("[", 1)
    base = base.strip()
    inner = bracket[:-1].strip()
    if ":" not in inner:
        return base, None
    left, right = inner.split(":", 1)
    left = left.strip()
    right = right.strip()
    start = int(left) if left else None
    stop = int(right) if right else None
    return base, slice(start, stop)


def _normalize_base_split(base_split: str) -> str:
    """
    BEIR NQ is not a classic train/dev/test QA split. It’s a single query/qrels set.
    We accept common names but normalize all to 'all'.
    """
    normalized = (base_split or "").strip().lower()
    if normalized in {"", "all", "train", "test", "validation", "dev"}:
        return "all"
    raise ValueError(
        "BEIR NQ has a single query/qrels partition. "
        f"Unsupported split base '{base_split}'. Use split like 'test[:100]' or 'all[:100]'."
    )


def _progress_iter(items: Iterable[Any], *, enabled: bool, desc: str):
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except Exception:
        return items
    return tqdm(items, desc=desc)


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


# ============================================================================
# Wikipedia fetch + HTML->text
# ============================================================================
class _HTMLToText(HTMLParser):
    """Tiny HTML -> text converter good enough for wiki parsed HTML."""
    def __init__(self):
        super().__init__()
        self._chunks: List[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        if tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"):
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "div", "li", "tr"):
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        text = (data or "").strip()
        if text:
            self._chunks.append(text + " ")

    def text(self) -> str:
        out = "".join(self._chunks)
        out = "\n".join(line.strip() for line in out.splitlines())
        out = "\n".join(line for line in out.splitlines() if line)
        return out.strip()


def _wiki_cache_paths(cache_dir: str, title: str) -> Tuple[str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.md5(title.encode("utf-8")).hexdigest()
    return (
        os.path.join(cache_dir, f"{digest}.json"),  # raw API response
        os.path.join(cache_dir, f"{digest}.txt"),   # extracted plain text
    )


def _fetch_wikipedia_article_text(
    title: str,
    *,
    cache_dir: str,
    language: str,
    user_agent: str,
    timeout_seconds: int,
    sleep_seconds: float,
    use_cache: bool,
    retries: int = 2,
) -> str:
    """
    Fetch current Wikipedia page text via MediaWiki parse API.
    Note: this is NOT guaranteed to match the BEIR snapshot.
    """
    raw_path, txt_path = _wiki_cache_paths(cache_dir, title)

    if use_cache and os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    api = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }
    url = api + "?" + urllib.parse.urlencode(params)

    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))

            if use_cache:
                with open(raw_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)

            html = ""
            try:
                html = payload["parse"]["text"]
            except Exception:
                return ""

            parser = _HTMLToText()
            parser.feed(html)
            text = parser.text()

            if use_cache:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)

            return text

        except Exception as exc:
            last_exc = exc
            # very small backoff
            time.sleep(0.25 * (attempt + 1))

    # If everything failed
    _ = last_exc
    return ""


# ============================================================================
# Loader: BEIR NQ queries/qrels + Wikipedia article docs (scraped)
# ============================================================================
@dataset("natural_questions")
@dataset("NaturalQuestions")
@dataset("beir_nq_wiki_articles")
def load_beir_nq_wiki_articles(
    split: str = "test",
    cache_dir: Optional[str] = None,  # kept for signature compatibility
    limit: Optional[int] = None,
    dataset_id: str = "beir/nq",
    doc_id_prefix: str = "wiki-",
    min_rel_score: float = 1.0,
    show_progress: bool = True,
    # Wikipedia fetch options
    wikipedia_language: str = "en",
    wikipedia_cache_dir: str = "data/wiki_cache",
    wikipedia_user_agent: str = "chunking-research/0.1 (contact: you@example.com)",
    wikipedia_timeout_seconds: int = 30,
    wikipedia_sleep_seconds: float = 0.0,
    wikipedia_use_cache: bool = True,
    wikipedia_retries: int = 2,
    skip_wikipedia_fetch: bool = False,
    # Evidence options (BEIR has relevance labels + passages, not gold QA answers)
    max_evidence_passages_per_query: int = 0,  # 0 => keep ALL
    # If scraping fails, optionally build "article" by concatenating BEIR passages for that title:
    fallback_to_beir_passage_concat: bool = True,
) -> Tuple[List[DocumentRecord], List[QueryRecord]]:
    """
    What this loader guarantees:
      - Query text comes from the BEIR NQ benchmark.
      - Relevance labels come from BEIR qrels.
      - Documents are full Wikipedia articles fetched online (current content),
        attached as DocumentRecords (one per Wikipedia title).
      - Query metadata includes:
          * qrels_article_scores: aggregated relevance per article-doc (max over its passages)
          * qrels_passage_scores: original passage-level qrels
          * evidence_passages: the full BEIR passage texts (no truncation)
    Important honesty:
      - BEIR NQ does NOT ship Natural Questions “short/long answers”.
        So there is no gold answer string/span to attach here—only relevance + evidence.
    """
    _ = cache_dir

    try:
        import ir_datasets
    except Exception as exc:
        raise ImportError(
            "This loader requires `ir_datasets`.\nInstall with: pip install ir-datasets"
        ) from exc

    base_split, qid_slice = _split_base_and_slice(split)
    _normalize_base_split(base_split)

    ds = ir_datasets.load(dataset_id)

    # ------------------------------------------------------------------------
    # 1) Read all qrels (build qrels_by_qid and set of qids that have qrels)
    # ------------------------------------------------------------------------
    qrels_by_qid: DefaultDict[str, List[Tuple[str, float]]] = defaultdict(list)
    qids_in_qrels: Set[str] = set()

    for qrel in _progress_iter(ds.qrels_iter(), enabled=show_progress, desc="Reading qrels"):
        qid = str(getattr(qrel, "query_id", ""))
        did = str(getattr(qrel, "doc_id", ""))
        if not qid or not did:
            continue
        score = float(getattr(qrel, "relevance", 0.0))
        if score < float(min_rel_score):
            continue
        qids_in_qrels.add(qid)
        qrels_by_qid[qid].append((did, score))

    # ------------------------------------------------------------------------
    # 2) Read queries in dataset order -> stable qid_order, plus query text map
    # ------------------------------------------------------------------------
    qid_order: List[str] = []
    query_text_by_id: Dict[str, str] = {}

    for query in _progress_iter(ds.queries_iter(), enabled=show_progress, desc="Reading queries"):
        # Safe attribute reads
        qid_val = getattr(query, "query_id", None)
        if qid_val is None:
            continue
        qid = str(qid_val)

        if qid not in qids_in_qrels:
            continue

        if qid not in query_text_by_id:
            text_val = getattr(query, "text", "")
            query_text_by_id[qid] = str(text_val or "")
            qid_order.append(qid)

    # Apply slice/limit on stable query order
    if qid_slice is not None:
        qid_order = qid_order[qid_slice]
    if limit is not None:
        qid_order = qid_order[: min(limit, len(qid_order))]

    selected_qids = set(qid_order)

    # ------------------------------------------------------------------------
    # 3) Determine needed BEIR passage ids for selected queries
    # ------------------------------------------------------------------------
    needed_passage_ids: Set[str] = set()
    for qid in qid_order:
        for did, _ in qrels_by_qid.get(qid, []):
            needed_passage_ids.add(str(did))

    # ------------------------------------------------------------------------
    # 4) Map passage doc_id -> (title, passage text) using docstore when possible
    # ------------------------------------------------------------------------
    try:
        docstore = ds.docs_store()
    except Exception:
        docstore = None

    passage_id_to_title: Dict[str, str] = {}
    passage_id_to_text: Dict[str, str] = {}
    title_to_passages: DefaultDict[str, List[str]] = defaultdict(list)
    title_to_passage_ids: DefaultDict[str, List[str]] = defaultdict(list)

    if docstore is not None:
        for did in _progress_iter(sorted(needed_passage_ids), enabled=show_progress, desc="Reading passage docs"):
            doc = docstore.get(did)
            if doc is None:
                continue
            title = str(getattr(doc, "title", "") or "").strip()
            text = str(getattr(doc, "text", "") or "").strip()
            if not title:
                continue
            did_s = str(did)
            passage_id_to_title[did_s] = title
            if text:
                passage_id_to_text[did_s] = text
                title_to_passages[title].append(text)
                title_to_passage_ids[title].append(did_s)
    else:
        # Slow fallback: scan corpus
        for doc in _progress_iter(ds.docs_iter(), enabled=show_progress, desc="Scanning passage corpus"):
            did_s = str(getattr(doc, "doc_id", "") or "")
            if did_s not in needed_passage_ids:
                continue
            title = str(getattr(doc, "title", "") or "").strip()
            text = str(getattr(doc, "text", "") or "").strip()
            if not title:
                continue
            passage_id_to_title[did_s] = title
            if text:
                passage_id_to_text[did_s] = text
                title_to_passages[title].append(text)
                title_to_passage_ids[title].append(did_s)

    needed_titles = set(title_to_passages.keys()) | set(passage_id_to_title.values())

    # ------------------------------------------------------------------------
    # 5) Fetch Wikipedia pages and build DocumentRecords (one per title)
    # ------------------------------------------------------------------------
    title_to_doc_id: Dict[str, str] = {}
    documents: List[DocumentRecord] = []

    for title in _progress_iter(sorted(needed_titles), enabled=show_progress, desc="Fetching Wikipedia articles"):
        source = "wikipedia_api_parse"
        article_text = ""

        if skip_wikipedia_fetch:
            source = "blank_stub"
        else:
            article_text = _fetch_wikipedia_article_text(
                title,
                cache_dir=wikipedia_cache_dir,
                language=wikipedia_language,
                user_agent=wikipedia_user_agent,
                timeout_seconds=wikipedia_timeout_seconds,
                sleep_seconds=wikipedia_sleep_seconds,
                use_cache=wikipedia_use_cache,
                retries=wikipedia_retries,
            )

            if not article_text and fallback_to_beir_passage_concat:
                # Guaranteed to be consistent with the BEIR corpus, but not a real full wiki page.
                article_text = "\n\n".join(title_to_passages.get(title, []))
                source = "beir_passage_concat"

            if not article_text:
                continue

        digest = hashlib.md5(title.encode("utf-8")).hexdigest()[:16]
        doc_id = f"{doc_id_prefix}{digest}"
        title_to_doc_id[title] = doc_id

        contents = "" if skip_wikipedia_fetch else f"{title}\n\n{article_text}".strip()
        documents.append(
            DocumentRecord(
                doc_id=doc_id,
                contents=contents,
                metadata={
                    "dataset": "beir_nq",
                    "title": title,
                    "language": wikipedia_language,
                    "source": source,
                    "passage_ids": title_to_passage_ids.get(title, []),
                },
            )
        )

    included_titles = set(title_to_doc_id.keys())

    # ------------------------------------------------------------------------
    # 6) Build QueryRecords:
    #     - relevant = wiki article doc_ids, aggregated from passage qrels
    #     - metadata includes evidence passages (NO truncation)
    # ------------------------------------------------------------------------
    queries: List[QueryRecord] = []

    for qid in qid_order:
        passage_pairs = sorted(qrels_by_qid.get(qid, []), key=lambda x: x[1], reverse=True)

        qrels_passage_scores: Dict[str, float] = {}
        qrels_article_scores: Dict[str, float] = {}
        evidence_passages: List[str] = []
        evidence_passage_ids: List[str] = []

        for passage_id, score in passage_pairs:
            pid = str(passage_id)
            qrels_passage_scores[pid] = float(score)

            title = passage_id_to_title.get(pid)
            if not title or title not in included_titles:
                continue

            article_doc_id = title_to_doc_id.get(title)
            if not article_doc_id:
                continue

            # Aggregate per-article relevance (max over its passages)
            prev = qrels_article_scores.get(article_doc_id, 0.0)
            qrels_article_scores[article_doc_id] = max(prev, float(score))

            # Evidence passages are the BEIR passages (no truncation)
            ptext = passage_id_to_text.get(pid, "").strip()
            if ptext:
                evidence_passages.append(ptext)
                evidence_passage_ids.append(pid)

            if max_evidence_passages_per_query > 0 and len(evidence_passages) >= max_evidence_passages_per_query:
                break

        relevant_article_ids = sorted(qrels_article_scores.keys(), key=lambda d: qrels_article_scores[d], reverse=True)

        # Skip queries that can't be linked to any fetched/created article doc
        if not relevant_article_ids:
            continue

        queries.append(
            QueryRecord(
                query_id=f"q.{qid}",
                contents=query_text_by_id.get(qid, ""),
                relevant=relevant_article_ids,
                metadata={
                    "dataset": "beir_nq",
                    "note": (
                        "BEIR NQ provides relevance judgments (qrels), not NaturalQuestions gold answer strings/spans. "
                        "Use qrels_* and evidence_* for retrieval evaluation; use original NQ JSONL for QA answers."
                    ),
                    "qrels_article_scores": qrels_article_scores,
                    "qrels_passage_scores": qrels_passage_scores,
                    "evidence_passage_ids": _dedupe_preserve_order(evidence_passage_ids),
                    "evidence_passages": _dedupe_preserve_order(evidence_passages),
                },
            )
        )

    return documents, queries
