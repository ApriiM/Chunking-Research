"""
Chunker Analyzer Backend
========================
Serves data from pirb_data/, results/, retrieved_documents/, and processed/ directories.

Configuration via environment variables:
  DATA_ROOT      - root dir containing pirb/ and processed/ subdirs (default: ../data)
  PORT           - port to listen on (default: 8000)

Directory layout assumed:
  <DATA_ROOT>/pirb/pirb_data/<exp>/passages/passage.json
  <DATA_ROOT>/pirb/pirb_data/<exp>/queries/queries.json
  <DATA_ROOT>/pirb/pirb_data/<exp>/metadata.json
  <DATA_ROOT>/pirb/results/<exp>           (JSON file, no extension)
  <DATA_ROOT>/pirb/retrieved_documents/<exp>.jsonl
  <DATA_ROOT>/processed/<dataset_slug>/documents/documents.jsonl
"""

import os
import json
import glob
from pathlib import Path
from flask import Flask, jsonify, abort, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data"))
PIRB_ROOT = DATA_ROOT / "pirb"
PROCESSED_ROOT = DATA_ROOT / "processed"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def get_all_experiments() -> list[str]:
    """
    Return only experiments that have BOTH pirb_data and results.
    """
    data_base = PIRB_ROOT / "pirb_data"
    results_base = PIRB_ROOT / "results"

    if not data_base.exists() or not results_base.exists():
        return []

    # eksperymenty z pirb_data
    data_exps = {d.name for d in data_base.iterdir() if d.is_dir()}

    # eksperymenty z results (uwzględniamy pliki bez rozszerzenia i .json)
    results_exps = set()
    for p in results_base.iterdir():
        if p.is_file():
            name = p.name.replace(".json", "")
            results_exps.add(name)
    # tylko wspólne
    valid_exps = data_exps & results_exps
    return sorted(valid_exps)


def get_experiment_metadata(exp: str) -> dict:
    path = PIRB_ROOT / "pirb_data" / exp / "metadata.json"
    if not path.exists():
        return {}
    return load_json(path)


def get_results(exp: str) -> dict:
    """Try results/<exp> (no ext) or results/<exp>.json"""
    for candidate in [
        PIRB_ROOT / "results" / exp,
        PIRB_ROOT / "results" / f"{exp}.json",
    ]:
        if candidate.exists():
            return load_json(candidate)
    return {}


def get_passages(exp: str) -> dict:
    """Returns {id: passage_dict}"""
    path = PIRB_ROOT / "pirb_data" / exp / "passages" / "passages.jsonl"
    if not path.exists():
        return {}
    data = load_jsonl(path)
    if isinstance(data, list):
        return {p["id"]: p for p in data}
    return data  # already a dict keyed by id


def get_queries(exp: str) -> list:
    path = PIRB_ROOT / "pirb_data" / exp / "queries" / "queries.json"
    if not path.exists():
        return []
    data = load_json(path)
    if isinstance(data, list):
        return data
    return list(data.values())

def get_queries_jsonl(exp: str) -> list:
    path = PIRB_ROOT / "pirb_data" / exp / "queries" / "queries.jsonl"    
    if not path.exists():
        return []
    data = load_jsonl(path)
    if isinstance(data, list):
        return {p["id"]: p for p in data}
    return data


def get_processed_query_meta(dataset_slug: str) -> dict:
    """
    Load extra per-query metadata (e.g. aspect, complexity) from
    processed/<dataset_slug>/queries/queries.jsonl.
    Returns {query_id: metadata_dict}.
    """
    path = PROCESSED_ROOT / dataset_slug / "queries" / "queries.jsonl"
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    result = {}
    for row in rows:
        qid = row.get("id")
        if qid is not None:
            result[str(qid)] = row.get("metadata") or {}
    return result


def get_retrieved(exp: str) -> list:
    path = PIRB_ROOT / "retrieved_documents" / f"{exp}.jsonl"
    if not path.exists():
        return []
    return load_jsonl(path)


def get_documents(dataset_slug: str) -> dict:
    """Returns {id: contents}"""
    path = PROCESSED_ROOT / dataset_slug / "documents" / "documents.jsonl"
    if not path.exists():
        return {}
    docs = load_jsonl(path)
    return {d["id"]: d["contents"] for d in docs}


def group_experiments_by_dataset() -> dict[str, list[str]]:
    """Returns {dataset_slug: [exp1, exp2, ...]}"""
    groups: dict[str, list[str]] = {}
    for exp in get_all_experiments():
        meta = get_experiment_metadata(exp)
        slug = meta.get("dataset_slug", "unknown")
        groups.setdefault(slug, []).append(exp)
    return groups


def get_paired_dataset(dataset_slug: str) -> str | None:
    """
    Return the paired dataset slug for a given slug, or None if no pair exists.

    Pairing logic (symmetric):
      - "<base>_merged"  ↔  "<base>"
      - "<base>"         ↔  "<base>_merged"

    Works for any prefix, e.g.:
      gutenqa_all_merged        ↔  gutenqa_all
      triviaqa_..._merged       ↔  triviaqa_...
    """
    if dataset_slug.endswith("_merged"):
        return dataset_slug[: -len("_merged")]
    else:
        return dataset_slug + "_merged"


def build_paired_query_map(partner_slug: str, exps_partner: list[str]) -> dict:
    """
    Build {query_id: {exp_name_without_slug: retrieved_relevant}}
    for the partner dataset.

    We key by chunker_name (not exp id) so the current dataset can match
    across different exp ids that share the same chunker.
    """
    result: dict = {}  # query_id → {chunker_name: retrieved_relevant}

    for exp in exps_partner:
        meta = get_experiment_metadata(exp)
        chunker_name = meta.get("chunker_name", exp)
        retrieved_list = get_retrieved(exp)

        for item in retrieved_list:
            qid = item["id"]
            if qid not in result:
                result[qid] = {}
            result[qid][chunker_name] = item.get("retrieved_relevant", False)

    return result


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/api/datasets")
def list_datasets():
    """List all dataset slugs with their experiments."""
    groups = group_experiments_by_dataset()
    result = []
    for slug, exps in sorted(groups.items()):
        chunkers = []
        for exp in exps:
            meta = get_experiment_metadata(exp)
            chunkers.append({
                "exp": exp,
                "chunker_name": meta.get("chunker_name", exp),
                "chunk_count": meta.get("chunk_count"),
                "query_count": meta.get("query_count"),
                "document_count": meta.get("document_count"),
            })
        result.append({"dataset_slug": slug, "chunkers": chunkers})
    return jsonify(result)


@app.get("/api/datasets/<dataset_slug>/metrics")
def dataset_metrics(dataset_slug: str):
    """Return metrics for all experiments of a given dataset."""
    groups = group_experiments_by_dataset()
    exps = groups.get(dataset_slug)
    if exps is None:
        abort(404, f"Dataset '{dataset_slug}' not found")

    out = []
    for exp in exps:
        meta = get_experiment_metadata(exp)
        results = get_results(exp)
        out.append({
            "exp": exp,
            "chunker_name": meta.get("chunker_name", exp),
            "metrics": results.get("metrics", {}),
            "metrics_random": results.get("metrics_random", {}),
            "chunk_count": meta.get("chunk_count"),
            "query_count": meta.get("query_count"),
        })
    return jsonify(out)


@app.get("/api/datasets/<dataset_slug>/queries")
def dataset_queries(dataset_slug: str):
    """
    Return all queries for every experiment of the dataset,
    merged so each query appears once with per-chunker data:
      - retrieved_relevant flag
      - retrieved chunk ids + scores
      - relevant: list of relevant chunk ids for THIS experiment
      - relevant_count, chunk_count, relevant_pct
    """
    groups = group_experiments_by_dataset()
    exps = groups.get(dataset_slug)
    if exps is None:
        abort(404)

    # Pre-load chunk counts from metadata
    chunk_counts: dict[str, int] = {}
    for exp in exps:
        meta = get_experiment_metadata(exp)
        chunk_counts[exp] = meta.get("chunk_count", 0)

    # Load extra query metadata from processed/ (aspect, complexity, etc.)
    extra_meta = get_processed_query_meta(dataset_slug)

    queries_map: dict = {}

    for exp in exps:
        meta = get_experiment_metadata(exp)
        chunker_name = meta.get("chunker_name", exp)
        total_chunks = chunk_counts.get(exp, 0)
        retrieved_list = get_retrieved(exp)

        for item in retrieved_list:
            qid = item["id"]
            qid_str = str(qid)
            if qid not in queries_map:
                item_meta = item.get("metadata") or {}
                processed_meta = extra_meta.get(qid_str, {})
                queries_map[qid] = {
                    "id": qid,
                    "contents": item.get("contents", ""),
                    "free_text_answer": item_meta.get("free_text_answer", ""),
                    # extra fields from processed queries (aspect, complexity, etc.)
                    "extra_meta": processed_meta,
                    "chunkers": {},
                }

            relevant_ids = item.get("relevant") or []
            relevant_count = len(relevant_ids)
            relevant_pct = round(relevant_count / total_chunks * 100, 4) if total_chunks else None

            queries_map[qid]["chunkers"][exp] = {
                "chunker_name": chunker_name,
                "retrieved_relevant": item.get("retrieved_relevant", False),
                "retrieved": item.get("retrieved", []),
                "retrieved_scores": item.get("retrieved_scores", []),
                "relevant": relevant_ids,
                "relevant_count": relevant_count,
                "chunk_count": total_chunks,
                "relevant_pct": relevant_pct,
            }

    # Also fill from queries.json for exps that may have no retrieved_documents
    for exp in exps:
        meta = get_experiment_metadata(exp)
        chunker_name = meta.get("chunker_name", exp)
        total_chunks = chunk_counts.get(exp, 0)

        for q in get_queries(exp):
            qid = q["id"]
            qid_str = str(qid)
            if qid not in queries_map:
                q_meta = q.get("metadata") or {}
                processed_meta = extra_meta.get(qid_str, {})
                queries_map[qid] = {
                    "id": qid,
                    "contents": q.get("contents", ""),
                    "free_text_answer": q_meta.get("free_text_answer", ""),
                    "extra_meta": processed_meta,
                    "chunkers": {},
                }
            # fill chunker entry if not already populated from retrieved_documents
            if exp not in queries_map[qid]["chunkers"]:
                relevant_ids = q.get("relevant") or []
                relevant_count = len(relevant_ids)
                relevant_pct = round(relevant_count / total_chunks * 100, 4) if total_chunks else None
                queries_map[qid]["chunkers"][exp] = {
                    "chunker_name": chunker_name,
                    "retrieved_relevant": False,
                    "retrieved": [],
                    "retrieved_scores": [],
                    "relevant": relevant_ids,
                    "relevant_count": relevant_count,
                    "chunk_count": total_chunks,
                    "relevant_pct": relevant_pct,
                }

    return jsonify(sorted(queries_map.values(), key=lambda q: q["id"]))


@app.get("/api/experiments/<exp>/query/<query_id>")
def query_detail(exp: str, query_id: str):
    """
    Full detail for one query in one experiment:
    retrieved chunks with scores and their text.
    """
    retrieved_list = get_retrieved(exp)
    passages = get_passages(exp)

    item = next((r for r in retrieved_list if r["id"] == query_id), None)
    if item is None:
        # Fall back to queries.json
        queries = get_queries(exp)
        item = next((q for q in queries if q["id"] == query_id), None)
    if item is None:
        abort(404, f"Query '{query_id}' not found in experiment '{exp}'")

    retrieved_ids = item.get("retrieved", [])
    scores = item.get("retrieved_scores", [])
    relevant_set = set(item.get("relevant", []))

    chunks = []
    for i, chunk_id in enumerate(retrieved_ids):
        passage = passages.get(chunk_id, {})
        score = scores[i] if i < len(scores) else None
        chunks.append({
            "rank": i + 1,
            "id": chunk_id,
            "score": score,
            "is_relevant": chunk_id in relevant_set,
            "contents": passage.get("contents", ""),
            "parent_id": (passage.get("metadata") or {}).get("parentId"),
            "original_id": (passage.get("metadata") or {}).get("original_id"),
        })

    meta = get_experiment_metadata(exp)

    return jsonify({
        "query_id": query_id,
        "exp": exp,
        "chunker_name": meta.get("chunker_name", exp),
        "dataset_slug": meta.get("dataset_slug", ""),
        "contents": item.get("contents", ""),
        "relevant": item.get("relevant", []),
        "free_text_answer": (item.get("metadata") or {}).get("free_text_answer", ""),
        "retrieved_relevant": item.get("retrieved_relevant", False),
        "chunks": chunks,
    })


DOCUMENT_PREVIEW_CHARS = 3000


@app.get("/api/chunks/<exp>/<chunk_id>")
def get_chunk(exp: str, chunk_id: str):
    """
    Return a chunk with a truncated document preview (first 3000 chars).
    Full document text is NOT included here — use the separate endpoints below.
    """
    passages = get_passages(exp)
    passage = passages.get(chunk_id)
    if passage is None:
        abort(404, f"Chunk '{chunk_id}' not found in experiment '{exp}'")

    meta_p = passage.get("metadata") or {}
    parent_id = meta_p.get("parentId")

    exp_meta = get_experiment_metadata(exp)
    dataset_slug = exp_meta.get("dataset_slug", "")

    document_preview = None
    document_total_len = None
    if parent_id and dataset_slug:
        docs = get_documents(dataset_slug)
        full = docs.get(parent_id)
        if full:
            document_preview = full[:DOCUMENT_PREVIEW_CHARS]
            document_total_len = len(full)

    return jsonify({
        "id": chunk_id,
        "contents": passage.get("contents", ""),
        "parent_id": parent_id,
        "original_id": meta_p.get("original_id"),
        "document_preview": document_preview,
        "document_total_len": document_total_len,
        "document_truncated": document_total_len is not None and document_total_len > DOCUMENT_PREVIEW_CHARS,
        "dataset_slug": dataset_slug,
    })


@app.get("/api/chunks/<exp>/<chunk_id>/full-document")
def get_chunk_full_document(exp: str, chunk_id: str):
    """Return the full parent document text for a chunk (lazy-loaded)."""
    passages = get_passages(exp)
    passage = passages.get(chunk_id)
    if passage is None:
        abort(404, f"Chunk '{chunk_id}' not found in experiment '{exp}'")

    meta_p = passage.get("metadata") or {}
    parent_id = meta_p.get("parentId")
    if not parent_id:
        abort(404, "Chunk has no parent document")

    exp_meta = get_experiment_metadata(exp)
    dataset_slug = exp_meta.get("dataset_slug", "")
    docs = get_documents(dataset_slug)
    full = docs.get(parent_id)
    if full is None:
        abort(404, f"Document '{parent_id}' not found in dataset '{dataset_slug}'")

    return jsonify({
        "parent_id": parent_id,
        "dataset_slug": dataset_slug,
        "contents": full,
    })


@app.get("/api/chunks/<exp>/<chunk_id>/relevant-queries")
def get_chunk_relevant_queries(exp: str, chunk_id: str):
    """
    Return all queries in this experiment that list chunk_id in their 'relevant' field.
    Lazy-loaded — not included in the main chunk response.
    """
    queries = get_queries_jsonl(exp)
    exp_meta = get_experiment_metadata(exp)
    dataset_slug = exp_meta.get("dataset_slug", "")
    extra_meta = get_processed_query_meta(dataset_slug)

    matching = []
    for q in queries.values() if isinstance(queries, dict) else queries:
        if chunk_id in (q.get("relevant") or []):
            qid = q["id"]
            q_meta = q.get("metadata") or {}
            processed = extra_meta.get(str(qid), {})
            matching.append({
                "id": qid,
                "contents": q.get("contents", ""),
                "free_text_answer": q_meta.get("free_text_answer", ""),
                "extra_meta": processed,
            })

    return jsonify({
        "chunk_id": chunk_id,
        "exp": exp,
        "chunker_name": exp_meta.get("chunker_name", exp),
        "count": len(matching),
        "queries": matching,
    })


@app.get("/api/documents/<dataset_slug>/<doc_id>")
def get_document(dataset_slug: str, doc_id: str):
    """Return a full document."""
    docs = get_documents(dataset_slug)
    text = docs.get(doc_id)
    if text is None:
        abort(404, f"Document '{doc_id}' not found in dataset '{dataset_slug}'")
    return jsonify({"id": doc_id, "contents": text, "dataset_slug": dataset_slug})


@app.get("/api/datasets/<dataset_slug>/pair-info")
def dataset_pair_info(dataset_slug: str):
    """
    If this dataset has a _merged partner (or is itself the _merged),
    return the partner slug and per-query per-chunker retrieved_relevant map
    from the partner dataset.

    Response:
    {
      "partner_slug": "gutenqa_all",          // null if no partner
      "partner_queries": {                     // null if no partner
        "<query_id>": {
          "<chunker_name>": true|false,
          ...
        },
        ...
      }
    }
    """
    groups = group_experiments_by_dataset()

    partner_slug = get_paired_dataset(dataset_slug)
    partner_exps = groups.get(partner_slug)

    if partner_exps is None:
        return jsonify({"partner_slug": None, "partner_queries": None})

    partner_map = build_paired_query_map(partner_slug, partner_exps)
    return jsonify({"partner_slug": partner_slug, "partner_queries": partner_map})


SIMILARITY_URL = os.environ.get("SIMILARITY_URL", "http://localhost:5051/similarity")


@app.post("/api/similarity")
def proxy_similarity():
    """
    Proxy to the external similarity/reranker service.
    Accepts: { "query": str, "documents": [str, ...] }
    Returns the raw response from the similarity service.
    """
    import urllib.request
    import urllib.error

    body = request.get_json(force=True)
    if not body or "query" not in body or "documents" not in body:
        abort(400, "Body must contain 'query' and 'documents'")

    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        SIMILARITY_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return jsonify(json.loads(resp.read().decode("utf-8")))
    except urllib.error.HTTPError as e:
        abort(e.code, f"Similarity service error: {e.reason}")
    except urllib.error.URLError as e:
        abort(502, f"Cannot reach similarity service at {SIMILARITY_URL}: {e.reason}")


@app.get("/api/experiments/<exp>/query/<query_id>/relevant-chunk-texts")
def get_relevant_chunk_texts(exp: str, query_id: str):
    """
    Return the text contents of all relevant chunks for a given query+experiment.
    Used by the similarity page to populate documents for reranking.
    """
    passages = get_passages(exp)

    # Find query from retrieved_documents first, fall back to queries.json
    retrieved_list = get_retrieved(exp)
    item = next((r for r in retrieved_list if str(r["id"]) == str(query_id)), None)
    if item is None:
        queries = get_queries(exp)
        item = next((q for q in queries if str(q["id"]) == str(query_id)), None)
    if item is None:
        abort(404, f"Query '{query_id}' not found in experiment '{exp}'")

    relevant_ids = item.get("relevant") or []
    retrieved_ids = item.get("retrieved") or []
    retrieved_scores = item.get("retrieved_scores") or []

    # Build a score map from retrieval results
    retrieval_score_map = {rid: retrieved_scores[i] if i < len(retrieved_scores) else None
                           for i, rid in enumerate(retrieved_ids)}

    chunks = []
    for chunk_id in relevant_ids:
        passage = passages.get(chunk_id, {})
        chunks.append({
            "id": chunk_id,
            "contents": passage.get("contents", ""),
            "retrieval_score": retrieval_score_map.get(chunk_id),
            "was_retrieved": chunk_id in retrieval_score_map,
        })

    exp_meta = get_experiment_metadata(exp)
    return jsonify({
        "query_id": query_id,
        "query_text": item.get("contents", ""),
        "exp": exp,
        "chunker_name": exp_meta.get("chunker_name", exp),
        "chunks": chunks,
    })


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "data_root": str(DATA_ROOT)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Chunker Analyzer backend on port {port}")
    print(f"Data root: {DATA_ROOT.resolve()}")
    app.run(host="0.0.0.0", port=port, debug=True)
