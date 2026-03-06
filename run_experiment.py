import argparse
import copy
import itertools
import json
import multiprocessing as mp
import os
import re
import shutil
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from src.chunking import get_chunker
from src.data_loader.core.schemas import (
    PassageRecord,
    load_document_records_jsonl,
    load_query_records_jsonl,
    save_passage_records_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run chunking experiment sweeps with timing and structured outputs "
            "(documents, queries, passages.jsonl, metadata.json)."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config for experiment runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only expand and print planned runs without executing chunking",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help=(
            "Resume latest session under output_root and rerun only runs that are "
            "missing or previously failed; successful runs are skipped."
        ),
    )
    return parser.parse_args()


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must decode to a mapping")
    return data


def _coerce_bool(value: Any, field_name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "on", "y"}:
            return True
        if lowered in {"false", "no", "0", "off", "n"}:
            return False
    raise ValueError(f"{field_name} must be boolean-like (got {value!r})")


def _coerce_optional_positive_float(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric (got {value!r})") from exc
    if numeric <= 0:
        raise ValueError(f"{field_name} must be > 0 (got {numeric!r})")
    return numeric


def _coerce_positive_int(value: Any, field_name: str, default: int) -> int:
    if value is None:
        return default
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer (got {value!r})") from exc
    if numeric <= 0:
        raise ValueError(f"{field_name} must be > 0 (got {numeric!r})")
    return numeric


def _as_list(value: Any, field_name: str) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("._")
    return slug or "run"


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_slug_from_documents_path(documents_path: str) -> str:
    p = Path(documents_path)
    if p.name == "documents.jsonl" and p.parent.name == "documents":
        split_name = p.parent.parent.name
        dataset_name = p.parent.parent.parent.name if p.parent.parent.parent.name else "dataset"
        return _sanitize_slug(f"{dataset_name}_{split_name}")
    return _sanitize_slug(p.stem)


def _default_queries_path(documents_path: str) -> str:
    p = Path(documents_path)
    if p.name == "documents.jsonl" and p.parent.name == "documents":
        return str(p.parent.parent / "queries" / "queries.jsonl")
    return str(p.with_name("queries.jsonl"))


def _expand_param_grid(param_grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    value_lists: List[List[Any]] = []
    for k in keys:
        raw = param_grid[k]
        values = raw if isinstance(raw, list) else [raw]
        if not values:
            raise ValueError(f"param_grid key '{k}' has an empty values list")
        value_lists.append(values)

    combos: List[Dict[str, Any]] = []
    for product_values in itertools.product(*value_lists):
        combos.append({k: v for k, v in zip(keys, product_values)})
    return combos


def _resolve_runs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Config must include non-empty runs: [ ... ]")
    global_timeout_minutes = _coerce_optional_positive_float(
        config.get("max_chunking_minutes"),
        "max_chunking_minutes",
    )
    global_batch_size_docs = _coerce_positive_int(
        config.get("chunking_batch_size_documents"),
        "chunking_batch_size_documents",
        default=1,
    )

    resolved: List[Dict[str, Any]] = []
    for run_idx, run_cfg in enumerate(runs, start=1):
        if not isinstance(run_cfg, dict):
            raise ValueError(f"runs[{run_idx - 1}] must be a mapping")

        name = str(run_cfg.get("name") or f"run_{run_idx}")
        chunker_names = _as_list(
            run_cfg.get("chunker_names", run_cfg.get("chunker_name")),
            f"runs[{run_idx - 1}].chunker_name",
        )
        if not chunker_names:
            raise ValueError(f"runs[{run_idx - 1}] must include chunker_name or chunker_names")

        documents_paths = _as_list(
            run_cfg.get("documents_paths", run_cfg.get("documents_path")),
            f"runs[{run_idx - 1}].documents_path",
        )
        if not documents_paths:
            raise ValueError(f"runs[{run_idx - 1}] must include documents_path or documents_paths")

        base_params = run_cfg.get("chunker_params", {}) or {}
        if not isinstance(base_params, dict):
            raise ValueError(f"runs[{run_idx - 1}].chunker_params must be a mapping")

        param_grid = run_cfg.get("param_grid", {}) or {}
        if not isinstance(param_grid, dict):
            raise ValueError(f"runs[{run_idx - 1}].param_grid must be a mapping")
        param_combos = _expand_param_grid(param_grid)

        repeats = int(run_cfg.get("repeats", 1))
        if repeats <= 0:
            raise ValueError(f"runs[{run_idx - 1}].repeats must be positive")
        timeout_minutes = _coerce_optional_positive_float(
            run_cfg.get("max_chunking_minutes", global_timeout_minutes),
            f"runs[{run_idx - 1}].max_chunking_minutes",
        )
        batch_size_docs = _coerce_positive_int(
            run_cfg.get("chunking_batch_size_documents", global_batch_size_docs),
            f"runs[{run_idx - 1}].chunking_batch_size_documents",
            default=global_batch_size_docs,
        )
        if timeout_minutes is None:
            batch_size_docs = 0

        copy_inputs = _coerce_bool(run_cfg.get("copy_inputs"), f"runs[{run_idx - 1}].copy_inputs", default=True)
        queries_override = run_cfg.get("queries_path")

        for docs_path in documents_paths:
            docs_path_str = str(docs_path)
            qpath = str(queries_override) if queries_override else _default_queries_path(docs_path_str)
            for chunker_name in chunker_names:
                for combo_idx, combo in enumerate(param_combos, start=1):
                    merged = dict(base_params)
                    merged.update(combo)
                    for rep in range(1, repeats + 1):
                        resolved.append(
                            {
                                "group_name": name,
                                "documents_path": docs_path_str,
                                "queries_path": qpath,
                                "chunker_name": str(chunker_name),
                                "chunker_params": merged,
                                "combo_index": combo_idx,
                                "repetition": rep,
                                "copy_inputs": copy_inputs,
                                "max_chunking_minutes": timeout_minutes,
                                "chunking_batch_size_documents": batch_size_docs,
                            }
                        )
    return resolved


def _copy_if_exists(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _prepare_output_root(path: str, overwrite: bool) -> str:
    root = os.path.abspath(path)
    if os.path.exists(root):
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {root}. "
                "Set overwrite: true or pick a new output_root."
            )
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)
    return root


def _run_dir(session_dir: str, run_number: int) -> str:
    return os.path.join(session_dir, f"run_{run_number:04d}")


def _list_session_dirs(output_root: str) -> List[str]:
    if not os.path.exists(output_root):
        return []
    dirs = []
    for name in os.listdir(output_root):
        candidate = os.path.join(output_root, name)
        if os.path.isdir(candidate) and name.startswith("session_"):
            dirs.append(candidate)
    return sorted(dirs)


def _latest_session_dir(output_root: str) -> Optional[str]:
    sessions = _list_session_dirs(output_root)
    if not sessions:
        return None
    return sessions[-1]


def _load_existing_run_metadata(session_dir: str, run_number: int) -> Optional[Dict[str, Any]]:
    meta_path = os.path.join(_run_dir(session_dir, run_number), "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class ChunkingTimeoutError(TimeoutError):
    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: float,
        chunking_seconds: float,
        partial_chunk_count: int,
        processed_document_count: int,
        document_count: int,
    ) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.chunking_seconds = chunking_seconds
        self.partial_chunk_count = partial_chunk_count
        self.processed_document_count = processed_document_count
        self.document_count = document_count


class ChunkingWorkerTimeoutError(TimeoutError):
    def __init__(self, message: str, *, processed_document_count: int) -> None:
        super().__init__(message)
        self.processed_document_count = processed_document_count


class _ProgressTrackingDocuments:
    def __init__(self, documents: Sequence[str], counter: Any) -> None:
        self._documents = documents
        self._counter = counter

    def __len__(self) -> int:
        return len(self._documents)

    def __getitem__(self, index: Any) -> Any:
        return self._documents[index]

    def __iter__(self):
        for doc in self._documents:
            with self._counter.get_lock():
                self._counter.value += 1
            yield doc


def _split_text_worker(
    chunker_name: str,
    chunker_params: Dict[str, Any],
    documents: List[str],
    documents_meta: List[Dict[str, Any]],
    result_path: str,
    progress_counter: Any,
) -> None:
    try:
        chunker = get_chunker(chunker_name, chunker_params)
        tracked_documents = _ProgressTrackingDocuments(documents, progress_counter)
        chunks = chunker.split_text(tracked_documents, documents_meta=documents_meta)
        payload = [
            {
                "text": chunk.text,
                "chunk_id": str(chunk.chunk_id),
                "metadata": dict(chunk.metadata or {}),
            }
            for chunk in chunks
        ]
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"ok": True, "chunks": payload}, f, ensure_ascii=False)
    except Exception as exc:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "error_traceback": traceback.format_exc(),
                },
                f,
                ensure_ascii=False,
            )


def _split_text_with_timeout(
    chunker_name: str,
    chunker_params: Dict[str, Any],
    documents: List[str],
    documents_meta: List[Dict[str, Any]],
    timeout_seconds: float,
) -> Tuple[List[Dict[str, Any]], int]:
    ctx = mp.get_context()
    progress_counter = ctx.Value("i", 0)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".chunker_result.json",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        result_path = tmp.name
    proc = ctx.Process(
        target=_split_text_worker,
        args=(
            chunker_name,
            chunker_params,
            documents,
            documents_meta,
            result_path,
            progress_counter,
        ),
    )
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
        if os.path.exists(result_path):
            os.remove(result_path)
        raise ChunkingWorkerTimeoutError(
            f"Chunking subprocess exceeded timeout ({timeout_seconds:.3f}s)",
            processed_document_count=int(progress_counter.value),
        )

    if not os.path.exists(result_path):
        raise RuntimeError(
            f"Chunking subprocess exited without result payload (exitcode={proc.exitcode})"
        )

    with open(result_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    os.remove(result_path)
    if not payload.get("ok"):
        err_type = payload.get("error_type", "RuntimeError")
        err_msg = payload.get("error_message", "Unknown worker error")
        err_tb = payload.get("error_traceback", "")
        raise RuntimeError(f"{err_type}: {err_msg}\n{err_tb}")

    return payload.get("chunks", []), int(progress_counter.value)


def _payload_to_passages(chunk_payload: Sequence[Dict[str, Any]]) -> List[PassageRecord]:
    passages: List[PassageRecord] = []
    for chunk in chunk_payload:
        meta = dict(chunk.get("metadata") or {})
        parent_id = str(meta.get("doc_id", ""))
        passages.append(
            PassageRecord(
                passage_id=str(chunk.get("chunk_id", "")),
                contents=str(chunk.get("text", "")),
                parent_id=parent_id,
                metadata=meta,
            )
        )
    return passages


def _run_one(
    run_number: int,
    total_runs: int,
    plan: Dict[str, Any],
    run_dir: str,
    docs_cache: Dict[str, Tuple[List[str], List[Dict[str, Any]], int]],
    queries_count_cache: Dict[str, Optional[int]],
) -> Dict[str, Any]:
    overall_start = perf_counter()
    started_at = _iso_utc_now()

    documents_path = plan["documents_path"]
    queries_path = plan["queries_path"]
    chunker_name = plan["chunker_name"]
    chunker_params = copy.deepcopy(plan["chunker_params"])
    max_chunking_minutes = plan.get("max_chunking_minutes")
    timeout_seconds = (
        float(max_chunking_minutes) * 60.0
        if max_chunking_minutes is not None
        else None
    )
    timeout_mode = "in_process" if timeout_seconds is None else "per_run_subprocess"

    # Cache document parsing to reduce repeated I/O for sweeps/repeats.
    if documents_path not in docs_cache:
        doc_records = load_document_records_jsonl(documents_path)
        texts = [doc.contents for doc in doc_records]
        metas = [{"doc_id": doc.doc_id, **(doc.metadata or {})} for doc in doc_records]
        docs_cache[documents_path] = (texts, metas, len(doc_records))
    base_texts, base_metas, document_count = docs_cache[documents_path]
    document_texts = list(base_texts)
    documents_meta = [dict(m) for m in base_metas]

    if queries_path not in queries_count_cache:
        if os.path.exists(queries_path):
            queries_count_cache[queries_path] = len(load_query_records_jsonl(queries_path))
        else:
            queries_count_cache[queries_path] = None
    query_count = queries_count_cache[queries_path]

    print(
        f"[{run_number}/{total_runs}] "
        f"group={plan['group_name']} chunker={chunker_name} "
        f"dataset={_dataset_slug_from_documents_path(documents_path)} "
        f"combo={plan['combo_index']} rep={plan['repetition']}"
        f"{'' if max_chunking_minutes is None else f' timeout_min={max_chunking_minutes}'}"
    )

    passages_path = os.path.join(run_dir, "passages.jsonl")
    passages: List[PassageRecord] = []
    processed_document_count = 0

    if timeout_seconds is None:
        init_start = perf_counter()
        chunker = get_chunker(chunker_name, chunker_params)
        init_seconds = perf_counter() - init_start

        chunking_start = perf_counter()
        chunks = chunker.split_text(document_texts, documents_meta=documents_meta)
        chunking_seconds = perf_counter() - chunking_start
        for chunk in chunks:
            meta = dict(chunk.metadata or {})
            parent_id = str(meta.get("doc_id", ""))
            passages.append(
                PassageRecord(
                    passage_id=str(chunk.chunk_id),
                    contents=chunk.text,
                    parent_id=parent_id,
                    metadata=meta,
                )
            )
        processed_document_count = document_count
    else:
        init_seconds = 0.0
        chunking_start = perf_counter()
        try:
            payload, processed_document_count = _split_text_with_timeout(
                chunker_name=chunker_name,
                chunker_params=chunker_params,
                documents=document_texts,
                documents_meta=documents_meta,
                timeout_seconds=timeout_seconds,
            )
        except ChunkingWorkerTimeoutError as exc:
            chunking_seconds = perf_counter() - chunking_start
            raise ChunkingTimeoutError(
                (
                    f"Chunking timed out after {timeout_seconds:.3f}s "
                    f"(processed_documents={exc.processed_document_count}/{document_count}, "
                    "partial_chunks=0)"
                ),
                timeout_seconds=timeout_seconds,
                chunking_seconds=chunking_seconds,
                partial_chunk_count=0,
                processed_document_count=int(exc.processed_document_count),
                document_count=document_count,
            ) from exc

        passages.extend(_payload_to_passages(payload))
        chunking_seconds = perf_counter() - chunking_start

    save_passage_records_jsonl(passages, passages_path)

    copied_documents = False
    copied_queries = False
    if plan["copy_inputs"]:
        copied_documents = _copy_if_exists(
            documents_path,
            os.path.join(run_dir, "documents", "documents.jsonl"),
        )
        copied_queries = _copy_if_exists(
            queries_path,
            os.path.join(run_dir, "queries", "queries.jsonl"),
        )

    finished_at = _iso_utc_now()
    total_runtime_seconds = perf_counter() - overall_start
    metadata = {
        "status": "success",
        "run_number": run_number,
        "total_runs": total_runs,
        "group_name": plan["group_name"],
        "dataset_slug": _dataset_slug_from_documents_path(documents_path),
        "documents_source_path": documents_path,
        "queries_source_path": queries_path,
        "copied_documents": copied_documents,
        "copied_queries": copied_queries,
        "chunker_name": chunker_name,
        "chunker_params": chunker_params,
        "combo_index": plan["combo_index"],
        "repetition": plan["repetition"],
        "output_dir": run_dir,
        "passages_path": passages_path,
        "metadata_path": os.path.join(run_dir, "metadata.json"),
        "document_count": document_count,
        "query_count": query_count,
        "chunk_count": len(passages),
        "processed_document_count": processed_document_count,
        "max_chunking_minutes": max_chunking_minutes,
        "timeout_mode": timeout_mode,
        "chunking_batch_size_documents": None,
        "timing": {
            "chunker_init_seconds": init_seconds,
            "chunking_seconds": chunking_seconds,
            "total_runtime_seconds": total_runtime_seconds,
        },
        "started_at": started_at,
        "finished_at": finished_at,
    }
    _write_json(metadata["metadata_path"], metadata)

    print(
        f"  -> chunks={len(passages)} "
        f"chunking_seconds={chunking_seconds:.3f} "
        f"total_seconds={total_runtime_seconds:.3f}"
    )
    return metadata


def _write_manifest(session_dir: str, entries: Sequence[Dict[str, Any]]) -> str:
    manifest_path = os.path.join(session_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in entries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest_path


def _write_quick_log(session_dir: str, entries: Sequence[Dict[str, Any]]) -> str:
    path = os.path.join(session_dir, "run_results.log")
    success_count = sum(1 for e in entries if e.get("status") == "success")
    failed_count = sum(1 for e in entries if e.get("status") == "failed")

    lines: List[str] = []
    lines.append(f"Run Results ({_iso_utc_now()})")
    lines.append(f"Total: {len(entries)}  Success: {success_count}  Failed: {failed_count}")
    lines.append("")
    for row in entries:
        status = row.get("status", "unknown")
        icon = "🟢" if status == "success" else "🔴"
        run_label = f"run_{int(row.get('run_number', 0)):04d}"
        base = (
            f"{icon} {run_label} {status.upper()} "
            f"group={row.get('group_name')} "
            f"chunker={row.get('chunker_name')} "
            f"dataset={row.get('dataset_slug')}"
        )
        if status == "success":
            timing = row.get("timing") or {}
            lines.append(
                f"{base} chunks={row.get('chunk_count')} "
                f"chunking_s={timing.get('chunking_seconds', 'n/a')}"
            )
        else:
            err_type = row.get("error_type", "Error")
            err_msg = row.get("error_message", "")
            partial = ""
            if row.get("chunk_count") is not None:
                partial = (
                    f" partial_chunks={row.get('chunk_count')}"
                    f" processed_docs={row.get('processed_document_count')}"
                )
            lines.append(f"{base}{partial} {err_type}: {err_msg}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    args = parse_args()
    cfg = _load_yaml(args.config)

    output_root = cfg.get("output_root")
    if not output_root:
        raise ValueError("Config must include output_root")
    overwrite = _coerce_bool(cfg.get("overwrite"), "overwrite", default=False)
    continue_on_error = _coerce_bool(cfg.get("continue_on_error"), "continue_on_error", default=True)
    resolved_runs = _resolve_runs(cfg)

    print(f"Expanded {len(resolved_runs)} concrete run(s)")
    for idx, plan in enumerate(resolved_runs, start=1):
        timeout_label = plan.get("max_chunking_minutes")
        timeout_suffix = (
            f" timeout_min={timeout_label}"
            if timeout_label is not None
            else ""
        )
        print(
            f"  [{idx}] group={plan['group_name']} "
            f"chunker={plan['chunker_name']} "
            f"documents={plan['documents_path']} "
            f"combo={plan['combo_index']} rep={plan['repetition']}"
            f"{timeout_suffix}"
        )

    if args.dry_run:
        print("Dry-run requested; no chunking executed.")
        return

    output_root_abs = os.path.abspath(output_root)
    session_dir: str
    if args.rerun_failed:
        os.makedirs(output_root_abs, exist_ok=True)
        latest = _latest_session_dir(output_root_abs)
        if latest is None:
            session_dir = os.path.join(output_root_abs, f"session_{_timestamp_utc()}")
            os.makedirs(session_dir, exist_ok=False)
            print(f"No existing session found. Created new session: {session_dir}")
        else:
            session_dir = latest
            print(f"Resuming latest session: {session_dir}")
    else:
        session_root = _prepare_output_root(output_root_abs, overwrite=overwrite)
        session_dir = os.path.join(session_root, f"session_{_timestamp_utc()}")
        os.makedirs(session_dir, exist_ok=False)

    with open(os.path.join(session_dir, "config.used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    docs_cache: Dict[str, Tuple[List[str], List[Dict[str, Any]], int]] = {}
    queries_count_cache: Dict[str, Optional[int]] = {}
    manifest_rows: List[Dict[str, Any]] = []

    batch_start = perf_counter()
    for run_number, plan in enumerate(resolved_runs, start=1):
        run_dir = _run_dir(session_dir, run_number)
        if args.rerun_failed:
            existing = _load_existing_run_metadata(session_dir, run_number)
            if existing and existing.get("status") == "success":
                print(
                    f"[{run_number}/{len(resolved_runs)}] "
                    f"SKIP already successful run_{run_number:04d}"
                )
                manifest_rows.append(existing)
                continue

        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        os.makedirs(run_dir, exist_ok=False)
        run_started_at = _iso_utc_now()
        run_start = perf_counter()
        try:
            row = _run_one(
                run_number=run_number,
                total_runs=len(resolved_runs),
                plan=plan,
                run_dir=run_dir,
                docs_cache=docs_cache,
                queries_count_cache=queries_count_cache,
            )
            manifest_rows.append(row)
        except Exception as exc:
            tb = traceback.format_exc()
            documents_path = str(plan.get("documents_path", ""))
            queries_path = str(plan.get("queries_path", ""))

            copied_documents = False
            copied_queries = False
            if plan.get("copy_inputs"):
                copied_documents = _copy_if_exists(
                    documents_path,
                    os.path.join(run_dir, "documents", "documents.jsonl"),
                )
                copied_queries = _copy_if_exists(
                    queries_path,
                    os.path.join(run_dir, "queries", "queries.jsonl"),
                )

            failed_row = {
                "status": "failed",
                "run_number": run_number,
                "total_runs": len(resolved_runs),
                "group_name": plan.get("group_name"),
                "dataset_slug": _dataset_slug_from_documents_path(documents_path) if documents_path else "unknown",
                "documents_source_path": documents_path,
                "queries_source_path": queries_path,
                "copied_documents": copied_documents,
                "copied_queries": copied_queries,
                "chunker_name": plan.get("chunker_name"),
                "chunker_params": plan.get("chunker_params", {}),
                "combo_index": plan.get("combo_index"),
                "repetition": plan.get("repetition"),
                "output_dir": run_dir,
                "passages_path": os.path.join(run_dir, "passages.jsonl"),
                "metadata_path": os.path.join(run_dir, "metadata.json"),
                "document_count": None,
                "query_count": queries_count_cache.get(queries_path),
                "chunk_count": None,
                "processed_document_count": None,
                "max_chunking_minutes": plan.get("max_chunking_minutes"),
                "timeout_mode": (
                    "in_process"
                    if plan.get("max_chunking_minutes") is None
                    else "per_run_subprocess"
                ),
                "chunking_batch_size_documents": None,
                "timing": {
                    "chunker_init_seconds": None,
                    "chunking_seconds": None,
                    "total_runtime_seconds": perf_counter() - run_start,
                },
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_traceback": tb,
                "started_at": run_started_at,
                "finished_at": _iso_utc_now(),
            }
            if documents_path in docs_cache:
                failed_row["document_count"] = docs_cache[documents_path][2]
            if isinstance(exc, ChunkingTimeoutError):
                failed_row["chunk_count"] = int(exc.partial_chunk_count)
                failed_row["processed_document_count"] = int(exc.processed_document_count)
                failed_row["document_count"] = int(exc.document_count)
                failed_row["timing"]["chunking_seconds"] = float(exc.chunking_seconds)
            _write_json(failed_row["metadata_path"], failed_row)
            with open(os.path.join(run_dir, "error.txt"), "w", encoding="utf-8") as f:
                f.write(tb)

            print(
                f"  -> FAILED {failed_row['error_type']}: {failed_row['error_message']}"
            )
            manifest_rows.append(failed_row)

            if not continue_on_error:
                raise

    manifest_path = _write_manifest(session_dir, manifest_rows)
    quick_log_path = _write_quick_log(session_dir, manifest_rows)
    batch_seconds = perf_counter() - batch_start
    success_count = sum(1 for row in manifest_rows if row.get("status") == "success")
    failed_count = sum(1 for row in manifest_rows if row.get("status") == "failed")
    summary = {
        "session_dir": session_dir,
        "manifest_path": manifest_path,
        "quick_log_path": quick_log_path,
        "run_count": len(manifest_rows),
        "success_count": success_count,
        "failed_count": failed_count,
        "total_batch_seconds": batch_seconds,
        "generated_at": _iso_utc_now(),
    }
    _write_json(os.path.join(session_dir, "summary.json"), summary)

    print(
        f"Finished {len(manifest_rows)} run(s) in {batch_seconds:.3f}s "
        f"(success={success_count}, failed={failed_count})"
    )
    print(f"Session dir: {session_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Quick log: {quick_log_path}")


if __name__ == "__main__":
    main()
