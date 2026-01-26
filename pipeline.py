from __future__ import annotations
import argparse
import json
import traceback
import yaml
import pandas as pd
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.chunking.prepare_passages import chunk_documents, _metadata_path
from src.data_loader.prepare_dataset import prepare_dataset
from src.eval_chunks import evaluate_chunks


"""
Pipeline runner for:
    - downloading datasets from data_loader/datasets
    - chunking documents using chunking/strategies
    - evaluating chunks via eval_chunks.py
    - collecting structured results to disk

Usage:
    python pipeline.py --config configs/experiments/run_pipeline.yaml
"""

DEFAULT_CONFIG_PATH = "configs/experiments/run_pipeline.yaml"
DEFAULT_DOCUMENTS_PATH = Path("documents") / "documents.jsonl"
DEFAULT_QUERIES_PATH = Path("queries") / "queries.jsonl"


def save_results(results: List[Dict[str, Any]], results_dir: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"results_{timestamp}.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    scores_summary_path = results_dir / f"scores_summary_{timestamp}.csv"
    save_scores_summary(results, scores_summary_path)

    print(f"Saved results to {results_path}, {scores_summary_path}")


def save_scores_summary(results: List[Dict[str, Any]], results_path: Path):
    data = []
    for result in results:
        dataset = result.get("dataset")
        chunker_name = result.get("chunker_name")
        metrics = result.get("scores", {})

        # Flatten metrics into a single dictionary
        row = {"dataset": dataset, "chunker_name": chunker_name}
        row.update(metrics)
        data.append(row)

    df = pd.DataFrame(data)

    cols = ["dataset", "chunker_name"] + [c for c in df.columns if c not in ["dataset", "chunker_name"]]
    df = df[cols]

    df.to_csv(results_path, index=False)


# ----------------------------
# Config handling
# ----------------------------
def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError("Config file must decode to a mapping")
        if "results_dir" not in cfg:
            raise ValueError("Config must include results_dir")
        if "datasets" not in cfg or not isinstance(cfg["datasets"], list):
            raise ValueError("Config must include datasets as a list")
        if "chunkers" not in cfg or not isinstance(cfg["chunkers"], list):
            raise ValueError("Config must include chunkers as a list")
        
        return cfg
    

# ----------------------------
# Dataset handling
# ----------------------------
def parse_datasets_config(datasets_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    datasets_conf: List[Dict[str, Any]] = []
    for ds_raw in datasets_raw:
        if not isinstance(ds_raw, dict) or "dataset" not in ds_raw:
            raise ValueError("Each dataset config entry must be a mapping with at least a 'dataset' key")
        
        datasets_conf.append({
            "dataset": ds_raw["dataset"],
            "split": ds_raw.get("split", "train"),
            "output_dir": ds_raw.get("output_dir", f"data/processed/{ds_raw['dataset']}"),
            "cache_dir": ds_raw.get("cache_dir", None),
            "limit": ds_raw.get("limit", None),
            "max_documents": ds_raw.get("max_documents", None),
            "loader_kwargs": ds_raw.get("loader_kwargs", {}),
            "overwrite": ds_raw.get("overwrite", False)})
    
    return datasets_conf


def limit_documents_in_corpus(output_dir: Path, max_documents: int):
    if max_documents is None:
        return
    
    documents_path = output_dir / DEFAULT_DOCUMENTS_PATH
    queries_path = output_dir / DEFAULT_QUERIES_PATH

    if not queries_path.exists() or not documents_path.exists():
        return

    relevant_docs_ids = set()
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                query_rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(query_rec, dict) and "relevant" in query_rec:
                relevant_docs_ids.update(query_rec["relevant"])

    relevant_docs = []
    remaining_docs = []
    with open(documents_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                document_rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if document_rec["id"] in relevant_docs_ids:
                relevant_docs.append(document_rec)
            else:
                remaining_docs.append(document_rec)
        
        additional_nr_of_docs = max_documents - len(relevant_docs)
        if additional_nr_of_docs > 0:
            relevant_docs.extend(random.sample(remaining_docs, min(additional_nr_of_docs, len(remaining_docs))))
    
    with open(documents_path, "w", encoding="utf-8") as f:
        for doc in relevant_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def run_data_loaders(
    datasets: List[Dict[str, Any]]):
    for ds in datasets:
        run_data_loader(
            dataset=ds["dataset"],
            split=ds["split"],
            output_dir=ds["output_dir"],
            cache_dir=ds["cache_dir"],
            limit=ds["limit"],
            loader_kwargs=ds["loader_kwargs"],
            overwrite=ds["overwrite"],
            max_documents=ds["max_documents"])


def run_data_loader(
    dataset: str,
    split: str,
    output_dir: str,
    cache_dir: Optional[str],
    limit: Optional[int],
    loader_kwargs: Dict,
    overwrite: bool,
    max_documents: Optional[int]) -> None:
    
    try:
        prepare_dataset(
            dataset=dataset,
            split=split,
            output_dir=output_dir,
            cache_dir=cache_dir,
            limit=limit,
            loader_kwargs=loader_kwargs,
            overwrite=overwrite)
        
        if max_documents is not None:
            limit_documents_in_corpus(Path(output_dir), max_documents)

    except Exception as exc:
        print(f"Dataset was not loaded for {dataset}: {str(exc)}")


# ----------------------------
# Chunking handling
# ----------------------------
def parse_chunkers_config(chunkers_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunkers_conf = []

    for chunker_raw in chunkers_raw:
        if not isinstance(chunker_raw, dict) or "name" not in chunker_raw:
            raise ValueError("Each chunker config entry must be a mapping with at least a 'name' key")

        chunkers_conf.append({
            "name": chunker_raw["name"],
            "params": chunker_raw.get("params", {})
        })

    return chunkers_conf


def run_chunker(
    documents_path: str,
    chunker_name: str,
    chunker_params: Dict,
    output_path: str,
    overwrite: bool) -> None:
    try:
        chunk_documents(
            documents_path=documents_path,
            chunker_name=chunker_name,
            chunker_params=chunker_params,
            output_path=output_path,
            overwrite=overwrite)
    except Exception as exc:
        raise RuntimeError(f"Chunking failed for {documents_path} with {chunker_name}") from exc


# ----------------------------
# Orchestration for a single experiment
# ----------------------------
def run_experiments(
    datasets_config: List[Dict[str, Any]],
    chunkers_config: List[Dict[str, Any]]):
    results = []
    for chunker in chunkers_config:
        chunker_name = chunker["name"]
        chunker_params = chunker["params"]

        for dataset in datasets_config:
            dataset_name = dataset["dataset"]
            output_dir = Path(dataset["output_dir"])

            documents_path = output_dir / DEFAULT_DOCUMENTS_PATH
            queries_path = output_dir / DEFAULT_QUERIES_PATH
            chunks_output_path = output_dir / "passages" / f"passages_{chunker_name}.jsonl"

            result = run_experiment(
                documents_path=str(documents_path),
                queries_path=str(queries_path),
                dataset_name=dataset_name,
                chunker_name=chunker_name,
                chunker_params=chunker_params,
                chunks_path=str(chunks_output_path),
                overwrite=True)
            results.append(result)
    return results
            

def run_experiment(
    documents_path: str,
    queries_path: str, 
    dataset_name: str,
    chunker_name: str,
    chunker_params: Dict,
    chunks_path: str,
    overwrite: bool):
    try:
        if not Path(documents_path).exists():
            raise FileNotFoundError(f"Documents file not found: {documents_path}")
        if not Path(queries_path).exists():
            raise FileNotFoundError(f"Queries file not found: {queries_path}")

        start_time = datetime.now().isoformat()

        run_chunker(
            documents_path=documents_path,
            chunker_name=chunker_name,
            chunker_params=chunker_params,
            output_path=chunks_path,
            overwrite=overwrite)
        
        end_time = datetime.now().isoformat()

        meta_path = _metadata_path(chunks_path)

        payload = evaluate_chunks(
            passages_meta_path=meta_path,
            documents_path=documents_path,
            queries_path=queries_path,
            passages_path=chunks_path)
        
        res = {
            "result": "success",
            "dataset": dataset_name,
            "chunker_name": chunker_name,
            "chunker_params": chunker_params,
            "start_time": start_time,
            "end_time": end_time}
        res.update(payload)
        return res
    except Exception as exc:
        print(f"Experiment failed for {dataset_name}, {chunker_name}: {str(exc)}")
        res = {
            "result": "failure",
            "dataset": dataset_name,
            "chunker_name": chunker_name,
            "chunker_params": chunker_params,
            "error": str(exc),
            "traceback": traceback.format_exc()}
        return res


# ----------------------------
# Runner
# ----------------------------
def run_pipeline(cfg: Dict[str, Any]):
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_config = parse_datasets_config(cfg["datasets"])
    chunkers_config = parse_chunkers_config(cfg["chunkers"])

    run_data_loaders(datasets_config)
    results = run_experiments(datasets_config, chunkers_config)
    save_results(results, results_dir)


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run dataset chunking experiments pipeline.")
    p.add_argument("--config", "-c", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config file.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    run_pipeline(cfg)
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
