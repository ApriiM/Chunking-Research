## Chunking Research

Toolkit for chunking text, running retrieval-style evaluations on QA datasets, and comparing chunkers and retrievers via config-driven pipelines.

## Setup
- Create/activate a virtual environment.
- Install dependencies: `pip install -r requirements.txt`.

## Repository layout
- Chunkers live in [src/chunking](src/chunking) with per-strategy defaults in [configs/chunkers](configs/chunkers).
- Datasets live in [src/data_loader/datasets](src/data_loader/datasets) and register through [src/data_loader/registry.py](src/data_loader/registry.py).
- Retrievers register via [src/retrieval/registry.py](src/retrieval/registry.py).
- Evaluations live in [src/evaluation](src/evaluation) with registry in [src/evaluation/registry.py](src/evaluation/registry.py).
- Runners: chunker-only [run_chunking.py](run_chunking.py), retrieval pipeline [run_retrieval_eval.py](run_retrieval_eval.py).
- Configs: chunker demos under [configs/experiments](configs/experiments) (fixed_size, passage, text_tiling); full pipelines under [configs/experiments/pipeline](configs/experiments/pipeline).

## Quickstart
- Chunker-only demo: `python run_chunking.py --config configs/experiments/fixed_size/demo.yaml`.
- Retrieval pipeline (PoQuAD): `python run_retrieval_eval.py --config configs/experiments/pipeline/poquad_demo.yaml`.

## Pipeline config schema (high level)
- dataset: name, params (loader kwargs), preprocessed_path, use_preprocessed, save_preprocessed.
- chunker: name, params, optional precomputed_chunks_path to reuse saved chunks.
- retrieval: method, top_k, relevance (currently substring fallback).
- evaluations: list of eval names (e.g., retrieval_at_k, chunk_stats).
- output: results_path, chunks_path, manifest_path, overwrite.

## Creating a new benchmark/dataset
1) Implement a loader in [src/data_loader/datasets](src/data_loader/datasets) returning List[QASample] and register it in [src/data_loader/registry.py](src/data_loader/registry.py).
2) Add a pipeline config under [configs/experiments/pipeline](configs/experiments/pipeline) setting dataset.name and dataset.params for your loader; set preprocessed_path for caching.
3) Run `python run_retrieval_eval.py --config <your_config>`; chunks/results/manifest will be written under results/.

## Creating or modifying a pipeline
- Copy an existing pipeline config (e.g., [configs/experiments/pipeline/poquad_demo.yaml](configs/experiments/pipeline/poquad_demo.yaml)).
- Swap chunker (chunker.name/params) or retriever (retrieval.method/top_k).
- Add evaluations under evaluations:; built-ins: retrieval_at_k, chunk_stats.
- To reuse existing chunks, set chunker.precomputed_chunks_path to a chunks JSONL produced earlier.

## Extending components
- Chunker: add a strategy under [src/chunking/strategies](src/chunking/strategies) and register in [src/chunking/__init__.py](src/chunking/__init__.py); optional defaults in configs/chunkers/<name>.yaml.
- Retriever: register a factory in [src/retrieval/registry.py](src/retrieval/registry.py) and reference via retrieval.method in configs.
- Evaluation: implement a function and register it in [src/evaluation/__init__.py](src/evaluation/__init__.py); selectable via evaluations in configs.
- Dataset: add a loader under [src/data_loader/datasets](src/data_loader/datasets) and register it; set dataset.name/params accordingly.

## Outputs
- Samples cache: data/processed/*.jsonl (preprocessed QA samples).
- Chunks: JSONL with chunk_id, text, metadata (includes sample_id).
- Retrieval results: per-sample retrieval outputs JSONL.
- Manifest: JSON with config snapshot and aggregated metrics.

## Notes
- Overwrite control via output.overwrite; if false, files get timestamped suffixes.
- Current relevance policy is substring-based; alternative policies can be added in [src/evaluation/retrieval.py](src/evaluation/retrieval.py).
