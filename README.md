## Chunking Research

Toolkit for loading QA-style datasets and producing chunked text outputs in a unified JSONL format. Evaluation is intentionally out of scope (run it in your target system).

## Setup
- Python 3.9+ recommended.
- Create/activate a virtual environment, then `pip install -r requirements.txt`.
- For TextTiling, ensure NLTK data is available: `python -m nltk.downloader punkt`.

## Quickstart (CLI)
- Single-document chunking: `python run_chunking.py --config configs/experiments/chunking/fixed_size_demo.yaml`.
- Other chunker presets: `configs/experiments/chunking/passage_demo.yaml` and `configs/experiments/chunking/text_tiling_demo.yaml`.
- Outputs are JSONL with `chunk_id`, `text`, and `metadata` (metadata includes propagated document info such as `sample_id`). When `output.overwrite` is false, existing targets get a timestamp suffix.

## Notebook workflow
1) `examples/01_load_dataset_unified.ipynb`: load a registered dataset slice (e.g., PoQuAD), preview a sample, and write unified QA JSONL to `data/processed/`.
2) `examples/02_chunk_unified.ipynb`: pick a chunker (`fixed_size`, `passage`, `text_tiling`), apply it to the unified samples, and save chunk records via `src.schemas.save_chunk_records_jsonl`.
3) Evaluate chunks externally in your target system; the repo stops at producing chunk JSONL.

## Repository layout
- Chunkers in `src/chunking` with defaults in `configs/chunkers`.
- Chunking experiment configs in `configs/experiments/chunking`.
- Dataset loaders in `src/data_loader/datasets` via the dataset registry.
- Shared serialization helpers for chunks in `src/schemas.py`.

## Extending components
- Chunker: add a strategy under `src/chunking/strategies` and either decorate with `@chunker("name")` (preferred) or register it manually in `src/chunking/__init__.py`; optional defaults in `configs/chunkers/<name>.yaml`.
- Dataset: add a loader under `src/data_loader/datasets` and decorate with `@dataset` to make it discoverable.

## Tips
- Hugging Face datasets cache under `data/hf_cache` by default; adjust paths in configs if needed.
- Use `run_chunking.ensure_output_path` when saving artifacts manually to avoid accidental overwrites.
