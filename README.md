## Chunking Research
Toolkit for loading QA-style datasets into a unified documents/queries format and producing chunked passages for retrieval evaluation. Evaluation is out of scope—run it in your target system.

## Setup
- Python 3.9+ recommended.
- Create/activate a virtual environment, then install: `pip install -r requirements.txt`.
- For TextTiling: `python -m nltk.downloader punkt`.

## Data loading (documents & queries)
- Notebook: run `examples/01_load_dataset_unified.ipynb` to load a registered dataset slice (e.g., PoQuAD), preview, and write `documents.jsonl` + `queries.jsonl` under `data/processed/poquad/example/`.
- CLI: `python -m src.data_loader.prepare_dataset --dataset poquad --split train[:200] --output-dir data/processed/poquad/example_2 --overwrite`
	- Loader returns `(documents, queries)` already normalized; files are written as JSONL.
- Add a dataset:
	1) Create `src/data_loader/datasets/<name>.py`.
	2) Decorate a loader with `@dataset("<name>")` and return `List[DocumentRecord], List[QueryRecord]`.
	3) Normalize inside the loader; keep per-dataset quirks there.

## Chunking (documents -> passages)
- Notebook: run `examples/02_chunk_unified.ipynb` to chunk `documents.jsonl` with a chosen strategy (`fixed_size`, `passage`, `text_tiling`) and save `passages.jsonl`.
- CLI (flags): `python -m src.chunking.prepare_passages --documents-path data/processed/poquad/example_2/documents.jsonl --chunker-name fixed_size --chunker-params "{chunk_size: 100, overlap: 50}" --output-path data/processed/poquad/example_2/passages.jsonl --overwrite`
- CLI (YAML config):
	- `python run_chunking.py --config configs/experiments/run_chunking_fixed_size.yaml` (expects `documents_path` JSONL of `DocumentRecord`s, plus `chunker` section and `output` section).
	- `python -m src.chunking.prepare_passages --config configs/experiments/run_chunking_fixed_size.yaml` for direct documents→passages with the same schema.
- Add a chunker:
	1) Create `src/chunking/strategies/<name>.py`.
	2) Decorate the class with `@chunker("<name>")` and implement `split_text`.
	3) Optionally add defaults in `configs/chunkers/<name>.yaml`.

## Repository layout (essentials)
- `src/data_loader/datasets/`: dataset-specific loaders (return documents/queries).
- `src/data_loader/core/schemas.py`: document/query/passage/chunk record shapes + JSONL helpers.
- `src/chunking/strategies/`: chunking strategies.
- `configs/chunkers/`: per-chunker default params.
- `configs/experiments/`: ready-to-run YAML configs for chunking CLI tools.

## Tips
- HF datasets cache under `data/hf_cache`; override via loader kwargs.
- `prepare_passages.py` writes `passages.jsonl` with `parentId` linking back to documents.
- Overwrite handling: the chunking CLI will refuse to overwrite unless `--overwrite` is passed; add timestamp logic if you need append-safe semantics.
