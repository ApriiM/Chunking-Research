## Chunking Research
Toolkit for loading QA-style datasets into a unified documents/queries format and producing chunked passages for retrieval evaluation. Evaluation is out of scope—run it in your target system.

## Setup
- Python 3.9+ recommended.
- Create/activate a virtual environment, then install: `pip install -r requirements.txt`.
- For TextTiling: `python -m nltk.downloader punkt`.
### Submodules
After cloning this repo normally, you may notice that code under `submodules/` is missing. Those folders are **Git submodules** (external repositories), and they require an extra step to fetch their contents.

If you clone fresh:
```bash
git clone --recurse-submodules <THIS_REPO_URL>
```
If you already cloned:
```bash
git submodule update --init --recursive
```

To add a new external chunking repository as a submodule, follow the steps in
[Add a submodule (maintainers)](#add-a-submodule-maintainers).

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
- CLI (YAML config): `python run_chunking.py --config configs/experiments/run_chunking_fixed_size.yaml` (expects `documents_path` JSONL of `DocumentRecord`s, plus `chunker` section and `output` section). Outputs `PassageRecord` JSONL with `parent_id` and a `.meta.json` summary; when `overwrite` is false, filenames are suffixed with a timestamp.
- Add a chunker:
	1) Create `src/chunking/strategies/<name>.py`.
	2) Decorate the class with `@chunker("<name>")` and implement `split_text`.
	3) Optionally add defaults in `configs/chunkers/<name>.yaml`.

## Passage evaluation (retrieval)
- Notebook: run `examples/03_eval_chunks.ipynb` to evaluate pre-chunked passages against queries.
- CLI: `python -m src.eval_chunks --passages-meta data/processed/poquad/example/passages_20251229T153044.meta.json --model-name jinaai/jina-embeddings-v2-small-en --output-path results/eval_chunks/poquad_fixed_size.json`
- Late chunking (pool spans from full documents): add `--late-chunking --late-docs-source passages` (rebuilds document context by concatenating passage texts; use `--passage-separator` to control glue). For long documents, use `--truncate-max-length` or `--long-late-chunking-embed-size` with `--long-late-chunking-overlap-size`.
- Requires the late-chunking dependencies (`mteb`, `transformers`, `torch`) from the submodule environment.

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

## Submodules (external repos)
We vendor third-party research code as **git submodules** under a single, consistent location:

- **All submodules live under:** `submodules/<name>/`
- Example: `submodules/late_chunking/` (jina-ai/late-chunking)

### Add a submodule (maintainers)
From the repo root:
```bash
git submodule add <REPO_URL> submodules/<name>
git add .gitmodules submodules/<name>
git commit -m "Add <name> submodule"
```
