# Chunking Research

INIT

## Setup
- Create/activate a virtual environment.
- Install deps: `pip install -r requirements.txt`.

## Running a demo
- Pick a config under `configs/experiments/<chunker>/demo.yaml` (e.g., `fixed_size/demo.yaml`, `passage/demo.yaml`, `text_tiling/demo.yaml`). Update `input_file` or `input_text` as needed.
- Choose evaluations by listing them under `evaluations:` (e.g., `- chunk_stats`). Multiple entries run sequentially.
- To persist chunks, set `output.save_chunks: true` and `output.chunks_path` to your desired file (JSONL). If the file exists, a timestamp suffix is added unless you set `output.overwrite: true`.
- Run: `python run_chunking.py --config configs/experiments/fixed_size/demo.yaml` (or another config path).
- Outputs the requested evaluation metrics to stdout.

## Extending
- Add new chunkers under `src/chunking/strategies/` and register them in `src/chunking/__init__.py`.
- Add metrics in `src/evaluation/` and register them in `src/evaluation/__init__.py` to make them selectable via the `evaluations` list.
- Chunker default parameters live in `configs/chunkers/defaults.yaml`. YAML `chunker.params` override any defaults.

### Available chunkers
- `fixed_size`: character-based windows with overlap.
- `passage`: sentence-based passages.
- `text_tiling`: NLTK TextTiling (Hearst, 1997) segmentation.

Example config snippet for passage chunking:
```
chunker:
	name: "passage"
	params:
		passage_length: 8  # sentences per chunk
```

Example config snippet for TextTiling chunking:
```
chunker:
	name: "text_tiling"
	params:
		w: 20
		k: 10
		similarity_method: block_comparison  # or vocabulary_introduction
		stopwords: []  # default empty to avoid NLTK download
		smoothing_method: default
		smoothing_width: 2
		smoothing_rounds: 1
		cutoff_policy: hc  # or lc
		demo_mode: false
```
