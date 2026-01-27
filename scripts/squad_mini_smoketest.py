# Minimal end-to-end run on SQuAD validation:
# 1) Chunk with fixed_size (300 chars, 40 overlap)
# 2) Standard evaluation
# 3) Late-chunking evaluation

python -m src.chunking.prepare_passages --documents-path data/processed/squad/validation/documents/documents.jsonl --chunker-name fixed_size --chunker-params "{chunk_size: 300, overlap: 40}" --output-path data/processed/squad/validation/passages_all/fixed_300_40.jsonl --overwrite
python -m src.eval_chunks --passages-meta data/processed/squad/validation/passages_all/fixed_300_40.meta.json --model-name jinaai/jina-embeddings-v2-small-en --batch-size 64 --output-path results/eval_chunks/squad_fixed_300_40.json --show-progress
python -m src.eval_chunks --passages-meta data/processed/squad/validation/passages_all/fixed_300_40.meta.json --model-name jinaai/jina-embeddings-v2-small-en --batch-size 64 --late-chunking --late-docs-source passages --passage-separator "\n" --output-path results/eval_chunks/squad_fixed_300_40_late.json --show-progress