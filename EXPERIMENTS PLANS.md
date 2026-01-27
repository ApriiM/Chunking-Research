Datasets to test:
- poquad - validation
- squad - validation
- scifacts - train
- gutenqa_concat - validation


Experiments plans:

poquad (validation split)
------------------------
Assume `data/processed/poquad/validation/documents/documents.jsonl` exists (run dataset prep first). Outputs go to `passages_all/` under the same folder.

fixed_size
- Small chunk, with overlap:  
  `python -m src.chunking.prepare_passages --documents-path data/processed/poquad/validation/documents/documents.jsonl --chunker-name fixed_size --chunker-params "{chunk_size: 150, overlap: 30}" --output-path data/processed/poquad/validation/passages_all/fixed_150_30.jsonl --overwrite`
- Small chunk, no overlap:  
  `python -m src.chunking.prepare_passages --documents-path data/processed/poquad/validation/documents/documents.jsonl --chunker-name fixed_size --chunker-params "{chunk_size: 150, overlap: 0}" --output-path data/processed/poquad/validation/passages_all/fixed_150_0.jsonl --overwrite`
- Medium chunk, with overlap:  
  `python -m src.chunking.prepare_passages --documents-path data/processed/poquad/validation/documents/documents.jsonl --chunker-name fixed_size --chunker-params "{chunk_size: 300, overlap: 40}" --output-path data/processed/poquad/validation/passages_all/fixed_300_40.jsonl --overwrite`
- Medium chunk, no overlap:  
  `python -m src.chunking.prepare_passages --documents-path data/processed/poquad/validation/documents/documents.jsonl --chunker-name fixed_size --chunker-params "{chunk_size: 300, overlap: 0}" --output-path data/processed/poquad/validation/passages_all/fixed_300_0.jsonl --overwrite`
- Large chunk, with overlap:  
  `python -m src.chunking.prepare_passages --documents-path data/processed/poquad/validation/documents/documents.jsonl --chunker-name fixed_size --chunker-params "{chunk_size: 512, overlap: 64}" --output-path data/processed/poquad/validation/passages_all/fixed_512_64.jsonl --overwrite`
- Large chunk, no overlap:  
  `python -m src.chunking.prepare_passages --documents-path data/processed/poquad/validation/documents/documents.jsonl --chunker-name fixed_size --chunker-params "{chunk_size: 512, overlap: 0}" --output-path data/processed/poquad/validation/passages_all/fixed_512_0.jsonl --overwrite`

passage_regexp
- Tight sentence groups (fine-grained):  
  `python -m src.chunking.prepare_passages --documents-path data/processed/poquad/validation/documents/documents.jsonl --chunker-name passage_regexp --chunker-params "{passage_length:5}" --output-path data/processed/poquad/validation/passages_all/regexp_len5.jsonl --overwrite`
- Default-length passages (coarser):  
  `python -m src.chunking.prepare_passages --documents-path data/processed/poquad/validation/documents/documents.jsonl --chunker-name passage_regexp --chunker-params "{passage_length:10}" --output-path data/processed/poquad/validation/passages_all/regexp_len10.jsonl --overwrite`

passage_spacy
- Sentence-accurate singles (max precision):  
  `python -m src.chunking.prepare_passages --documents-path data/processed/poquad/validation/documents/documents.jsonl --chunker-name passage_spacy --chunker-params "{passage_length:1, spacy_model: en_core_web_sm, use_sentencizer: true}" --output-path data/processed/poquad/validation/passages_all/spacy_len1.jsonl --overwrite`
- Small bundles for recall:  
  `python -m src.chunking.prepare_passages --documents-path data/processed/poquad/validation/documents/documents.jsonl --chunker-name passage_spacy --chunker-params "{passage_length:3, spacy_model: en_core_web_sm, use_sentencizer: true}" --output-path data/processed/poquad/validation/passages_all/spacy_len3.jsonl --overwrite`

text_tiling
- Default Hearst settings (baseline):  
  `python -m src.chunking.prepare_passages --documents-path data/processed/poquad/validation/documents/documents.jsonl --chunker-name text_tiling --chunker-params "{w:20, k:10, similarity_method:block_comparison, smoothing_method:default, smoothing_width:2, smoothing_rounds:1, cutoff_policy:hc, demo_mode:false}" --output-path data/processed/poquad/validation/passages_all/texttiling_default.jsonl --overwrite`
- Smaller tiles for finer topical shifts:  
  `python -m src.chunking.prepare_passages --documents-path data/processed/poquad/validation/documents/documents.jsonl --chunker-name text_tiling --chunker-params "{w:14, k:8, similarity_method:block_comparison, smoothing_method:default, smoothing_width:2, smoothing_rounds:1, cutoff_policy:hc, demo_mode:false}" --output-path data/processed/poquad/validation/passages_all/texttiling_small.jsonl --overwrite`
