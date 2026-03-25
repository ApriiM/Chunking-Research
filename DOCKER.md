# Docker Usage

## Build Image

```bash
docker build -t chunkowanie:final .
```

This image includes:
- project code (`src/`, `scripts/`, top-level runners),
- configs (including your `FINAL_EXPERIMESNTS_*` YAML files),
- prepared datasets from `data/processed/**`,
- GPU-ready NVIDIA base image: `nvcr.io/nvidia/pytorch:24.02-py3`,
- CUDA PyTorch install line:
  - `torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`,
- virtualenv paths in the image:
  - `/app/.venv`,
  - `/app/.venv-literaryqa` (symlink to `.venv` in this Docker build).

## Run Experiments (No Mounts)

Run directly with Python from the baked venv:

```bash
docker run --rm -it chunkowanie:final \
  python run_experiment.py --config configs/experiments/FINAL_EXPERIMESNTS_processed_all_methods.yaml
```

Dry run:

```bash
docker run --rm -it chunkowanie:final \
  python run_experiment.py --config configs/experiments/FINAL_EXPERIMESNTS_processed_all_methods.yaml --dry-run
```

Open shell:

```bash
docker run --rm -it chunkowanie:final bash
```

## Export to PIRB

```bash
docker run --rm -it chunkowanie:final python -m src.results_converter.pirb_export --help
```

## Pack / Share Image

Save:

```bash
docker save chunkowanie:final | gzip > chunkowanie_final_image.tar.gz
```

Load on another machine:

```bash
gunzip -c chunkowanie_final_image.tar.gz | docker load
```
