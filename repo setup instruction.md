# Set hf token
`git config --global credential.helper store`

`hf auth login --token "<your hf token>"  --add-to-git-credential`

`git clone https://huggingface.co/datasets/NovelQA/NovelQA downloads/NovelQA`

# .venv

Run `./init.sh --download-novelqa`

# Downloading datsets

```
PYTHON_BIN="$PWD/.venv/bin/python" \
PYTHON_BIN_LITERARYQA="$PWD/.venv-literaryqa/bin/python" \
./download_datasets.sh
```