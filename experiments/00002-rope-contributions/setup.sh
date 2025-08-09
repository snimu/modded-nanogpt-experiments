pip install uv
uv init
uv add numpy tqdm torch huggingface-hub matplotlib rich
uv run data/cached_fineweb10B.py