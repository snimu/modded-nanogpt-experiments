pip install uv
uv venv
source .venv/bin/activate
uv pip install numpy tqdm torch huggingface-hub matplotlib rich
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
uv run data/cached_fineweb10B.py

cd runs
torchrun --standalone --nproc_per_node=8 0-2025-08-17-x00-x01.py
torchrun --standalone --nproc_per_node=8 1-2025-08-20-x00-x01-x02.py
torchrun --standalone --nproc_per_node=8 2-2025-08-21-x00-x01-x02-x03.py
torchrun --standalone --nproc_per_node=8 3-2025-08-21-x00-x01-x02-x03-x04.py
