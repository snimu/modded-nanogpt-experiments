pip install uv
uv venv
source .venv/bin/activate
uv pip install numpy tqdm torch huggingface-hub matplotlib rich
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
uv run data/cached_fineweb10B.py

cd runs
torchrun --standalone --nproc_per_node=8 0-baseline.py
torchrun --standalone --nproc_per_node=8 1-skip-05-05-init.py
torchrun --standalone --nproc_per_node=8 2-skip-00-10-init.py
torchrun --standalone --nproc_per_node=8 3-skip-10-00-init.py
