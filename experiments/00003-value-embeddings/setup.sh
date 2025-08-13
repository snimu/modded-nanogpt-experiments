pip install uv
uv venv
source .venv/bin/activate
uv pip install numpy tqdm torch huggingface-hub matplotlib rich
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
uv run data/cached_fineweb10B.py

cd runs
torchrun --standalone --nproc_per_node=8 0-no-valemb-0.py
torchrun --standalone --nproc_per_node=8 1-no-valemb-1.py
torchrun --standalone --nproc_per_node=8 2-no-valemb-2.py
torchrun --standalone --nproc_per_node=8 3-no-valemb-13.py
torchrun --standalone --nproc_per_node=8 4-no-valemb-14.py
torchrun --standalone --nproc_per_node=8 5-no-valemb-15.py
torchrun --standalone --nproc_per_node=8 6-no-valemb-0-1.py
torchrun --standalone --nproc_per_node=8 7-no-valemb-0-1-2.py
torchrun --standalone --nproc_per_node=8 8-no-valemb-0-13.py
torchrun --standalone --nproc_per_node=8 9-no-valemb-1-14.py
torchrun --standalone --nproc_per_node=8 10-no-valemb-2-15.py