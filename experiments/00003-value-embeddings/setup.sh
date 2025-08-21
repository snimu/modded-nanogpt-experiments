pip install uv
uv venv
source .venv/bin/activate
uv pip install numpy tqdm torch huggingface-hub matplotlib rich
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
uv run data/cached_fineweb10B.py

cd runs
torchrun --standalone --nproc_per_node=8 0-2025-08-13-no-valemb-0.py
torchrun --standalone --nproc_per_node=8 1-2025-08-13-no-valemb-1.py
torchrun --standalone --nproc_per_node=8 2-2025-08-13-no-valemb-2.py
torchrun --standalone --nproc_per_node=8 3-2025-08-13-no-valemb-13.py
torchrun --standalone --nproc_per_node=8 4-2025-08-13-no-valemb-14.py
torchrun --standalone --nproc_per_node=8 5-2025-08-13-no-valemb-15.py
torchrun --standalone --nproc_per_node=8 6-2025-08-13-no-valemb-0-1.py
torchrun --standalone --nproc_per_node=8 7-2025-08-13-no-valemb-0-1-2.py
torchrun --standalone --nproc_per_node=8 8-2025-08-13-no-valemb-0-13.py
torchrun --standalone --nproc_per_node=8 9-2025-08-13-no-valemb-1-14.py
torchrun --standalone --nproc_per_node=8 10-2025-08-13-no-valemb-2-15.py
torchrun --standalone --nproc_per_node=8 11-2025-08-17-shared-valemb-012-131415.py
torchrun --standalone --nproc_per_node=8 12-2025-08-17-shared-valemb-01-1415.py
torchrun --standalone --nproc_per_node=8 13-2025-08-20-baseline.py
torchrun --standalone --nproc_per_node=8 14-2025-08-21-new-valemb-15.py
torchrun --standalone --nproc_per_node=8 15-2025-08-21-new-valemb-3-15.py