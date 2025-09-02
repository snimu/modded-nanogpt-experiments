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
torchrun --standalone --nproc_per_node=8 8-improved-skip-saving.py
torchrun --standalone --nproc_per_node=8 9-improved-skip-saving-x00-x01.py
torchrun --standalone --nproc_per_node=8 10-skip-to-head-improved-skip-saving.py
torchrun --standalone --nproc_per_node=8 11-skip-to-head-improved-skip-saving-x00-and-x01.py

for ((i=0; i<40; i++)); do
  export RUN_ID=$i
  torchrun --standalone --nproc-per-node=8 12-previous-record.py
done

for ((i=0; i<40; i++)); do
  export RUN_ID=$i
  torchrun --standalone --nproc-per-node=8 13-skip-10-00-init-from-previous-record.py
done
