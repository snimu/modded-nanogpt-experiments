pip install uv
uv venv
source .venv/bin/activate
uv pip install numpy tqdm torch huggingface-hub matplotlib rich
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
uv run data/cached_fineweb10B.py

cd runs

torchrun --standalone --nproc-per-node=8 21-concat-x00-from-previous-record.py
torchrun --standalone --nproc-per-node=8 22-concat-x02-from-previous-record.py
torchrun --standalone --nproc-per-node=8 24-concat-x-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2300-concat-skip0-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2301-concat-skip1-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2302-concat-skip2-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2303-concat-skip3-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2304-concat-skip4-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2305-concat-skip5-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2306-concat-skip6-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2307-concat-skip7-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2308-concat-skip8-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2309-concat-skip9-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2310-concat-skip10-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2311-concat-skip11-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2312-concat-skip12-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2313-concat-skip13-from-previous-record.py
torchrun --standalone --nproc-per-node=8 2314-concat-skip14-from-previous-record.py

torchrun --standalone --nproc_per_node=8 0-baseline.py
torchrun --standalone --nproc_per_node=8 1-skip-05-05-init.py
torchrun --standalone --nproc_per_node=8 2-skip-00-10-init.py
torchrun --standalone --nproc_per_node=8 3-skip-10-00-init.py
torchrun --standalone --nproc_per_node=8 8-improved-skip-saving.py
torchrun --standalone --nproc_per_node=8 9-improved-skip-saving-x00-x01.py
torchrun --standalone --nproc_per_node=8 10-skip-to-head-improved-skip-saving.py
torchrun --standalone --nproc_per_node=8 11-skip-to-head-improved-skip-saving-x00-and-x01.py

for ((i=0; i<4; i++)); do
  export RUN_ID=$i
  torchrun --standalone --nproc-per-node=8 16-previous-record-x00-x01.py
  torchrun --standalone --nproc-per-node=8 18-previous-record-x00-x01-increased-lr.py
  torchrun --standalone --nproc-per-node=8 19-previous-record-x00-x01-increased-lr-100percent-decay.py
  torchrun --standalone --nproc-per-node=8 20-previous-record-x00-x01-increased-lr-100percent-decay-fewer-steps.py
done
