pip install uv
uv venv
source .venv/bin/activate
uv pip install numpy tqdm torch huggingface-hub matplotlib rich sentence_transformers tiktoken
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
uv run data/cached_fineweb10B.py

cd runs
torchrun --standalone --nproc_per_node=8 4-2025-08-22-x00-x01-with-word-logging.py

torchrun --standalone --nproc_per_node=8 0-2025-08-17-x00-x01.py
torchrun --standalone --nproc_per_node=8 1-2025-08-20-x00-x01-x02.py
torchrun --standalone --nproc_per_node=8 2-2025-08-21-x00-x01-x02-x03.py
torchrun --standalone --nproc_per_node=8 3-2025-08-21-x00-x01-x02-x03-x04.py
torchrun --standalone --nproc_per_node=8 5-2025-08-28-x00-x01-with-extra-valembs.py
torchrun --standalone --nproc_per_node=8 999-2025-08-20-baseline.py
torchrun --standalone --nproc_per_node=8 999-2025-08-20-baseline-x00-x01.py

torchrun --standalone --nproc-per-node=8 62-2025-09-03-record-5-valembs-x01.py
for ((i=0; i<4; i++)); do
  export RUN_ID=$i
  torchrun --standalone --nproc-per-node=8 60-2025-09-03-baseline-5-valembs.py
  torchrun --standalone --nproc-per-node=8 61-2025-09-03-record-4-valembs-x01.py
  torchrun --standalone --nproc-per-node=8 62-2025-09-03-record-5-valembs-x01.py
done

for ((i=0; i<2; i++)); do
  export RUN_ID=$i
  torchrun --standalone --nproc-per-node=8 63-2025-09-03-record-5-valembs-x01-5690steps-higher-emb-lr.py
  torchrun --standalone --nproc-per-node=8 64-2025-09-03-record-5-valembs-x01-5690steps-lr-mult.py
  torchrun --standalone --nproc-per-node=8 65-2025-09-03-record-5-valembs-x01-5690steps.py
done

git pull
for ((i=0; i<35; i++)); do
  export RUN_ID=$i
  torchrun --standalone --nproc-per-node=8 65-2025-09-03-record-5-valembs-x01-5690steps.py
done

git pull
for ((i=0; i<35; i++)); do
  export RUN_ID=$i
  torchrun --standalone --nproc-per-node=8 66-2025-09-03-record-5-valembs-x01-5640steps.py
done

for ((i=0; i<35; i++)); do
  export RUN_ID=$i
  torchrun --standalone --nproc-per-node=8 67-2025-09-03-record-5-valembs-x01-5660steps.py
done

for ((i=0; i<40; i++)); do
  export RUN_ID=$i
  torchrun --standalone --nproc-per-node=8 68-2025-09-03-record-5-valembs-x01-5675steps.py
done

for ((i=0; i<23; i++)); do
  export RUN_ID=$i
  torchrun --standalone --nproc-per-node=8 690-2025-09-03-record-5-valembs-x01-5680steps.py
  torchrun --standalone --nproc-per-node=8 691-2025-09-03-record-5-valembs-x01-5680steps-inverse-valemb-share.py
done

for ((i=0; i<30; i++)); do
  export RUN_ID=$i
  torchrun --standalone --nproc-per-node=8 692-2025-09-03-record-5-valembs-x01-5690steps.py
  torchrun --standalone --nproc-per-node=8 693-2025-09-03-record-5-valembs-x01-5690steps-inverse-valemb-share.py
done

for ((i=0; i<30; i++)); do
  export RUN_ID=$i
  torchrun --standalone --nproc-per-node=8 692-2025-09-03-record-5-valembs-x01-5690steps.py
done

torchrun --standalone --nproc-per-node=8 train_gptm.py
