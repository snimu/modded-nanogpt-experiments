pip install uv
uv venv
source .venv/bin/activate
uv pip install numpy tqdm torch huggingface-hub matplotlib rich scipy torchinfo
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
uv run data/cached_fineweb10B.py

cd runs
torchrun --standalone --nproc-per-node=8 8001-add-skip-rolling.py

# NEW ABLATIONS
# 1. Try out changed seq-len schedule -> DONE; unsuccessful
cd runs
torchrun --standalone --nproc-per-node=8 7004-add-skip11-record-from-updated-record-changed-seq-len-schedule.py
cd .. && python plot_results.py --print-final-stats --path=logs
# 2. Try out norm-sum-norm -> DONE; unsuccessful
cd runs
torchrun --standalone --nproc-per-node=8 7005-add-skip11-record-from-updated-record-norm-sum-norm.py
cd .. && python plot_results.py --print-final-stats --path=logs
# 3. Time old and new record -> DONE; ~8.5 seconds time reduction
for ((i=0; i<5; i++)); do
  export RUN_ID=$i
  cd runs
  torchrun --standalone --nproc-per-node=8 7000-updated-record.py
  cd .. && python plot_results.py --print-final-stats --path=logs
  cd runs
  torchrun --standalone --nproc-per-node=8 7002-add-skip11-record-from-updated-record.py
  cd .. && python plot_results.py --print-final-stats --path=logs
done
4. Time the record (taking previous runs into account)
for ((i=5; i<35; i++)); do -> DONE; PR raised
  export RUN_ID=$i
  cd runs
  torchrun --standalone --nproc-per-node=8 7002-add-skip11-record-from-updated-record.py
  cd .. && python plot_results.py --print-final-stats --path=logs
done
# 5. Ablate multiple skips
for ((i=1; i<3; i++)); do
  for ((j=2; j<15; j++)); do
    export RUN_ID=$i
    cd runs
    torchrun --standalone --nproc-per-node=8 8000-add-skip-multiple.py -n=$j -c=random
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 8000-add-skip-multiple.py -n=$j -c=btw
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 8000-add-skip-multiple.py -n=$j -c=wtb
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 8000-add-skip-multiple.py -n=$j -c=lth
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 8000-add-skip-multiple.py -n=$j -c=htl
    cd .. && python plot_results.py --print-final-stats --path=logs
  done
done

# Compare new to old record
for ((i=0; i<2; i++)); do
  export RUN_ID=$i
  cd runs
  torchrun --standalone --nproc-per-node=8 7000-updated-record.py
  cd .. && python plot_results.py --print-final-stats --path=logs
  cd runs
  torchrun --standalone --nproc-per-node=8 7002-add-skip11-record-from-updated-record.py
  cd .. && python plot_results.py --print-final-stats --path=logs
done

# Ablate adding multiple skips
for ((i=0; i<3; i++)); do
  for ((j=2; i<15; i++)); do
    export RUN_ID=$i
    cd runs
    torchrun --standalone --nproc-per-node=8 8000-add-skip-multiple.py -n=$j -c=random
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 8000-add-skip-multiple.py -n=$j -c=btw
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 8000-add-skip-multiple.py -n=$j -c=wtb
    cd .. && python plot_results.py --print-final-stats --path=logs
  done
done

# Record attempt
for ((i=0; i<35; i++)); do
  export RUN_ID=$i
  cd runs
  torchrun --standalone --nproc-per-node=8 7002-add-skip11-record-from-updated-record.py
  cd .. && python plot_results.py --print-final-stats --path=logs
done

cd runs
torchrun --standalone --nproc-per-node=8 7000-updated-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 7001-add-skip11-from-updated-record.py
cd .. && python plot_results.py --print-final-stats --path=logs

cd runs
torchrun --standalone --nproc-per-node=8 3211-concat-skip11-last-mlp-with-cutoff-and-activation.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3311-concat-skip11-last-mlp-with-cutoff.py
cd .. && python plot_results.py --print-final-stats --path=logs

for ((i=0; i<15; i++)); do
  cd runs
  torchrun --standalone --nproc-per-node=8 6000-concat-skips-last-mlp-with-projection.py --skip-layer=$i --compressed-dim=128
  torchrun --standalone --nproc-per-node=8 6000-concat-skips-last-mlp-with-projection.py --skip-layer=$i --compressed-dim=256
  cd .. && python plot_results.py --print-final-stats --path=logs
done

cd runs
torchrun --standalone --nproc-per-node=8 3111-concat-skip11-last-mlp-with-projection.py
cd .. && python plot_results.py --print-final-stats --path=logs

for ((i=0; i<35; i++)); do
  cd runs
  export RUN_ID=$i
  torchrun --standalone --nproc-per-node=8 5002-add-normed-skip11-to-x-out-record.py
  cd .. && python plot_results.py --print-final-stats --path=logs
done

for ((i=0; i<15; i++)); do
  cd runs
  torchrun --standalone --nproc-per-node=8 5001-add-normed-skips-to-x-out.py --skip-layer=$i
  cd .. && python plot_results.py --print-final-stats --path=logs
done

cd runs
torchrun --standalone --nproc-per-node=8 5000-add-skips-to-x-out.py --skip-layer=10
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 5000-add-skips-to-x-out.py --skip-layer=11
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 5000-add-skips-to-x-out.py --skip-layer=12
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 5000-add-skips-to-x-out.py --skip-layer=13
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 5000-add-skips-to-x-out.py --skip-layer=14
cd .. && python plot_results.py --print-final-stats --path=logs

cd runs
torchrun --standalone --nproc-per-node=8 21-concat-x00-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 22-concat-x02-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 24-concat-x-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2300-concat-skip0-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2301-concat-skip1-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2302-concat-skip2-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2303-concat-skip3-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2304-concat-skip4-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2305-concat-skip5-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2306-concat-skip6-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2307-concat-skip7-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2308-concat-skip8-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2309-concat-skip9-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2310-concat-skip10-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2311-concat-skip11-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2312-concat-skip12-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2313-concat-skip13-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 2314-concat-skip14-from-previous-record.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3000-concat-x00-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3001-concat-x01-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3002-concat-x02-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3010-concat-skip0-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3011-concat-skip1-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3012-concat-skip2-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3013-concat-skip3-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3014-concat-skip4-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3015-concat-skip5-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3016-concat-skip6-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3017-concat-skip7-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3018-concat-skip8-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3019-concat-skip9-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3020-concat-skip10-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3021-concat-skip11-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3022-concat-skip12-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3023-concat-skip13-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3024-concat-skip14-last-mlp.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 3025-concat-skip15-last-mlp.py
torchrun --standalone --nproc-per-node=8 4022-concat-skip12-last-mlp-compressed.py

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
