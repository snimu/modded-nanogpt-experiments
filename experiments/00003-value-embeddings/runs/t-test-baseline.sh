for ((i=0; i<40; i++)); do
  export RUN_ID=i
  torchrun --standalone --nproc-per-node=8 25-baseline-for-t-test.py
done
