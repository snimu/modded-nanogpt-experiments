for ((i=0; i<40; i++)); do
  export RUN_ID=i
  torchrun --standalone --nproc-per-node=8 26-two-new-valemb-for-t-test.py
done
