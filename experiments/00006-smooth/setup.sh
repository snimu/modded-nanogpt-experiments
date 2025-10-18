pip install uv
uv venv
source .venv/bin/activate
uv pip install numpy tqdm torch huggingface-hub matplotlib rich scipy torchinfo setuptools
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
uv run data/cached_fineweb10B.py

cd runs
torchrun --standalone --nproc-per-node=8 0005-mtp-and-smear-inputs.py -l=11
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 0005-mtp-and-smear-inputs.py -l=15
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 0004-mtp.py -l=11
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 0004-mtp.py -l=15
cd .. && python plot_results.py --print-final-stats --path=logs

cd runs
torchrun --standalone --nproc-per-node=8 0003-smear-inputs-and-outputs.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 0002-smear-outputs.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 0001-smear-inputs.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 0000-baseline.py
cd .. && python plot_results.py --print-final-stats --path=logs
