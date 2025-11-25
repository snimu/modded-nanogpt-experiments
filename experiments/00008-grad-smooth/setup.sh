curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install numpy tqdm torch huggingface-hub matplotlib rich scipy torchinfo setuptools
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
uv run data/cached_fineweb10B.py

cd runs
torchrun --standalone --nproc_per_node=8 00000-baseline.py
cd .. && python plot_results.py --print-final-stats --path=logs 
cd runs
torchrun --standalone --nproc_per_node=8 00001-grad-smooth.py -g 0.05
cd .. && python plot_results.py --print-final-stats --path=logs 
cd runs
torchrun --standalone --nproc_per_node=8 00001-grad-smooth.py -g 0.1
cd .. && python plot_results.py --print-final-stats --path=logs 
cd runs
torchrun --standalone --nproc_per_node=8 00001-grad-smooth.py -g 0.2
cd .. && python plot_results.py --print-final-stats --path=logs 
