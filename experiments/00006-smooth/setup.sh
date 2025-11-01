pip install uv
uv venv
source .venv/bin/activate
uv pip install numpy tqdm torch huggingface-hub matplotlib rich scipy torchinfo setuptools
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
uv run data/cached_fineweb10B.py

for ((idx=0; idx<20; idx++)); do
    cd runs
    torchrun --standalone --nproc-per-node=8 0018-record.py
    cd .. && python plot_results.py --print-final-stats --path=logs  
done

for ((idx=0; idx<10; idx++)); do
    cd runs
    torchrun --standalone --nproc-per-node=8 0014-mtp-with-trafo-no-gate.py
    cd .. && python plot_results.py --print-final-stats --path=logs  
done

cd runs
torchrun --standalone --nproc-per-node=8 0004-mtp.py -l=7
cd .. && python plot_results.py --print-final-stats --path=logs 
cd runs
torchrun --standalone --nproc-per-node=8 0013-mtp-no-gate.py
cd .. && python plot_results.py --print-final-stats --path=logs 
cd runs
torchrun --standalone --nproc-per-node=8 0015-scale-up-1B-baseline.py
cd .. && python plot_results.py --print-final-stats --path=logs 
cd runs
torchrun --standalone --nproc-per-node=8 0017-scale-up-1B-mtp-no-gate.py
cd .. && python plot_results.py --print-final-stats --path=logs 

for ((idx=0; idx<10; idx++)); do
    cd runs
    torchrun --standalone --nproc-per-node=8 0000-baseline.py
    cd .. && python plot_results.py --print-final-stats --path=logs  
    cd runs
    torchrun --standalone --nproc-per-node=8 0009-mtp-same-layer-with-trafo.py -l=7
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 0010-mtp-with-trafo.py -l=7
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 0013-mtp-no-gate.py
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 0004-mtp.py -l=7
    cd .. && python plot_results.py --print-final-stats --path=logs  
done

for ((idx=0; idx<5; idx++)); do
    cd runs
    torchrun --standalone --nproc-per-node=8 0013-mtp-no-gate.py -l=7
    cd .. && python plot_results.py --print-final-stats --path=logs  
    cd runs
    torchrun --standalone --nproc-per-node=8 0009-mtp-same-layer-with-trafo.py -l=7
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 0010-mtp-with-trafo.py -l=7
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 0011-mtp-no-mlp.py -l=7
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 0004-mtp.py -l=7
    cd .. && python plot_results.py --print-final-stats --path=logs  
done
for ((idx=0; idx<5; idx++)); do
    cd runs
    torchrun --standalone --nproc-per-node=8 0009-mtp-same-layer-with-trafo.py -l=4
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 0010-mtp-with-trafo.py -l=4
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 0011-mtp-no-mlp.py -l=4
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 0004-mtp.py -l=4
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 0013-mtp-no-gate.py -l=4
    cd .. && python plot_results.py --print-final-stats --path=logs   
done
for ((idx=0; idx<5; idx++)); do
    cd runs
    torchrun --standalone --nproc-per-node=8 0010-mtp-with-trafo.py -l=4
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 0000-baseline.py
    cd .. && python plot_results.py --print-final-stats --path=logs   
done

for ((layer=1; layer<16; layer++)); do
    cd runs
    torchrun --standalone --nproc-per-node=8 0006-mtp-difficulty-estimation.py -l=$layer
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 0004-mtp.py -l=$layer
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 0007-mtp-from-last-token.py -l=$layer
    cd .. && python plot_results.py --print-final-stats --path=logs    
    cd runs
    torchrun --standalone --nproc-per-node=8 0000-baseline.py
    cd .. && python plot_results.py --print-final-stats --path=logs    
done

cd runs
torchrun --standalone --nproc-per-node=8 0002-smear-outputs.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 0001-smear-inputs.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 0000-baseline.py
cd .. && python plot_results.py --print-final-stats --path=logs
