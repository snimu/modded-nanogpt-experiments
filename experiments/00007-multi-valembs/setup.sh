pip install uv
uv venv
source .venv/bin/activate
uv pip install numpy tqdm torch huggingface-hub matplotlib rich scipy polars seaborn setuptools
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
uv run data/cached_fineweb10B.py

for ((num_embs_per_ve=3; num_embs_per_ve<6; num_embs_per_ve++)); do
    cd runs
    torchrun --standalone --nproc-per-node=8 00005-extra-embs-one-valemb.py --num-ve=1 --num-embs-per-ve=$num_embs_per_ve
    cd .. && python plot_results.py --print-final-stats --path=logs
done

# For 8xH200
for ((num_embs_per_ve=1; num_embs_per_ve<6; num_embs_per_ve++)); do
    cd runs
    torchrun --standalone --nproc-per-node=8 00000-extra-embs.py --num-ve=5 --num-embs-per-ve=$num_embs_per_ve
    cd .. && python plot_results.py --print-final-stats --path=logs
    cd runs
    torchrun --standalone --nproc-per-node=8 00005-extra-embs-one-valemb.py --num-ve=5 --num-embs-per-ve=$num_embs_per_ve
    cd .. && python plot_results.py --print-final-stats --path=logs
done

# For 8xH100
for ((num_ve=4; num_ve>0; num_ve--)); do
    for ((num_embs_per_ve=1; num_embs_per_ve<6; num_embs_per_ve++)); do
        cd runs
        torchrun --standalone --nproc-per-node=8 00005-extra-embs-one-valemb.py --num-ve=$num_ve --num-embs-per-ve=$num_embs_per_ve
        cd .. && python plot_results.py --print-final-stats --path=logs
    done
done

cd runs
torchrun --standalone --nproc-per-node=8 00003-multi-emb-via-projection-relu.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 00000-extra-embs.py --num-ve=4 --num-embs-per-ve=5
cd .. && python plot_results.py --print-final-stats --path=logs

cd runs
torchrun --standalone --nproc-per-node=8 00001-multi-emb-via-projection.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 00002-multi-small-emb-up-projected.py -d 2
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 00002-multi-small-emb-up-projected.py -d 4
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 00002-multi-small-emb-up-projected.py -d 8
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 00002-multi-small-emb-up-projected.py -d 16
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 00003-multi-emb-via-projection-relu.py
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 00004-multi-small-emb-up-projected-relu.py -d 2
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 00004-multi-small-emb-up-projected-relu.py -d 4
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 00004-multi-small-emb-up-projected-relu.py -d 8
cd .. && python plot_results.py --print-final-stats --path=logs
cd runs
torchrun --standalone --nproc-per-node=8 00004-multi-small-emb-up-projected-relu.py -d 16
cd .. && python plot_results.py --print-final-stats --path=logs

for ((num_ve=4; num_ve>0; num_ve--)); do
    for ((num_embs_per_ve=1; num_embs_per_ve<6; num_embs_per_ve++)); do
    cd runs
    torchrun --standalone --nproc-per-node=8 00000-extra-embs.py --num-ve=$num_ve --num-embs-per-ve=$num_embs_per_ve
    cd .. && python plot_results.py --print-final-stats --path=logs
    done
done
