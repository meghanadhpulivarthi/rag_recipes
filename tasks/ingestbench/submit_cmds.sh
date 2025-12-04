#!/bin/zsh
GEN_CFG="${1:-mini}"

cd /dccstor/meghanadhp/projects/rag_recipes
if [ ! -d "logs" ]; then
    mkdir logs
fi

export HF_HOME=/dccstor/meghanadhp/hf_home
export VLLM_CACHE_ROOT=/dccstor/meghanadhp/.cache/vllm
export HOME=/dccstor/meghanadhp



# bsub -J rag_index_sparse_mini_job -M 256G -n 1 -oo logs/rag_index_sparse_mini_job.out -eo logs/rag_index_sparse_mini_job.err  uv run src/indexing.py --task_config configs/ingestbench_sparse.yaml --gen_cfg $GEN_CFG

# bsub -J rag_index_dense_mini_job -gpu "num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" -M 256G -n 1 -oo logs/rag_index_dense_mini_job.out -eo logs/rag_index_dense_mini_job.err  uv run src/indexing.py --task_config configs/ingestbench_dense.yaml --gen_cfg $GEN_CFG

# bsub -J rag_index_ensemble_mini_job -gpu "num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" -M 256G -n 1 -oo logs/rag_index_ensemble_mini_job.out -eo logs/rag_index_ensemble_mini_job.err  uv run src/indexing.py --task_config configs/ingestbench_ensemble.yaml --gen_cfg $GEN_CFG

# generate

# for now use an interactive job with 2 gpus
# bsub -J installer -U infusion -gpu "num=2/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" -M 100G -W 360 -Is /bin/bash

ks=(10 20 30 40)
# ks=(6)

for k in "${ks[@]}"; do
    uv run src/generation.py --task_config configs/ingestbench_sparse.yaml --wandb_run_name "rag-$GEN_CFG-sparse-k${k}" --llm_provider "vllm" --gen_cfg $GEN_CFG --top_k "$k" --filename "${GEN_CFG}_sparse_k${k}.jsonl"

    uv run src/generation.py --task_config configs/ingestbench_dense.yaml --wandb_run_name "rag-$GEN_CFG-dense-k${k}" --llm_provider "vllm" --gen_cfg $GEN_CFG --top_k "$k" --filename "${GEN_CFG}_dense_k${k}.jsonl"

    uv run src/generation.py --task_config configs/ingestbench_ensemble.yaml --wandb_run_name "rag-$GEN_CFG-ensemble-k${k}" --llm_provider "vllm" --gen_cfg $GEN_CFG --top_k "$(( k / 2 ))" --filename "${GEN_CFG}_ensemble_k${k}.jsonl"
done
