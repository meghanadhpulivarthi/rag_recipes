#!/bin/zsh
GEN_CFG="${1:-$GEN_CFG}"

cd /dccstor/meghanadhp/projects/rag_recipes
mkdir -p logs/ingestbench/$GEN_CFG/retrieval
mkdir -p logs/ingestbench/$GEN_CFG/generation
mkdir -p logs/ingestbench/$GEN_CFG/indexing

# export HF_HOME=/dccstor/meghanadhp/hf_home
# export VLLM_CACHE_ROOT=/dccstor/meghanadhp/.cache/vllm
# export HOME=/dccstor/meghanadhp



# bsub -J rag_index_sparse_$GEN_CFG_job -M 256G -n 1 -oo logs/ingestbench/$GEN_CFG/indexing/sparse.out -eo logs/ingestbench/$GEN_CFG/indexing/sparse.err  uv run src/indexing.py --task_config configs/ingestbench_sparse.yaml --gen_cfg $GEN_CFG

# bsub -J rag_index_dense_$GEN_CFG_job -gpu "num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" -M 256G -n 1 -oo logs/ingestbench/$GEN_CFG/indexing/dense.out -eo logs/ingestbench/$GEN_CFG/indexing/dense.err  uv run src/indexing.py --task_config configs/ingestbench_dense.yaml --gen_cfg $GEN_CFG

# bsub -J rag_index_ensemble_$GEN_CFG_job -gpu "num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" -M 256G -n 1 -oo logs/ingestbench/$GEN_CFG/indexing/ensemble.out -eo logs/ingestbench/$GEN_CFG/indexing/ensemble.err  uv run src/indexing.py --task_config configs/ingestbench_ensemble.yaml --gen_cfg $GEN_CFG

# generate

# for now use an interactive job with 2 gpus
# bsub -J installer -U infusion -gpu "num=2/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" -M 100G -W 360 -Is /bin/bash

# ks=(5 10 20 30 40)
ks=(5)

for k in "${ks[@]}"; do
    # bsub -J rag_retrieval_sparse_k${k}_$GEN_CFG_job -M 256G -n 1 -oo logs/ingestbench/$GEN_CFG/retrieval/sparse_k${k}_$GEN_CFG.out -eo logs/ingestbench/$GEN_CFG/retrieval/sparse_k${k}_$GEN_CFG.err uv run src/retrieval.py --task_config configs/ingestbench_sparse.yaml --wandb_run_name "rag-$GEN_CFG-sparse-k${k}" --gen_cfg $GEN_CFG --top_k "$k" --filename "${GEN_CFG}_sparse_k${k}.jsonl"

    # bsub -J rag_retrieval_dense_k${k}_$GEN_CFG_job -gpu "num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" -M 256G -n 1 -oo logs/ingestbench/$GEN_CFG/retrieval/dense_k${k}_$GEN_CFG.out -eo logs/ingestbench/$GEN_CFG/retrieval/dense_k${k}_$GEN_CFG.err uv run src/retrieval.py --task_config configs/ingestbench_dense.yaml --wandb_run_name "rag-$GEN_CFG-dense-k${k}" --gen_cfg $GEN_CFG --top_k "$k" --filename "${GEN_CFG}_dense_k${k}.jsonl"

    # bsub -J rag_retrieval_ensemble_k${k}_$GEN_CFG_job -gpu "num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" -M 256G -n 1 -oo logs/ingestbench/$GEN_CFG/retrieval/ensemble_k${k}_$GEN_CFG.out -eo logs/ingestbench/$GEN_CFG/retrieval/ensemble_k${k}_$GEN_CFG.err uv run src/retrieval.py --task_config configs/ingestbench_ensemble.yaml --verbose --wandb_run_name "rag-$GEN_CFG-ensemble-k${k}" --gen_cfg $GEN_CFG --top_k "$k" --filename "${GEN_CFG}_ensemble_k${k}.jsonl"

    # bsub -J rag_generation_sparse_k${k}_$GEN_CFG_job -M 256G -n 1 -oo logs/ingestbench/$GEN_CFG/generation/sparse_k${k}_$GEN_CFG.out -eo logs/ingestbench/$GEN_CFG/generation/sparse_k${k}_$GEN_CFG.err uv run src/generation.py --task_config configs/ingestbench_sparse.yaml --wandb_run_name "rag-$GEN_CFG-sparse-k${k}" --llm_provider "rits" --gen_cfg $GEN_CFG --top_k "$k" --filename "${GEN_CFG}_sparse_k${k}.jsonl" --model_name deepseek-ai/DeepSeek-V3.2

    bsub -J rag_generation_dense_k${k}_$GEN_CFG_job -M 256G -n 1 -oo logs/ingestbench/$GEN_CFG/generation/dense_k${k}_$GEN_CFG.out -eo logs/ingestbench/$GEN_CFG/generation/dense_k${k}_$GEN_CFG.err uv run src/generation.py --task_config configs/ingestbench_dense.yaml --wandb_run_name "rag-$GEN_CFG-dense-k${k}" --llm_provider "rits" --gen_cfg $GEN_CFG --top_k "$k" --filename "${GEN_CFG}_dense_k${k}.jsonl" --model_name deepseek-ai/DeepSeek-V3.2

    # bsub -J rag_generation_ensemble_k${k}_$GEN_CFG_job -M 256G -n 1 -oo logs/ingestbench/$GEN_CFG/generation/ensemble_k${k}_$GEN_CFG.out -eo logs/ingestbench/$GEN_CFG/generation/ensemble_k${k}_$GEN_CFG.err 
    # uv run src/generation.py --task_config configs/ingestbench_ensemble.yaml --wandb_run_name "rag-$GEN_CFG-ensemble-k${k}" --llm_provider "rits" --gen_cfg $GEN_CFG --top_k "$k" --filename "${GEN_CFG}_ensemble_k${k}.jsonl" --model_name deepseek-ai/DeepSeek-V3.2
done
