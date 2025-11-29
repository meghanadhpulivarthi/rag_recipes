#!/bin/zsh

cd /dccstor/meghanadhp/projects/rag_recipes
if [ ! -d "logs" ]; then
    mkdir logs
fi

# bsub -J rag_index_job -gpu "num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" -M 256G -n 1 -oo logs/rag_index_job.out -eo logs/rag_index_job.err  uv run src/indexing.py --task_config configs/ingestbench.yaml

# bsub -J rag_job -gpu "num=2/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" -M 256G -n 1 -oo logs/rag_job.out -eo logs/rag_job.err  uv run src/generation.py --task_config configs/ingestbench.yaml --wandb_enabled --wandb_run_name "rag" --llm_provider "hf"

# bsub -J rag_structured_job -gpu "num=2/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" -M 256G -n 1 -oo logs/rag_structured_job.out -eo logs/rag_structured_job.err  uv run src/generation.py --task_config configs/ingestbench.yaml --wandb_enabled --wandb_run_name "rag-structured-output" --llm_provider "vllm"

uv run src/generation.py --task_config configs/ingestbench.yaml --wandb_enabled --wandb_run_name "rag-structured-output" --llm_provider "vllm"
