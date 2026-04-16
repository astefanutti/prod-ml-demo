#!/bin/bash
#SBATCH --job-name=smartshop-llm-finetune
#SBATCH --partition=slinky
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/llm-finetune-%j.out
#SBATCH --error=logs/llm-finetune-%j.err

# SmartShop AI - LLM Fine-Tuning on Slurm
# This script is called by Kubeflow Trainer via Kueue/Slinky integration
# or can be submitted directly: sbatch infrastructure/slurm/sbatch_llm_finetune.sh

set -euo pipefail

echo "=========================================="
echo "SmartShop LLM Fine-Tuning"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
echo "=========================================="

# Environment setup
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))

# NCCL tuning
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# S3 access for data and model artifacts
export S3_ENDPOINT="${S3_ENDPOINT:-http://minio.smartshop.svc.cluster.local:9000}"

# Data and output paths
DATA_DIR="${DATA_DIR:-s3://smartshop-features/llm_data}"
OUTPUT_DIR="${OUTPUT_DIR:-s3://smartshop-models/llm-adapter}"

mkdir -p logs

# Launch distributed training with torchrun
srun torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=${SLURM_GPUS_PER_NODE} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    training/llm/finetune.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 3 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --lr 2e-4 \
    --lora-r 16 \
    --lora-alpha 32

echo "Fine-tuning complete. Adapter saved to ${OUTPUT_DIR}"
