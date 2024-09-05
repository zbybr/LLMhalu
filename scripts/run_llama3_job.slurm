#!/bin/bash
#SBATCH -J llmhalu
#SBATCH -t 0-12:00:00
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/research/LLMhalu/scripts/out/llama3_%j.out

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate llmhalu && \
python /home/mdafifal.mamun/research/LLMhalu/llama3/llama3_pipeline.py
# python /home/mdafifal.mamun/research/LLMhalu/evaluation/automated_evaluation.py --eval_type llama3 --dataset truthfulqa1.1