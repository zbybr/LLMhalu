#!/bin/bash
#SBATCH -J llmhalu
#SBATCH -t 0-12:00:00
#SBATCH -n 1
#SBATCH --mem=80G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/research/LLMhalu/scripts/out/%j.out

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate llmhalu && \
python /home/mdafifal.mamun/research/LLMhalu/llama3/llama3_pipeline.py