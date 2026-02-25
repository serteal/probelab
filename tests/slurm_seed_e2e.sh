#!/bin/bash
#SBATCH --job-name=seed-e2e
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=tests/seed_e2e_%j.log

set -euo pipefail
export HF_HOME=~/huggingface
export HF_TOKEN=$(grep HF_TOKEN ~/.bashrc | cut -d= -f2)
cd /mnt/nw/home/a.terre/scaling-probes/probelab

echo "=== $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi -L | head -1)"
echo ""

uv run pytest tests/test_seed_e2e.py -v -s 2>&1

echo ""
echo "=== Done $(date) ==="
