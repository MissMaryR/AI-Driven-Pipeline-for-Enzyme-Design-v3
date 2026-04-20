#!/bin/bash --norc
# Generated with Siegel Lab HIVE Cluster Skill v1.1
#SBATCH --job-name=symmetrize
#SBATCH --partition=high
#SBATCH --account=publicgrp
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/out_symmetrize_%A_%a.out
#SBATCH --error=logs/err_symmetrize_%A_%a.err
#SBATCH --array=0-9   # Must equal NUM_RFD_TASKS * NUM_DESIGNS - 1 (matches 1_RfDiff.sh)

set -euo pipefail

mkdir -p logs

##### Set variables #####
NUM_DESIGNS=1          # Must match inference.num_designs in 1_RfDiff.sh

round=$(basename "$PWD")

# Decode 2D index to match the {task}_{design} naming from 1_RfDiff.sh
num1=$(( SLURM_ARRAY_TASK_ID / NUM_DESIGNS ))
num2=$(( SLURM_ARRAY_TASK_ID % NUM_DESIGNS ))
filename="${num1}_${num2}"

# Paths
RFDIFF_PDB="/quobyte/jbsiegelgrp/<username>/${round}/outputs/${filename}.pdb"
DOCKED_PDB="/quobyte/jbsiegelgrp/<username>/${round}/docked.pdb"
OUTPUT_PDB="/quobyte/jbsiegelgrp/<username>/${round}/outputs_sym/${filename}.pdb"

mkdir -p "$(dirname "$OUTPUT_PDB")"

if [[ ! -f "$RFDIFF_PDB" ]]; then
    echo "Error: $RFDIFF_PDB not found."
    exit 1
fi
if [[ ! -f "$DOCKED_PDB" ]]; then
    echo "Error: $DOCKED_PDB not found."
    exit 1
fi

# Activate PyMOL environment
eval "$(conda shell.bash hook)"
conda activate pymol_env

# symmetrize.py lives alongside this sbatch script in the round directory
SYM_SCRIPT="${SLURM_SUBMIT_DIR:-$PWD}/symmetrize.py"
if [[ ! -f "$SYM_SCRIPT" ]]; then
    echo "Error: symmetrize.py not found at $SYM_SCRIPT"
    exit 1
fi

echo "Symmetrizing ${filename}.pdb using C3 symmetry from docked.pdb..."
pymol -cq "$SYM_SCRIPT" -- "$DOCKED_PDB" "$RFDIFF_PDB" "$OUTPUT_PDB"

echo "Done: $OUTPUT_PDB"
