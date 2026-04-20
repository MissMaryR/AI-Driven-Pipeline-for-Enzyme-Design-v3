#!/bin/bash --norc
# Generated with Siegel Lab HIVE Cluster Skill v1.1
#SBATCH --job-name=top5
#SBATCH --partition=high
#SBATCH --account=publicgrp
#SBATCH --requeue
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/out_top5_%A_%a.out
#SBATCH --error=logs/err_top5_%A_%a.err

set -euo pipefail

mkdir -p logs

##### Set variables #####
round=$(basename "$PWD")
NUM_RFD_TASKS=10      # Must match --array range in 1_RfDiff.sh (0-9 → 10 tasks)
NUM_DESIGNS=1         # Must match inference.num_designs in 1_RfDiff.sh
NUM_RUNS=10           # Must match NUM_RUNS in 2_LigMPNN.sh

NUM_CHAINS=3          # Number of chains for AF3 prediction
                      # 1 = monomer          -> id: ["A"]
                      # 2 = dimer            -> id: ["A","B"]
                      # 3 = homotrimer       -> id: ["A","B","C"]

# Build chain ID string based on NUM_CHAINS
case $NUM_CHAINS in
    1) CHAIN_IDS='"A"' ;;
    2) CHAIN_IDS='"A","B"' ;;
    3) CHAIN_IDS='"A","B","C"' ;;
    *) echo "ERROR: NUM_CHAINS must be 1, 2, or 3."; exit 1 ;;
esac

# Initialize results array and counters
results=()
total_fa_found=0
total_fa_missing=0
declare -A per_design_best   # best overall_confidence seen per filename (e.g. "8_0")
declare -A per_design_count  # how many runs were found per filename

# Collect results across all array jobs and designs
# Mirrors the 2D index decoding in 2_LigMPNN.sh:
#   num1 = SLURM_ARRAY_TASK_ID / NUM_DESIGNS  (RFDiffusion task)
#   num2 = SLURM_ARRAY_TASK_ID % NUM_DESIGNS  (design index)
for task_id in $(seq 0 $(( NUM_RFD_TASKS * NUM_DESIGNS - 1 ))); do
    num1=$(( task_id / NUM_DESIGNS ))
    num2=$(( task_id % NUM_DESIGNS ))
    filename="${num1}_${num2}"
    per_design_count["$filename"]=0

    for i in $(seq 1 $NUM_RUNS); do

        fa_file="/quobyte/jbsiegelgrp/<username>/${round}/MPNN_outputs/${filename}_run${i}/seqs/${filename}.fa"

        if [[ -f "$fa_file" ]]; then
            (( total_fa_found++ )) || true
            (( per_design_count["$filename"]++ )) || true
            line=$(sed -n '3p' "$fa_file")

            overall_confidence=$(echo "$line" | grep -oP "(?<=overall_confidence=)[0-9.]+")
            id=$(echo "$line" | grep -oP "(?<=id=)[^,]+")
            T=$(echo "$line" | grep -oP "(?<=T=)[^,]+")
            seed=$(echo "$line" | grep -oP "(?<=seed=)[^,]+")
            ligand_confidence=$(echo "$line" | grep -oP "(?<=ligand_confidence=)[0-9.]+")
            seq_rec=$(echo "$line" | grep -oP "(?<=seq_rec=)[0-9.]+")

            results+=("$overall_confidence ${filename}_run${i} $filename $id $T $seed $ligand_confidence $seq_rec $fa_file")

            # Track best overall_confidence per design (filename = e.g. "8_0")
            current_best="${per_design_best[$filename]:-0}"
            if (( $(echo "$overall_confidence > $current_best" | bc -l) )); then
                per_design_best["$filename"]="$overall_confidence"
            fi
        else
            (( total_fa_missing++ )) || true
        fi
    done
done

if [[ ${#results[@]} -eq 0 ]]; then
    echo "ERROR: No .fa files found. Check that 2_LigMPNN.sh completed successfully."
    exit 1
fi

total_designs_seen=${#per_design_best[@]}
echo "Scan complete: $total_fa_found .fa files found, $total_fa_missing missing, across $total_designs_seen unique designs."

# -----------------------------------------------------------------------
# Select top 5 by picking the BEST RUN per unique design (filename),
# then ranking those per-design winners. This prevents one design's
# multiple runs from crowding out other designs.
# -----------------------------------------------------------------------

# Step 1: for each unique design, keep only its best-scoring run entry
declare -A best_entry_per_design
while IFS= read -r entry; do
    conf=$(echo "$entry" | awk '{print $1}')
    fname=$(echo "$entry" | awk '{print $3}')   # column 3 is the design filename (e.g. 8_0)
    current="${best_entry_per_design[$fname]:-}"
    if [[ -z "$current" ]]; then
        best_entry_per_design["$fname"]="$entry"
    else
        current_conf=$(echo "$current" | awk '{print $1}')
        if (( $(echo "$conf > $current_conf" | bc -l) )); then
            best_entry_per_design["$fname"]="$entry"
        fi
    fi
done < <(printf "%s\n" "${results[@]}")

# Step 2: sort the per-design winners and take top 5
sorted_results=$(printf "%s\n" "${best_entry_per_design[@]}" | sort -k1,1nr | head -n 5)

# -----------------------------------------------------------------------
# Build the full-scan summary table (all results, sorted)
# -----------------------------------------------------------------------
all_sorted=$(printf "%s\n" "${results[@]}" | sort -k1,1nr)

# Save ranking summary
output_file="/quobyte/jbsiegelgrp/<username>/${round}/top_5_overall_confidence.txt"
{
    echo "====================================================="
    echo "  Top-5 Selection Summary  —  Round: ${round}"
    echo "  Generated: $(date)"
    echo "====================================================="
    echo ""
    echo "SCAN STATISTICS"
    echo "---------------"
    echo "  RFDiffusion tasks scanned : $NUM_RFD_TASKS"
    echo "  Designs per task          : $NUM_DESIGNS"
    echo "  LigMPNN runs per design   : $NUM_RUNS"
    echo "  Total .fa files expected  : $(( NUM_RFD_TASKS * NUM_DESIGNS * NUM_RUNS ))"
    echo "  Total .fa files found     : $total_fa_found"
    echo "  Total .fa files missing   : $total_fa_missing"
    echo "  Unique designs with data  : $total_designs_seen"
    echo ""
    echo "SELECTION METHOD"
    echo "----------------"
    echo "  For each unique design (e.g. 8_0), the run with the"
    echo "  highest overall_confidence is selected as that design's"
    echo "  representative. The top 5 designs are then ranked by"
    echo "  their representative overall_confidence score."
    echo "  This ensures 5 DISTINCT designs are always chosen."
    echo ""
    echo "TOP 5 RESULTS (one winner per unique design)"
    echo "---------------------------------------------"
    printf "%-6s %-22s %-8s %-8s %-8s %-8s %-20s %-10s %-8s\n" \
        "Rank" "Run" "Design" "OverConf" "LigConf" "SeqRec" "ID" "T" "Seed"
    echo "------  ----------------------  --------  --------  --------  --------  --------------------  ----------  --------"
    rank=1
    echo "$sorted_results" | while IFS= read -r result; do
        conf=$(echo "$result"     | awk '{print $1}')
        run_label=$(echo "$result"  | awk '{print $2}')
        design=$(echo "$result"   | awk '{print $3}')
        id=$(echo "$result"       | awk '{print $4}')
        T=$(echo "$result"        | awk '{print $5}')
        seed=$(echo "$result"     | awk '{print $6}')
        lig=$(echo "$result"      | awk '{print $7}')
        seqrec=$(echo "$result"   | awk '{print $8}')
        printf "%-6s %-22s %-8s %-8s %-8s %-8s %-20s %-10s %-8s\n" \
            "$rank" "$run_label" "$design" "$conf" "$lig" "$seqrec" "$id" "$T" "$seed"
        rank=$(( rank + 1 ))
    done
    echo ""
    echo "PER-DESIGN BEST CONFIDENCE SCORES (all designs)"
    echo "------------------------------------------------"
    printf "%-10s  %-10s  %-6s\n" "Design" "BestConf" "Runs"
    echo "----------  ----------  ------"
    for fname in $(echo "${!per_design_best[@]}" | tr ' ' '\n' | sort); do
        printf "%-10s  %-10s  %-6s\n" \
            "$fname" "${per_design_best[$fname]}" "${per_design_count[$fname]}"
    done
    echo ""
    echo "FULL SCAN — ALL ENTRIES (sorted by overall_confidence desc)"
    echo "------------------------------------------------------------"
    printf "%-22s %-8s %-8s %-8s %-8s %-20s %-10s %-8s\n" \
        "Run" "OverConf" "LigConf" "SeqRec" "Design" "ID" "T" "Seed"
    echo "----------------------  --------  --------  --------  --------  --------------------  ----------  --------"
    echo "$all_sorted" | while IFS= read -r result; do
        conf=$(echo "$result"     | awk '{print $1}')
        run_label=$(echo "$result"  | awk '{print $2}')
        design=$(echo "$result"   | awk '{print $3}')
        id=$(echo "$result"       | awk '{print $4}')
        T=$(echo "$result"        | awk '{print $5}')
        seed=$(echo "$result"     | awk '{print $6}')
        lig=$(echo "$result"      | awk '{print $7}')
        seqrec=$(echo "$result"   | awk '{print $8}')
        printf "%-22s %-8s %-8s %-8s %-8s %-20s %-10s %-8s\n" \
            "$run_label" "$conf" "$lig" "$seqrec" "$design" "$id" "$T" "$seed"
    done
} > "$output_file"

echo "Top 5 summary written to: $output_file"

# Write top 5 AlphaFold3 JSON files
output_dir="/quobyte/jbsiegelgrp/<username>/${round}/top_5_af3_inputs"
mkdir -p "$output_dir"

rank=1
echo "$sorted_results" | while IFS= read -r result; do
    fa_file_path=$(echo "$result" | awk '{print $9}')

    # Line 4 is the full sequence — the monomer is repeated 3 times (homotrimer).
    # Find the repeat boundary by searching for the second occurrence of the
    # first 20 characters (a unique anchor) within the full sequence.
    full_seq=$(sed -n '4p' "$fa_file_path")
    anchor="${full_seq:0:20}"
    monomer_len=$(python3 -c "s='${full_seq}'; anchor=s[:20]; pos=s.find(anchor,1); print(pos)")
    if [[ -z "$monomer_len" || "$monomer_len" -le 0 ]]; then
        echo "Warning: top_${rank} could not find repeat boundary, falling back to full_len/3"
        monomer_len=$(( ${#full_seq} / 3 ))
    fi
    monomer_seq="${full_seq:0:$monomer_len}"

    actual_len=${#monomer_seq}
    if [[ $actual_len -lt $monomer_len ]]; then
        echo "Warning: top_${rank} sequence is only ${actual_len} residues (expected ${monomer_len})"
    fi

    # Name uses round and rank, e.g. "HIVE_1"
    name="${round}_${rank}"

    json_file="${output_dir}/top_${rank}.json"
    cat > "$json_file" << JSONEOF
{
  "name": "${name}",
  "sequences": [
    {
      "protein": {
          "id": [${CHAIN_IDS}],
        "sequence": "${monomer_seq}"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
JSONEOF

    echo "Created: $json_file (${actual_len} residues, ${NUM_CHAINS} chain(s))"
    rank=$((rank + 1))
done

echo "Done. Top 5 AlphaFold3 JSON files saved to: $output_dir"

# Auto-submit AF3 array job for the top 5 JSONs
AF3_SCRIPT=""
for candidate_dir in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$(dirname "$0")"; do
    if [[ -n "$candidate_dir" && -f "$candidate_dir/4_AF3_bulk.py" ]]; then
        AF3_SCRIPT="$candidate_dir/4_AF3_bulk.py"
        break
    fi
done

if [[ -z "$AF3_SCRIPT" ]]; then
    echo "WARNING: 4_AF3_bulk.py not found in \$SLURM_SUBMIT_DIR, \$PWD, or script dir — skipping AF3 submission."
    echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    echo "  PWD=$PWD"
    echo "  script dir=$(dirname "$0")"
    echo "To submit manually: python 4_AF3_bulk.py $output_dir"
else
    echo ""
    echo "Found AF3 submitter: $AF3_SCRIPT"
    echo "Submitting AF3 array job for top 5 JSONs..."
    python "$AF3_SCRIPT" "$output_dir"
fi
