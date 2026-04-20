"""
AlphaFold 3 Array Job Submission Script
Generated with Siegel Lab HIVE Cluster Skill v1.1

Finds all JSON files in the current directory (or a provided directory) and
submits them as a SLURM array job with GPU monitoring and runtime reporting.

Usage:
    python AF3_bulk.py               # uses current working directory
    python AF3_bulk.py <directory>   # uses specified directory
"""

import sys
import subprocess
from pathlib import Path
import argparse

# Concurrency cap: max simultaneous array tasks
MAX_CONCURRENT = 20


def main():
    parser = argparse.ArgumentParser(description='Submit AlphaFold 3 predictions as SLURM array job')
    parser.add_argument('directory', nargs='?', default='.', help='Directory containing JSON files (default: current directory)')
    args = parser.parse_args()

    input_dir = Path(args.directory).resolve()

    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        sys.exit(1)

    n = len(json_files)
    print(f"Found {n} JSON file(s) to process")
    print(f"Directory: {input_dir}")
    print()

    # Setup directories
    base_output_dir = input_dir / f"{input_dir.name}_output"
    logs_dir = input_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    print(f"Base output directory: {base_output_dir}")
    print(f"Logs directory: {logs_dir}")
    print()

    # Write 0-indexed file list for array job (sed uses 1-based, so task_id + 1)
    json_list_file = logs_dir / "json_files_list.txt"
    with open(json_list_file, 'w') as f:
        for json_file in json_files:
            f.write(f"{json_file.name}\n")

    print(f"Created file list: {json_list_file}")

    concurrency = min(MAX_CONCURRENT, n)
    array_spec = f"0-{n - 1}%{concurrency}" if n > 1 else "0"
    print(f"Array job spec: --array={array_spec}")
    print()

    af3_dir = "/quobyte/jbsiegelgrp/software/alphafold3"
    slurm_script_path = input_dir / "af3_array_job.sbatch"

    slurm_script_content = f"""\
#!/bin/bash --norc
# Generated with Siegel Lab HIVE Cluster Skill v1.1
#SBATCH --job-name=af3_array
#SBATCH --partition=gpu-a100
#SBATCH --account=genome-center-grp
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array={array_spec}
#SBATCH --output={logs_dir}/af3_%A_%a.out
#SBATCH --error={logs_dir}/af3_%A_%a.err

set -euo pipefail

module load apptainer/latest

mkdir -p "{logs_dir}"

# 0-indexed: SLURM_ARRAY_TASK_ID=0 → line 1
JSON_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{json_list_file}")
if [ -z "$JSON_FILE" ]; then
    echo "Error: Could not get JSON file for array task ${{SLURM_ARRAY_TASK_ID}}"
    exit 1
fi

BASE_NAME="${{JSON_FILE%.json}}"
SAFE_NAME=$(echo "$BASE_NAME" | tr ' ' '_' | tr -cd '[:alnum:]._-')

INPUT_DIR="{input_dir}"
OUTPUT_DIR="{base_output_dir}/${{SAFE_NAME}}"
LOGS_DIR="{logs_dir}"

mkdir -p "$OUTPUT_DIR"

START_TIME=$(date +%s)
echo "Starting AlphaFold 3 prediction for $JSON_FILE at $(date)"
echo "Job ID:          ${{SLURM_JOB_ID}}"
echo "Array Task ID:   ${{SLURM_ARRAY_TASK_ID}}"
echo "Running on node: ${{SLURM_NODELIST}}"
echo "Input JSON:      $INPUT_DIR/$JSON_FILE"
echo "Output dir:      $OUTPUT_DIR"

# Background GPU monitoring
(
    while true; do
        nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \\
            --format=csv,noheader \\
            >> "${{LOGS_DIR}}/${{SAFE_NAME}}_gpu_${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.csv" 2>/dev/null
        sleep 5
    done
) &
MONITOR_PID=$!

# Run AlphaFold 3
singularity exec \\
    --bind "$INPUT_DIR:/input" \\
    --bind "$OUTPUT_DIR:/output" \\
    --bind "{af3_dir}:/models" \\
    --bind "{af3_dir}/public_databases:/databases" \\
    --bind "$LOGS_DIR:/logs" \\
    --nv \\
    "{af3_dir}/alphafold3.sif" \\
    python /app/alphafold/run_alphafold.py \\
        --json_path="/input/$JSON_FILE" \\
        --model_dir=/models \\
        --output_dir=/output \\
        --db_dir=/databases

# Stop GPU monitor
kill $MONITOR_PID 2>/dev/null || true

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
HOURS=$((RUNTIME / 3600))
MINUTES=$(( (RUNTIME % 3600) / 60 ))
SECONDS=$((RUNTIME % 60))
echo "Prediction completed at $(date)"
echo "Total runtime: ${{HOURS}}h ${{MINUTES}}m ${{SECONDS}}s"

# Summarise GPU usage
GPU_CSV="${{LOGS_DIR}}/${{SAFE_NAME}}_gpu_${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.csv"
if [ -f "$GPU_CSV" ]; then
    echo ""
    echo "=== Resource Usage Summary ==="

    PEAK_VRAM=$(awk -F', ' '{{gsub(" MiB","",$3); if($3>max) max=$3}} END {{printf "%.1f", max/1024}}' "$GPU_CSV" 2>/dev/null || echo "N/A")
    TOTAL_VRAM=$(awk -F', ' 'NR==1 {{gsub(" MiB","",$4); printf "%.1f", $4/1024}}' "$GPU_CSV" 2>/dev/null || echo "N/A")
    AVG_UTIL=$(awk -F', ' '{{gsub(" %","",$5); sum+=$5; count++}} END {{if(count>0) printf "%.1f", sum/count; else print "N/A"}}' "$GPU_CSV" 2>/dev/null || echo "N/A")
    GPU_NAME=$(awk -F', ' 'NR==1 {{print $2}}' "$GPU_CSV" 2>/dev/null || echo "Unknown")

    echo "GPU:                   ${{GPU_NAME}}"
    echo "Peak VRAM:             ${{PEAK_VRAM}} GB / ${{TOTAL_VRAM}} GB"
    echo "Avg GPU utilisation:   ${{AVG_UTIL}}%"
    echo ""
    echo "CPU/Memory efficiency (from SLURM):"
    seff "${{SLURM_JOB_ID}}" 2>/dev/null \\
        | grep -E "(CPU Efficiency|Memory Efficiency|Memory Utilized)" \\
        || echo "Unable to retrieve SLURM efficiency data"
    echo "=============================="

    rm -f "$GPU_CSV"
fi

echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR/"
echo ""
echo "AlphaFold 3 prediction for $JSON_FILE finished successfully!"
"""

    with open(slurm_script_path, 'w') as f:
        f.write(slurm_script_content)

    print(f"Created SLURM script: {slurm_script_path}")
    print()

    print("Submitting array job...")
    try:
        result = subprocess.run(
            ["sbatch", str(slurm_script_path)],
            cwd=input_dir,
            capture_output=True,
            text=True,
            check=True
        )

        print("Job submitted successfully!")
        print(f"SLURM output: {result.stdout.strip()}")

        if "Submitted batch job" in result.stdout:
            job_id = result.stdout.split()[-1]
            print()
            print("Useful commands:")
            print(f"  Check job status:    squeue -j {job_id}")
            print(f"  Check all tasks:     squeue -j {job_id} -t all")
            print(f"  Cancel job:          scancel {job_id}")
            print(f"  Monitor a log:       tail -f {logs_dir}/af3_{job_id}_<task>.out")
            print(f"  Check efficiency:    sacct -j {job_id} --format=JobID,Elapsed,CPUTime,MaxRSS")

    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: sbatch command not found. Make sure SLURM is available.")
        sys.exit(1)


if __name__ == "__main__":
    main()
