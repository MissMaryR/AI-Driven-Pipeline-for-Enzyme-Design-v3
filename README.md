# AI-Driven Pipeline for Enzyme Design with a Trimeric Structure
## RFDiffusion → Symmetrize → LigandMPNN → AlphaFold3
### Design enzymes with novel inserted sequences

This pipeline takes a docked protein-ligand complex (`.pdb`) and:
1. Uses **RFDiffusion All-Atom** to generate backbone structures with a novel inserted loop on chain A
2. **Symmetrizes** the RFDiffusion output into a true C3 homotrimer using the C3 symmetry from `docked.pdb` (PyMOL-based pre-processing step)
3. Uses **LigandMPNN** to design sequences for the inserted region with cross-chain symmetry enforced across all three chains
4. Selects the **top 5** designs by confidence score
5. Runs **AlphaFold3** structure prediction on the top designs

---

## Requirements

The following tools must be installed and accessible on your HPC cluster:

- [RFDiffusion All-Atom](https://github.com/baker-laboratory/rf_diffusion_all_atom) — with Apptainer/Singularity container (`rf_se3_diffusion.sif`)
- [PyMOL](https://pymol.org/) — installed in a conda env named `pymol_env`, with `pymol` executable on `PATH` (used by `1b_symmetrize.sh`)
- [LigandMPNN](https://github.com/dauparas/LigandMPNN) — with a conda environment
- [AlphaFold3](https://github.com/google-deepmind/alphafold3) — with Singularity container, model weights, and `public_databases/`
- SLURM workload manager
- Apptainer/Singularity


---

## Runtime Estimates

Wall-clock time depends heavily on cluster GPU availability. The numbers below assume the `gpu-a100` partition with ~10 concurrent A100 GPUs available and no heavy queue contention. Real times can be longer if the queue is busy.

Per-task rough costs (observed):

- RFDiffusion: **~20–40 min per design** (1 GPU, sequential within an array task)
- Symmetrize: **< 30 s per design** (CPU only — PyMOL alignment + coordinate rotation)
- LigandMPNN: **~30 s per run**, so `NUM_RUNS=10` → ~5 min per array task (1 GPU)
- Top 5 selection: **< 1 min** regardless of scale (CPU-only file parsing)
- AlphaFold3: **~30–45 min per prediction** for a ~270 aa homotrimer (1 GPU). Only 5 predictions ever run here, so this stage has a fixed cost.

End-to-end estimates at `NUM_DESIGNS=1`, `NUM_RUNS=10`:

| RFDiff designs | LigMPNN total runs | 1) RFDiff | 1b) Sym | 2) LigMPNN | 3) Top 5 | 4) AF3 (top 5) | **End-to-end (parallel)** |
|---|---|---|---|---|---|---|---|
| 10 | 100 | ~30–40 min | <1 min | ~5–10 min | <1 min | ~30–45 min | **~1–1.5 h** |
| 100 | 1,000 | ~5–7 h (10× batches of 10) | ~1–2 min | ~30–60 min | ~1 min | ~30–45 min | **~6–9 h** |
| 1,000 | 10,000 | ~2–3 days (100× batches of 10) | ~5–10 min | ~5–10 h | ~5 min | ~30–45 min | **~2.5–4 days** |

Notes:
- "End-to-end (parallel)" assumes SLURM lets you run ~10 array tasks concurrently. A busier cluster will stretch the RFDiff/LigMPNN rows proportionally; a less busy one (or a bigger share of GPUs) can collapse them toward the single-task cost.
- `1_RfDiff.sh` sets `--time=01:30:00` per array task, `1b_symmetrize.sh` sets `--time=00:10:00`, `2_LigMPNN.sh` sets `--time=00:15:00`. If you increase `inference.num_designs` so that a single task runs multiple designs serially, bump script 1's wall time accordingly (30–40 min per design).
- Step 4 runs a fixed 5 AF3 predictions regardless of how many RFDiff designs you generated, so it doesn't scale with the other steps.

---

## Setup: Configure Paths and Parameters

Before submitting any job, open each script and update the following hardcoded values:

| What to edit | Script | Description |
|---|---|---|
| `#SBATCH --array=...` | 1 | Number of RFDiffusion array tasks (scaling — see "Scaling Up" below) |
| `conda activate ...` | 1 | Path to your RFDiffusion conda env (e.g. `SE3nv`) |
| `SCRIPT_DIR=...` | 1 | Path to your RFDiffusion All-Atom installation |
| `input_pdb_path=...` | 1 | Full path to your `docked.pdb` input file. The scripts use a `<username>` placeholder (e.g. `/quobyte/jbsiegelgrp/<username>/${round}/docked.pdb`) — replace `<username>` with your actual cluster username and change the base path to your project root |
| `output_prefix=...` | 1 | Full path prefix for RFDiffusion output PDBs (same `<username>`/base path as `input_pdb_path`) |
| `inference.ligand=...` | 1 | 3-letter ligand residue code from your PDB (e.g. `ATP`, `HEM`, `4EP`) |
| `contigmap.contigs=...` | 1 | Contig string describing your protein topology and insertion |
| `inference.num_designs=...` | 1 | Number of designs per array task |
| `#SBATCH --array=...` | 1b | Must equal `NUM_RFD_TASKS * NUM_DESIGNS - 1` (same as script 1) |
| `NUM_DESIGNS=...` | 1b | Must match `inference.num_designs` in script 1 |
| `conda activate pymol_env` | 1b | Name/path of your PyMOL conda env (must have `pymol` executable on `PATH`) |
| `RFDIFF_PDB=...`, `DOCKED_PDB=...`, `OUTPUT_PDB=...` | 1b | Paths follow the `<username>`/`${round}` convention — replace `<username>` with your cluster username |
| `#SBATCH --array=...` | 2 | Must equal `NUM_RFD_TASKS * NUM_DESIGNS - 1` (e.g. `0-9`) |
| `NUM_RFD_TASKS=...` | 2 | Must match `--array` range in script 1 |
| `NUM_DESIGNS=...` | 2 | Must match `inference.num_designs` in script 1 |
| `NUM_RUNS=...` | 2 | Number of independent LigandMPNN sequences to generate per design |
| `TORCH_HOME=...` | 2 | Path to the LigandMPNN torch cache directory |
| `conda activate ...` | 2 | Path to your LigandMPNN conda environment |
| `LIGAND_MPNN_DIR=...` | 2 | Path to your LigandMPNN installation |
| `pdb_file=...` | 2 | Base path to the **symmetrized** PDBs from step 1b (default `outputs_sym/${num1}_${num2}.pdb`) |
| `out_folder=...` | 2 | Base path for LigandMPNN output folders |
| `--redesigned_residues ...` | 2 | Space-separated residues to redesign **across all three chains** (A, B, C) — use `update_MPNN.py` to generate automatically |
| `--symmetry_residues ...` | 2 | Cross-chain symmetry groups linking equivalent positions on A/B/C (format: `A151,B151,C151\|A152,B152,C152\|...`) — generated by `update_MPNN.py` |
| `--symmetry_weights ...` | 2 | Matching weights per group (e.g. `0.33,0.33,0.33\|0.33,0.33,0.33\|...`) — generated by `update_MPNN.py` |
| `NUM_RFD_TASKS=...` | 3 | Number of RFDiffusion array tasks (must match `--array` in script 1) |
| `NUM_DESIGNS=...` | 3 | Must match `inference.num_designs` in script 1 |
| `NUM_RUNS=...` | 3 | Must match `NUM_RUNS` in script 2 |
| `NUM_CHAINS=...` | 3 | Number of chains for AF3 prediction (1=monomer, 2=dimer, 3=homotrimer) |
| Base path `/quobyte/jbsiegelgrp/<username>/${round}/...` | 3 | Replace `<username>` with your cluster username (appears in `fa_file`, `output_file`, and `output_dir`) and change the base path if your `MPNN_outputs/` live elsewhere |
| `af3_dir = ...` | 4 | Path to your AlphaFold3 installation — inside the `main()` function |
| `MAX_CONCURRENT = ...` | 4 | Max simultaneous AF3 array tasks (default `20`) |
| `#SBATCH` directives | 4 | `--partition`, `--account`, `--time`, `--mem`, `--cpus-per-task`, `--gres=gpu:1` inside the `slurm_script_content` string in `main()` |

Also update `--account` and `--partition` in every `#SBATCH` header to match your cluster.

---

## Directory Structure

Each design round gets its own directory. All scripts use `basename "$PWD"` to detect the round name automatically, so **always run scripts from within the round directory**.

```
your_project/
└── round_name/          # e.g. "round1" — name this whatever you like
    ├── docked.pdb                   # Input: homotrimer + ligand (provides C3 symmetry)
    ├── 1_RfDiff.sh                  # Script 1
    ├── 1b_symmetrize.sh             # Script 1b (SLURM wrapper for PyMOL step)
    ├── symmetrize.py                # PyMOL helper called by 1b
    ├── 2_LigMPNN.sh                 # Script 2
    ├── 3_Top5.sh                    # Script 3
    ├── 4_AF3_bulk.py                # Script 4
    ├── update_MPNN.py               # Populates residue/symmetry flags in 2_LigMPNN.sh
    ├── logs/                        # All SLURM logs (auto-created)
    ├── outputs/                     # RFDiffusion backbone outputs (A has insert; B, C are WT context)
    ├── outputs_sym/                 # C3-symmetric homotrimers produced by step 1b
    ├── MPNN_outputs/                # LigandMPNN sequence designs
    └── top_5_af3_inputs/            # Top 5 AF3 JSON inputs + results
```

Your `docked.pdb` should contain:
- A C3-symmetric homotrimer (chains A, B, C — this is what `1b_symmetrize.sh` reads to recover the C3 operation)
- A ligand (the HETATM residue you specify as `inference.ligand` in `1_RfDiff.sh`)

---

Useful monitoring commands:
```bash
squeue -u <username>         #shows all of your running jobs
squeue -u <username> -o%j -h | sort | uniq -c | sort -rn    #shows your jobs organized by job name
squeue -j <job_id>                     # Check job status
squeue -j <job_id> -t all              # Check all array tasks
scancel <job_id>                       # Cancel job
tail -f /logs/rf_out_*.out  # Monitor live logs
```

## 1) RFDiffusion — Backbone Generation

Configure `1_RfDiff.sh`:

- **`CONTIGS`**: Defines the protein topology and where the new loop is inserted.
  - Format: `'[ChainResStart-ResEnd,insert_len-insert_len,ChainResStart-ResEnd,...]'`
  - Multimer Example: `['A1-100,12-12,A102-200,B1-200,C1-200']` inserts 12 residues between A100 and A102 on chain A, with chains B and C passed in as fixed structural context
  - note - RFdiffusion does not apply design/insertions to all three chains simultaneously. All three chains (A, B, C) are provided so RFDiffusion sees the full trimeric environment, but the insertion happens only on chain A. This is **intentional** — the insert should be identical across all three chains in the final homotrimer, so we generate it once on chain A and then use `1b_symmetrize.sh` (Step 1b below) to build a true C3 homotrimer by rotating chain A into the B and C positions using the C3 axis from `docked.pdb`. LigandMPNN then designs that symmetric trimer with cross-chain symmetry constraints.
  - Monomer Example: `['A1-100,12-12,A102-200']` inserts 12 residues between A100 and A102 in a single chain A
  - The insert amount can be adjusted to a range ex: 10-20 and will generate a range of structures, but deciding on one length is critical for this pipeline as LigandMPNN needs specific residues to design on which can vary when running a range. We use a script later on that reads the 1_RFDiff script and automatically updates the designed residues in 2_LigMPNN but it must be in the 12-12 format. 
  - See the [RFDiffusion All-Atom docs](https://github.com/baker-laboratory/rf_diffusion_all_atom) for full syntax
- **`--array`**: adjust this to decide how many designs to generate
  - each design takes 20-40 minutes to run

Run:
```bash
sbatch 1_RfDiff.sh
```

Output: `outputs/` folder with PDBs named `{array_id}_{design_id}.pdb` (e.g. `0_0.pdb`, `0_1.pdb`). In each file, chain A has the new insertion and chains B/C are the wild-type context carried over from `docked.pdb`.


---

## 1b) Symmetrize — Build a True C3 Homotrimer

Between RFDiffusion and LigandMPNN we rebuild the PDB into a proper C3 homotrimer so that LigandMPNN sees all three chains with the insert in symmetric positions. This is what lets LigandMPNN actually enforce cross-chain symmetry and make good decisions at the inter-chain interface the insert will sit at.

`1b_symmetrize.sh` is a SLURM array wrapper that activates the `pymol_env` conda env and calls `symmetrize.py` on each RFDiffusion output PDB. `symmetrize.py`:
1. Loads `docked.pdb` (your original homotrimer with ligand) and the RFDiffusion output.
2. Extracts chain A (with the new insertion) plus any hetatm/ligand atoms as a template.
3. Makes three copies of the template and aligns each copy onto the corresponding chain (A, B, C) in `docked.pdb` using PyMOL's `cmd.align`. Because `cmd.align` does sequence-aware structural alignment, the 12-residue insert region is skipped during alignment (docked has no insert to match it to) and the shared fixed residues drive the C3 rotation. The ligand atoms in each copy are transformed along with their protein chain, so all three chains end up with their own ligand copy.
4. Relabels the chains A/B/C and writes the merged PDB to `outputs_sym/{array_id}_{design_id}.pdb`.

`symmetrize.py` runs its entry point at **module level** (no `if __name__ == "__main__":` guard) because `pymol -cq script.py` does not reliably set `__name__` to `"__main__"` — guarding the entry point that way causes PyMOL to exit cleanly having done nothing. It also prints per-step atom counts and cmd.align RMSDs, and verifies the output file exists and is non-empty before returning; if anything goes wrong (empty template, empty final object, missing output file) it `sys.exit`s with a clear error. If you edit `symmetrize.py`, keep the top-level call and avoid re-introducing a `__main__` guard.

Configure `1b_symmetrize.sh`:

- **`#SBATCH --array=...`**: Must match `1_RfDiff.sh`.
- **`NUM_DESIGNS=...`**: Must match `inference.num_designs` in `1_RfDiff.sh`.
- **`conda activate pymol_env`**: Name of your PyMOL conda env. `pymol` must be available on `PATH` in that env.
- **Paths** (`RFDIFF_PDB`, `DOCKED_PDB`, `OUTPUT_PDB`): Follow the `<username>`/`${round}` convention. Replace `<username>` with your cluster username.

Run:
```bash
sbatch 1b_symmetrize.sh
```

Output: `outputs_sym/{array_id}_{design_id}.pdb` — each file is a C3-symmetric homotrimer where chains A, B, and C share the same coordinates (rotated) and residue count, and the ligand appears three times (once per chain).

---

## 2) LigandMPNN — Sequence Design

LigandMPNN now runs on the symmetric trimers from step 1b and is constrained to produce **one** sequence that applies to all three chains simultaneously (so the insert residues are identical on A, B, and C in the output).

Configure `2_LigMPNN.sh`:

- **update_MPNN.py**: run this on your mac before submitting the job. It reads the contigs from `1_RfDiff.sh` and rewrites three flags in `2_LigMPNN.sh` for you:
  ```
  python3 update_MPNN.py
  ```
  - **`--redesigned_residues`** — expanded to cover every insert position on all three chains (e.g. `A151…A162 B151…B162 C151…C162` for a 12-residue insert).
  - **`--symmetry_residues`** — cross-chain groups linking equivalent positions, format `A151,B151,C151|A152,B152,C152|...|A162,B162,C162`.
  - **`--symmetry_weights`** — `0.33,0.33,0.33|0.33,0.33,0.33|...` (one weight per chain per group).

  This replaces what used to be a manual step for chain-A-only redesign. It's common to also design a couple of residues on either side of the insert — if you want that, edit the residue list by hand after running `update_MPNN.py` (or extend the contig in `1_RfDiff.sh` so the widened region is captured automatically).
- **`pdb_file`**: Points at `outputs_sym/${num1}_${num2}.pdb` (the symmetrized homotrimers from step 1b), not `outputs/`.
- **`--array`**: Must match `1_RfDiff.sh` / `1b_symmetrize.sh`.
- **`NUM_RUNS`**: How many independent LigandMPNN sequences to generate per design (default: 10).
- See [LigandMPNN](https://github.com/dauparas/LigandMPNN) for additional options.

Run:
```bash
sbatch 2_LigMPNN.sh
```

Output: `MPNN_outputs/` with one subfolder per design per run, each containing a `seqs/` directory with `.fa` files. Because the input is a symmetric trimer and cross-chain symmetry is enforced, the designed sequence on line 4 of each `.fa` file should be the monomer repeated three times — i.e. a true homotrimer.

---

## 3) Top 5 Selection

`3_Top5.sh` walks every `MPNN_outputs/{task}_{design}_run{i}/seqs/*.fa` file and parses the header for each sequence. It then selects the top 5 in two stages to ensure 5 **distinct** designs are chosen (not 5 runs of the same design):

1. For each unique design (e.g. `8_0`), it keeps only the single run with the highest **`overall_confidence`** as that design's representative.
2. Those per-design winners are ranked by `overall_confidence` (descending), and the top 5 are selected.

An AlphaFold3 JSON input is written for each of the top 5.

Configure `3_Top5.sh`:

```bash
NUM_RFD_TASKS=10   # Must match --array range in 1_RfDiff.sh (0-9 → 10 tasks)
NUM_DESIGNS=1      # Must match inference.num_designs in 1_RfDiff.sh
NUM_RUNS=10        # Must match NUM_RUNS in 2_LigMPNN.sh
NUM_CHAINS=3       # 1=monomer, 2=dimer, 3=homotrimer
```

The monomer sequence is **not** hardcoded by length. The script reads line 4 of each `.fa` file (the full designed sequence, which for a homotrimer is the monomer repeated 3×) and finds the repeat boundary by searching for the second occurrence of the first 20 residues. If that fails, it falls back to `len(full_seq) / 3`.

Run:
```bash
sbatch 3_Top5.sh
```

Output:
- `top_5_overall_confidence.txt` — a full report for the round containing:
  - **Scan statistics**: number of RFDiffusion tasks, designs per task, LigMPNN runs per design, total `.fa` files expected vs. found vs. missing, and the number of unique designs with data
  - **Selection method**: a short description of the "one winner per unique design" logic used to pick the top 5
  - **Top 5 results**: Rank, Run, Design, OverConf, LigConf, SeqRec, ID, T, Seed
  - **Per-design best confidence scores**: for every unique design, the best `overall_confidence` seen and how many runs contributed
  - **Full scan**: every `.fa` entry found, sorted by `overall_confidence` (descending)
- `top_5_af3_inputs/` — one AlphaFold3 JSON file per top design (`top_1.json` … `top_5.json`); check that these files were generated correctly


Example JSON:
```json
{
  "name": "round1_1",
  "sequences": [
    {
      "protein": {
        "id": ["A","B","C"],
        "sequence": "GEVRHLKMYAE..."
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

---

## 4) AlphaFold3 Structure Prediction

**Auto-submission:** After `3_Top5.sh` finishes writing the top 5 JSON files, it automatically looks for `4_AF3_bulk.py` in `$SLURM_SUBMIT_DIR`, `$PWD`, or the directory the script was launched from, and submits the AF3 array job for you. If it can't find `4_AF3_bulk.py`, it prints a warning and falls back to manual submission.

To submit manually (if auto-submission is skipped or you want to re-run):

```bash
python 4_AF3_bulk.py /path/to/your/project/round_name/top_5_af3_inputs
```

> **Note:** The path to the AlphaFold3 installation (`af3_dir`) is hardcoded inside the `main()` function of `4_AF3_bulk.py`. Edit that variable directly before running. Default: `/quobyte/jbsiegelgrp/software/alphafold3`.

This script scans the target directory for `*.json` files, writes a file list to `logs/json_files_list.txt`, generates `af3_array_job.sbatch` with `--array=0-{N-1}%{MAX_CONCURRENT}` (default concurrency cap: 20), and submits it with `sbatch`. Each array task runs one AlphaFold3 prediction inside a Singularity container on 1 GPU, logs GPU utilization every 5 s via `nvidia-smi`, and prints a resource-usage summary (runtime, peak VRAM, average GPU utilization, and `seff`-based CPU/memory efficiency) at the end of each task log.

SLURM defaults inside the generated sbatch (edit in `4_AF3_bulk.py` to change): partition `gpu-a100`, account `genome-center-grp`, 1 h wall time, 16 CPUs, 64 GB RAM, 1 GPU.

Output:
- `top_5_af3_inputs/top_5_af3_inputs_output/<name>/` — AF3 predictions per design (structures, pLDDT / PAE / pTM / ipTM, auxiliary files)
- `top_5_af3_inputs/logs/af3_<job_id>_<task_id>.{out,err}` — per-task SLURM logs with runtime + resource summaries
- `top_5_af3_inputs/logs/json_files_list.txt` — 1-indexed JSON file list used by the array job
- `top_5_af3_inputs/af3_array_job.sbatch` — the generated submission script (useful for debugging or re-submitting)
