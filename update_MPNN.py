#!/usr/bin/env python3
"""
Parses contigmap.contigs from 1_RfDiff.sh to find the inserted (gap) residues
on chain A, computes their output PDB residue numbers (accounting for
cumulative insertions), and then expands the list across chains A/B/C to
match the symmetric homotrimer PDB produced by 1b_symmetrize.sh + symmetrize.py.

It overwrites three flags in 2_LigMPNN.sh:

  --redesigned_residues : every insert position on A, B, and C
                           e.g. "A151 A152 ... A162 B151 B152 ... B162 C151 ... C162"

  --symmetry_residues   : cross-chain symmetry groups tying equivalent
                           positions across all three chains
                           e.g. "A151,B151,C151|A152,B152,C152|...|A162,B162,C162"

  --symmetry_weights    : matching 1/N weights per group (N = NUM_CHAINS)
                           e.g. "0.33,0.33,0.33|0.33,0.33,0.33|...|0.33,0.33,0.33"

use:
  python3 update_MPNN.py

Output numbering logic (per chain):
  - Fixed segments keep their original PDB numbers, but are SHIFTED upward
    by the cumulative number of inserted residues seen so far on that chain.
  - Inserted residues are assigned new numbers that immediately follow the
    last output residue of the preceding segment.
  - Only residues from insertions go into --redesigned_residues; the same
    positions are then replicated onto chains B and C for the symmetric
    homotrimer.
"""

import re
import sys
from pathlib import Path


# Homotrimer: A/B/C are C3-symmetric copies produced by symmetrize.py.
# Change if your pipeline ever targets a different oligomeric state.
CHAINS = ('A', 'B', 'C')


def parse_contigs(contig_str):
    """
    Given a contig string like:
      A1-151,12-12,A155-183,1-1,A185-262,B1-262,C1-262
    Return a list of dicts, each describing one segment:
      { 'type': 'fixed'|'insertion',
        'chain': 'A'|'B'|...|None,
        'pdb_start': int|None, 'pdb_end': int|None,
        'length': int }
    """
    segments = []
    for token in contig_str.split(','):
        token = token.strip()
        # Fixed segment: optional chain letter(s) then start-end
        fixed_match = re.match(r'^([A-Za-z]+)(\d+)-(\d+)$', token)
        # Insertion: digits-digits (no leading letter)
        insert_match = re.match(r'^(\d+)-(\d+)$', token)

        if fixed_match:
            chain = fixed_match.group(1)
            start = int(fixed_match.group(2))
            end   = int(fixed_match.group(3))
            segments.append({
                'type': 'fixed',
                'chain': chain,
                'pdb_start': start,
                'pdb_end': end,
                'length': end - start + 1,
            })
        elif insert_match:
            length = int(insert_match.group(1))   # e.g. 12 from "12-12"
            segments.append({
                'type': 'insertion',
                'chain': None,
                'pdb_start': None,
                'pdb_end': None,
                'length': length,
            })
        else:
            print(f"Warning: unrecognised contig token '{token}', skipping.")

    return segments


def compute_redesigned_residues(segments):
    """
    Walk through segments in order, tracking:
      - current output residue number for the active chain
      - cumulative insertion offset

    Returns a list of strings like ['A152', 'A153', ..., 'A196']
    (chain A only — expansion to B and C happens in expand_to_all_chains()).
    """
    redesigned = []

    # We need to track the next output residue number.
    # Output number = pdb_number + cumulative_insertions_so_far (for fixed segs)
    # For insertions we just continue numbering from where we left off.

    current_output_num = 0   # last output residue number assigned
    current_chain = None

    for seg in segments:
        if seg['type'] == 'fixed':
            chain = seg['chain']

            if chain != current_chain:
                # Switching to a new chain — reset tracking
                # Output numbers for the new chain start from pdb_start
                # (no prior insertions on this chain)
                current_chain = chain
                current_output_num = seg['pdb_start'] - 1  # will be incremented below

            # For fixed segments the output numbers are:
            #   current_output_num+1 .. current_output_num + length
            # but they must align with where we left off (insertions shift things)
            # Actually the rule is simpler: output numbers run contiguously.
            # After any insertion the next fixed segment resumes immediately after.
            # So we just advance by the segment length.
            current_output_num += seg['length']

        else:  # insertion
            # Inserted residues: assign consecutive numbers following last output
            for _ in range(seg['length']):
                current_output_num += 1
                redesigned.append(f"{current_chain}{current_output_num}")

    return redesigned


def expand_to_all_chains(chain_a_residues, chains=CHAINS):
    """
    Given chain-A insert positions like ['A151', 'A152', ..., 'A162'],
    return:
      flat_list        — every position on every chain, A-block then
                         B-block then C-block, e.g.
                         ['A151', ..., 'A162', 'B151', ..., 'B162',
                          'C151', ..., 'C162']
      symmetry_groups  — list of tuples linking equivalent positions
                         across chains, e.g.
                         [('A151','B151','C151'), ('A152','B152','C152'), ...]
    """
    # Strip leading chain letter off each entry so we can re-prefix
    numeric_positions = [r[1:] for r in chain_a_residues]

    flat_list = []
    for c in chains:
        flat_list.extend(f"{c}{num}" for num in numeric_positions)

    symmetry_groups = [
        tuple(f"{c}{num}" for c in chains) for num in numeric_positions
    ]

    return flat_list, symmetry_groups


def build_symmetry_strings(symmetry_groups, num_chains=len(CHAINS)):
    """
    Build the --symmetry_residues and --symmetry_weights strings for
    LigandMPNN. Format:
      residues: "A151,B151,C151|A152,B152,C152|..."
      weights : "0.33,0.33,0.33|0.33,0.33,0.33|..."
    Weights within a group are 1/num_chains each (two decimal places,
    matching the 0.33,0.33,0.33 convention used in 2_LigMPNN.sh).
    """
    sym_res_str = '|'.join(','.join(g) for g in symmetry_groups)

    per_group_weight = round(1.0 / num_chains, 2)
    weight_group_str = ','.join(f"{per_group_weight:.2f}" for _ in range(num_chains))
    sym_weights_str = '|'.join(weight_group_str for _ in symmetry_groups)

    return sym_res_str, sym_weights_str


def extract_contig_string(rfdiff_path):
    """Read 1_RfDiff.sh and extract the contig list string."""
    text = Path(rfdiff_path).read_text()
    # Match contigmap.contigs="['...']"  (single or double quotes around the list)
    match = re.search(r"contigmap\.contigs=['\"]?\[['\"](.*?)['\"]\]['\"]?", text)
    if not match:
        sys.exit(f"Error: could not find contigmap.contigs in {rfdiff_path}")
    return match.group(1)


def replace_flag_line(text, flag, new_value, script_path):
    """
    Replace a `--flag "value"` line in a bash script while preserving its
    original leading whitespace (so indentation inside the continued
    python command stays consistent).
    """
    pattern = r'([ \t]*)' + re.escape(flag) + r'\s+"[^"]*"'
    match = re.search(pattern, text)
    if not match:
        sys.exit(f"Error: could not find {flag} in {script_path}")
    indent = match.group(1)
    replacement = f'{indent}{flag} "{new_value}"'
    return re.sub(pattern, replacement, text, count=1)


def update_ligmpnn(ligmpnn_path, redesigned, sym_res_str, sym_weights_str):
    """Overwrite --redesigned_residues, --symmetry_residues, and
    --symmetry_weights lines in 2_LigMPNN.sh with the computed values."""
    text = Path(ligmpnn_path).read_text()

    text = replace_flag_line(text, '--redesigned_residues',
                             ' '.join(redesigned), ligmpnn_path)
    text = replace_flag_line(text, '--symmetry_residues',
                             sym_res_str, ligmpnn_path)
    text = replace_flag_line(text, '--symmetry_weights',
                             sym_weights_str, ligmpnn_path)

    Path(ligmpnn_path).write_text(text)


def main():
    script_dir = Path(__file__).parent

    rfdiff_path  = script_dir / '1_RfDiff.sh'
    ligmpnn_path = script_dir / '2_LigMPNN.sh'

    for p in (rfdiff_path, ligmpnn_path):
        if not p.exists():
            sys.exit(f"Error: {p} not found. Make sure this script is in the same "
                     "directory as 1_RfDiff.sh and 2_LigMPNN.sh.")

    contig_str = extract_contig_string(rfdiff_path)
    print(f"Found contigs: {contig_str}")

    segments = parse_contigs(contig_str)
    print("\nParsed segments:")
    for s in segments:
        if s['type'] == 'fixed':
            print(f"  Fixed   {s['chain']}{s['pdb_start']}-{s['pdb_end']}  (len {s['length']})")
        else:
            print(f"  Insert  {s['length']} residues")

    chain_a_residues = compute_redesigned_residues(segments)
    print(f"\nChain-A insert residues ({len(chain_a_residues)}): {' '.join(chain_a_residues)}")

    flat_list, symmetry_groups = expand_to_all_chains(chain_a_residues)
    sym_res_str, sym_weights_str = build_symmetry_strings(symmetry_groups)

    print(f"\nRedesigned residues across chains {CHAINS} "
          f"({len(flat_list)} total):")
    print(f"  {' '.join(flat_list)}")

    print(f"\nSymmetry groups ({len(symmetry_groups)}):")
    print(f"  {sym_res_str}")

    print(f"\nSymmetry weights:")
    print(f"  {sym_weights_str}")

    update_ligmpnn(ligmpnn_path, flat_list, sym_res_str, sym_weights_str)
    print(f"\nUpdated {ligmpnn_path}:")
    print(f"  --redesigned_residues : {len(flat_list)} positions across chains {CHAINS}")
    print(f"  --symmetry_residues   : {len(symmetry_groups)} cross-chain groups")
    print(f"  --symmetry_weights    : matching weights (1/{len(CHAINS)} per member)")


if __name__ == '__main__':
    main()
