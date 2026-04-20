"""
symmetrize.py

Build a C3-symmetric homotrimer PDB from an RFDiffusion single-chain output,
using the C3 symmetry of the original docked.pdb.

Why:
  1_RfDiff.sh runs RFDiffusion with chain A as the design chain and chains
  B/C as fixed structural context. RFDiffusion All-Atom **flattens the
  entire complex onto a single chain A** in its output PDB, so the file
  doesn't have three separate chains — it has one long chain A whose
  residues break down as:
     residues 1                     .. designed_len            → designed chain A (with insert)
     residues designed_len+1        .. designed_len+ctx_len    → wild-type chain B context
     residues designed_len+ctx_len+1 .. end                    → wild-type chain C context
  where ctx_len = docked.pdb chain-A length (chain B and C context are
  unchanged copies of the wild-type monomer).

  LigandMPNN needs to design as a true homotrimer, so it needs to see all
  three chains with the insert and in C3-symmetric positions. This script
  builds that symmetric PDB by slicing off just the designed portion
  (residues 1..designed_len) of rfdiff chain A, creating two rotated
  copies of it (and of the ligand), and aligning each copy to one of the
  three chain positions from docked.pdb.

Inputs:
  docked.pdb  — original homotrimer + ligand (provides C3 symmetry axes)
  rfdiff.pdb  — RFDiffusion output (chain A has the new insert; chains
                B and C are discarded)

Output:
  output.pdb  — three C3-symmetric copies of chain A on chains A/B/C,
                each paired with its own ligand copy.

Usage (inside a conda env with pymol executable on PATH, e.g. pymol_env):
  pymol -cq symmetrize.py -- <docked_pdb> <rfdiff_pdb> <output_pdb>

NOTE: This script runs at MODULE level (no `if __name__ == "__main__"` guard).
PyMOL's `-cq script.py` execution does not always set __name__ to "__main__",
so guarding the entry point that way would cause the script to do nothing.
"""

import sys
import os
from pymol import cmd


def symmetrize(docked_pdb, rfdiff_pdb, output_pdb):
    cmd.reinitialize()

    # Load inputs
    cmd.load(docked_pdb, "docked")
    cmd.load(rfdiff_pdb, "rfdiff")
    print(f"[symmetrize] loaded docked ({cmd.count_atoms('docked')} atoms, "
          f"chains={cmd.get_chains('docked')}), "
          f"rfdiff ({cmd.count_atoms('rfdiff')} atoms, "
          f"chains={cmd.get_chains('rfdiff')})")

    # RFDiffusion All-Atom collapses all contig chains onto a single
    # chain A in its output PDB. So rfdiff's chain A is really:
    #   [designed chain A (with insert)] + [WT chain B] + [WT chain C]
    # We need to slice off just the designed portion. We derive the
    # designed length from residue counts instead of parsing contigs:
    #     designed_len = (rfdiff chain A polymer count)
    #                  - 2 * (docked chain A polymer count)
    # because the last (2 * ctx_len) residues of rfdiff chain A are
    # verbatim copies of docked chains B and C.
    total_rf_poly = cmd.count_atoms("rfdiff and chain A and polymer and name CA")
    ctx_len       = cmd.count_atoms("docked and chain A and polymer and name CA")
    designed_len  = total_rf_poly - 2 * ctx_len
    print(f"[symmetrize] rfdiff chain A polymer: {total_rf_poly} residues; "
          f"docked chain A: {ctx_len} residues; "
          f"computed designed chain A length: {designed_len}")
    if designed_len <= 0 or designed_len >= total_rf_poly:
        sys.exit(f"[symmetrize] ERROR: designed_len ({designed_len}) is "
                 f"nonsensical. Check that rfdiff chain A really contains "
                 f"designed-A + chain-B-context + chain-C-context "
                 f"(total={total_rf_poly}, ctx_len={ctx_len}).")

    # Strip rfdiff destructively to ONLY the designed portion of chain A
    # (residues 1 .. designed_len) plus any hetatm (ligand). Everything
    # else — WT B/C context carried inside rfdiff's chain A, plus anything
    # from docked — is dropped so unit copies can only inherit the
    # designed monomer + ligand.
    cmd.remove(f"rfdiff and not ((chain A and polymer and resi 1-{designed_len}) "
               f"or hetatm)")
    n_poly = cmd.count_atoms("rfdiff and polymer and name CA")
    n_het  = cmd.count_atoms("rfdiff and hetatm")
    print(f"[symmetrize] rfdiff stripped to designed chain A + ligand: "
          f"{n_poly} residues (CA count), {n_het} hetatm atoms, "
          f"chains={cmd.get_chains('rfdiff')}")
    if n_poly != designed_len:
        sys.exit(f"[symmetrize] ERROR: after strip, got {n_poly} residues "
                 f"but expected {designed_len}")
    if n_poly == 0:
        sys.exit("[symmetrize] ERROR: stripped rfdiff has no chain A polymer")

    # Three independent copies — one per C3 position.
    cmd.copy("unit_A", "rfdiff")
    cmd.copy("unit_B", "rfdiff")
    cmd.copy("unit_C", "rfdiff")
    for u in ("unit_A", "unit_B", "unit_C"):
        r = cmd.count_atoms(f"{u} and polymer and name CA")
        h = cmd.count_atoms(f"{u} and hetatm")
        print(f"[symmetrize] {u}: {r} residues + {h} hetatm atoms")

    # Align each unit's polymer onto the corresponding chain in docked.pdb.
    # cmd.align is sequence-aware, so the insert region (which has no match
    # in docked) is automatically skipped and the shared fixed residues
    # drive the alignment. The ligand atoms in each unit move along with
    # the polymer because cmd.align moves the whole mobile object.
    r_a = cmd.align("unit_A and polymer", "docked and chain A and polymer")
    r_b = cmd.align("unit_B and polymer", "docked and chain B and polymer")
    r_c = cmd.align("unit_C and polymer", "docked and chain C and polymer")
    print(f"[symmetrize] align RMSDs: A={r_a[0]:.3f} B={r_b[0]:.3f} "
          f"C={r_c[0]:.3f} Å (over {r_a[1]}/{r_b[1]}/{r_c[1]} atoms)")

    # cmd.align can leave behind temporary alignment objects. Delete
    # everything that isn't one of our three unit objects so that when
    # we save, only unit_A / unit_B / unit_C are in play.
    for obj in list(cmd.get_object_list()):
        if obj not in ("unit_A", "unit_B", "unit_C"):
            cmd.delete(obj)

    # Relabel chain IDs so the saved PDB reads as A / B / C.
    cmd.alter("unit_A", "chain='A'")
    cmd.alter("unit_B", "chain='B'")
    cmd.alter("unit_C", "chain='C'")
    cmd.sort()

    # Post-relabel sanity check: each unit should have exactly one chain,
    # named A / B / C, with the expected residue count.
    for u, expected in (("unit_A", "A"), ("unit_B", "B"), ("unit_C", "C")):
        chains = cmd.get_chains(u)
        r = cmd.count_atoms(f"{u} and polymer and name CA")
        print(f"[symmetrize] {u} post-alter: chains={chains}, residues={r}")
        if chains != [expected]:
            sys.exit(f"[symmetrize] ERROR: {u} has chains {chains}, "
                     f"expected ['{expected}']")

    # Save straight from the explicit selection — no intermediate
    # cmd.create object — so we know exactly what ends up on disk.
    os.makedirs(os.path.dirname(os.path.abspath(output_pdb)), exist_ok=True)
    cmd.save(output_pdb, "unit_A or unit_B or unit_C")

    if not os.path.exists(output_pdb):
        sys.exit(f"[symmetrize] ERROR: cmd.save reported success but "
                 f"{output_pdb} does not exist")
    size = os.path.getsize(output_pdb)
    if size == 0:
        sys.exit(f"[symmetrize] ERROR: {output_pdb} is 0 bytes")
    print(f"[symmetrize] wrote {output_pdb} ({size} bytes)")


# --- module-level entry point ---
# PyMOL's `pymol -cq script.py -- a b c` passes ['script.py', 'a', 'b', 'c']
# in sys.argv. We run directly at module level so this works regardless of
# what PyMOL sets __name__ to.
if len(sys.argv) < 4:
    sys.exit(
        f"Usage: pymol -cq symmetrize.py -- <docked_pdb> <rfdiff_pdb> "
        f"<output_pdb>\nGot sys.argv = {sys.argv}"
    )

symmetrize(sys.argv[1], sys.argv[2], sys.argv[3])
