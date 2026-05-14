"""Fetch the Tumor Deconvolution DREAM Challenge validation set + ground truth.

Synapse project syn15589870. Auth: requires ~/.synapseConfig with PAT, or
SYNAPSE_AUTH_TOKEN env var.

Usage:
    python scripts/bench_real/fetch_dream.py --dest /vf/users/parks34/projects/0sigdiscov/data/dream
"""

from __future__ import annotations

import argparse
import os
import sys

DREAM_SYN_IDS = {
    # In-vitro validation expression (DS1..DS4, hugo/ensg, tpm/counts) — 16 files
    "in_vitro_expression": [f"syn{n}" for n in range(21821101, 21821117)],
    # In-silico validation, combined matrices
    "in_silico_coarse": "syn21752552",
    "in_silico_fine": "syn21752551",
    # Ground truth
    "gt_coarse": "syn21820375",
    "gt_fine": "syn21820376",
    "gt_coarse_from_fine": "syn22267267",
    # Per-dataset sample map
    "sample_map": "syn21590362",
    # In-silico metadata + spike-in annotations
    "in_silico_metadata": "syn21773017",
    "spike_in_coarse": "syn21763908",
    "spike_in_fine": "syn21763907",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="local destination directory")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of keys from DREAM_SYN_IDS to fetch (default: all)")
    args = ap.parse_args()

    try:
        import synapseclient
    except ImportError:
        sys.exit("synapseclient not installed. Run: pip install synapseclient")

    os.makedirs(args.dest, exist_ok=True)
    syn = synapseclient.Synapse()
    syn.login()  # picks up ~/.synapseConfig or SYNAPSE_AUTH_TOKEN

    keys = args.only or list(DREAM_SYN_IDS)
    for k in keys:
        ids = DREAM_SYN_IDS[k]
        if isinstance(ids, str):
            ids = [ids]
        subdir = os.path.join(args.dest, k)
        os.makedirs(subdir, exist_ok=True)
        for sid in ids:
            print(f"[fetch] {k}: {sid} -> {subdir}")
            syn.get(sid, downloadLocation=subdir, ifcollision="keep.local")

    print("done.")


if __name__ == "__main__":
    main()
