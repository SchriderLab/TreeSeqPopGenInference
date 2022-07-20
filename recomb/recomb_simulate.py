import subprocess
import multiprocessing as mp
import os, sys
import gzip
from itertools import cycle
import argparse

import numpy as np
from tqdm import tqdm


def separate_output(ms_output):
    """Separates tree sequence from ms output for separated dictionary storage."""
    _ms = ms_output.split("\n")
    for idx in range(len(_ms)):
        if _ms[idx] == "//":
            start_index = idx
        elif "segsites:" in _ms[idx]:
            end_index = idx
        else:
            pass

    trees = _ms[start_index + 1 : end_index]
    ms_out = _ms[: start_index + 1] + _ms[end_index:]

    return "\n".join(trees), "\n".join(ms_out)


def create_sim_cmd(morgans_per_bp, Ne, n_chrom=50, mu=1.5e-8, bp=20001):
    """Generates an ms-style command for simulating a replicate."""
    total_gen_dist = morgans_per_bp * (bp - 1)
    rho = 4 * Ne * total_gen_dist
    theta = 4 * Ne * mu * bp

    return f"mspms {n_chrom} 1 -t {theta} -T -r {rho} {bp}"


def worker(args):
    rep, cmd, msdir, treedir = args
    ms_output = subprocess.check_output(cmd.split()).decode("utf-8")
    trees, ms = separate_output(ms_output)

    with gzip.open(os.path.join(msdir, f"{rep}.msOut.gz"), "w") as msfile:
        msfile.write(bytes(ms, "utf-8"))
    with gzip.open(os.path.join(treedir, f"{rep}.nwk.gz"), "w") as treefile:
        treefile.write(bytes(trees, "utf-8"))


def main():
    ap = argparse.ArgumentParser(
        description="Simulates replicates from a randomized set of Ne/rho/theta for training a model to predict rho."
    )
    ap.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        default="./sims",
        help="Directory to create subdirectories in and write simulations to. Defaults to './sims/'.",
    )
    ap.add_argument(
        "-n",
        "--num-reps",
        dest="num_reps",
        default=10000,
        help="Number of simulation replicates. Defaults to 10,000.",
    )
    ap.add_argument(
        "--threads",
        dest="threads",
        default=mp.cpu_count() - 1 or 1,
        help="Number of threads to parallelize across.",
    )
    ua = ap.parse_args()

    msdir = os.path.join(ua.outdir, "ms")
    treedir = os.path.join(ua.outdir, "trees")

    # Make outdir if not present
    if not os.path.exists(msdir):
        os.makedirs(msdir)
        os.makedirs(treedir)

    # Sim params
    n_sims = 100
    Ne_opts = np.array([1000, 2000, 5000, 10000, 15000, 20000, 50000])
    morgans_per_bp = np.power(10, np.random.uniform(-8, -6, n_sims))
    Ne_vals = np.random.choice(Ne_opts, size=n_sims)

    print(f"[Info] Output dir: {ua.outdir}")
    print(f"[Info] Number of sims: {n_sims}")
    print(f"[Info] Ne options: {Ne_opts}")
    print(f"[Info] CPUs used: {ua.threads}")

    # Sim reps
    cmds = [create_sim_cmd(morgans_per_bp[rep], Ne_vals[rep]) for rep in range(n_sims)]

    # Write cmds to file
    with open(f"{ua.outdir}/cmds.txt", "w") as cmdfile:
        for r in range(n_sims):
            cmdfile.write(f"{r} {cmds[r]}\n")

    pool = mp.Pool(ua.threads)
    list(
        tqdm(
            pool.imap(
                worker, zip(range(n_sims), cmds, cycle([msdir]), cycle([treedir]))
            ),
            total=n_sims,
        )
    )


if __name__ == "__main__":
    main()
