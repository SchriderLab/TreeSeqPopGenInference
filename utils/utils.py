"""Utility functions used across all simulation modules."""
import argparse
import gzip
import multiprocessing as mp
import os

import numpy as np


def make_dirs(msdir, treedir, dumpdir):
    """Creates output directories for different sim files."""
    if not os.path.exists(msdir):
        os.makedirs(msdir)
        os.makedirs(treedir)
        os.makedirs(dumpdir)


def log_status(ua):
    """Logs some basic runtime variables given user arguments."""
    print(f"[Info] Output dir: {ua.outdir}")
    print(f"[Info] Number of sims: {ua.num_reps}")
    print(f"[Info] CPUs used: {ua.threads}\n")


def get_ua():
    """Generalized argparser setup."""
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
        type=int,
        help="Number of simulation replicates. Defaults to 10,000.",
    )
    ap.add_argument(
        "--threads",
        dest="threads",
        default=mp.cpu_count() - 1 or 1,
        help="Number of threads to parallelize across.",
    )
    return ap.parse_args()


def compress_file(file, overwrite=True):
    """Gzip compresses a file, will delete original if overwrite==True (default)."""
    with open(file, "r") as ifile:
        istring = ifile.read()

    with gzip.open(file + ".gz", "w") as ofile:
        ofile.write(bytes(istring, "utf-8"))

    if overwrite:
        os.remove(file)


def get_seeds(num_reps):
    """Generates an array of random seeds using numpy rng streams."""
    seedseq = np.random.SeedSequence(123456)
    child_seeds = seedseq.spawn(num_reps)
    seeds = [np.random.default_rng(s).integers(0, int(1e8)) for s in child_seeds]

    return seeds
