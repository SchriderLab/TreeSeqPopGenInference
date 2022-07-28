"""Utility functions used across all simulation modules."""
import argparse
import gzip
import multiprocessing as mp
import os

# General utils
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


def get_reps(ua):
    if ua.rep_range:
        reps = range(ua.rep_range[0], ua.rep_range[1] + 1)
    else:
        reps = range(ua.num_reps)

    return reps


def get_ua():
    """Generalized argparser setup."""
    ap = argparse.ArgumentParser(
        description="Simulates replicates from a randomized set of Ne/rho/theta for training a model to predict rho."
    )
    ap.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        help="Directory to create subdirectories in and write simulations to.",
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
        "-r",
        "--rep-range",
        dest="rep_range",
        nargs=2,
        type=int,
        help="Specify a subset of replicates to run, will override --num-reps.",
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
