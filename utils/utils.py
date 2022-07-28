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


def log_params(outdir, param_names, params_list, outfile_name="sim_params.txt"):
    with open(os.path.join(outdir, outfile_name), "w") as ofile:
        ofile.write(
            "\t".join(param_names) + "\n",
        )
        for p in params_list:
            ofile.write("\t".join([str(i) for i in p]))
            ofile.write("\n")

    print(f"[Info] Params logged to: {os.path.join(outdir, outfile_name)}")


def log_cmds(outdir, cmd_list, outfile_name="sim_cmds.txt"):
    with open(os.path.join(outdir, outfile_name), "w") as ofile:
        ofile.write("rep\tcmd\n")
        for c in cmd_list:
            ofile.write("\t".join([str(i) for i in c]))

    print(f"[Info] Cmds logged to: {os.path.join(outdir, outfile_name)}")
