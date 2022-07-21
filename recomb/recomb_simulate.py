import gzip
import multiprocessing as mp
import os
from itertools import cycle

import msprime
import numpy as np
import tskit
from tqdm import tqdm

from ..utils import utils


def log_params(outdir, params_list):
    with open(os.path.join(outdir, "sim_params.txt"), "w") as ofile:
        ofile.write(
            "\t".join(
                [
                    "rep",
                    "Ne",
                    "L",
                    "bp",
                    "mu",
                    "r",
                    "seed",
                    "n_chrom",
                ]
            )
            + "\n",
        )
        for p in params_list:
            ofile.write("\t".join([str(i) for i in p]))
            ofile.write("\n")


def inject_nwk(msfile, nwk_lines, overwrite=True):
    combo_name = msfile.replace(".notree", "")

    with open(msfile, "r+") as ifile:
        mslines = ifile.readlines()

    with open(combo_name, "w") as ofile:
        ofile.writelines(mslines[:4])
        ofile.write("\n".join(nwk_lines) + "\n")
        ofile.writelines(mslines[4:])

    if overwrite:
        os.remove(msfile)


def sim_ts(Ne, mu, r, L, seed, n_chrom):
    """
    Simulate a single replicate given a set of parameters.
    Need to explicitly set/log more params than they'll let us oob.
    """
    ts = msprime.sim_ancestry(
        samples=n_chrom,
        recombination_rate=r,
        sequence_length=L,
        ploidy=1,
        population_size=Ne,
        random_seed=seed,
        model="hudson",
    )
    mut_ts = msprime.sim_mutations(ts, rate=mu)

    return mut_ts


def worker(args):
    params, msdir, treedir, dumpdir = args
    (rep, ne, mu, r, L, seed, n_chrom) = params

    ts = sim_ts(ne, mu, r, L, seed, n_chrom)

    ts_nwk = []
    for tree in ts.trees():
        newick = tree.newick()
        ts_nwk.append(f"[{int(tree.span)}] {newick}")

    try:
        with open(os.path.join(msdir, f"{rep}.notree.msOut"), "w+") as msfile:
            # Why is write_ms written like this? Just let me compress with a bytestream
            tskit.write_ms(ts, output=msfile)

        inject_nwk(os.path.join(msdir, f"{rep}.notree.msOut"), ts_nwk)

        utils.compress_file(os.path.join(msdir, f"{rep}.msOut"), overwrite=True)

        with gzip.open(os.path.join(treedir, f"{rep}.nwk.gz"), "w") as treefile:
            treefile.write(bytes("\n".join(ts_nwk), "utf-8"))

        ts.dump(os.path.join(dumpdir, f"{rep}.dump"))

    except ValueError as e:
        print(e)


def main():
    ua = utils.get_ua()
    msdir = os.path.join(ua.outdir, "ms")
    treedir = os.path.join(ua.outdir, "trees")
    dumpdir = os.path.join(ua.outdir, "tsdump")

    # Make outdir if not present
    utils.make_dirs(msdir, treedir, dumpdir)

    # Randomize
    seeds = utils.get_seeds(ua.num_reps)

    # Pre-specified params
    Ne_opts = np.array([1000, 2000, 5000, 10000, 15000, 20000, 50000])
    L = 1e6
    mu = 1.5e-8
    n_chrom = 50

    # Randomized params
    Ne = np.random.choice(Ne_opts, size=ua.num_reps)
    morgans_per_bp = np.power(10, np.random.uniform(-8, -6, ua.num_reps))

    print("[Info] Calculating parameters")
    params = [
        (rep, ne, mu, r, L, seeds[rep], n_chrom)
        for (rep, r, ne) in zip(range(ua.num_reps), morgans_per_bp, Ne)
    ]

    # Write cmds to file
    print(f"[Info] Logging parameters to {os.path.join(ua.outdir, 'params.txt')}\n")
    log_params(ua.outdir, params)

    # Simulate
    pool = mp.Pool(ua.threads)
    list(
        tqdm(
            pool.imap(
                worker,
                zip(
                    params,
                    cycle([msdir]),
                    cycle([treedir]),
                    cycle([dumpdir]),
                ),
            ),
            total=ua.num_reps,
            desc="Simulating",
        )
    )
    pool.close()


if __name__ == "__main__":
    main()
