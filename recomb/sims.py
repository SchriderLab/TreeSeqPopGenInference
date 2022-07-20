import subprocess
import multiprocessing as mp

import h5py
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
    rep, cmd = args
    ms_output = subprocess.check_output(cmd.split()).decode("utf-8")
    trees, ms = separate_output(ms_output)

    return rep, cmd, trees, ms


def main():
    # Chunk size for faster processing
    chunk_size = 50
    cpus = mp.cpu_count() - 1

    # Init fresh hdf5 file
    filename = "recomb_sims.hdf5"
    hf = h5py.File(filename, "w")
    hf.close()

    # Sim params
    n_sims = 1000  # 0000
    Ne_opts = np.array([1000, 2000, 5000, 10000, 15000, 20000, 50000])
    morgans_per_bp = np.power(10, np.random.uniform(-8, -6, n_sims))
    Ne_vals = np.random.choice(Ne_opts, size=n_sims)

    print(f"[Info] Output file: {filename}")
    print(f"[Info] Number of sims: {n_sims}")
    print(f"[Info] Ne options: {Ne_opts}")
    print(f"[Info] Chunk size: {chunk_size}")
    print(f"[Info] CPUs used: {cpus}")

    # Sim reps
    for chunk in tqdm(
        range(
            0,
            n_sims - chunk_size,
            chunk_size,
        ),
        desc=f"Submitting sims in chunks",
    ):
        chunk_reps = range(chunk, chunk + chunk_size)
        cmds = [create_sim_cmd(morgans_per_bp[rep], Ne_vals[rep]) for rep in chunk_reps]

        pool = mp.Pool(cpus)
        worker_res = pool.map(worker, zip(chunk_reps, cmds))

        hf = h5py.File(filename, "a")
        for res in worker_res:
            rep, cmd, trees, ms = res
            rep_group = hf.create_group(str(rep))
            rep_group.create_dataset("cmd", data=cmd)
            rep_group.create_dataset("trees", data=trees)
            rep_group.create_dataset("ms", data=ms)

    hf.close()


if __name__ == "__main__":
    main()
