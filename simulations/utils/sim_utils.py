"""Simulation utilities for parameter handling, tree processing, etc."""
import os
import numpy as np

#Sim utils
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
