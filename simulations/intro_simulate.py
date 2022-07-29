import gzip
import multiprocessing as mp
import os
import random
from pprint import pprint

import demes
import demesdraw
import msprime
import numpy as np
import tskit

from utils import sim_utils, utils


# Parameter handling and utility
def drawUnif(m, fold=0.5):
    x = m * fold
    return random.uniform(m - x, m + x)


def rescale_Ne(theta, mu, numsites):
    """
    Converts from basewise coalescent units to population size.
    Formula: theta = 4Ne*mu*numsites
    """
    return theta / (4 * mu * numsites)


def rescale_T(T, N_anc):
    return T * 4 * N_anc


def drawParams(
    thetaMean,
    morgans_pbp,
    nu1Mean,
    nu2Mean,
    m12Times2Mean,
    m21Times2Mean,
    TMean,
    mig=False,
):
    """
    Draws parameter sets from a uniform distribution with range {<val>-(<val>*0.5); <val>+(<val>*0.5)}.

    Returns:
        List(float): Parameters for simulations.
            If mig=True the final two will be migration time and probability, otherwise None.
    """
    theta = drawUnif(thetaMean)
    morgans_pbp = drawUnif(morgans_pbp)
    nu1 = drawUnif(nu1Mean)
    nu2 = drawUnif(nu2Mean)
    T = drawUnif(TMean)
    m12 = drawUnif(m12Times2Mean)
    m21 = drawUnif(m21Times2Mean)
    if mig:
        migTime = random.uniform(0, T / 4)
        migProb = 1 - random.random()
        return theta, morgans_pbp, nu1, nu2, T, m12, m21, migTime, migProb
    else:
        return theta, morgans_pbp, nu1, nu2, T, m12, m21, None, None


def create_param_sets(
    num_reps,
    thetaMean,
    morgans_pbpMean,
    nu1Mean,
    nu2Mean,
    m12Times2Mean,
    m21Times2Mean,
    TMean,
    sampleSize1,
    sampleSize2,
    numSites,
):
    """
    Creates both parameter sets and commands for all replicates for:
        - No migration
        - 1 -> 2 migration
        - 2 -> 1 migration.

        Returns:
            list[dict[list[list[str]]]]: Dictionary of labeled parameter lists,
                each entry in each list is a set of parameters for a single replicate.
    """
    # Rescale Ne by theta/(numsites*4*mu*)

    # Create parameter sets
    noMigParams, mig12Params, mig21Params = [], [], []
    for rep in range(num_reps):
        theta, morgans_pbp, nu1, nu2, splitTime, m12, m21, _, _ = drawParams(
            thetaMean,
            morgans_pbpMean,
            nu1Mean,
            nu2Mean,
            m12Times2Mean,
            m21Times2Mean,
            TMean,
            mig=False,
        )

        # paramVec = [theta, rho, nu1, nu2, m12, m21, splitTime, splitTime]
        noMigParams.append(
            [rep, theta, morgans_pbp, nu1, nu2, 0, 0, splitTime, splitTime]
        )

        # Mig 1 -> 2
        theta, rho, nu1, nu2, splitTime, m12, m21, migTime, migProb = drawParams(
            thetaMean,
            morgans_pbp,
            nu1Mean,
            nu2Mean,
            m12Times2Mean,
            m21Times2Mean,
            TMean,
            mig=True,
        )

        mig12Params.append(
            [
                rep,
                theta,
                morgans_pbp,
                nu1,
                nu2,
                0,
                0,
                splitTime,
                splitTime,
                migTime,
                migProb,
            ]
        )

        # Mig 2 -> 1
        theta, rho, nu1, nu2, splitTime, m12, m21, migTime, migProb = drawParams(
            thetaMean,
            morgans_pbp,
            nu1Mean,
            nu2Mean,
            m12Times2Mean,
            m21Times2Mean,
            TMean,
            mig=True,
        )

        mig21Params.append(
            [
                rep,
                theta,
                morgans_pbp,
                nu1,
                nu2,
                0,
                0,
                splitTime,
                splitTime,
                migTime,
                migProb,
            ]
        )

    return {"noMig": noMigParams, "mig12": mig12Params, "mig21": mig21Params}


# Simulation
def worker(args):
    params, msdir, treedir, dumpdir = args
    (
        rep,
        N_anc,
        nu1,
        nu2,
        splitTime,
        migProb,
        migTime,
        n_samples,
        morgans_pbp,
        L,
        seed,
        mig,
    ) = params

    """
    msmove {sampleSize1 + sampleSize2} {num_reps} 
        -t {theta} 
        -r {morgans_pbp} {numSites} 
        -I 2 {sampleSize1} {sampleSize2} 
        -n 1 {nu1} 
        -n 2 {nu2} 
        -eg 0 1 6.576808 
        -eg 0 2 -7.841388 
        -ma x 0 0 x 
        -ej {splitTime} 2 1 
        -en {splitTime} 1 1 
    """


def create_demo(
    nu1=9279970.1758,
    nu2=26691.779717,
    nu1_0=117605.344694,
    nu2_0=4878347.23386,
    N_anc=487835.088398,
    splitTime=86404.6219829,
    mig=None,
    migTime=None,
    migProb=None,
    n_samples=32,
    L=1e6,
    r=3e-9,
    seed=42,
):
    """https://tskit.dev/tutorials/introgression.html"""
    demography = msprime.Demography()
    demography.add_population(
        name="ancestral",
        initial_size=N_anc,
    )
    demography.add_population(
        name="simulans",
        initial_size=nu1,
        growth_rate=6.576808,
    )
    demography.add_population(
        name="sechelia",
        initial_size=nu2,
        growth_rate=-7.841388,
    )
    demography.add_population_split(
        splitTime,
        derived=["simulans", "sechelia"],
        ancestral="ancestral",
    )
    return demography

    if mig == "mig12":
        demography.add_mass_migration(
            migTime,
            source="simulans",
            dest="sechelia",
            proportion=migProb,
        )

    elif mig == "mig21":
        demography.add_mass_migration(
            migTime,
            source="sechelia",
            dest="simulans",
            proportion=migProb,
        )
    else:
        pass

    ts = msprime.sim_ancestry(
        demography=demography,
        samples=n_samples,
        recombination_rate=r,
        sequence_length=L,
        ploidy=1,
        population_size=N_anc,
        random_seed=seed,
        model="hudson",
    )

    ts_nwk = []
    for tree in ts.trees():
        newick = tree.newick()
        ts_nwk.append(f"[{int(tree.span)}] {newick}")

    try:
        with open(os.path.join(msdir, f"{rep}.notree.msOut"), "w+") as msfile:
            # Why is write_ms written like this? Just let me compress with a bytestream
            tskit.write_ms(ts, output=msfile)

        sim_utils.inject_nwk(os.path.join(msdir, f"{rep}.notree.msOut"), ts_nwk)

        utils.compress_file(os.path.join(msdir, f"{rep}.msOut"), overwrite=True)

        with gzip.open(os.path.join(treedir, f"{rep}.nwk.gz"), "w") as treefile:
            treefile.write(bytes("\n".join(ts_nwk), "utf-8"))

        ts.dump(os.path.join(dumpdir, f"{rep}.dump"))

    except ValueError as e:
        print(f"[Error] {e}")


def main():
    ua = utils.get_ua()

    msdir = os.path.join(ua.outdir, "ms")
    treedir = os.path.join(ua.outdir, "trees")
    dumpdir = os.path.join(ua.outdir, "tsdump")

    # Make outdir if not present
    utils.make_dirs(msdir, treedir, dumpdir)

    reps = utils.get_reps(ua)

    # Generate parameter and command sets
    """
    dadi parameter estimates
    AIC: 10303.1237186
    with u = 3.500000e-09
    Nref: 487835.088398
    nu1_0 : 117605.344694
    nu2_0 : 4878347.23386
    nu1 : 9279970.1758
    nu2 : 26691.779717
    T : 86404.6219829
    2Nref_m12 : 0.0128753943002
    2Nref_m21 : 0.0861669095413
    """
    N_anc = 487835.088398
    sampleSize1 = 20
    sampleSize2 = 14
    numSites = 10000
    (thetaMean, morgans_pbp, nu1Mean, nu2Mean, m12Times2Mean, m21Times2Mean, TMean) = (
        68.29691232,
        0.2,
        19.022761,
        0.054715,
        0.025751,
        0.172334,
        0.664194,
    )
    rho = thetaMean / 0.2
    morgans_pbp = rho / (4 * N_anc * numSites)
    TMean_rescaled = rescale_T(TMean, 487835.088398)
    Ne_rescaled = rescale_Ne(thetaMean, 3.500000e-09, numSites)

    paramsDict = create_param_sets(
        len(reps),
        thetaMean,
        morgans_pbp,
        nu1Mean,
        nu2Mean,
        m12Times2Mean,
        m21Times2Mean,
        TMean_rescaled,
        sampleSize1,
        sampleSize2,
        numSites,
    )

    # Log everything
    for lab in paramsDict:
        sim_utils.log_params(
            ua.outdir,
            [
                "rep",
                "theta",
                "morgans_pbp",
                "nu1",
                "nu2",
                "m12",
                "m21",
                "splitTime",
                "splitTime",
                "migTime",
                "migProb",
            ],
            paramsDict[lab],
            f"{lab}_params.txt",
        )

    # Simulate
    # TODO wrap in MP
    # TODO fix r and L values
    for scenario in paramsDict.keys():
        print("[Debug]", scenario)
        print("[Debug]", pprint(paramsDict[scenario]))
        for rep, cmd in zip(reps, paramsDict[scenario]):
            worker(
                (
                    (
                        rep,
                        cmd,
                        Ne_rescaled,
                        sampleSize1 + sampleSize2,
                        np.power(10, np.random.uniform(-8, -6, 1)),
                        1e6,
                        sim_utils.get_seeds(1)[0],
                    ),
                    msdir,
                    treedir,
                    dumpdir,
                )
            )


if __name__ == "__main__":
    main()
