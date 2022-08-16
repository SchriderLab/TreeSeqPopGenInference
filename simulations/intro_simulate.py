import gzip
import multiprocessing as mp
import os
import random
from itertools import cycle

import demes
import demesdraw
import msprime
import numpy as np
import tskit
from tqdm import tqdm

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
    Nref,
    L,
    thetaOverRho,
    thetaMean,
    nu1Mean,
    nu2Mean,
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
    rho = theta / thetaOverRho
    mean_morgans_pbp = rho / (4 * Nref * L)
    morgans_pbp = drawUnif(mean_morgans_pbp)
    nu1 = drawUnif(nu1Mean)
    nu2 = drawUnif(nu2Mean)
    T = drawUnif(TMean)

    if mig:
        migTime = random.uniform(0, T / 4)
        migProb = 1 - random.random()
        return theta, morgans_pbp, nu1, nu2, T, migTime, migProb
    else:
        return theta, morgans_pbp, nu1, nu2, T, None, None


def create_param_sets(
    reps, theta, morgans_pbp, nu1, nu2, T_gens, thetaOverRho, L, Nref
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
    for rep in reps:
        theta, morgans_pbp, nu1, nu2, T_gens, migTime, migProb = drawParams(
            Nref, L, thetaOverRho, theta, nu1, nu2, T_gens, False
        )

        noMigParams.append(
            [
                rep,
                theta,
                morgans_pbp,
                nu1,
                nu2,
                T_gens,
                "noMig",
                migTime,
                migProb,
                sim_utils.get_seeds(1)[0],
            ]
        )

        # Mig 1 -> 2
        theta, morgans_pbp, nu1, nu2, T_gens, migTime, migProb = drawParams(
            Nref, L, thetaOverRho, theta, nu1, nu2, T_gens, True
        )

        mig12Params.append(
            [
                rep,
                theta,
                morgans_pbp,
                nu1,
                nu2,
                T_gens,
                "mig12",
                migTime,
                migProb,
                sim_utils.get_seeds(1)[0],
            ]
        )

        # Mig 2 -> 1
        theta, morgans_pbp, nu1, nu2, T_gens, migTime, migProb = drawParams(
            Nref, L, thetaOverRho, theta, nu1, nu2, T_gens, True
        )

        mig21Params.append(
            [
                rep,
                theta,
                morgans_pbp,
                nu1,
                nu2,
                T_gens,
                "mig21",
                migTime,
                migProb,
                sim_utils.get_seeds(1)[0],
            ]
        )

    return {"noMig": noMigParams, "mig12": mig12Params, "mig21": mig21Params}


# Simulation
def worker(args):
    (
        params,
        Nref,
        L,
        mu,
        sampleSize1,
        sampleSize2,
        growth_rate_1,
        growth_rate_2,
        msdir,
        treedir,
        dumpdir,
    ) = args
    (rep, theta, r, nu1, nu2, T_gens, mig, migTime, migProb, seed) = params

    demo = create_demo(
        nu1, nu2, growth_rate_1, growth_rate_2, Nref, T_gens, mig, migTime, migProb
    )

    sim(rep, msdir, treedir, dumpdir, demo, sampleSize1, sampleSize2, r, mu, L, seed)


def create_demo(
    nu1,
    nu2,
    growth_1,
    growth_2,
    N_anc,
    splitTime,
    mig,
    migTime,
    migProb,
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
        growth_rate=growth_1,
    )
    demography.add_population(
        name="sechelia",
        initial_size=nu2,
        growth_rate=growth_2,
    )
    demography.add_population_split(
        splitTime,
        derived=["simulans", "sechelia"],
        ancestral="ancestral",
    )

    if mig == "mig12":
        demography.add_mass_migration(
            migTime,
            source="sechelia",
            dest="simulans",
            proportion=migProb,
        )

    elif mig == "mig21":
        demography.add_mass_migration(
            migTime,
            source="simulans",
            dest="sechelia",
            proportion=migProb,
        )
    else:
        pass

    demography.sort_events()

    return demography


def sim(rep, msdir, treedir, dumpdir, demography, n_samps_1, n_samps_2, r, mu, L, seed):
    ts = msprime.sim_ancestry(
        demography=demography,
        samples={"simulans": n_samps_1, "sechelia": n_samps_2},
        recombination_rate=r,
        sequence_length=L,
        ploidy=1,
        random_seed=seed,
        model="hudson",
    )
    mut_ts = msprime.sim_mutations(ts, rate=mu, discrete_genome=False)

    ts_nwk = []
    for tree in mut_ts.trees():
        newick = tree.newick()
        ts_nwk.append(f"[{int(tree.span)}] {newick}")

    try:
        with open(os.path.join(msdir, f"{rep}.msOut"), "w+") as msfile:
            # Why is write_ms written like this? Just let me compress with a bytestream
            tskit.write_ms(mut_ts, output=msfile)

        #sim_utils.inject_nwk(os.path.join(msdir, f"{rep}.notree.msOut"), ts_nwk)

        utils.compress_file(os.path.join(msdir, f"{rep}.msOut"), overwrite=True)

        with gzip.open(os.path.join(treedir, f"{rep}.nwk.gz"), "w") as treefile:
            treefile.write(bytes("\n".join(ts_nwk), "utf-8"))

        mut_ts.dump(os.path.join(dumpdir, f"{rep}.dump"))

    except ValueError as e:
        print(f"[Error] {e}")


def main():
    ua = utils.get_ua()
    os.makedirs(ua.outdir, exist_ok=True)
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
    # Assumed params
    gen_time = 1 / 15  # 15 gens/year
    L = 10000
    sampleSize1 = 20
    sampleSize2 = 14
    mu = 3.5e-09
    thetaOverRho = 0.2

    # dadi params
    # nu1 and nu2 (before)
    # nu1_0 and nu2_0 (after split)
    # migration rates (Nref_m12, Nref_m21)
    Nref = 487835.088398
    nu1_0 = 117605.344694  # Size before split
    nu2_0 = 4878347.23386  # Size before split
    nu1 = 9279970.1758  # Size after split
    nu2 = 26691.779717  # Size after split
    T = 86404.6219829  # Split time

    # Derived params
    T_gens = T / gen_time
    theta = 4 * Nref * mu * L
    rho = 1 / (theta / thetaOverRho)
    morgans_pbp = rho / (4 * Nref * L)
    growth_rate_1 = np.log(nu1 / nu1_0) / T_gens  # Growth rates constant based on means
    growth_rate_2 = np.log(nu2 / nu2_0) / T_gens  # Growth rates constant based on means

    # OG Script params - scaled by Nref
    # thetaMean = 68.29691232
    # thetaOverRhoMean = 0.2
    # nu1Mean = 19.022761
    # nu2Mean = 0.054715
    # m12Times2Mean = 0.025751
    # m21Times2Mean = 0.172334
    # TMean = 0.664194

    paramsDict = create_param_sets(
        reps, theta, morgans_pbp, nu1, nu2, T_gens, thetaOverRho, L, Nref
    )

    # Log everything
    for lab in paramsDict:
        sim_utils.log_params(
            ua.outdir,
            [
                "rep",
                "theta",
                "r",
                "nu1",
                "nu2",
                "split_T_gens",
                "migTime",
                "migProb",
                "seed",
            ],
            paramsDict[lab],
            f"{lab}_{reps[0]}-{reps[-1]}_params.txt",
        )

    # Simulate
    # TODO wrap in MP
    # TODO fix r and L values
    for scenario in paramsDict.keys():

        msdir = os.path.join(ua.outdir, scenario, "ms")
        treedir = os.path.join(ua.outdir, scenario, "trees")
        dumpdir = os.path.join(ua.outdir, scenario, "tsdump")

        # Make outdir if not present
        utils.make_dirs(msdir, treedir, dumpdir)

        # Simulate
        serial = True
        if serial:
            for p in tqdm(
                paramsDict[scenario], total=len(reps), desc=f"Simulating {scenario}"
            ):
                worker(
                    (
                        p,
                        Nref,
                        L,
                        mu,
                        sampleSize1,
                        sampleSize2,
                        growth_rate_1,
                        growth_rate_2,
                        msdir,
                        treedir,
                        dumpdir,
                    )
                )
        else:
            pool = mp.Pool(ua.threads)
            tqdm(
                pool.map(
                    worker,
                    zip(
                        paramsDict[scenario],
                        cycle([Nref]),
                        cycle([L]),
                        cycle([mu]),
                        cycle([sampleSize1]),
                        cycle([sampleSize2]),
                        cycle([growth_rate_1]),
                        cycle([growth_rate_2]),
                        cycle([msdir]),
                        cycle([treedir]),
                        cycle([dumpdir]),
                    ),
                ),
                total=len(reps),
                desc="Simulating",
                chunksize=5,
            )
            pool.close()


if __name__ == "__main__":
    main()
