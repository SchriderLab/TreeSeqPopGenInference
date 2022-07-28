import multiprocessing as mp
import os
import random

import demes
import demesdraw
import msprime

from ..utils import utils

# Parameter handling and utility
def drawUnif(m, fold=0.5):
    x = m * fold
    return random.uniform(m - x, m + x)


def rescale_Ne(theta, mu, numsites):
    return theta / (4 * mu * numsites)


def drawParams(
    thetaMean,
    thetaOverRhoMean,
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
    thetaOverRho = drawUnif(thetaOverRhoMean)
    rho = theta / thetaOverRho
    nu1 = drawUnif(nu1Mean)
    nu2 = drawUnif(nu2Mean)
    T = drawUnif(TMean)
    m12 = drawUnif(m12Times2Mean)
    m21 = drawUnif(m21Times2Mean)
    if mig:
        migTime = random.uniform(0, T / 4)
        migProb = 1 - random.random()
        return theta, rho, nu1, nu2, T, m12, m21, migTime, migProb
    else:
        return theta, rho, nu1, nu2, T, m12, m21, None, None


def create_param_sets(
    num_reps,
    thetaMean,
    thetaOverRhoMean,
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
            list[dict[list[str]]] : Dictionary of labeled msmove cmd lists.
    """
    # Example:
    # ./msmove 34 1000 -t 68.29691232 -r 341.4845616 10000 -I 2 20 14 -n 1 19.022761 -n 2 0.054715 -eg 0 1 6.576808 -eg 0 2 -7.841388 -ma x 0.025751 0.172334 x -ej 0.664194 2 1 -en 0.664194 1 1

    # Rescale Ne by theta/(numsites*4*mu*)

    # Create parameter sets
    noMigParams, mig12Params, mig21Params = [], [], []
    noMigCmds, mig12Cmds, mig21Cmds = [], [], []
    for rep in range(num_reps):
        theta, rho, nu1, nu2, splitTime, m12, m21, _, _ = drawParams(
            thetaMean,
            thetaOverRhoMean,
            nu1Mean,
            nu2Mean,
            m12Times2Mean,
            m21Times2Mean,
            TMean,
            mig=False,
        )

        # paramVec = [theta, rho, nu1, nu2, m12, m21, splitTime, splitTime]
        noMigCmds.append(
            (
                rep,
                f"""msmove {sampleSize1 + sampleSize2} {num_reps} \
            -t {theta} \
            -r {rho} {numSites} \
            -I 2 {sampleSize1} {sampleSize2} \
            -n 1 {nu1} \
            -n 2 {nu2} \
            -eg 0 1 6.576808 \
            -eg 0 2 -7.841388 \
            -ma x 0 0 x \
            -ej {splitTime} 2 1 \
            -en {splitTime} 1 1
        """,
            )
        )
        noMigParams.append([rep, theta, rho, nu1, nu2, 0, 0, splitTime, splitTime])

        # Mig 1 -> 2
        theta, rho, nu1, nu2, splitTime, m12, m21, migTime, migProb = drawParams(
            thetaMean,
            thetaOverRhoMean,
            nu1Mean,
            nu2Mean,
            m12Times2Mean,
            m21Times2Mean,
            TMean,
            mig=True,
        )

        mig12Params.append(
            [rep, theta, rho, nu1, nu2, 0, 0, splitTime, splitTime, migTime, migProb]
        )

        mig12Cmds.append(
            (
                rep,
                f"""msmove {sampleSize1 + sampleSize2} {num_reps} \
            -t {theta} \
            -r {rho} {numSites} \
            -I 2 {sampleSize1} {sampleSize2} \
            -n 1 {nu1} \
            -n 2 {nu2} \
            -eg 0 1 6.576808 \
            -eg 0 2 -7.841388 \
            -ma x 0 0 x \
            -ej {splitTime} 2 1 \
            -en {splitTime} 1 1 \
            -ev {migTime} 1 2 {migProb}
            """,
            )
        )

        # Mig 2 -> 1
        theta, rho, nu1, nu2, splitTime, m12, m21, migTime, migProb = drawParams(
            thetaMean,
            thetaOverRhoMean,
            nu1Mean,
            nu2Mean,
            m12Times2Mean,
            m21Times2Mean,
            TMean,
            mig=True,
        )

        mig21Params.append(
            [rep, theta, rho, nu1, nu2, 0, 0, splitTime, splitTime, migTime, migProb]
        )

        mig21Cmds.append(
            (
                rep,
                f"""msmove {sampleSize1 + sampleSize2} {num_reps} \
            -t {theta} \
            -r {rho} {numSites} \
            -I 2 {sampleSize1} {sampleSize2} \
            -n 1 {nu1} \
            -n 2 {nu2} \
            -eg 0 1 6.576808 \
            -eg 0 2 -7.841388 \
            -ma x 0 0 x \
            -ej {splitTime} 2 1 \
            -en {splitTime} 1 1 \
            -ev {migTime} 2 1 {migProb}
            """,
            )
        )

    return (
        {"noMig": noMigParams, "mig12": mig12Params, "mig21": mig21Params},
        {"noMig": noMigCmds, "mig12": mig12Cmds, "mig21": mig21Cmds},
    )


# Simulation
def sim_rep(ms_cmd, N0):
    """https://tskit.dev/tutorials/introgression.html"""

    demo_deme = demes.from_ms(ms_cmd, N0=N0, deme_names=["simulans", "sechelia"])
    demo = msprime.Demography(demo_deme)
    ts = msprime.sim_ancestry(demography=demo)


def main():
    ua = utils.get_ua()

    msdir = os.path.join(ua.outdir, "ms")
    treedir = os.path.join(ua.outdir, "trees")
    dumpdir = os.path.join(ua.outdir, "tsdump")

    # Make outdir if not present
    utils.make_dirs(msdir, treedir, dumpdir)

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

    sampleSize1 = 20
    sampleSize2 = 14
    numSites = 10000
    (
        thetaMean,
        thetaOverRhoMean,
        nu1Mean,
        nu2Mean,
        m12Times2Mean,
        m21Times2Mean,
        TMean,
    ) = (
        68.29691232,
        0.2,
        19.022761,
        0.054715,
        0.025751,
        0.172334,
        0.664194,
    )

    paramsDict, cmdsDict = create_param_sets(
        ua.um_reps,
        thetaMean,
        thetaOverRhoMean,
        nu1Mean,
        nu2Mean,
        m12Times2Mean,
        m21Times2Mean,
        TMean,
        sampleSize1,
        sampleSize2,
        numSites,
    )

    # Log everything
    for lab, cmd_list in cmdsDict:
        utils.log_cmds(ua.out_dir, cmd_list, f"{lab}_cmds.txt")

    for lab, params_list in paramsDict:
        utils.log_params(
            ua.out_dir,
            [
                "rep",
                "theta",
                "rho",
                "nu1",
                "nu2",
                "m12",
                "m21",
                "splitTime",
                "splitTime",
                "migTime",
                "migProb",
            ],
            params_list,
            f"{lab}_params.txt",
        )

    # Simulate


if __name__ == "__main__":
    main()
