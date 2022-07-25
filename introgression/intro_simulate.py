import multiprocessing as mp
import os
import random
import subprocess

import msprime

from ..utils import utils


def drawUnif(m, fold=0.5):
    x = m * fold
    return random.uniform(m - x, m + x)


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


def writeTbsFile(params, outFileName):
    with open(outFileName, "w") as outFile:
        for paramVec in params:
            outFile.write(" ".join([str(x) for x in paramVec]) + "\n")


def sim_no_mig():
    """https://tskit.dev/tutorials/introgression.html"""
    demography = msprime.Demography()
    demography.add_population(name="Africa", initial_size=Ne)
    demography.add_population(name="Eurasia", initial_size=Ne)
    demography.add_population(name="Neanderthal", initial_size=Ne)


def main():
    ua = utils.get_ua()

    msdir = os.path.join(ua.outdir, "ms")
    treedir = os.path.join(ua.outdir, "trees")
    dumpdir = os.path.join(ua.outdir, "tsdump")

    # Make outdir if not present
    utils.make_dirs(msdir, treedir, dumpdir)

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

    # Example:
    # ./msmove 34 1000 -t 68.29691232 -r 341.4845616 10000 -I 2 20 14 -n 1 19.022761 -n 2 0.054715 -eg 0 1 6.576808 -eg 0 2 -7.841388 -ma x 0.025751 0.172334 x -ej 0.664194 2 1 -en 0.664194 1 1

    noMigParams, mig12Params, mig21Params = [], [], []
    for rep in range(ua.num_reps):
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
        paramVec = [theta, rho, nu1, nu2, 0, 0, splitTime, splitTime]

        noMigTbsFileName = f"{ua.out_dir}/noMig.tbs"
        noMigSimCmd = f"./msmove {sampleSize1} {sampleSize2} -t tbs -r tbs {numSites} -I 2 {sampleSize1} {sampleSize2} -n 1 tbs -n 2 tbs -eg 0 1 6.576808 -eg 0 2 -7.841388 -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1"

        noMigParams.append(paramVec)
        print(noMigSimCmd)

        writeTbsFile(noMigParams, noMigTbsFileName)
        os.system(
            'echo "%s > %s/noMig.msOut" | ./msmove -o /dev/null -e /dev/null'
            % (noMigSimCmd, ua.out_dir)
        )
        ms_output = subprocess.check_output(cmd.split()).decode("utf-8")

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
        paramVec = [
            theta,
            rho,
            nu1,
            nu2,
            0,
            0,
            splitTime,
            splitTime,
            migTime,
            migProb,
        ]
        mig12Params.append(paramVec)
        mig12TbsFileName = "%s/mig12.tbs" % (ua.out_dir)
        mig12SimCmd = (
            "./msmove %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 6.576808 -eg 0 2 -7.841388 -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -ev tbs 1 2 tbs < %s"
            % (
                sampleSize1 + sampleSize2,
                ua.num_reps,
                numSites,
                sampleSize1,
                sampleSize2,
                mig12TbsFileName,
            )
        )

        writeTbsFile(mig12Params, mig12TbsFileName)
        os.system(
            'echo "%s > %s/mig12.msOut" |./msmove -o /dev/null -e /dev/null'
            % (mig12SimCmd, ua.out_dir)
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
        paramVec = [
            theta,
            rho,
            nu1,
            nu2,
            0,
            0,
            splitTime,
            splitTime,
            migTime,
            migProb,
        ]
        mig21Params.append(paramVec)

        mig21TbsFileName = "%s/mig21.tbs" % (ua.out_dir)
        mig21SimCmd = (
            "./msmove %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 6.576808 -eg 0 2 -7.841388 -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -ev tbs 2 1 tbs < %s"
            % (
                sampleSize1 + sampleSize2,
                ua.num_reps,
                numSites,
                sampleSize1,
                sampleSize2,
                mig21TbsFileName,
            )
        )
        writeTbsFile(mig21Params, mig21TbsFileName)
        os.system(
            'echo "%s > %s/mig21.msOut" | ./msmove -o /dev/null -e /dev/null'
            % (mig21SimCmd, ua.out_dir)
        )


if __name__ == "__main__":
    main()
