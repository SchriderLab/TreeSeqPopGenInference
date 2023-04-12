# -*- coding: utf-8 -*-
import sys, os, random, math

import math, random

def selectPopSize(meanFoldChange, maxFoldChange):
    foldChange = 1+randBoundedExp(meanFoldChange-1, maxFoldChange-1)
    if random.random() <= 0.5:
        return foldChange
    else:
        return 1/foldChange

def selectGridOfTimesLogScale(gridSize, maxTime):
    gridTimes = []
    interval = 1.0/(gridSize-1)
    for i in range(0, gridSize, 1):
        gridTimes.append(math.exp(interval * i * math.log(maxTime+1))-1)
    return gridTimes

def randBoundedExp(mean, upper):
    val = upper+1
    while val > upper:
        val = random.expovariate(1.0/mean)
    return val

def writeParamFile(paramOutLines, paramFilePath):
    with open(paramFilePath, "w") as paramFile:
        for line in paramOutLines:
            paramFile.write(line)


baseOutDir, filePrefix, numReps = sys.argv[1:]
numReps = int(numReps)

def selectVal(minVal, maxVal):
    size = maxVal-minVal
    return (random.random()*size)+minVal

msOutDir = baseOutDir
logDir = "%s/logs" %(baseOutDir)
msOutPath = msOutDir+"/"+filePrefix+".msOut"
logPath = logDir+"/"+filePrefix+".log"
popSizeHistDir = "%s/popSizeHistories/" %(baseOutDir)
popSizeHistPath = popSizeHistDir+"/"+filePrefix+".popSize"
tbsDir = "%s/tbsFiles" %(baseOutDir)
tbsPath = tbsDir+"/"+filePrefix+".tbs"

os.system('mkdir -p {}'.format(msOutDir))
os.system('mkdir -p {}'.format(logDir))
os.system('mkdir -p {}'.format(popSizeHistDir))
os.system('mkdir -p {}'.format(tbsDir))

L = 1500000
msPath = "msdir/ms"
sampleSize = 50
u=1.2e-8/10
r=1e-8
meanPopSize=1e4
theta = 4*meanPopSize*u*L
rho = 4*meanPopSize*r*L
paramOutLines=["N0\tt1\tN1\tt2\tN2\n"]
demogStr = "-en 0 1 tbs -en tbs 1 tbs -en tbs 1 tbs "
tbsOutLines=[]
for repI in range(numReps):
    N0 = selectVal(100, 40000)
    N0Fold = N0/meanPopSize
    N1 = selectVal(100, 5000)
    N1Fold = N1/meanPopSize
    t1 = selectVal(100, 3499.99)/(4*N0)
    N2 = selectVal(100, 20000)
    N2Fold = N2/meanPopSize
    t2 = t1 + selectVal(1, 3500)/(4*N0)
    #t2 = 3500/(4*N0)
    paramOutLines.append("%f\t%f\t%f\t%f\t%f\n" %(N0, t1, N1, t2, N2))
    tbsOutLines.append("%f %f %f %f %f\n" %(N0Fold, t1, N1Fold, t2, N2Fold))
writeParamFile(paramOutLines, popSizeHistPath)
writeParamFile(tbsOutLines, tbsPath)

cmd = "%s %d %d -t %f -r %f %d %s< %s | gzip > %s.gz" %(msPath, sampleSize, numReps, theta, rho, L/100, demogStr, tbsPath, msOutPath)
os.system(cmd)
