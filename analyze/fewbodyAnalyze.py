#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:42:46 2019

@author: lucas
"""

import pandas as pd
from collections import Counter
import numpy as np
from numpy import format_float_scientific
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from scipy import interpolate
from scipy.signal import find_peaks
import seaborn as sns





def massRatCalc(allSysRat, changedSysRat, changedSysKRat, changedSysKLRat):
    allSys = np.zeros(len(allSysRat))
    changedSys = np.zeros(len(changedSysRat))
    changedSysK = np.zeros(len(changedSysKRat))
    changedSysKL = np.zeros(len(changedSysKLRat))

    singleHigherAll = np.where((allSysRat['m0'] < allSysRat['m10'] + allSysRat['m11']))
    binHigherAll = np.where((allSysRat['m0'] > allSysRat['m10'] + allSysRat['m11']))

    singleHigherks = np.where((changedSysRat['m0'] < changedSysRat['m10'] + changedSysRat['m11']))
    binHigherks = np.where((changedSysRat['m0'] > changedSysRat['m10'] + changedSysRat['m11']))

    singleHigherK = np.where((changedSysKRat['m0'] < changedSysKRat['m10'] + changedSysKRat['m11']))
    binHigherK = np.where((changedSysKRat['m0'] > changedSysKRat['m10'] + changedSysKRat['m11']))

    singleHigherKL = np.where((changedSysKLRat['m0'] < changedSysKLRat['m10'] + changedSysKLRat['m11']))
    binHigherKL = np.where((changedSysKLRat['m0'] > changedSysKLRat['m10'] + changedSysKLRat['m11']))

    allSys[singleHigherAll] = allSysRat['m0'].iloc[singleHigherAll] / (allSysRat['m10'].iloc[singleHigherAll] + allSysRat['m11'].iloc[singleHigherAll])
    allSys[binHigherAll] =  (allSysRat['m10'].iloc[binHigherAll] + allSysRat['m11'].iloc[binHigherAll]) / allSysRat['m0'].iloc[binHigherAll]

    changedSys[singleHigherks] = changedSysRat['m0'].iloc[singleHigherks] / (changedSysRat['m10'].iloc[singleHigherks] + changedSysRat['m11'].iloc[singleHigherks])
    changedSys[binHigherks] = (changedSysRat['m10'].iloc[binHigherks] + changedSysRat['m11'].iloc[binHigherks]) / changedSysRat['m0'].iloc[binHigherks]

    changedSysK[singleHigherK] = changedSysKRat['m0'].iloc[singleHigherK] / (changedSysKRat['m10'].iloc[singleHigherK] + changedSysKRat['m11'].iloc[singleHigherK])
    changedSysK[binHigherK] = (changedSysKRat['m10'].iloc[binHigherK] + changedSysKRat['m11'].iloc[binHigherK]) / changedSysKRat['m0'].iloc[binHigherK]

    changedSysKL[singleHigherKL] = changedSysKLRat['m0'].iloc[singleHigherKL] / (changedSysKLRat['m10'].iloc[singleHigherKL] + changedSysKLRat['m11'].iloc[singleHigherKL])
    changedSysKL[binHigherKL] = (changedSysKLRat['m10'].iloc[binHigherKL] + changedSysKLRat['m11'].iloc[binHigherKL]) / changedSysKLRat['m0'].iloc[binHigherKL]

    return allSys, changedSys, changedSysK


def compRuns(param, pltTitle, xMin, xMax, xStep, xLabel, dfs, dfCols):
    df = pd.DataFrame()
    count = 0
    for i in dfs:
        if includeIMBH == 0:
            allSys = i.iloc[0:,21:24].astype('float')
            massMask1 = (allSys['m0'] < IMBHMassLimit) &  (allSys['m10'] < IMBHMassLimit) & (allSys['m11'] < IMBHMassLimit)
            if param == 'rMin':
                data = i[param][massMask1].astype('float') * i['l Unit'][massMask1].astype('float')

            else:
                data = i[param][massMask1]

        elif onlyLowMass == 1:
            allSysRat = i.iloc[0:,21:24].astype('float')
            allSys = np.zeros(len(allSysRat))

            singleHigherAll = np.where((allSysRat['m0'] < allSysRat['m10'] + allSysRat['m11']))
            binHigherAll = np.where((allSysRat['m0'] > allSysRat['m10'] + allSysRat['m11']))

            allSys[singleHigherAll] = allSysRat['m0'].iloc[singleHigherAll] / (allSysRat['m10'].iloc[singleHigherAll] + allSysRat['m11'].iloc[singleHigherAll])
            allSys[binHigherAll] =  (allSysRat['m10'].iloc[binHigherAll] + allSysRat['m11'].iloc[binHigherAll]) / allSysRat['m0'].iloc[binHigherAll]

            allLow = np.where(allSys < lowMass)

            if param == 'rMin':
                data = i[param].iloc[allLow].astype('float') * i['l Unit'].iloc[allLow].astype('float')

            else:
                data = i[param].iloc[allLow]
        else:
            if param == 'rMin':
                rSunAU = 0.00465047
                data = i[param].astype('float') * i['l Unit'].astype('float') / rSunAU
                confs = i['conf'].to_numpy(dtype='str')
                data[np.core.defchararray.find(confs, ':') != -1] = 'nan'

            elif param == 'e_f1' or param == 'a_f1':
                if param == 'e_f1':
                    data = i[param].astype('float')[i[param].astype('float') != 0]**2
                else:
                    data = i[param].astype('float')[i[param].astype('float') != 0]
            else:
                data = i[param].astype('float')


        if onlyHighE == 1:
            data = data[data['e1'] == '0.99'].astype('float')


        data.rename(dfCols[count], inplace=True)
        data = data.astype(float)
        df = pd.concat([df, data], axis=1)
        count += 1

    binEdges = np.arange(xMin, xMax, xStep)
    if param == 'dEE0' or param == 'dLL0':
        df = df.abs()



        # bins1 = np.arange(0, 1e-5, 1e-7)
        bins1 = np.logspace(np.log10(xMin), np.log10(xMax), 1000)

        df.plot.hist(bins=bins1, alpha=1, rwidth=2, histtype='step', density=False)
        plt.xscale('log')
        plt.legend(loc='upper left')
        plt.title(pltTitle)
        plt.xlabel(xLabel)
        plt.ylabel('Count')



    else:
        df[df.columns[0]].plot.hist(bins=binEdges, alpha=1, rwidth=1, title=pltTitle, histtype='step', density=False)
        plt.legend()
        plt.xlabel('Rmin [R$_{\odot}$]')
        plt.ylabel('Count')

        plt.figure()
        df[df.columns[1]].plot.hist(bins=binEdges, alpha=5, rwidth=1, title=pltTitle, histtype='step', density=False)
        plt.legend()
        plt.xlabel('Rmin [R$_{\odot}$]')
        plt.ylabel('Count')

        # plt.figure()
        # df[df.columns[2]].plot.hist(bins=binEdges, alpha=5, rwidth=1, title=pltTitle, histtype='step', density=False)
        # plt.legend()
        # plt.xlabel('Rmin [R$_{\odot}$]')
        # plt.ylabel('Count')

        # plt.legend(loc='upper right')
        # plt.xlabel(xLabel)
        # plt.ylabel('Fraction')

        plt.tight_layout()


def compRunsOutcome(param, pltTitle, xMin, xMax, xStep, xLabel, outcome, dfs, dfNames):
    dfArray = pd.DataFrame()
    for i in range(len(dfs)):
        outk, outcomes = outcomeInteger(dfs[i], dfs[i]['conf'])
        df = outk[outcomes.index(outcome)]

        if param == 'rMin':
            rSunAU = 0.00465047
            dfSys = df[param].astype('float') * df['l Unit'].astype('float') / rSunAU
        else:
            dfSys = df[param]

        if onlyHighE == 1:
            dfSys = dfSys[df['e1'] == '0.99'].astype('float')

        else:
            dfSys = dfSys.astype('float')

        dfArray = pd.concat([dfArray, dfSys], axis=1)

    dfArray.columns = dfNames
    binEdges = np.arange(xMin, xMax, xStep)
#    dfArray = dfArray.fillnan(0)

    dfArray.plot.hist(bins=binEdges, alpha=0.8, rwidth=1, title=pltTitle, histtype='step')
    plt.legend(loc='upper left')
    plt.xlabel(xLabel)
    plt.ylabel('Count')
    plt.tight_layout()


def semiAxisdE(params, pltTitle):
    if includeIMBH == 0:
        kSys = params.iloc[0:,21:24].astype('float')
        massMask = (kSys['m0'] < IMBHMassLimit) &  (kSys['m10'] < IMBHMassLimit) & (kSys['m11'] < IMBHMassLimit)

        semiK = abs(params['a1'][massMask].astype('float'))
        dEK = abs(params['dE'][massMask].astype('float'))
    else:
        semiK = abs(params['a1'].astype('float'))
        dEK = abs(params['dE'].astype('float'))

#    fig = plt.figure()
#    ax = plt.gca()

#    ax.scatter(dEK, semiK, c=z, s=4)
#    ax.set_xlim([10e-20, 10e5])
#    ax.set_yscale('log')
#    ax.set_xscale('log')
#    ax.set_title(pltTitle)


    xbins = 10**np.linspace(-19, 5, 200)
    ybins = 10**np.linspace(-2, 5, 200)

    counts, _, _ = np.histogram2d(dEK, semiK, bins=(xbins, ybins))

    countsMa = np.ma.masked_where(counts==0,counts)

    fig, ax = plt.subplots()
    H = ax.pcolormesh(xbins, ybins, countsMa.T)
    cbar = plt.colorbar(H)
    cbar.set_label('Count')

    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.axvline(x=energyLimit)
    ax.tick_params(axis='x', which='minor', bottom=False)

#    formatter = ScalarFormatter()
#    formatter.set_scientific(False)
#    ax.set_major_formatter(formatter)

    ax.get_xaxis().get_major_formatter().labelOnlyBase = False

    ax.set_xlabel('dE')
    ax.set_ylabel('a [AU]')
    plt.tight_layout()


def fracCol(param):
    # dE = param['dE'].astype('float')
#    sysdE = K['nStar'][dE > energyLimit].astype('float')
    sysdE = param['nStar'].astype('float')

    numMerge = len(sysdE[sysdE < 3])
    print(str(numMerge) + ' collisions out of ' + str(len(sysdE)) + ' interactions (' + "{0:.4f}".format(numMerge/len(sysdE) * 100) + '%)')

def crossSec(df, run):
    vInf = df['vInf'].to_numpy(dtype='float')
    vCrit = np.unique(vInf)

    confVec, outcomes = outcomeInteger(df, df['conf'])
    sigma = np.array([])
    sigmaErr= np.array([])
    vInfPlot = np.array([])
    confs = np.array([])


    for j in vCrit:
        count = 0
        totV = np.sum(vInf == j)

        for i in confVec:
            vCritTemp = i['vInf'].to_numpy(dtype='float')
            iTemp = i[vCritTemp == j]

            confCount = len(iTemp['conf'])
            if len(iTemp) > 0:
                nRat = confCount/totV
                nRatErr = np.sqrt(confCount)/totV
            else:
                count += 1
                continue


            bTemp = iTemp['bImp'].to_numpy(dtype='float')
            bMax = np.max(bTemp)
            vInfTemp = iTemp['vInf'].iloc[np.argmax(bTemp)]

            sigmaTemp = bMax**2 * nRat
            sigmaErrT = bMax**2 * nRatErr
            sigma = np.append(sigma, sigmaTemp)
            sigmaErr = np.append(sigmaErr, sigmaErrT)
            vInfPlot = np.append(vInfPlot, vInfTemp)
            confs = np.append(confs, outcomes[count])
            count += 1

    confUn = np.unique(confs)


    vInfPlot = vInfPlot.astype('float')

    confsMan = ['Flyby', 'Exchange', 'Ionization', 'Bound merger', 'Unbound merger', 'Triple merger', 'Bound triple']
    plt.figure()
    ax = plt.gca()
    for i in range(len(confUn)):
        if confsMan[i] in confs:
            if confsMan[i] == 'Bound triple' or confsMan[i] == 'Triple merger':
                continue
            mask = (confs == confsMan[i])
            ax.errorbar(vInfPlot[mask], sigma[mask], yerr=sigmaErr[mask], fmt='-o', label=confsMan[i], ms=5)

    ax.loglog()
    ax.set_title(run)
    ax.set_ylim(1, 1.2e2)
    ax.set_xlabel('$v_{\infty}/v_c$', fontsize=40)
    ax.set_ylabel('$\sigma / \pi a^2$', fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(bbox_to_anchor=(0.31, 0.85))


    return sigma

def crossSecMult(dfs, run, acc):
    fig, ax = plt.subplots()
    dfC = 0
    for df in dfs:
        vInf = df['vInf'].to_numpy(dtype='float')

        vCrit = np.unique(vInf)

        confVec, outcomes = outcomeInteger(df, df['conf'])
        sigma = np.array([])
        sigmaErr= np.array([])
        vInfPlot = np.array([])
        confs = np.array([])
        aPlot = np.array([])


        for j in vCrit:
            count = 0
            totV = np.sum(vInf == j)

            for i in confVec:
                vCritTemp = i['vInf'].to_numpy(dtype='float')
                iTemp = i[vCritTemp == j]

                confCount = len(iTemp['conf'])
                if len(iTemp) > 0:
                    nRat = confCount/totV
                    nRatErr = np.sqrt(confCount)/totV
                else:
                    count += 1
                    continue


                bTemp = iTemp['bImp'].to_numpy(dtype='float')
                bMax = np.max(bTemp)
                vInfTemp = iTemp['vInf'].iloc[np.argmax(bTemp)]
                aTemp = iTemp['a1'].iloc[np.argmax(bTemp)]

                sigmaTemp = bMax**2 * nRat
                sigmaErrT = bMax**2 * nRatErr
                sigma = np.append(sigma, sigmaTemp)
                sigmaErr = np.append(sigmaErr, sigmaErrT)
                aPlot = np.append(aPlot, aTemp)
                vInfPlot = np.append(vInfPlot, vInfTemp)
                confs = np.append(confs, outcomes[count])
                count += 1

        confUn = np.unique(confs)

        vInfPlot = vInfPlot.astype('float')
        for i in range(len(confUn)):
            mask = (confs == confUn[i])
            if acc[dfC] == 1:
                ax.errorbar(aPlot[mask], sigma[mask], fmt='o-', yerr = sigmaErr[mask], markeredgecolor='none',
                            label=confUn[i] + ' (PN)', alpha=0.7)     # vInfPlot[mask]
            else:
                ax.errorbar(aPlot[mask], sigma[mask], fmt='o--', yerr = sigmaErr[mask], markeredgecolor='none',
                            label=confUn[i], alpha=0.7)                     # vInfPlot[mask]
        dfC += 1
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax.loglog()
    ax.set_xlabel('$v_{\infty}/v_c$', fontsize=20)
    ax.set_ylabel('$\sigma / \pi a^2$', fontsize=20)
    plt.title(run)
    plt.legend(fontsize=20, bbox_to_anchor=(1, 1))

def crossSecConf(dfs, dfNames, confsV):
    nam = 0
    for df in dfs:
        vInf = df['vInf'].to_numpy(dtype='float')
        vCrit = np.unique(vInf)

        confVec, outcomes = outcomeInteger(df, df['conf'])
        sigma = np.array([])
        sigmaErr= np.array([])
        vInfPlot = np.array([])
        confs = np.array([])


        for j in vCrit:
            count = 0
            totV = np.sum(vInf == j)

            for i in confVec:
                vCritTemp = i['vInf'].to_numpy(dtype='float')
                iTemp = i[vCritTemp == j]

                confCount = len(iTemp['conf'])
                if len(iTemp) > 0 and outcomes[count] in confsV:
                    nRat = confCount/totV
                    nRatErr = np.sqrt(confCount)/totV
                else:
                    count += 1
                    continue


                bTemp = iTemp['bImp'].to_numpy(dtype='float')
                bMax = np.max(bTemp)
                vInfTemp = iTemp['vInf'].iloc[np.argmax(bTemp)]

                sigmaTemp = bMax**2 * nRat
                sigmaErrT = bMax**2 * nRatErr
                sigma = np.append(sigma, sigmaTemp)
                sigmaErr = np.append(sigmaErr, sigmaErrT)
                vInfPlot = np.append(vInfPlot, vInfTemp)
                confs = np.append(confs, outcomes[count])
                count += 1


        confUn = np.unique(confs)
        cRange = np.arange(0, len(confUn), 1)
        cRange = ['g', 'r', 'b', 'purple', 'orange', 'black', 'c', 'm', 'y', 'grey']


        plt.figure(figsize=(15,10))
        ax = plt.gca()
        vInfPlot = vInfPlot.astype('float')
        vInfUn = np.unique(vInfPlot)

        theoSpace = np.logspace(-1, 1, num=100)
        sigIon = 40/9 / vInfUn**2
        sigEx = 640/81 / theoSpace**6

        for i in range(len(confUn)):
            mask = (confs == confUn[i])
            ax.errorbar(vInfPlot[mask], sigma[mask], fmt='o', yerr = sigmaErr[mask], markeredgecolor='none', c=cRange[i], label=confUn[i], alpha=0.7)
#            ax.plot(vInfPlot[mask], sigma[mask], 'o', markeredgecolor='none', c=cRange[i], label=confUn[i], alpha=0.7)

        ax.plot(vInfUn, sigIon, '--')
        ax.plot(theoSpace, sigEx, '.')
        ax.set_xlabel('$v_{\infty}/v_c$', fontsize=50)
        ax.set_ylabel('$\sigma / \pi a^2$', fontsize=50)
        plt.legend(fontsize=55)
        plt.title(dfNames[nam], fontsize=55)
        plt.tight_layout()
        ax.set_yscale('log')
        ax.set_xscale('log')
        nam += 1
        ax.set_ylim(0.01, 12)
        ax.set_xlim(np.amin(vInfUn), np.amax(vInfUn))
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)

def compOutcome(param, pltTitle, xMin, xMax, xStep, xLabel, dfs, dfNames):
    sysVec = []
    confVec = []

    for i in range(len(dfs)):
        sysVec.append(dfs[i][param].astype(float))
        confVec.append(dfs[i]['conf'])

    if param == 'bImp':
        binData = sysVec[0]
        diffH = np.diff(np.unique(binData)).min()
        lfb = binData.min() - float(diffH)/2
        rlb = binData.max() + float(diffH)/2
        binEdges = np.arange(lfb, rlb + diffH, diffH)
    else:
        binEdges = np.arange(xMin, xMax, xStep)

    for i in range(len(sysVec)):
        out, outcomes = outcomeIntegerOutcomeOrder2(sysVec[i], confVec[i])

        df = pd.concat(out, axis=1)
        df = df[(df != 0).all(1)]
        df.columns = outcomes
        df = df.dropna(axis=1, how='all')

        hiFly, temp = np.histogram(df['Flyby'], bins=binEdges)
        hiExc, temp = np.histogram(df['Exchange'], bins=binEdges)
        hiBtr, temp = np.histogram(df['Bound triple'], bins=binEdges)
        hi, temp = np.histogram(df, bins=binEdges)


        # weig = np.ones(np.shape(df))/hi[0]
        plt.figure()
        for col in df.columns:
            hiTemp = np.histogram(df[col], bins=binEdges)[0]/hi
            plt.step(binEdges[:-1], hiTemp, label=col)

            if col == 'Flyby' or col == 'Exchange':
                print(col)

        # plt.axvline(x=4.35, c='black', linestyle='--')
        plt.ylim(0, 1)
        plt.xlim(xMin, xMax)
        plt.legend(loc='center right', fontsize=19)
        plt.xlabel(xLabel, fontsize=35)
        plt.ylabel('Fraction', fontsize=35)
        plt.title(dfNames[i], fontsize=40)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        # plt.tight_layout()

        # df.plot.hist(bins=binEdges, alpha=1, rwidth=1, title=dfNames[i],
        #               figsize = [13,15], histtype='step') #title=pltTitle + ' (' + nameVec[i] + ')'
        # plt.legend(loc='center right', fontsize=19)
        # plt.xlabel(xLabel, fontsize=35)
        # plt.ylabel('Count', fontsize=35)
        # plt.title(dfNames[i], fontsize=40)
        # plt.xticks(fontsize=28)
        # plt.yticks(fontsize=28)
        # plt.yscale('log')
    #    plt.ticklabel_format(style=tickStyle, axis='x', scilimits=(0,0))
        # plt.tight_layout()


        if param == 'bImp':
            fb = np.histogram(df['Flyby'].dropna(), bins=binEdges)
            ex = np.histogram(df['Exchange'].dropna(), bins=binEdges)
            idx = np.argwhere(np.diff(np.sign(fb[0] - ex[0]))).flatten()
            print(dfNames[i])
            print(fb[1][idx][0])

def  compOutcomeSamePlot(param, pltTitle, xMin, xMax, xStep, xLabel, dfs, dfNames):
    sysVec = []
    confVec = []
    plt.figure()
    colours = ['blue', 'orange', 'green', 'red']
    styles = ['-', 'dashed', 'dotted']
    accuracies = ['High', 'Default', 'Low']

    sCount = 0
    aCount= 0

    for i in range(len(dfs)):
        sysVec.append(dfs[i][param].astype(float))
        confVec.append(dfs[i]['conf'])

    if param == 'bImp':
        binData = sysVec[0]
        diffH = np.diff(np.unique(binData)).min()
        lfb = binData.min() - float(diffH)/2
        rlb = binData.max() + float(diffH)/2
        binEdges = np.arange(lfb, rlb + diffH, diffH)
    else:
        binEdges = np.arange(xMin, xMax, xStep)

    for i in range(len(sysVec)):
        out, outcomes = outcomeIntegerOutcomeOrder2(sysVec[i], confVec[i])

        df = pd.concat(out, axis=1)
        df = df[(df != 0).all(1)]
        df.columns = outcomes
        df = df.dropna(axis=1, how='all')

        hiFly, temp = np.histogram(df['Flyby'], bins=binEdges)
        hiExc, temp = np.histogram(df['Exchange'], bins=binEdges)
        hiBtr, temp = np.histogram(df['Bound triple'], bins=binEdges)
        hi, temp = np.histogram(df, bins=binEdges)


        # weig = np.ones(np.shape(df))/hi[0]
        cCount = 0
        for col in df.columns:
            hiTemp = np.histogram(df[col], bins=binEdges)[0]/hi
            plt.step(binEdges[:-1], hiTemp, label=col + ' (' + accuracies[aCount] + ')', c=colours[cCount], linestyle=styles[sCount])
            cCount += 1

            if col == 'Flyby' or col == 'Exchange':
                print(col)
        sCount += 1
        aCount += 1

    plt.ylim(0, 1.05)
    plt.xlim(xMin, xMax)
    plt.legend(loc='center right', fontsize=19)
    plt.xlabel(xLabel, fontsize=35)
    plt.ylabel('Fraction', fontsize=35)
    plt.title(dfNames[0], fontsize=40)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)



def lineIntersect(line1, line2, run):
    """ Find intersection between flyby and Exchange (https://stackoverflow.com/a/51145981) """
    def upsample_coords(coord_list):
        # s is smoothness, set to zero
        # k is degree of the spline. setting to 1 for linear spline
        tck, u = interpolate.splprep(coord_list, k=1, s=0.0)
        upsampled_coords = interpolate.splev(np.linspace(0, 1, 100), tck)
        return upsampled_coords

    # target line
    x_targ = line1[0].astype('float')
    y_targ = line1[1].astype('float')
    targ_upsampled = upsample_coords([x_targ, y_targ])
    targ_coords = np.column_stack(targ_upsampled)


    # line two
    x2 = line2[0].astype('float')
    y2 = line2[1].astype('float')
    l2_upsampled = upsample_coords([x2, y2])
    l2_coords = np.column_stack(l2_upsampled)


    # find intersections
    lowestDiff = 1000
    xLowDiff = 0
    yLowDiff = 0
    for i in range(len(l2_coords)):
        diffY = abs(l2_coords[i] - targ_coords[i])
        if diffY[1] <= lowestDiff:
            lowestDiff = diffY[1]
            xLowDiff = targ_coords[i][0]
            yLowDiff = targ_coords[i][1]
    print(run)
    print('x = ', xLowDiff, 'y = ', yLowDiff)

def outcomeInteger(sysVec, confVec):
    flyby = sysVec[(confVec == '0 [1 2]')]
    ion = sysVec[(confVec == '0 1 2')]
    exch = sysVec[((confVec == '[0 2] 1') | (confVec == '[0 1] 2'))]
    mergBound = sysVec[((confVec == '[0:1 2]') | (confVec == '[0:2 1]') | (confVec == '[0 1:2]') | (confVec == '[1 2:0]') | (confVec == '[1:0 2]') | (confVec == '[0 2:1]')) ]
    mergIon = sysVec[((confVec == '0:1 2') | (confVec == '0:2 1') | (confVec == '0 1:2') | (confVec == '2:0 1') | (confVec == '2 1:0') | (confVec == '0 2:1'))]
    tripMerge = sysVec[(confVec == '0:1:2') | (confVec == '2:0:1') | (confVec == '0:2:1')]
    tripple = sysVec[(confVec == '[[0 2] 1]') | (confVec == '[[0 1] 2]') | (confVec == '[[1 2] 0]')]
    inspiral = sysVec[(confVec == 'Inspiral')]
    mergers = sysVec[(confVec == 'Merger')]
    PIMerger = sysVec[(confVec == 'Post-interaction merger')]

    return [flyby, exch, ion, mergBound, mergIon, tripMerge, tripple, inspiral, mergers, PIMerger], ['Flyby',  'Exchange', 'Ionization', 'Bound merger', 'Unbound merger',
               'Triple merger', 'Bound triple', 'Inspiral', 'Mergers', 'Post-interaction merger']

def outcomeInteger2BH(sysVec, confVec):
    flyby = sysVec[(confVec == '0 [1 2]')]
    ion = sysVec[(confVec == '0 1 2')]
    exch = sysVec[((confVec == '[0 2] 1') | (confVec == '[0 1] 2'))]
    mergBHStar = sysVec[((confVec == '[0:1 2]') | (confVec == '[0:2 1]') | (confVec == '[0 1:2]') | (confVec == '[1 2:0]') | (confVec == '[1:0 2]') | (confVec == '[0 2:1]') | (confVec == '0:1 2') | (confVec == '0:2 1') | (confVec == '0 1:2') | (confVec == '2:0 1') | (confVec == '2 1:0') | (confVec == '0 2:1') | (confVec == 'Merger')) & ~sysVec['BHBin']]
    mergBHBH = sysVec[((confVec == '[0:1 2]') | (confVec == '[0:2 1]') | (confVec == '[0 1:2]') | (confVec == '[1 2:0]') | (confVec == '[1:0 2]') | (confVec == '[0 2:1]') | (confVec == '0:1 2') | (confVec == '0:2 1') | (confVec == '0 1:2') | (confVec == '2:0 1') | (confVec == '2 1:0') | (confVec == '0 2:1') | (confVec == 'Merger')) & sysVec['BHBin']]
    # mergIon = sysVec[((confVec == '0:1 2') | (confVec == '0:2 1') | (confVec == '0 1:2') | (confVec == '2:0 1') | (confVec == '2 1:0') | (confVec == '0 2:1'))]
    tripMerge = sysVec[(confVec == '0:1:2') | (confVec == '2:0:1') | (confVec == '0:2:1')]
    tripple = sysVec[(confVec == '[[0 2] 1]') | (confVec == '[[0 1] 2]') | (confVec == '[[1 2] 0]')]
    inspiral = sysVec[(confVec == 'Inspiral')]
    inspiralStar = sysVec[(confVec == 'Inspiral star')]
    # mergers = sysVec[(confVec == 'Merger')]
    PIMerger = sysVec[(confVec == 'Post-interaction merger')]

    return [flyby, exch, ion, mergBHStar, mergBHBH, tripMerge, tripple, inspiral, inspiralStar, PIMerger], ['Flyby',  'Exchange', 'Ionization', 'BH-star merger', 'BH-BH merger',
               'Triple merger', 'Bound triple', 'Inspiral', 'Inspiral star', 'Post-interaction merger']

def outcomeIntegerOutcomeOrder2(sysVec, confVec):
    flyby = sysVec[(confVec == '0 [1 2]')]
    ion = sysVec[(confVec == '0 1 2')]
    exch = sysVec[((confVec == '[0 2] 1') | (confVec == '[0 1] 2'))]
    mergBound = sysVec[((confVec == '[0:1 2]') | (confVec == '[0:2 1]') | (confVec == '[0 1:2]') | (confVec == '[1 2:0]') | (confVec == '[1:0 2]') | (confVec == '[0 2:1]')) ]
    mergIon = sysVec[((confVec == '0:1 2') | (confVec == '0:2 1') | (confVec == '0 1:2') | (confVec == '2:0 1') | (confVec == '2 1:0') | (confVec == '0 2:1'))]
    tripMerge = sysVec[(confVec == '0:1:2') | (confVec == '2:0:1') | (confVec == '0:2:1')]
    tripple = sysVec[(confVec == '[[0 2] 1]') | (confVec == '[[0 1] 2]') | (confVec == '[[1 2] 0]')]

    return [flyby, exch, tripple, ion, mergBound, mergIon, tripMerge], ['Flyby',  'Exchange', 'Bound triple', 'Ionization', 'Bound merger',
                                                                                 'Unbound merger', 'Triple merger' ]

def confSimple(pltTitle, dfs, colNames):
    df = pd.DataFrame()
    for i in range(len(dfs)):
        data, outcomes = outcomeInteger(dfs[i], dfs[i]['conf'])

        numI = np.array([])
        confs = np.array(outcomes)

        for j in range(len(data)):
            numI = np.append(numI, len(data[j].index))

        dfSort = pd.DataFrame(numI, columns=[colNames[i]])
        dfSort = dfSort.set_index(confs)

        df = pd.concat([df, dfSort], axis=1, sort=False)


    df = df.fillna(0)
    df = df.loc[(df != 0).any(axis=1), :]
    dfNum = df
    df /= np.sum(df)

    sns.set_style("ticks")

    # sortInd = ['Flyby', 'Exchange', 'Bound merger', 'Unbound merger', 'Triple merger',  'Bound triple']
    sortInd = ['Flyby', 'Exchange', 'Ionization', 'Bound merger', 'Unbound merger', 'Triple merger',  'Bound triple']
    # sortInd = ['Flyby','Exchange', 'Bound triple', 'Ionization', 'Bound merger', 'Unbound merger', 'Triple merger']

    # sortInd = ['Flyby', 'Exchange', 'Ionization', 'Bound triple', 'Bound merger'] # 3 BH: low a, high e
    # sortInd = ['Flyby', 'Exchange', 'Bound merger', 'Unbound merger', 'Triple merger', 'Bound triple'] # 2b: Rmin peaks
    # sortInd = ['Flyby', 'Exchange', 'Bound merger', 'Unbound merger', 'Triple merger', 'Ionization', 'Bound triple'] # 2b: accuracy
    # sortInd = ['Flyby', 'Exchange', 'Bound triple', 'Ionization', 'Bound merger', 'Unbound merger', 'Triple merger'] # 3BH - manually setup


    # sortInd = ['Exchange', 'Flyby',  'Ionization', 'Bound triple']
    # sortInd = ['Exchange', 'Flyby', 'Bound triple']
    # sortInd = ['Exchange', 'Flyby', 'Inspiral', 'Mergers', 'Bound triple']
    # sortInd = ['Flyby', 'Exchange', 'Bound triple']

    # sortInd = ['Exchange', 'Flyby', 'Bound merger', 'Unbound merger', 'Triple merger', 'Bound triple']
    # sortInd = ['Exchange', 'Flyby', 'Bound merger', 'Ionization', 'Unbound merger', 'Triple merger', 'Bound triple']
    # sortInd = ['Flyby', 'Exchange', 'Bound merger', 'Unbound merger', 'Triple merger']
    # sortInd = ['Flyby', 'Exchange', 'Ionization', 'Bound merger', 'Bound triple']
    # sortInd = ['Flyby', 'Exchange', 'Bound triple', 'Ionization', 'Bound merger']

    # sortInd = ['Flyby', 'Exchange', 'Ionization', 'Bound merger', 'Unbound merger', 'Bound triple']     # man setup 5 seeds
    # sortInd = ['Flyby', 'Exchange', 'Ionization',  'Unbound merger', 'Bound triple']                    # man setup 1 seed
    # sortInd = ['Flyby', 'Exchange', 'Bound triple', 'Ionization', 'Bound merger', 'Unbound merger']     # phase angle







    df.loc[sortInd].plot(kind='bar',  legend=False, rot=0, title=pltTitle)
    # plt.legend(prop={'size': 30}, bbox_to_anchor=(1, 1))
#    df.sort_values(by='No regularisation',ascending=False).plot(kind='bar',  legend=True, rot=0, title=pltTitle)
    plt.title(pltTitle, size=30)
    plt.ylabel('Fraction', size=25)
    # plt.ylabel('Count', size=25)
    plt.xticks(fontsize=25, rotation=20)
    # plt.tight_layout()

    # a = plt.axes([.51, .45, .28, .2])
    # a = plt.axes([.395, .45, .425, .2])     # 2b accuracy
    # a = plt.axes([.43, .45, .405, .2])     # 2b accuracy good energy consv

    # a = plt.axes([.398, .45, .42, .2])     # 2a accuracy
    # a = plt.axes([.42, .45, .40, .2])     # 2a accuracy good energyConsv


    # a = plt.axes([.55, .35, .32, .2])     # 2b PM ks1
    # a = plt.axes([.65, .45, .25, .2])     # 2b PM ks0
    # a = plt.axes([.53, .45, .29, .2])     # 2b
    # a = plt.axes([.40, .45, .425, .2])     # 2a normal vs PM
    # a = plt.axes([.47, .45, .32, .2])     # 1BH
    # a = plt.axes([.337, .44, .48, .2])   # 2BH & 1BH
    # a = plt.axes([.34, .45, .45, .2])   # 2BH old vs new
    # a = plt.axes([.315, .50, .515, .2])   # handmade 2BH set
    # a = plt.axes([.565, .43, .24, .2])   # handmade 2BH set (fewer included in zoom)

    # a = plt.axes([.478, .45, .335, .2])     # 3BH: low a high
    # a = plt.axes([.53, .45, .31, .2])     # 2b: Rmin peaks

    # a = plt.axes([.5, .45, .31, .2])     # 3BH
    # a = plt.axes([.70, .40, .15, .2])     # 3BH seed vs no seed
    # a = plt.axes([.64, .45, .195, .2])     # manually setup BH

    # a = plt.axes([.555, .45, .21, .2])     # 2b rMin peak

    # a = plt.axes([.5, .45, .22, .2])     # 1 BH new tsunami
    # a = plt.axes([.485, .45, .355, .2])     # 3 BH man setup new tsunami
    # a = plt.axes([.51, .46, .255, .2])     # phase angle
    # a = plt.axes([.7, .40, .1, .2])     # 3BH seed vs no seed
    # a = plt.axes([.55, .40, .15, .2])     # 3BH seed vs no seed

    a1 = plt.axes([.46, .25, .30, .2])     # 2 BH MOCCA set (post-PN fix)





    """ 2a """
    # df.loc[['Bound merger', 'Unbound merger', 'Triple merger', 'Ionization', 'Bound triple']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)

    """ 2a accuracy """
    # df.loc[['Ionization', 'Bound merger', 'Unbound merger', 'Triple merger', 'Bound triple']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)

    """ 2b """
    # df.loc[['Unbound merger', 'Triple merger', 'Bound triple']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)

    """ 2b PM """
    # df.loc[['Ionization', 'Bound triple']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)    # ks1
    # df.loc[['Bound triple']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)    # ks0


    """ 2b accuracy """
    # df.loc[['Bound merger', 'Unbound merger', 'Triple merger', 'Ionization', 'Bound triple']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)

    """ 2b acc good dEE0 """
    # df.loc[['Bound merger', 'Unbound merger', 'Triple merger', 'Bound triple']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)


    """ 2b Rmin peaks """
    # df.loc[['Unbound merger', 'Triple merger', 'Bound triple']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)

    """ 1 BH """
    # df.loc[[ 'Bound merger', 'Unbound merger', 'Triple merger', 'Bound triple']].plot(kind='bar',  legend=False, rot=0, ax=a)

    """ 2 BH """
    # df.loc[['Exchange', 'Ionization', 'Bound merger', 'Unbound merger', 'Triple merger', 'Bound triple']].plot(kind='bar',  legend=True, rot=0, title=pltTitle, ax=a)
    # df.loc[['Unbound merger', 'Triple merger', 'Bound triple']].sort_values(by='No regularisation',ascending=False).plot(kind='bar',  legend=True, rot=0, title=pltTitle, ax=a)

    """ handmade 2 BH """
    # df.loc[['Exchange', 'Ionization', 'Bound merger', 'Unbound merger', 'Triple merger', 'Bound triple']].plot(kind='bar',  legend=True, rot=0, title=pltTitle, ax=a)
    # df.loc[['Bound merger', 'Unbound merger', 'Triple merger']].plot(kind='bar',  legend=True, rot=0, title=pltTitle, ax=a)


    """ 3 BH """
    # df.loc[['Ionization', 'Bound triple', 'Bound merger']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)

    """ manually setup BH """
    # df.loc[['Bound merger', 'Unbound merger', 'Triple merger']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)


    """ 2b bad energy consv """
    # df.loc[['Triple merger', 'Bound triple']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)


    """ 3 BH select interactions """
    # df.loc[['Ionization', 'Bound merger', 'Bound triple']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)
    # df.loc[['Ionization', 'Bound merger']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)


    """ rMin peaks """
    # df.loc[['Unbound merger', 'Triple merger']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)

    """ manually setup 3BH """
    # df.loc[['Ionization', 'Bound merger', 'Unbound merger']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)        # 5 seeds
    # df.loc[['Ionization', 'Unbound merger', 'Bound triple']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)        # 1 seed
    # df.loc[['Ionization', 'Bound merger', 'Bound triple' ]].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)          # lower a
    # df.loc[[ 'Ionization', 'Bound merger']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)        # johan set, log a


    """ 3 BH seed vs no seed new Tsunami """
    # df.loc[['Bound triple']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)
    # df.loc[['Ionization']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)
    # df.loc[['Bound merger']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a)

    """ MOCCA 2 BH (post-PN fix) """
    df.loc[['Bound merger', 'Unbound merger', 'Triple merger']].plot(kind='bar',legend=False, rot=0, title=pltTitle, ax=a1)
    plt.xticks([])
    plt.yticks([0, 0.0001], ['0', '1e-4'], rotation='vertical')
    plt.yticks(rotation='vertical')
    plt.title('')
    a2 = plt.axes([.80, .25, .10, .2])     # 2 BH MOCCA set (post-PN fix)


    df.loc[['Bound triple']].plot(kind='bar',legend=True, rot=0, title=pltTitle, ax=a2)




    plt.xticks([])
    plt.yticks([0, 0.004], ['0', '4e-3'], rotation='vertical')
    # plt.yticks([0, 0.03], ['0', '0.03'], rotation='vertical')   # 2b

    plt.yticks(rotation='vertical')
    a2.legend(prop={'size': 45}, bbox_to_anchor=(1, 3))
    # a.legend(prop={'size': 45}, bbox_to_anchor=(.65, 2.6))
    # a.legend(prop={'size': 25}, bbox_to_anchor=(1.2, 2.25), title='$\sigma = $')




    # a.get_legend().remove()
    plt.title('')

    # plt.legend()
def confSimpleMult(pltTitle, dfs, addLabel):
    dfPerm = pd.DataFrame()
    count = 0
    for df in dfs:

        ks0Sys = df[0]
        ks1Sys = df[1]
        KSys = df[2]


        outks0, outcomesks0 = outcomeInteger(ks0Sys, ks0Sys['conf'])
        outks1, outcomesks1 = outcomeInteger(ks1Sys, ks1Sys['conf'])
        outK, outcomesK = outcomeInteger(KSys, KSys['conf'])


        numIks0 = np.array([])
        numIks1 = np.array([])
        numIK = np.array([])
        confsks0 = np.array([])
        confsks1 = np.array([])
        confsK = np.array([])

        for i in range(len(outks0)):
            numIks0 = np.append(numIks0, len(outks0[i].index))
            confsks0 = np.append(confsks0, outcomesks0[i])

            numIks1 = np.append(numIks1, len(outks1[i].index))
            confsks1 = np.append(confsks1, outcomesks1[i])

            numIK = np.append(numIK, len(outK[i].index))
            confsK = np.append(confsK, outcomesK[i])



        dfks0 = pd.DataFrame(numIks0,  columns=['No regularisation ' + addLabel[count]])
        dfks1 = pd.DataFrame(numIks1,  columns=['KS regularisation '  + addLabel[count]])
        dfK1 = pd.DataFrame(numIK,  columns=['AR chain '  + addLabel[count]])


        dfks0 = dfks0.set_index(confsks0)
        dfks1 = dfks1.set_index(confsks1)
        dfK1= dfK1.set_index(confsK)


        dfTemp = pd.concat([dfks0/len(ks0Sys), dfks1/len(ks1Sys), dfK1/len(KSys)], axis=1, sort=False)
        dfPerm = pd.concat([dfPerm, dfTemp], axis=1, sort=False)
        count += 1


    dfPerm = dfPerm.fillna(0)

    dfPerm.sort_values(by='No regularisation ',ascending=False).plot(kind='bar',  legend=True, rot=0, title=pltTitle)
    plt.title(pltTitle, size=30)
    plt.legend(prop={'size': 20})
    plt.ylabel('Fraction', size=25)
    plt.xticks(fontsize=10)
    plt.tight_layout()

def confSankey(dfs, names):
    for i in range(len(dfs)):
        sys = dfs[i]
        out, outcomes = outcomeInteger(sys, sys['conf'])

        num = np.array([])
        confs = np.array([])

        for j in range(len(out)):
            num = np.append(num, len(out[j].index))
            confs = np.append(confs, outcomes[j])


        df = pd.DataFrame(num,  columns=['No regularisation'])
        df = df.set_index(confs)

        df = df.fillna(0)
        df = df.loc[(df != 0).any(axis=1), :]

        file = open(names[i] + '_sankey.txt', 'w')


        for index, row in df.iterrows():
            # print(index, row.iloc[0])
            file.write(names[i] + ' [' + str(row.iloc[0]) + '] ' + index + '\n')

        file.close()

def paramVS(df, param1, param2, title):

    out, outcomes = outcomeInteger(df, df['conf'])

    k = 0
    for i in out:
        p1 = i[param1].astype('float')
        p2 = i[param2].astype('float')

#        plt.figure()
#        plt.scatter(p1, p2, s=0.1)
        if len(p1) > 1:
            plt.figure()
            h = plt.hist2d(p1, p2, bins=20, norm=LogNorm())
            cbar= plt.colorbar(h[3], label='Count')
            cbar.set_clim(0,50)
            plt.title(outcomes[k] + ' (' + title + ')')
            plt.xlabel('e$_{\mathrm{initial}}$')
            plt.ylabel('e$_{\mathrm{final}}$')
            plt.tight_layout()
            k += 1
        else:
            k += 1
    # fig, axes=plt.subplots(nrows=2, ncols=1)
    # for i in out:
    #     if outcomes[k] == 'Flyby' or outcomes[k] == 'Exchange':
    #         p1 = i[param1].astype('float')
    #         p2 = i[param2].astype('float')

    # #        plt.figure()
    # #        plt.scatter(p1, p2, s=0.1)
    #         if len(p1) > 1:
    #             if outcomes[k] == 'Flyby':
    #                 ax1 = plt.subplot(121)
    #                 plt.ylabel('e$_{\mathrm{final}}$')

    #             elif outcomes[k] == 'Exchange':
    #                 ax2 = plt.subplot(122, sharey=ax1)

    #             h = plt.hist2d(p1, p2, bins= 40, norm=LogNorm())

    #             plt.title(outcomes[k] + ' (' + title + ')')
    #             plt.xlabel('e$_{\mathrm{initial}}$')
    #             k += 1
    #         else:
    #             k += 1
    # # plt.colorbar(h[3], label='Count')
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(h[3], cax=cbar_ax)


def paramVSMinus(df, param1, param2, title, xlab, ylab):

    out, outcomes = outcomeInteger(df, df['conf'])

    k = 0
    for i in out:
        if param1 == 'e1' or param1 == 'e_f1':
            p1 = 1 - i[param1].astype('float')
        else:
            p1 = i[param1].astype('float')

        if param2 == 'e_f1' or param2 == 'e1':
            p2 = 1- i[param2].astype('float')
        else:
            p2 = i[param2].astype('float')

        binsX=np.logspace(np.log10(np.amin(p1)),np.log10(np.amax(p1)), 100)
        binsY=np.logspace(np.log10(np.amin(p2)),np.log10(np.amax(p2)), 100)

#        plt.figure()
#        plt.scatter(p1, p2, s=0.1)
        if len(p1) > 1:
            plt.figure()
            h = plt.hist2d(p1, p2, bins=[binsX, binsY], norm=LogNorm())
            plt.colorbar(h[3], label='Count')
            if len(title) > 1:
                plt.title(outcomes[k] + ' (' + title + ')')
            else:
                plt.title(outcomes[k])

            if param1 == 'e1' and param2 == 'e_f1':
                beforeHighE = p1[p1 < 1e-2]
                afterHighE = p2[p2 < 1e-2]
                print(outcomes[k])
                print('Before: ' + str(len(beforeHighE)) + '\nAfter: ' + str(len(afterHighE)) + '\n')

                x = [0, 100]
                y = [0, 100]

                plt.plot(x,y, c='black', linestyle='--')


            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.tight_layout()
            plt.xscale('log')
            plt.yscale('log')
            if param1 == 'e1':
                plt.xlim(binsX[0], 1)
            if param2 == 'e_f1':
                plt.ylim(binsY[0], 1.2)
            # plt.tight_layout()
            k += 1
        else:
            k += 1



def paramDiff(dfs, param1, param2, xmax, xstep, lbls, xlab, pTitle):
    plt.figure()
    plt.axvline(x=0, c='black', linestyle='--', alpha=0.2)
    i = 0
    for df in dfs:
        mask = (df[param1] != 0) &(df['conf'] == '0 [1 2]')
        diff =  df[param1][mask] - df[param2][mask].to_numpy(dtype='float')
    #    diffE = df['e_f1'][mask] - df['e1'][mask].to_numpy(dtype='float')
        ##

        plt.hist(diff, bins=np.arange(-xmax, xmax, xstep), label=lbls[i], alpha=0.5, histtype='step')
        i += 1


    plt.xlabel(xlab)
    plt.ylabel('count')
    plt.title(pTitle)
    plt.legend()
    plt.tight_layout()

def paramDiffConf(dfs, param1, param2, xmax, xstep, lbls, xlab, pTitle):
    plt.figure()
    plt.axvline(x=0, c='black', linestyle='--', alpha=0.2)

    outks0, outcomesks0 = outcomeInteger(dfs[0], dfs[0]['conf'])
    outks1, outcomesks1 = outcomeInteger(dfs[1], dfs[1]['conf'])
    outK, outcomesK = outcomeInteger(dfs[2], dfs[2]['conf'])
    for i in range(len(outks0)):
        mask1 = (outks0[i][param1] != 0)
        mask2 = (outks1[i][param1] != 0)
        mask3 = (outK[i][param1] != 0)

        diffks0 =  outks0[i][param1][mask1].to_numpy(dtype='float') - outks0[i][param2][mask1].to_numpy(dtype='float')
        diffks1 =  outks1[i][param1][mask2].to_numpy(dtype='float') - outks1[i][param2][mask2].to_numpy(dtype='float')
        diffK =  outK[i][param1][mask3].to_numpy(dtype='float') - outK[i][param2][mask3].to_numpy(dtype='float')


        plt.figure()
        plt.hist(diffks0, bins=np.arange(-xmax, xmax, xstep), alpha=0.5, histtype='step', label='ks0')
        plt.hist(diffks1, bins=np.arange(-xmax, xmax, xstep), alpha=0.5, histtype='step', label='ks1')
        plt.hist(diffK, bins=np.arange(-xmax, xmax, xstep), alpha=0.5, histtype='step', label='K')

        plt.title(outcomesks0[i])
        plt.xlabel(xlab)
        plt.ylabel('count')
        plt.legend(loc='upper left')
        plt.tight_layout()

def compInputConfMinusDouble(param1, param2, pTitle, xMin, xMax, xStep, xlbl, sqr):
    outks0, outcomesks0 = outcomeInteger(K, K['conf'])

    for i in range(len(outks0)):
        if sqr == 1:
            p1 = 1 - outks0[i][param1][outks0[i]['a_f1'].astype('float') != 0].astype('float')**2
            p2 = 1 - outks0[i][param2][outks0[i]['a_f1'].astype('float') != 0].astype('float')**2
        else:
            p1 = 1 - outks0[i][param1][outks0[i]['a_f1'].astype('float') != 0].astype('float')
            p2 = 1 - outks0[i][param2][outks0[i]['a_f1'].astype('float') != 0].astype('float')

        if len(p1) == 0 and len(p2) == 0:
            continue

        bins1 = np.logspace(np.log10(np.amin(p1)),np.log10(np.amax(p1)), 100)
        bins2 = np.logspace(np.log10(np.amin(p2)),np.log10(np.amax(p2)), 100)


        plt.figure()
        plt.hist(p1,  histtype='step', alpha=1, label='Initial', bins=bins1)
        plt.hist(p2, histtype='step', alpha=1, label='Final', bins=bins2)
        plt.ylabel('Count')
        plt.xlabel('1 - ' + xlbl)
        plt.xscale('log')
        plt.title(outcomesks0[i])
        plt.legend(loc='upper left')
        plt.tight_layout()

def compV(dfs, flags):
    i = 0

    for df in dfs:
        plt.figure()
        out, outcome = outcomeInteger(df, df['conf'])

        uBMergers = out[4][['vInf', 'vCrit Unit', 'vx1', 'vy1', 'vz1']].astype('float')
        bMergers = out[3][['vInf', 'vCrit Unit', 'vx1', 'vy1', 'vz1']].astype('float')

        velUBM = (np.sqrt(uBMergers['vx1']**2 + uBMergers['vy1']**2 + uBMergers['vz1']**2) - uBMergers['vInf']) * uBMergers['vCrit Unit']
        velBM = (np.sqrt(bMergers['vx1']**2 + bMergers['vy1']**2 + bMergers['vz1']**2) - bMergers['vInf']) * bMergers['vCrit Unit']

        plt.title(flags[i])
        plt.hist(velUBM, histtype='step', label='Unbound')
        plt.hist(velBM, histtype='step', label='Bound')
        plt.xlabel('$\Delta v$ [km/s]')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        print(flags[i])
        print(np.mean(abs(velUBM)), np.mean(abs(velBM)))
        i += 1


def triplesBHStarInitial(df, origConf):
    masses = df[['m0', 'm10', 'm11']].astype(float)
    radii = df[['r0', 'r10', 'r11']]
    minRad = np.argmin(masses.to_numpy(), axis=1)

    BHBHInit = np.ones(len(minRad))
    for i in range(len(origConf)):
        outerIndex = int(origConf.iloc[i].split(']')[1])
        if minRad[i] == 0:
            BHBHInit[i] = True
        else:
            BHBHInit[i] = False


    print(np.sum(BHBHInit))

def inspiralBH(df, origConf):
    binMembers = pd.Series(origConf).apply(lambda st: st[st.find("[")+1:st.find("]")]).str.split(' ', expand=True)
    BM1 = binMembers[0].str[1]
    BM2 = binMembers[1]


    # radii = np.array([df['r0'], df['r10'], df['r11']])
    # masses = np.array([df['m0'].astype(float), df['m10'].astype(float), df['m11'].astype(float)]).T
    radii = np.array([df['r0'].astype(float), df['r10'].astype(float), df['r11'].astype(float)]).T

    # maxRad = np.argmax(radii, axis=0)
    minMass = np.argmax(radii, axis=1)

    BHBin = ((minMass.astype(str) != BM1) & (minMass.astype(str) != BM2))

    return BHBin

def starInInnerBin(df, origConf):
    binMembers = pd.Series(origConf).apply(lambda st: st[st.find("[")+1:st.find("]")]).str.split(' ', expand=True)
    BM1 = binMembers[0].str[1]
    BM2 = binMembers[1]


    # radii = np.array([df['r0'], df['r10'], df['r11']])
    masses = np.array([df['m0'].astype(float), df['m10'].astype(float), df['m11'].astype(float)]).T
    radii = np.array([df['r0'].astype(float), df['r10'].astype(float), df['r11'].astype(float)]).T

    # maxRad = np.argmax(radii, axis=0)
    minMass = np.argmax(radii, axis=1)

    BHBin = ((minMass.astype(str) == BM1) | (minMass.astype(str) == BM2))

    return BHBin

def BHBHConf(df, conf):
    outcomes, outnames = isolateOutcome(df, conf)
    minMass = np.argmin([outcomes['m0'].astype(float), outcomes['m10'].astype(float), outcomes['m11'].astype(float)], axis=0)
    minSingle = (minMass == 0)
    print(len(df))
    print('init star single: ' + str(np.sum(minSingle)))
    print('init BH single: ' + str(np.sum((minMass != 0))) + '\n')


def paramScatterConf(df, dfName, param1, param2, xlbl, ylbl, outC):
    c = 63239.7263  # AU/yr
    G = 39.478      # AU3 * yr-2 * Msun-1

    # t = (5/256 * c**5 * df['a_f1']**4 * (1-df['e_f1']**2)**(7/2)/(G**3 * 10 * 10 * (10 + 10)))
    t = df['mergerTime'].astype(float)
    periDist = df['a_f1'] * (1 - df['e_f1'])
    periDist[(df['a_f1'] < 0) | (df['e_f1'] > 1)] = 1e7
    outerPeriod = 2 * np.pi * np.sqrt(df['a_f1_O']**3 / ( G * 10))
    dfCopy = df.copy()

    merger = (periDist < df['mergeDist'])

    mergersDF = ((periDist < df['mergeDist']) & (df['conf'] != '[[0 1] 2]') & (df['conf'] != '[[0 2] 1]') & (df['conf'] != '[[1 2] 0]') & (df['conf'] != '0 1 2')
                 & (df['conf'] != '0:1:2') & (df['conf'] != '0:2:1') & (df['conf'] != '1:0:2') & (df['conf'] != '1:2:0') & (df['conf'] != '2:1:0') & (df['conf'] != '0 1 2'))


    dfCopy['conf'][mergersDF] = 'Merger'
    # dfTemp['conf'][(t <= 1) & (dfTemp['conf'] != '[[0 1] 2]') & (dfTemp['conf'] != '[[0 2] 1]') &
    #                 (dfTemp['conf'] != '[[1 2] 0]') & (dfTemp['conf'] != 'Merger')] = ['Inspiral']
    # dfCopy['conf'][(~np.isnan(df['mergeDist'])) & (periDist > 0) & (periDist < df['mergeDist']) & ((df['conf'] == '[[0 2] 1]') | (df['conf'] == '[[0 1] 2]') | (df['conf'] == '[[1 2] 0]')) & (~df['BHBin'])] = 'Inspiral star'

    # inspiralsOrgConf = dfCopy['conf'][(~np.isnan(df['mergeDist'])) & (periDist > 0) & (periDist < df['mergeDist']) & ((df['conf'] == '[[0 2] 1]') | (df['conf'] == '[[0 1] 2]') | (df['conf'] == '[[1 2] 0]'))]

    # dfCopy['conf'][(~np.isnan(df['mergeDist'])) & (periDist > 0) & (periDist < df['mergeDist']) & ((df['conf'] == '[[0 2] 1]') | (df['conf'] == '[[0 1] 2]') | (df['conf'] == '[[1 2] 0]'))] = 'Inspiral'
    # inspiralsOrgConf = np.concatenate((inspiralsOrgConf, dfCopy['conf'][(~np.isnan(t)) & (t < outerPeriod) & (outerPeriod > 0)  & (dfCopy['conf'] != 'Merger') & (dfCopy['conf'] != 'Inspiral')]))

    # dfCopy['conf'][(~np.isnan(t)) & (t < outerPeriod) & (outerPeriod > 0)  & (dfCopy['conf'] != 'Merger')] = ['Inspiral']
    # inspiralsOrgConf = np.concatenate((inspiralsOrgConf, dfCopy['conf'][(~np.isnan(t)) & (t < np.amax(outerPeriod)) & (df['e_f1_O'] > 1)  & (dfCopy['conf'] != 'Merger') & (dfCopy['conf'] != 'Inspiral')]))

    # dfCopy['conf'][(~np.isnan(t)) & (t < np.amax(outerPeriod)) & (df['e_f1_O'] > 1)  & (df['conf'] != 'Merger')] = ['Inspiral']
    dfCopy['conf'][(t < 14e9) & (t != 0) & (df['conf'] != 'Inspiral') & (df['conf'] != '[[0 1] 2]') & (df['conf'] != '[[0 2] 1]') &
                    (df['conf'] != '[[1 2] 0]') & (df['conf'] != 'Merger') & (df['BHBin'])] = ['Post-interaction merger']

    out, outcomes = outcomeInteger2BH(dfCopy, dfCopy['conf'])


    # num BHBH/BHStar exchanges/flyby
    # BHBHConf(out[0], 'Flyby')
    BHBHConf(out[1], 'Exchange')



    BHBHMergers = out[4]

    # radii = [BHBHMergers['r0'].astype(float), BHBHMergers['r10'].astype(float), BHBHMergers['r11'].astype(float)]

    # minRad = np.argmin(radii, axis=0)
    # binMembers = pd.Series(BHBHMergers['conf']).apply(lambda st: st[st.find("[")+1:st.find("]")]).str.split(' ', expand=True)

    # inspiralBHStarInitial(out[-4], inspiralsOrgConf)
    BHBoundTriples = out[-4][inspiralBH(out[-4], out[-4]['conf']).to_numpy()]
    BHStarBoundTriples = out[-4][starInInnerBin(out[-4], out[-4]['conf']).to_numpy()]

    # out[-4] = pd.concat([out[-4], out[-3][~BHBinInspirals.to_numpy()]])
    # out[-3] = out[-3][BHBinInspirals.to_numpy()]

    triplesBHStarInitial(out[-4], out[-4]['conf'])


    # print(len(out[-4]))
    # print(len(out[-3]))

    indexInspirals = out[-3].index
    for i in range(len(out)):
        print(outcomes[i] + ': ' + str(len(out[i]) / len(df)))
    # postInterMerger = out[-1]
    print('\n')

    plt.figure()
    colors = ['magenta', 'crimson', 'black', 'green', 'magenta', 'purple', 'brown', 'red', 'blue', 'orange']

    for i in range(len(out)):
        if outC == 'all':
            if len(out[i][param1]) == 0 or outcomes[i] == 'Flyby' or outcomes[i] == 'Exchange' or outcomes[i] == 'Ionization' or outcomes[i] == 'Bound triple':
                continue

            if param2 == 'bImp':
                if outcomes[i] == 'Inspiral' or outcomes[i] == 'Inspiral star':
                    plt.scatter(out[i][param1].astype('float'), out[i][param2].astype('float')*out[i]['a1'].astype('float'), s=50, alpha=1, label=outcomes[i], color=colors[i], zorder=1)
                else:
                    plt.scatter(out[i][param1].astype('float'), out[i][param2].astype('float')*out[i]['a1'].astype('float'), s=50, alpha=.5, label=outcomes[i], color=colors[i], zorder=-1)

            elif param2 == 'e1':
                if outcomes[i] == 'Inspiral' or outcomes[i] == 'Inspiral star':
                    plt.scatter(out[i][param1].astype('float'), 1-out[i][param2].astype('float'), s=50, alpha=1, label=outcomes[i], color=colors[i], zorder=1)
                else:
                    plt.scatter(out[i][param1].astype('float'), 1-out[i][param2].astype('float'), s=50, alpha=.5, label=outcomes[i], color=colors[i], zorder=-1)
            else:
                if outcomes[i] == 'Inspiral' or outcomes[i] == 'Inspiral star':
                    plt.scatter(out[i][param1].astype('float'), out[i][param2].astype('float'), s=50, alpha=1, label=outcomes[i], color=colors[i], zorder=1)
                else:
                    plt.scatter(out[i][param1].astype('float'), out[i][param2].astype('float'), s=50, alpha=.5, label=outcomes[i], color=colors[i], zorder=-1)
    #         plt.xlabel(xlbl)
    #         plt.ylabel(ylbl)
    #         # plt.xlim(0.1,100)
    # #            plt.title(outcomes[i] + ' (' + dfName + ') (1 M$_{\odot}$ star ejected)')
    #         plt.title(outcomes[i] + ' (' + dfName + ')')
    #         plt.xscale('log')
    #         plt.yscale('log')
    #         plt.tight_layout()
            # plt.axvline(x=1, linestyle='--', c='black', zorder=2)
        elif outcomes[i] == outC:
            plt.figure()
            if param2 == 'bImp':
                plt.scatter(out[i][param1].astype('float'), out[i][param2].astype('float')*out[i]['a1'].astype('float'), s=0.4, alpha=0.5)
            else:
                plt.scatter(out[i][param1].astype('float'), out[i][param2].astype('float'), s=0.4, alpha=0.5)

    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.legend()

#            plt.title(outcomes[i] + ' (' + dfName + ') (1 M$_{\odot}$ star ejected)')
    plt.title(dfName)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-5,1)
    plt.xlim(2e-2, 1e4)

    inspirals = out[-3]
    # bins=np.logspace(-3, 1, 50)
    # # bins=np.linspace(0, 1, 50)

    # paramAddInnerBinary(inspirals, 'radius', dfName, inspiralsOrgConf)


    # radii = pd.concat([df['r0'], df['r10'], df['r11']]).astype(float)
    # plt.figure()
    # # inspirals['a1'].astype(float).hist(bins=bins, histtype='step', label='Initial', linewidth=3)
    # # inspirals['a_f1'].astype(float).hist(bins=bins, histtype='step', label='Final', linewidth=3)
    # plt.scatter(inspirals['a1'].astype(float), inspirals['e1'].astype(float), label='Initial', s=50)
    # plt.scatter(inspirals['a_f1'].astype(float), inspirals['e_f1'].astype(float), label='Final', s=50)

    # plt.title(dfName + ' inspirals')
    # plt.legend()
    # # plt.ylim(bins[0], bins[-1])
    # plt.xlim(bins[0], bins[-1])
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.xlabel('a [AU]')
    # plt.ylabel('e')
    # plt.grid(False)

    return BHBHMergers




def paramScatter(df, dfName, param1, param2, xlbl, ylbl):
    out = df

    plt.figure()
    plt.scatter(out[param1].astype('float'), 1-out[param2].astype('float'), s=25, alpha=1)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(dfName)
    plt.tight_layout()

def paramScatterMult(dfs, dfNames, param1, param2, xlbl, ylbl, title):
    plt.figure()

    for i in range(len(dfs)):
        out = dfs[i]
        xDat = out[param1].astype('float') * out['l Unit'].astype(float)
        yDat = out[param2].astype('float') * out['vCrit Unit'].astype(float)
        # print(yDat[0:10])
        plt.scatter(xDat, yDat, s=45, alpha=1, zorder=3+i, label=dfNames[i])
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(title)
    plt.xlim(0.001, 100)
    plt.ylim(0.001, 100)
    # plt.ylim(0, 1)
    plt.legend()


    # plt.title("Exchange")
    # plt.tight_layout()


def timeMergeGWGivenT(m1, m2, tVec, e, labels):
    c = 63198       # AU/yr
    G = 39.478      # AU3 * yr-2 * Msun-1
    aVec = pd.DataFrame()
    k = 0
    for t in tVec:
        a = pd.Series((256/5 * G**3 * m1*m2*(m1+m2)*t / (c**5 * (1-e**2)**(7/2)))**(1/4), name=labels[k])
        aVec = pd.concat([aVec, a], axis=1)
        # t = 5/256 * c^5 * a^4 * (1-e^2)^(7/2)/(G^3 * m1 * m2 * (m1 + m2))
        k += 1
    return aVec

def paramScatterConfMinus(df, dfName, param1, param2, xlbl, ylbl, outC):
    out, outcomes = outcomeInteger(df, df['conf'])

    for i in range(len(out)):
        if outcomes[i] == outC:
            if param1 == 'e1' or param1 == 'e_f1':
                p1 = 1 - out[i][param1].astype('float')
            else:
                p1 = out[i][param1].astype('float')

            if param2 == 'e_f1' or param2 == 'e1':
                p2 = 1- out[i][param2].astype('float')
            else:
                p2 = out[i][param2].astype('float')

            # if (param1 == 'e_f1' and param2 == 'a_f1') or (param1 == 'e1' and param2 == 'a1'):
            #     tVec = [1e6, 1e8, 1e9, 1e10]
            #     tLabels = ['1 Myr', '100 Myr', '1 Gyr', '10 Gyr']
            #     eccVec = np.logspace(np.log10(0.99999), -7, 100)
            #     times = timeMergeGWGivenT(20, 20, tVec, eccVec, tLabels)

            plt.figure()
            # colors = ['purple', 'orange', 'green', 'red']
            # for j in range(len(times.columns)):
            #     plt.plot(1-eccVec, times.iloc[:,j], label=tLabels[j], c=colors[j], zorder=2)
            plt.scatter(p1, p2, s=1.5, alpha=1, zorder=1)


            plt.xlabel(xlbl)
            plt.ylabel('1 - ' + ylbl)
            # plt.legend(title='Time to merge:')
            if len(dfName) > 0:
                plt.title(outcomes[i] + ' (' + dfName + ')')
            else:
                plt.title(outcomes[i])

            plt.xlim(0.5e-7, 1.1)
            plt.ylim(1e-4, 11)
            plt.yscale('log')
            plt.xscale('log')
            plt.tight_layout()

def paramScatterConfColor(df, dfName, param1, param2, cParam, xlbl, ylbl):
    # out, outcomes = outcomeInteger(df, df['conf'])
    out = df

    # for i in range(len(out)):
        # if outcomes[i] == outC:
    if param1 == 'e1' or param1 == 'e_f1':
        p1 = out[param1].astype('float')
    else:
        p1 = out[param1].astype('float')

    if param2 == 'e_f1' or param2 == 'e1':
        p2 = out[param2].astype('float')
    elif param2 == 'bImp':
        p2 = out[param2].astype(float) * out['a1'].astype(float)
    else:
        p2 = out[param2].astype('float')


    # if (param1 == 'e_f1' and param2 == 'a_f1') or (param1 == 'e1' and param2 == 'a1'):
    #     tVec = [1e6, 1e8, 1e9, 1e10]
    #     tLabels = ['1 Myr', '100 Myr', '1 Gyr', '10 Gyr']
    #     eccVec = np.logspace(np.log10(0.99999), -7, 100)
    #     times = timeMergeGWGivenT(20, 20, tVec, eccVec, tLabels)

    plt.figure()
    # colors = ['purple', 'orange', 'green', 'red']
    # for j in range(len(times.columns)):
    #     plt.plot(1-eccVec, times.iloc[:,j], label=tLabels[j], c=colors[j], zorder=2)
    plt.scatter(p1, p2, c=out[cParam].astype(float)*214.93946938, s=30, alpha=1, zorder=1, cmap='magma_r')


    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    # plt.legend(title='Time to merge:')
    # if len(dfName) > 0:
    #     plt.title(outcomes[i] + ' (' + dfName + ')')
    # else:
    #     plt.title(outcomes[i])
    plt.colorbar(label='R$_{min}$ [R$_{\odot}$]')

    # plt.xlim(1e-5, 1.1)
    # plt.ylim(1e-2, 15)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.tight_layout()

def paramScatterConfMinusDouble(dfs, dfNames, param1, param2, xlbl, ylbl, outC):

    # out2, outcomes = outcomeInteger(df2, df2['conf'])

    # dfs = [df1, df2]
    # dfNames = [dfName1, dfName2]

    if (param1 == 'e_f1' and param2 == 'a_f1') or (param1 == 'e1' and param2 == 'a1'):
        tVec = [1e6, 1e8, 1e9, 1e10]
        tLabels = ['1 Myr', '100 Myr', '1 Gyr', '10 Gyr']
        eccVec = np.logspace(np.log10(0.99999), -7, 100)
        times = timeMergeGWGivenT(20, 20, tVec, eccVec, tLabels)

    legHandles = []
    plt.figure()
    for k in range(len(dfs)):
        out, outcomes = outcomeInteger(dfs[k], dfs[k]['conf'])
        for i in range(len(out)):
            if outcomes[i] == outC:
                if param1 == 'e1' or param1 == 'e_f1':
                    p1 = 1 - out[i][param1].astype('float')
                else:
                    p1 = out[i][param1].astype('float')

                if param2 == 'e_f1' or param2 == 'e1':
                    p2 = 1- out[i][param2].astype('float')
                else:
                    p2 = out[i][param2].astype('float')



                line = plt.scatter(p1, p2, s=0.4, alpha=1, zorder=1)
                legHandles.append(line)


    colors = ['purple', 'orange', 'green', 'red']

    for j in range(len(times.columns)):
        plt.plot(1-eccVec, times.iloc[:,j], label=tLabels[j], c=colors[j], zorder=2)


    plt.xlabel('1 - ' + xlbl)
    plt.ylabel(ylbl)

    ax = plt.gca()
    leg1 = ax.legend(title='Time to merge:')
    leg2 = ax.legend(legHandles, ['Reduced', 'Increased'], loc='upper left', markerscale=10)
    ax.add_artist(leg1)

    if len(dfNames[0]) > 0:
        plt.title(outcomes[i] + ' (' + dfNames[0] + ')')
    else:
        plt.title(outcomes[i])

    plt.xlim(1e-5, 1.1)
    plt.ylim(1e-2, 15)
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()


def paramScatterConfMinusInitialAfter(dfs, dfNames, param1, param2, param3, param4, pLabel1, pLabel2, xlbl, ylbl, outC):

    if (param1 == 'e_f1' and param2 == 'a_f1') or (param1 == 'e1' and param2 == 'a1'):
        tVec = [1e6, 1e8, 1e9, 1e10]
        tLabels = ['1 Myr', '100 Myr', '1 Gyr', '10 Gyr']
        eccVec = np.logspace(np.log10(0.99999), -7, 100)
        times = timeMergeGWGivenT(20, 20, tVec, eccVec, tLabels)

    legHandles = []
    plt.figure()
    out, outcomes = outcomeInteger(dfs, dfs['conf'])
    for i in range(len(out)):
        if outcomes[i] == outC:
            if param1 == 'e1' or param1 == 'e_f1':
                p1 = 1 - out[i][param1].astype('float')
            else:
                p1 = out[i][param1].astype('float')

            if param2 == 'e_f1' or param2 == 'e1':
                p2 = 1- out[i][param2].astype('float')
            else:
                p2 = out[i][param2].astype('float')

            if param3 == 'e1' or param3 == 'e_f1':
                p3 = 1 - out[i][param3].astype('float')
            else:
                p3 = out[i][param3].astype('float')

            if param4 == 'e_f1' or param4 == 'e1':
                p4 = 1- out[i][param4].astype('float')
            else:
                p4 = out[i][param4].astype('float')




            line1 = plt.scatter(p1, p2, s=0.6, alpha=0.6, zorder=1)
            line2 = plt.scatter(p3, p4, s=0.6, alpha=0.6, zorder=1)
            legHandles.append(line1)
            legHandles.append(line2)


    colors = ['purple', 'orange', 'green', 'red']

    for j in range(len(times.columns)):
        plt.plot(1-eccVec, times.iloc[:,j], label=tLabels[j], c=colors[j], zorder=2)


    plt.xlabel('1 - ' + xlbl)
    plt.ylabel(ylbl)

    ax = plt.gca()
    leg1 = ax.legend(title='Time to merge:', loc='lower left')
    leg2 = ax.legend(legHandles, [pLabel1, pLabel2], loc='upper left', markerscale=10)
    ax.add_artist(leg1)

    if len(dfNames[0]) > 0:
        plt.title(outC + ' (' + dfNames[0] + ')')
    else:
        plt.title(outC)

    plt.xlim(1e-5, 1.1)
    plt.ylim(1e-2, 15)
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()

def paramScatterConfMinusInitialOrAfter(dfs, dfNames, param1, param2, pLabel1, pLabel2, xlbl, ylbl, outC):

    if (param1 == 'e_f1' and param2 == 'a_f1') or (param1 == 'e1' and param2 == 'a1'):
        tVec = [1e6, 1e8, 1e9, 1e10]
        tLabels = ['1 Myr', '100 Myr', '1 Gyr', '10 Gyr']
        eccVec = np.logspace(np.log10(0.99999), -7, 100)
        times = timeMergeGWGivenT(20, 20, tVec, eccVec, tLabels)

    legHandles = []
    plt.figure()
    out, outcomes = outcomeInteger(dfs, dfs['conf'])
    for i in range(len(out)):
        if outcomes[i] == outC:
            if param1 == 'e1' or param1 == 'e_f1':
                p1 = 1 - out[i][param1].astype('float')
            else:
                p1 = out[i][param1].astype('float')

            if param2 == 'e_f1' or param2 == 'e1':
                p2 = 1- out[i][param2].astype('float')
            else:
                p2 = out[i][param2].astype('float')




            line1 = plt.scatter(p1, p2, s=0.6, alpha=0.6, zorder=1)
            legHandles.append(line1)


    colors = ['purple', 'orange', 'green', 'red']

    for j in range(len(times.columns)):
        plt.plot(1-eccVec, times.iloc[:,j], label=tLabels[j], c=colors[j], zorder=2)


    plt.xlabel('1 - ' + xlbl)
    plt.ylabel(ylbl)

    ax = plt.gca()
    leg1 = ax.legend(title='Time to merge:', loc='lower left')
    # leg2 = ax.legend(legHandles, [pLabel1, pLabel2], loc='upper left', markerscale=10)
    ax.add_artist(leg1)

    if len(dfNames[0]) > 0:
        # plt.title(outC + ' (' + dfNames[0] + ')')
        plt.title('Initial')
    else:
        plt.title(outC)

    plt.xlim(1e-5, 1.1)
    plt.ylim(1e-2, 15)
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()

def paramScatterConfXMinus(df, dfName, param1, param2, xlbl, ylbl, outC):
    out, outcomes = outcomeInteger(df, df['conf'])

#    binData = df[param1].astype('float')
#    diffH = np.diff(np.unique(binData)).min()
#    lfb = binData.min() - float(diffH)/2
#    rlb = binData.max() + float(diffH)/2
#    binsHE = np.arange(lfb, rlb + diffH, diffH*2)

    for i in range(len(out)):
        if outcomes[i] == outC:
            plt.figure()
            x = 1 - out[i][param1].astype('float')
            y = out[i][param2].astype('float')
            plt.scatter(x, y, s=0.4, alpha=0.5)
            plt.yscale('log')
            plt.xscale('log')

            plt.xlabel('1 - ' + xlbl)
            plt.ylabel(ylbl)
    #            plt.title(outcomes[i] + ' (' + dfName + ') (1 M$_{\odot}$ star ejected)')
            plt.title(outcomes[i] + ' (' + dfName + ')')
            plt.ylim(1e-3, 13)
            plt.xlim(1e-6, 1)

            plt.tight_layout()

def isolateOutcome(df, conf):
    out, outcome = outcomeInteger(df, df['conf'])
    outRet = pd.DataFrame()
    outcomesRet = np.array([])
    for i in range(len(outcome)):
        if outcome[i] in conf:
            # if outcome[i] == 'Ionization':
            #     out[i] = out[i][out[i]['vInf'].astype(float) > 1]

            outRet = pd.concat([outRet, out[i]])
            outcomesRet = np.append(outcomesRet, outcome[i])

    return outRet, outcomesRet

def massesOutcome(df):
    m0 = df['m0'].astype('float')
    m10 = df['m10'].astype('float')
    m11 = df['m11'].astype('float')
    binMass = m10 + m11

    binsArr = np.linspace(0, 50, 30)

    plt.hist(m0, histtype='step', bins=binsArr, label='Single')
    plt.hist(m10, histtype='step', bins=binsArr, label='Bin 1')
    plt.hist(m11, histtype='step', bins=binsArr, label='Bin 2')
    plt.xlabel('Mass [M$_{\odot}$]')
    plt.ylabel('Count')
    plt.legend()

    binsArr = np.linspace(0, 100, 50)

    plt.figure()
    plt.hist(m0, histtype='step', bins=binsArr, label='Single')
    plt.hist(binMass, histtype='step', bins=binsArr, label='Binary')
    plt.xlabel('Mass [M$_{\odot}$]')
    plt.ylabel('Count')
    plt.legend()


def paramHistSingle(df, param, xlab, binLims, pltTitle):
    data = df[param].astype('float')

    binArr = np.linspace(binLims[0], binLims[1], binLims[2])

    print('Mean: ' + str(np.mean(data)))
    print('Median: ' + str(np.median(data)))
    plt.figure()
    plt.hist(data, bins=binArr)
    if param == 'vInf':
        plt.axvline(x=1, linestyle='--', c='black')
    plt.xlabel(xlab)
    plt.ylabel('Count')
    plt.title(pltTitle)
    plt.tight_layout()


def tripMergeAnalyse(dfs, dfNames):
    df = pd.DataFrame()
    for i in range(len(dfs)):
        out, outcome = outcomeInteger(dfs[i], dfs[i]['conf'])
        posT = outcome.index('Triple merger')

        tripM = out[posT]['conf'].to_numpy(dtype='str')
        objs = np.stack(np.core.defchararray.split(tripM, sep=':')).astype('int')

        r0 = out[posT]['r0']
        r10 = out[posT]['r10']
        r11 = out[posT]['r11']
        radii = np.array([r0, r10, r11]).T
        minVal = radii.argmax(axis=1)

        sbbMerge = 0
        bsbMerge = 0
        bbsMerge = 0
        unSorted = 0

        for j in range(len(minVal)):
            if np.where(objs[j] == minVal[j])[0][0] == 0:
                sbbMerge += 1
            elif np.where(objs[j] == minVal[j])[0][0] == 1:
                bsbMerge += 1
            elif np.where(objs[j] == minVal[j])[0][0] == 2:
                bbsMerge += 1
            else:
                unSorted += 1

        sbbMerge /= len(tripM)
        bsbMerge /= len(tripM)
        bbsMerge /= len(tripM)

        dfTemp = pd.DataFrame([sbbMerge, bsbMerge, bbsMerge])
        dfTemp.columns = [dfNames[i]]
        dfTemp.index = ['S-BH-BH', 'BH-S-BH', 'BH-BH-S']

        df = pd.concat([df, dfTemp], axis=1)

        if unSorted != 0:
            print(unSorted)

    df.plot(kind='bar')
    plt.xticks(rotation=25)
    plt.ylabel('Fraction')
    plt.margins(y=0.5)
    plt.tight_layout()


def initialConf(df):
    r0 = df['r0'].astype(float)
    r10 = df['r10'].astype(float)
    r11 = df['r11'].astype(float)

    minR = np.argmax([r0, r10, r11], axis=0)

    count = Counter(minR)
    dfCount = pd.DataFrame.from_dict(count, orient='index')

    numBHBH = dfCount.loc[0][0]
    numBHStar = dfCount.loc[1][0] + dfCount.loc[2][0]

    print('BH-BH: ' + str(numBHBH) + '\nBH-Star: ' + str(numBHStar))


def energyFilter(df):
    energyConsv = df['dEE0'].astype(float)

    return df[abs(energyConsv) < 0.1]

def initParamSpace(df,conf):
    data, outnames = isolateOutcome(df, conf)

    aSpace = [np.amin(data['a1'].astype(float)), np.amax(data['a1'].astype(float)), np.mean(data['a1'].astype(float)), np.median(data['a1'].astype(float))]
    eSpace = [np.amin(data['e1'].astype(float)), np.amax(data['e1'].astype(float)), np.mean(data['e1'].astype(float)), np.median(data['e1'].astype(float))]
    vSpace = [np.amin(data['vInf'].astype(float)), np.amax(data['vInf'].astype(float)), np.mean(data['vInf'].astype(float)), np.median(data['vInf'].astype(float))]
    bSpace = [np.amin(data['bImp'].astype(float)), np.amax(data['bImp'].astype(float)), np.mean(data['bImp'].astype(float)), np.median(data['bImp'].astype(float))]

    return pd.DataFrame(data=[aSpace, eSpace, vSpace, bSpace], index=['a', 'e', 'vInf', 'b'], columns=['min', 'max', 'mean', 'median'])

def printParams(df, param, name):
    # format_float_scientific(f, exp_digits=3)

    print(name + ': max=' + format_float_scientific(np.amax(abs(df[param].astype(float))), precision=5) + ', min=' +
          format_float_scientific(np.amin(abs(df[param].astype(float))), precision=5) +
          ', mean=' + format_float_scientific(np.mean(abs(df[param].astype(float))), precision=5) +
          ', median=' + format_float_scientific(np.median(abs(df[param].astype(float))), precision=5))


def rMinPeaks(df, yLowLim):
    param = 'rMin'
    rSunAU = 0.00465047
    rMin = df[param].astype('float') * df['l Unit'].astype('float') / rSunAU

    confs = df['conf'].to_numpy(dtype='str')
    rMin[np.core.defchararray.find(confs, ':') != -1] = np.nan

    hBins = np.arange(0,100,0.1)
    rMin = rMin.dropna()
    rMinHist = np.histogram(rMin, bins=hBins)

    # test = find_peaks(rMinHist[0], height=20)
    histDiff = np.diff(rMinHist[0])
    peaks = rMinHist[0][0:-1][histDiff < -15]
    peaksBin = rMinHist[1][1:-1][histDiff < -15]

    # rMinHistVals = rMinHist[0]
    # rMinHistBins = rMinHist[1]

    # plt.figure()
    # plt.step(rMinHistBins[:-1], rMinHistVals, zorder=1)
    # plt.scatter(peaksBin, peaks, c='red', zorder=2)
    # plt.xlabel('R$_{min}$ [R$_{\odot}$]')
    # plt.ylabel('Count')

    peaksDF = pd.DataFrame()
    for i in range(len(peaks)):
        interactions = df.iloc[np.argwhere((rMin > peaksBin[i] - 0.1) & (rMin < peaksBin[i])).flatten()]
        if len(interactions) != peaks[i]:
            print('false')
        peaksDF = peaksDF.append(interactions)


    # peaksAll = df
    return peaksDF

def paramComps(param1, param2, df, dfName, p1Lab, p2Lab):
    # out, outcomes = outcomeInteger(df, df['conf'])

    data = df[df[param2].astype(float) > df[param1].astype(float)]

    relV = data['vInf'].astype(float) * data['vCrit Unit'].astype(float)
    impParam = data['bImp'].astype(float) * data['a1'].astype(float)
    rMin = data['rMin'].astype(float) * data['l Unit'].astype(float) / 	0.0046524726

    lims = pd.DataFrame([[np.amin(relV), np.amax(relV), np.mean(relV), np.median(relV)],
                         [np.amin(impParam), np.amax(impParam), np.mean(impParam), np.median(impParam)],
                         [np.amin(rMin), np.amax(rMin), np.mean(rMin), np.median(rMin)]],
                        index=['relV', 'b', 'rMin'], columns=['min', 'max', 'mean', 'median'])

    # plt.figure()
    # relV.plot.hist(histtype='step', bins=np.linspace(0,100, 100))
    # plt.title(dfName)
    # plt.xlabel('v$_{\infty}$ [km/s]')
    # plt.ylabel('Count')
    # plt.tight_layout()

    # plt.figure()
    # impParam.plot.hist(histtype='step', bins=np.linspace(0,1000, 100))
    # plt.title(dfName)
    # plt.xlabel('b [AU]')
    # plt.ylabel('Count')
    # plt.tight_layout()

    # plt.figure()
    # rMin.plot.hist(histtype='step', bins=np.linspace(0,100, 100))
    # plt.title(dfName)
    # plt.xlabel('R$_{min}$')
    # plt.ylabel('Count')
    # plt.tight_layout()
    return lims

def paramCompsMult(param1, param2, dfs, dfName, p1Lab, p2Lab):
    # out, outcomes = outcomeInteger(df, df['conf'])
    relV = pd.DataFrame()
    impParam = pd.DataFrame()
    rMin = pd.DataFrame()

    integers = pd.DataFrame()
    for df in dfs:
        integersT = df[param2].astype(float) > df[param1].astype(float)
        data = df[df[param2].astype(float) > df[param1].astype(float)]

        relVT = data['vInf'].astype(float) * data['vCrit Unit'].astype(float)
        impParamT = data['bImp'].astype(float) * data['a1'].astype(float)
        rMinT = data['rMin'].astype(float) * data['l Unit'].astype(float) / 	0.0046524726

        relV = pd.concat([relV, relVT], axis=1)
        impParam = pd.concat([impParam, impParamT], axis=1)
        rMin = pd.concat([rMin, rMinT], axis=1)
        integers = pd.concat([integers, integersT], axis=1)


    relV.columns = dfName
    impParam.columns = dfName
    rMin.columns = dfName
    integers.columns = dfName

    relV.plot.hist(histtype='step', bins=np.linspace(0,100, 100))
    plt.xlabel('v$_{\infty}$ [km/s]')
    plt.ylabel('Count')
    plt.title('e$_{final}$ > e$_{initial}$')
    plt.tight_layout()

    impParam.plot.hist(histtype='step', bins=np.linspace(0,1000, 100))
    plt.xlabel('b [AU]')
    plt.ylabel('Count')
    plt.title('e$_{final}$ > e$_{initial}$')
    plt.tight_layout()

    rMin.plot.hist(histtype='step', bins=np.linspace(0,100, 100))
    plt.xlabel('R$_{min}$')
    plt.ylabel('Count')
    plt.title('e$_{final}$ > e$_{initial}$')
    plt.tight_layout()

    ar = dfs[0].loc[((integers['ARChain'] == False) & (integers['PN'] == True))]
    pn = dfs[1].loc[((integers['ARChain'] == False) & (integers['PN'] == True))]

    # arE = ar[['e1', 'e_f1']]
    # pnE = pn[['e1', 'e_f1']]

    dfsE = [ar, pn]
    relVE = pd.DataFrame()
    impParamE = pd.DataFrame()
    rMinE = pd.DataFrame()

    for df in dfsE:
        relVT = df['vInf'].astype(float) * df['vCrit Unit'].astype(float)
        impParamT = df['bImp'].astype(float) * df['a1'].astype(float)
        rMinT = df['rMin'].astype(float) * df['l Unit'].astype(float) / 	0.0046524726

        relVE = pd.concat([relVE, relVT], axis=1)
        impParamE = pd.concat([impParamE, impParamT], axis=1)
        rMinE = pd.concat([rMinE, rMinT], axis=1)

    relVE.columns = dfName
    impParamE.columns = dfName
    rMinE.columns = dfName

    # relVE.plot.hist(histtype='step', bins=np.linspace(0,100, 100))
    plt.figure()
    plt.hist(relVE['PN'], histtype='step', bins=np.linspace(0,100, 100), density=True, label='Change in PN')
    plt.hist(relV['PN'], histtype='step', bins=np.linspace(0,100, 100), density=True, label='All')
    plt.xlabel('v$_{\infty}$ [km/s]')
    plt.ylabel('Count')
    plt.title('$\Delta e > 0$ for PN')
    plt.legend()
    plt.tight_layout()

    # impParamE.plot.hist(histtype='step', bins=np.linspace(0,1000, 100))
    plt.figure()
    plt.hist(impParamE['PN'], histtype='step', bins=np.linspace(0,100, 100), density=True, label='Change in PN')
    plt.hist(impParam['PN'], histtype='step', bins=np.linspace(0,100, 100), density=True, label='All')
    plt.xlabel('b [AU]')
    plt.ylabel('Count')
    plt.title('$\Delta e > 0$ for PN')
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.hist(rMinE['PN'], histtype='step', bins=np.linspace(0,100, 100), density=True, label='Change in PN')
    plt.hist(rMin['PN'], histtype='step', bins=np.linspace(0,100, 100), density=True, label='All')
    plt.xlabel('R$_{min}$')
    plt.ylabel('Count')
    plt.title('$\Delta e > 0$ for PN')
    plt.legend()
    plt.tight_layout()

    # rMinE.plot.hist(histtype='step', bins=np.linspace(0,100, 100))
    # plt.xlabel('R$_{min}$')
    # plt.ylabel('Count')
    # plt.title('$\Delta e > 0$ for PN')
    # plt.tight_layout()

def timeMergeGW(df, conf, time):
    # data, outnames = isolateOutcome(df, conf)
    data = df

    if conf == ['Exchange']:
        m2 = np.zeros(len(data))
        confs = data['conf']
        ex1 = confs.str.find('[0 1]')
        ex2 = confs.str.find('[0 2]')

        ex1.replace(0, True, inplace=True)
        ex2.replace(0, True, inplace=True)

        ex1.replace(-1, False, inplace=True)
        ex2.replace(-1, False, inplace=True)

        m1 = data['m0'].astype(float)
        m2[ex1] = data['m10'].astype(float)[ex1]
        m2[ex2] = data['m11'].astype(float)[ex2]

    else:
        m1 = data['m10'].astype(float)
        m2 = data['m11'].astype(float)

    if time == 'Before':
        a = data['a1'].astype(float)
        e = data['e1'].astype(float)
    elif time == 'After':
        a = data['a_f1'].astype(float)
        e = data['e_f1'].astype(float)
    else:
        print('Invalid time')
        return


    c = 63198       # AU/yr
    G = 39.478      # AU3 * yr-2 * Msun-1

    t = 5/256 * c**5 * a**4 * (1-e**2)**(7/2)/(G**3 * m1 * m2 * (m1 + m2))
    return t/(10**6)


def bbhFilter(df):
    r0 = df['r0'].astype(float)
    r10 = df['r10'].astype(float)
    r11 = df['r11'].astype(float)

    rad = pd.concat([r0, r10, r11], axis=1)

    maxim = rad.idxmax(axis=1)

    bbhs = df[maxim == 'r0']

    return bbhs

def BHStarFilter(df):
    r0 = df['r0'].astype(float)
    r10 = df['r10'].astype(float)
    r11 = df['r11'].astype(float)

    rad = pd.concat([r0, r10, r11], axis=1)

    maxim = rad.idxmax(axis=1)

    BHStar = df[maxim != 'r0']

    return BHStar



def findBHMergers(df):
    mergers = df[df['conf'].str.contains(':')]
    r0 = mergers['r0'].astype(float)
    r10 = mergers['r10'].astype(float)
    r11 = mergers['r11'].astype(float)
    radii = pd.concat([r0, r10, r11], axis=1)
    radii.columns = [0, 1, 2]

    indStar = radii.idxmax(axis=1)
    confs = mergers['conf'].str.split(':', expand=True)
    merger1 = confs[0].str.strip().str[-1]
    merger2 = confs[1].str.strip().str[0]

    mergerObj = pd.concat([merger1, merger2], axis=1).astype(float)

    starInMerger = ~mergerObj.isin(indStar)

    bhbh = mergers[starInMerger.all(axis=1)]


    return bhbh

def whichObjectsMergedFirst(df):
    radii = df[['r0', 'r10', 'r11']].astype(float)
    radii.columns = ['0', '1', '2']
    indMax = radii.idxmax(axis=1)
    confs = df['conf'].str.split(':', expand=True)

    whereBHBH = confs[2] == indMax

    print('t')



def initialParamPlots(dfs, param, dfLabels, xlabel, xMin, xMax, xStep):
    plt.figure()
    bins = np.arange(xMin, xMax, xStep)
    param = 'e1'
    # bins = np.logspace(-1,np.log10(1000), 30)
    color = ['blue', 'orange', 'green']
    for i in range(len(dfs)):
        if param == 'vInf':
            data = dfs[i][param].astype(float) * dfs[i]['vCrit Unit'].astype(float)
        elif param == 'bImp':
            data = dfs[i][param].astype(float) * dfs[i]['a1'].astype(float)

        elif param == 'm0':
            m0 = dfs[i]['m0'].astype(float)
            m1 = dfs[i]['m10'].astype(float)
            m2 = dfs[i]['m11'].astype(float)

            data = pd.concat((m0, m1, m2))
        elif param == 'a_f1':
            data = dfs[i][param].astype(float)[dfs[i][param].astype(float) != 0]
            dataInit = dfs[i]['a1'].astype(float)[dfs[i]['a1'].astype(float) != 0]

        elif param == 'e_f1':
            data = dfs[i][param].astype(float)
            # dataInit = dfs[i]['e1'].astype(float)

        elif param == 'rMinSun':
            dataAll = dfs[i][param].str.split(' ', expand=True)
            data = dataAll[0].astype(float)

            # if i == 0:
            #     # schRad = [0.00004246, 0.00008492, 0.00012738]
            #     # labs = ['10 M$_{\odot}$', '20 M$_{\odot}$', '30 M$_{\odot}$']

            #     schRad = [0.00004246]
            #     labs = ['Schwarzschild radius']
            #     for j in range(len(schRad)):
            #         plt.axvline(schRad[j], label=labs[j], c='black')
        else:
            data = dfs[i][param].astype(float)

        weights = np.ones_like(data)/len(data)
        # weightsInit = np.ones_like(dataInit)/len(dataInit)

        data.hist(bins=bins, label=dfLabels[i], histtype='step', linewidth=5, weights=weights, color=color[i])
        # dataInit.hist(bins=bins, label=dfLabels[i] + ' (initial)', histtype='step', linewidth=5, linestyle='--', weights=weightsInit, color=color[i])
        print(np.amin(data))
        print(dfLabels[i])

    plt.ylabel('Fraction', fontsize=40)
    plt.xlabel('e$_{initial}$', fontsize=40)
    plt.legend(loc='upper left', fontsize=40)
    plt.grid(b=None)
    plt.title('Exchange')
    # plt.xlim(xMin, xMax)
    # plt.xscale('log')


def bPrime(dfs, names, bins):
    plt.figure()
    i = 0
    for df in dfs:
        # bP = df['bImp'].astype(float) / df['a1'].astype(float) * (df['vInf'].astype(float) / df['vCrit Unit'].astype(float))
        bP = df['bImp'].astype(float) * (df['vInf'].astype(float))

        bP.hist(bins=bins, label=names[i], histtype='step')
        i += 1

    plt.ylabel('Count', fontsize=40)
    plt.xlabel('b$^{\prime}$', fontsize=40)
    plt.legend(loc='upper left', fontsize=40)
    plt.grid(b=None)
    plt.xscale('log')


def crossSecConfSemiMajorMOCCA(dfs, dfNames, conf):
    # plt.figure()

    numInBin = np.array([])
    for i in range(len(dfs)):
        plt.figure()
        dfTemp = dfs[i].copy()
        vInf = dfs[i]['vInf'].astype(float)/dfs[i]['vCrit Unit'].astype(float)
        vInfUn = vInf.unique()

        c = 63239.7263  # AU/yr
        G = 39.478      # AU3 * yr-2 * Msun-1

        dataOrig, outcomesOrig = outcomeInteger(dfTemp, dfTemp['conf'])
        # inspAnalytial = np.unique(dfs[i]['a1'].astype(float))**(2/7) * 20**(12/7) / 5**2
        t = (5/256 * c**5 * dfs[i]['a_f1']**4 * (1-dfs[i]['e_f1']**2)**(7/2)/(G**3 * 10 * 10 * (10 + 10)))
        periDist = dfs[i]['a_f1'] * (1 - dfs[i]['e_f1'])
        outerPeriod = 2 * np.pi * np.sqrt(dfs[i]['a_f1_O']**3 / ( G * 10))
        dfTemp['conf'][(periDist < 2 * 0.00000019746)] = 'Merger'
        # dfTemp['conf'][(periDist < 2 * 0.00000019746) & (dfTemp['conf'] != '[[0 1] 2]') & (dfTemp['conf'] != '[[0 2] 1]') &
        #                 (dfTemp['conf'] != '[[1 2] 0]')] = 'Merger'
        # dfTemp['conf'][(t <= 1) & (dfTemp['conf'] != '[[0 1] 2]') & (dfTemp['conf'] != '[[0 2] 1]') &
        #                 (dfTemp['conf'] != '[[1 2] 0]') & (dfTemp['conf'] != 'Merger')] = ['Inspiral']

        dfTemp['conf'][(periDist < 2 * 0.00000019746) & ((dfTemp['conf'] == '[[0 2] 1]') | (dfTemp['conf'] == '[[0 1] 2]') | (dfTemp['conf'] == '[[1 2] 0]'))] = 'Inspiral'

        dfTemp['conf'][(t < outerPeriod) & (outerPeriod > 0)  & (dfTemp['conf'] != 'Merger')] = ['Inspiral']
        dfTemp['conf'][(t < np.amax(outerPeriod)) & (dfTemp['e_f1_O'] > 1)  & (dfTemp['conf'] != 'Merger')] = ['Inspiral']
        dfTemp['conf'][(t < 14e9) & (dfTemp['conf'] != 'Inspiral') & (dfTemp['conf'] != '[[0 1] 2]') & (dfTemp['conf'] != '[[0 2] 1]') &
                        (dfTemp['conf'] != '[[1 2] 0]') & (dfTemp['conf'] != 'Merger')] = ['Post-interaction merger']

        data, outcomes = outcomeInteger(dfTemp, dfTemp['conf'])

        sigmaArray = np.array([])
        aArray = np.array([])



        eDF = pd.DataFrame(columns=outcomes)
        aBins = np.logspace(-4, 3, 8)
        aLowerBins = np.logspace(-5, 2, 8)
        aLowerBins[0] = 0

        vBins = np.logspace(-8, 2, 9)
        vLowerBins = np.logspace(-15, 1, 9)
        vLowerBins[0] = 0

        numInBinTemp = np.ones(len(aBins))


        for j in range(len(aBins)):
            maskA = ((dfTemp['a1'].astype(float) < aBins[j]) & (dfTemp['a1'].astype(float) > aLowerBins[j]))
            maskV = ((vInf < vBins[j]) & (vInf > vLowerBins[j]))
            dfV = dfTemp[maskA]
            dfT, names = outcomeInteger(dfV, dfV['conf'])

            numInBinTemp[j] = len(dfV)

            if len(dfV) == 0:
                continue

            aIndex = np.mean(dfV['a1'].astype(float))
            for j in range(len(dfT)):
                if len(dfV) == 0:
                    sigmaArray = np.append(sigmaArray, 0)
                    continue

                bImp = dfT[j]['bImp'].astype(float)*dfT[j]['a1'].astype(float)
                maxB = np.amax(bImp)

                sigma = np.pi * maxB**2 * len(bImp)/len(dfV)
                sigmaArray = np.append(sigmaArray, sigma)
                aArray = np.append(aArray, float(aIndex))

            seriesTemp = pd.Series(sigmaArray, name=aIndex, index=outcomes)
            eDF = eDF.append(seriesTemp)
            sigmaArray = np.array([])

        sumSigma = eDF.sum(axis=1)
        # sumSigmaMinExch = sumSigma - eDF['Exchange']
        # exchOverSum = eDF['Exchange'] / sumSigmaMinExch
        eDF = eDF[['Flyby', 'Exchange', 'Ionization', 'Bound merger', 'Unbound merger',
       'Triple merger', 'Mergers', 'Post-interaction merger', 'Inspiral', 'Bound triple']]
        for k in eDF.columns:
            if np.sum(eDF[k]) > 0:
                if k == 'Flyby' or k == 'Exchange' or k == 'Bound triple':
                    continue
                sumTemp = sumSigma - eDF[k]
                plt.plot(eDF.index.to_numpy(dtype=float), eDF[k], '-o', label=k)
                # plt.scatter(eDF.index.to_numpy(dtype=float), eDF[k]/sumTemp)


        # plt.plot(exchOverSum.index.to_numpy(dtype=float), exchOverSum, label=dfNames[i])
        # plt.plot(eDF.index.to_numpy(dtype=float), eDF['Inspiral'], label='Inspiral')
        # plt.plot(eDF.index.to_numpy(dtype=float), eDF['Mergers'], label='Merger')

        inspAnalytial = eDF.index.to_numpy(dtype=float)**(2/7) * 20**(12/7) / 5**2
        # print(numInBin)

        plt.plot(eDF.index.to_numpy(dtype=float), inspAnalytial, '--o', label='Analytical inspiral')
        plt.title(dfNames[i])
        plt.legend(fontsize=22)
        plt.xlabel('a$_{0}$ [AU]')
        # plt.ylabel('$\sigma / \sigma_{others}$')
        plt.ylabel('$\sigma$')
        plt.xscale('log')
        plt.yscale('log')

        numInBin = np.append(numInBin, numInBinTemp[numInBinTemp != 0])

    plt.figure()
    plt.plot(eDF.index.to_numpy(dtype=float), numInBin[0:5], '--o', label='ARChain')
    plt.plot(eDF.index.to_numpy(dtype=float), numInBin[5:10], '--o', label='PN')
    plt.plot(eDF.index.to_numpy(dtype=float), numInBin[10:15], '--o', label='PN + tides')
    # plt.title(dfNames[i])
    plt.legend()
    plt.xlabel('a$_0$ [AU]')
    plt.xscale('log')
    plt.ylabel('# interactions')

# def crossSecConfSemiMajor(dfs, dfNames, conf):
#     # plt.figure()

#     for i in range(len(dfs)):
#         plt.figure()
#         dfTemp = dfs[i].copy()
#         vInf = dfs[i]['vInf'].astype(float)/dfs[i]['vCrit Unit'].astype(float)
#         vInfUn = vInf.unique()

#         c = 63239.7263  # AU/yr
#         G = 39.478      # AU3 * yr-2 * Msun-1

#         dataOrig, outcomesOrig = outcomeInteger(dfTemp, dfTemp['conf'])
#         inspAnalytial = np.unique(dfs[i]['a1'].astype(float))**(2/7) * 20**(12/7) / 5**2
#         t = (5/256 * c**5 * dfs[i]['a_f1']**4 * (1-dfs[i]['e_f1']**2)**(7/2)/(G**3 * 10 * 10 * (10 + 10)))
#         periDist = dfs[i]['a_f1'] * (1 - dfs[i]['e_f1'])
#         outerPeriod = 2 * np.pi * np.sqrt(dfs[i]['a_f1_O']**3 / ( G * 10))
#         # dfTemp['conf'][(periDist < 2 * 0.00000019746)] = 'Merger'
#         dfTemp['conf'][(periDist < 2 * 0.00000019746) & (dfTemp['conf'] != '[[0 1] 2]') & (dfTemp['conf'] != '[[0 2] 1]') &
#                         (dfTemp['conf'] != '[[1 2] 0]')] = 'Merger'
#         # dfTemp['conf'][(t <= 1) & (dfTemp['conf'] != '[[0 1] 2]') & (dfTemp['conf'] != '[[0 2] 1]') &
#         #                 (dfTemp['conf'] != '[[1 2] 0]') & (dfTemp['conf'] != 'Merger')] = ['Inspiral']

#         dfTemp['conf'][(periDist < 2 * 0.00000019746) & ((dfTemp['conf'] == '[[0 2] 1]') | (dfTemp['conf'] == '[[0 1] 2]') | (dfTemp['conf'] == '[[1 2] 0]'))] = 'Inspiral'

#         dfTemp['conf'][(t < outerPeriod) & (outerPeriod > 0)  & (dfTemp['conf'] != 'Merger')] = ['Inspiral']
#         dfTemp['conf'][(t < np.amax(outerPeriod)) & (dfTemp['e_f1_O'] > 1)  & (dfTemp['conf'] != 'Merger')] = ['Inspiral']
#         dfTemp['conf'][(t < 14e9) & (dfTemp['conf'] != 'Inspiral') & (dfTemp['conf'] != '[[0 1] 2]') & (dfTemp['conf'] != '[[0 2] 1]') &
#                         (dfTemp['conf'] != '[[1 2] 0]') & (dfTemp['conf'] != 'Merger')] = ['Post-interaction merger']

#         data, outcomes = outcomeInteger(dfTemp, dfTemp['conf'])

#         sigmaArray = np.array([])
#         aArray = np.array([])

#         eDF = pd.DataFrame(columns=outcomes)

#         for j in range(len(vInfUn)):
#             dfV = dfTemp[vInf == vInfUn[j]]
#             dfT, names = outcomeInteger(dfV, dfV['conf'])

#             aIndex = dfV['a1'].iloc[0]
#             for j in range(len(dfT)):
#                 if len(dfV) == 0:
#                     sigmaArray = np.append(sigmaArray, 0)
#                     continue

#                 bImp = dfT[j]['bImp'].astype(float)*dfT[j]['a1'].astype(float)
#                 maxB = np.amax(bImp)

#                 sigma = np.pi * maxB**2 * len(bImp)/len(dfV)
#                 sigmaArray = np.append(sigmaArray, sigma)
#                 aArray = np.append(aArray, float(aIndex))

#             seriesTemp = pd.Series(sigmaArray, name=aIndex, index=outcomes)
#             eDF = eDF.append(seriesTemp)
#             sigmaArray = np.array([])

#         sumSigma = eDF.sum(axis=1)
#         # sumSigmaMinExch = sumSigma - eDF['Exchange']
#         # exchOverSum = eDF['Exchange'] / sumSigmaMinExch
#         eDF = eDF[['Flyby', 'Exchange', 'Ionization', 'Bound merger', 'Unbound merger',
#        'Triple merger', 'Mergers', 'Post-interaction merger', 'Inspiral', 'Bound triple']]
#         for k in eDF.columns:
#             if np.sum(eDF[k]) > 0:
#                 if k == 'Flyby' or k == 'Exchange' or k == 'Bound triple':
#                     continue
#                 sumTemp = sumSigma - eDF[k]
#                 plt.plot(eDF.index.to_numpy(dtype=float), eDF[k], '-o', label=k)
#                 # plt.scatter(eDF.index.to_numpy(dtype=float), eDF[k]/sumTemp)


#         # plt.plot(exchOverSum.index.to_numpy(dtype=float), exchOverSum, label=dfNames[i])
#         # plt.plot(eDF.index.to_numpy(dtype=float), eDF['Inspiral'], label='Inspiral')
#         # plt.plot(eDF.index.to_numpy(dtype=float), eDF['Mergers'], label='Merger')

#         plt.plot(eDF.index.to_numpy(dtype=float), inspAnalytial, '--o', label='Analytical inspiral')
#         plt.title(dfNames[i])
#         plt.legend(fontsize=22)
#         plt.xlabel('a$_{0}$ [AU]')
#         # plt.ylabel('$\sigma / \sigma_{others}$')
#         plt.ylabel('$\sigma$')
#         plt.xscale('log')
#         plt.yscale('log')

def verySmallRmin(dfs):
    dataAll = dfs[0]['rMinSun'].str.split(' ', expand=True)
    data = dataAll[0].astype(float)

    lowR_AR = dfs[0][(data < 0.00004246) & (data != 0)]
    lowR_PN = dfs[1][(data < 0.00004246) & (data != 0)]

    confs_AR = lowR_AR['conf']
    confs_PN = lowR_PN['conf']

    confs_Array = [confs_AR, confs_PN]
    colNames = ['ARChain', 'PN']
    sortInd = ['Flyby', 'Exchange', 'Ionization', 'Bound triple']

    df = pd.DataFrame()

    for i in range(len(confs_Array)):
        data, outcomes = outcomeInteger(confs_Array[i], confs_Array[i])

        numI = np.array([])
        confs = np.array(outcomes)

        for j in range(len(data)):
            numI = np.append(numI, len(data[j].index))

        dfSort = pd.DataFrame(numI, columns=[colNames[i]])
        dfSort = dfSort.set_index(confs)

        df = pd.concat([df, dfSort], axis=1, sort=False)

    df.loc[sortInd].plot(kind='bar',  legend=True, rot=0, title='R$_{min}$ < R$_{obj}$')
    plt.ylabel('Count')
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

def paramConf(dfs, dfNames, param, xlab):
    binLog = np.logspace(-5, 2, 100)

    bTriples = 0

    for i in range(len(dfs)):
        plt.figure()
        data, outcomes = outcomeInteger(dfs[i], dfs[i]['conf'])
        for j in range(len(data)):
            if len(data[j]) > 0:
                if param == 'rMinSun':
                    dC = data[j]['rMinSun'].str.split(' ', expand=True)[0].astype(float)
                else:
                    dC = data[j][param].astype(float)

                weigths = np.ones(len(dC)) / len(dC)
                dC.hist(histtype='step', bins=binLog, linewidth=2, label=outcomes[j], weights=weigths)
                if outcomes[j] == 'Bound triple':
                    bTriples = data[j][dC < 5e-1]

        plt.ylabel('Frac')
        plt.xlabel(xlab)
        plt.xscale('log')
        plt.legend(loc='upper left')
        plt.grid(None)
        plt.title(dfNames[i])

    return bTriples


def starMassDistribution(dfs, dfNames):
    plt.figure()
    bins = np.logspace(-3,2, 100)
    for i in range(len(dfs)):
        masses = dfs[i][['r0', 'r10', 'r11']].astype(float)
        starMass = np.amax(masses, axis=1)
        weights = np.ones_like(starMass) / len(starMass)
        plt.hist(starMass, bins=bins, label=dfNames[i])#, weights=weights, histtype='step', linewidth=5)
        plt.xscale('log')
        plt.title('All interactions')
        plt.ylabel('Count')
        plt.xlabel('R$_{\star}$ [R$_{\odot}$]')
        # plt.legend()


""" Plot parameters """
plt.rcParams['xtick.minor.visible'], plt.rcParams['xtick.top'] = True,True
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.right'] = True,True
plt.rcParams['xtick.direction'], plt.rcParams['ytick.direction'] = 'in','in'
plt.rcParams.update({'font.size': 30})
sns.set_style("ticks")

""" constants """
GAU = 39.478        # AU3 /yr2 / Msun
G = 1.90809e5       # Rsun/Msun * (km/s)^2
c = 299792.458      # km/s

""" Read data """
try:
    K
except NameError:
#    crossSecIn = 0
    """ old sets """
    # ks0 = pd.read_pickle('./ssh_code/dataframes/2BH/noSeed/df_1BH_ks0_K0_L0_l0')
    # ks1 = pd.read_pickle('./ssh_code/dataframes/2BH/noSeed/df_1BH_ks1_K0_L0_l0')
    # K = pd.read_pickle('./ssh_code/dataframes/2BH/noSeed/df_1BH_ks0_K1_L0_l0')
    # KL = pd.read_pickle('./ssh_code/dataframes/2BH/noSeed/df_1BH_ks0_K1_L1_l0')

    """ cross section """
    # ks0 = pd.read_pickle('./ssh_code/dataframes/cross/sec/df_cross1_ks0_K0_L0_l0')
    # ks1 = pd.read_pickle('./ssh_code/dataframes/cross/sec/df_cross2_ks1_K0_L0_l0')
    # K = pd.read_pickle('./ssh_code/dataframes/cross/sec/df_cross3_ks0_K1_L0_l0')
    # KLT =  pd.read_pickle('./ssh_code/dataframes/cross/sec/df_cross1_ks0_K1_L1_l1')

    # ks0 = pd.read_pickle('./ssh_code/dataframes/PM/df_cross1_ks0_K0_L0_l0')
    # ks1 = pd.read_pickle('./ssh_code/dataframes/PM/df_cross2_ks1_K0_L0_l0')
    # K = pd.read_pickle('./ssh_code/dataframes/PM/df_cross3_ks0_K1_L0_l0')
    # crossSecIn = 1


    """ 2a """
    # ks0 = pd.read_pickle('./ssh_code/dataframes/cross/sec/df_cross1_ks0_K0_L0_l0')
    # ks1 = pd.read_pickle('./ssh_code/dataframes/cross/sec/df_cross2_ks1_K0_L0_l0')
    # K = pd.read_pickle('./ssh_code/dataframes/cross/sec/df_cross3_ks0_K1_L0_l0')

    # # ks0PM = pd.read_pickle('./ssh_code/dataframes/PM/df_cross1_ks0_K0_L0_l0')
    # # ks1PM = pd.read_pickle('./ssh_code/dataframes/PM/df_cross2_ks1_K0_L0_l0')
    # # KPM = pd.read_pickle('./ssh_code/dataframes/PM/df_cross3_ks0_K1_L0_l0')

    # ks0H = pd.read_pickle('./ssh_code/dataframes/cross/sec/highAcc/df_acc1_ks0_K0_L0_l0')
    # ks1H = pd.read_pickle('./ssh_code/dataframes/cross/sec/highAcc/df_acc2_ks1_K0_L0_l0')
    # KH = pd.read_pickle('./ssh_code/dataframes/cross/sec/highAcc/df_acc3_ks0_K1_L0_l0')

    # ks0L = pd.read_pickle('./ssh_code/dataframes/cross/sec/lowAcc/df_acc1_ks0_K0_L0_l0')
    # ks1L = pd.read_pickle('./ssh_code/dataframes/cross/sec/lowAcc/df_acc2_ks1_K0_L0_l0')
    # # # KL = pd.read_pickle('./ssh_code/dataframes/cross/sec/lowAcc/df_acc3_ks0_K1_L0_l0')


    """ 2b """
    # ks0b = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/df_cross1_ks0_K0_L0_l0')
    # ks1b = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/df_cross2_ks1_K0_L0_l0')
    # Kb = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/df_cross3_ks0_K1_L0_l0')
    # # # KLT = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/df_cross1_ks0_K1_L1_l1')

    # ks0H = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/highAcc/df_acc1_ks0_K0_L0_l0')
    # ks1H = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/highAcc/df_acc2_ks1_K0_L0_l0')
    # # # KH = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/highAcc/df_acc3_ks0_K1_L0_l0')

    # ks0L = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/lowAcc/df_acc1_ks0_K0_L0_l0')
    # ks1L = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/lowAcc/df_acc2_ks1_K0_L0_l0')
    # KL = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/lowAcc/df_acc3_ks0_K1_L0_l0')

    """ PM """
    # ks0 = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/PM/df_cross1_ks0_K0_L0_l0')
    # ks1 = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/PM/df_cross2_ks1_K0_L0_l0')
    # K = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/PM/df_cross3_ks0_K1_L0_l0')

    # ks0H = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/PM/highAcc/df_acc1_ks0_K0_L0_l0')
    # ks1H = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/PM/highAcc/df_acc2_ks1_K0_L0_l0')
    # # # KH = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/PM/highAcc/df_acc3_ks0_K1_L0_l0')

    # ks0L = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/PM/lowAcc/df_acc1_ks0_K0_L0_l0')
    # ks1L = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/PM/lowAcc/df_acc2_ks1_K0_L0_l0')
    # KL = pd.read_pickle('./ssh_code/dataframes/cross/smaller-vFix/PM/lowAcc/df_acc3_ks0_K1_L0_l0')

    """ Fregeau cross-sections """
    # ks0 = pd.read_pickle('./ssh_code/dataframes/cross/freg/df_cross1_ks0_K0_L0_l0')
    # ks1 = pd.read_pickle('./ssh_code/dataframes/cross/freg/df_cross2_ks1_K0_L0_l0')
    # K = pd.read_pickle('./ssh_code/dataframes/cross/freg/df_cross3_ks0_K1_L0_l0')

    """1 BH """
    # ks0 = pd.read_pickle('./ssh_code/dataframes/1BH/df_1BH_ks0_K0_L0_l0')
    # ks1 = pd.read_pickle('./ssh_code/dataframes/1BH/df_1BH_ks1_K0_L0_l0')
    # KO = pd.read_pickle('./../masters/code/ssh_code/dataframes/1BH/df_1BH_ks0_K1_L0_l0')
    # KLO = pd.read_pickle('./../masters/code/ssh_code/dataframes/1BH/df_1BH_ks0_K1_L1_l0')
    # KLlO = pd.read_pickle('./../masters/code/ssh_code/dataframes/1BH/df_1BH_ks0_K1_L1_l1')
    # KT = pd.read_pickle('./ssh_code/dataframes/1BH/df_1BH_ks0_K1_L0_l1')

    # crossSecIn = 0

    """2 BH """
    # ks0 = pd.read_pickle('./ssh_code/dataframes/2BH/df_2BH_ks0_K0_L0_l0')
    # ks1 = pd.read_pickle('./ssh_code/dataframes/2BH/df_2BH_ks1_K0_L0_l0')
    # KO = pd.read_pickle('./dataframes/2BH/oldTsunami/df_2BH_ks0_K1_L0_l0')
    # KLO = pd.read_pickle('./dataframes/2BH/oldTsunami/df_2BH_ks0_K1_L1_l0')
    # KlO = pd.read_pickle('./dataframes/2BH/oldTsunami/df_2BH_ks0_K1_L0_l1')
    # KLlO = pd.read_pickle('./dataframes/2BH/oldTsunami/df_2BH_ks0_K1_L1_l1')

    # ks0NS = pd.read_pickle('./ssh_code/dataframes/2BH/noSeed/df_2BH_ks0_K0_L0_l0')
    # ks1NS = pd.read_pickle('./ssh_code/dataframes/2BH/noSeed/df_2BH_ks1_K0_L0_l0')
    # KNS = pd.read_pickle('./ssh_code/dataframes/2BH/noSeed/df_2BH_ks0_K1_L0_l0')
    # KLNS = pd.read_pickle('./ssh_code/dataframes/2BH/noSeed/df_2BH_ks0_K1_L1_l0')
    # KLTNS = pd.read_pickle('./ssh_code/dataframes/2BH/noSeed/df_2BH_ks0_K1_L1_l1')
    crossSecIn = 0


    """3 BH"""
    # ks0 = pd.read_pickle('./ssh_code/dataframes/3BH/df_3BH_ks0_K0_L0_l0')
    # ks1 = pd.read_pickle('./ssh_code/dataframes/3BH/df_3BH_ks1_K0_L0_l0')
    # KO = pd.read_pickle('./ssh_code/dataframes/3BH/df_3BH_ks0_K1_L0_l0')
    # KLO = pd.read_pickle('./ssh_code/dataframes/3BH/df_3BH_ks0_K1_L1_l0')

    # ks0NS = pd.read_pickle('./ssh_code/dataframes/3BH/noSeed/df_3BH_ks0_K0_L0_l0')
    # ks1NS = pd.read_pickle('./ssh_code/dataframes/3BH/noSeed/df_3BH_ks1_K0_L0_l0')
    # KNS = pd.read_pickle('./ssh_code/dataframes/3BH/noSeed/df_3BH_ks0_K1_L0_l0')
    # KLNS = pd.read_pickle('./ssh_code/dataframes/3BH/noSeed/df_3BH_ks0_K1_L1_l0')
    # crossSecIn = 0

    # BH1 = pd.read_pickle('./ssh_code/dataframes/1BH/df_1BH_ks0_K0_L0_l0')
    # BH2 = pd.read_pickle('./ssh_code/dataframes/2BH/df_2BH_ks0_K0_L0_l0')
    # BH3 = pd.read_pickle('./ssh_code/dataframes/3BH/df_3BH_ks0_K0_L0_l0')

    """ new tsunami mocca 2BH"""
    # KN = pd.read_pickle('./ssh_code/dataframes/new_tsunami/2BH/df5_2_ks0_K1_L0_l0')
    # KLN = pd.read_pickle('./ssh_code/dataframes/new_tsunami/2BH/df6_2_ks0_K1_L1_l0')
    # KTN = pd.read_pickle('./ssh_code/dataframes/new_tsunami/2BH/df2_2_ks0_K1_L0_l1')
    # KLTN = pd.read_pickle('./ssh_code/dataframes/new_tsunami/2BH/df1_2_ks0_K1_L1_l1')

    """ handmade BH sets"""
    # K = pd.read_pickle('./ssh_code/dataframes/handmade_BH/df_BH6_ks0_K1_L0_l0')
    # KL = pd.read_pickle('./ssh_code/dataframes/handmade_BH/df_BH5_ks0_K1_L1_l0')
    # KLT = pd.read_pickle('./ssh_code/dataframes/handmade_BH/df_BH7_ks0_K1_L1_l1')

    """ select 3BH interactions (a < 10 AU, e > 0.95) """
    # K = pd.read_pickle('./ssh_code/dataframes/lowAhighEBH/df_BH7_ks0_K1_L0_l0')
    # KL = pd.read_pickle('./ssh_code/dataframes/lowAhighEBH/df_BH6_ks0_K1_L1_l0')

    """ select 3BH interactions (a < 10 AU, e < 0.95) """
    # K = pd.read_pickle('./ssh_code/dataframes/lowAlowEBH/df_BH1_ks0_K1_L0_l0')
    # KL = pd.read_pickle('./ssh_code/dataframes/lowAlowEBH/df_BH2_ks0_K1_L1_l0')

    """ article diff """
    # K = pd.read_pickle('./data/df_diff1__ks0_K1_L0_l0')
    # KLDiff = pd.read_pickle('./data/df_diff2__ks0_K1_L1_l0')

    """ article merge """
    # K = pd.read_pickle('./dataframes/mergers/df_merge3__ks0_K1_L0_l0')
    # KLDiff = pd.read_pickle('./dataframes/mergers/df_merge4__ks0_K1_L1_l0')

    # K = pd.read_pickle('./dataframes/mergers/df_merge4_ks0_K1_L0_l0')
    # KLl = pd.read_pickle('./dataframes/mergers/df_merge5_ks0_K1_L1_l1')


    """ 1 BH New tsunami"""
    # ks0 = pd.read_pickle('./dataframes/1BH/df_1BH_ks0_K0_L0_l0')      # old
    # K = pd.read_pickle('./dataframes/1BH/df1_1_ks0_K1_L0_l0')
    # KL = pd.read_pickle('./dataframes/1BH/df2_1_ks0_K1_L1_l0')
    # KLl = pd.read_pickle('./dataframes/1BH/df3_1_ks0_K1_L1_l1')

    """ 2 BH New tsunami"""
    # K = pd.read_pickle('./dataframes/2BH/df5_2_ks0_K1_L0_l0')
    # KL = pd.read_pickle('./dataframes/2BH/df6_2_ks0_K1_L1_l0')
    # KLl = pd.read_pickle('./dataframes/2BH/df1_2_ks0_K1_L1_l1')
    # Kl = pd.read_pickle('./dataframes/2BH/df2_2_ks0_K1_L0_l1')

    """ 3 BH New tsunami """
    # K = pd.read_pickle('./dataframes/3BH/df1_3_ks0_K1_L0_l0')
    # KL = pd.read_pickle('./dataframes/3BH/df2_3_ks0_K1_L1_l0')

    # old sets
    # ks0 = pd.read_pickle('./dataframes/3BH/oldTsunami/df_3BH_ks0_K0_L0_l0')
    # KO = pd.read_pickle('./dataframes/3BH/oldTsunami/df_3BH_ks0_K1_L0_l0')
    # KLO = pd.read_pickle('./dataframes/3BH/oldTsunami/df_3BH_ks0_K1_L1_l0')

    # low A
    # noReg = pd.read_pickle('./dataframes/3BH/lowA_new/df1_3_ks0_K0_L0_l0')
    # K = pd.read_pickle('./dataframes/3BH/lowA_new/df2_3_ks0_K1_L0_l0')
    # KL = pd.read_pickle('./dataframes/3BH/lowA_new/df3_3_ks0_K1_L1_l0')

    # bound triples in old version
    # K = pd.read_pickle('./dataframes/3BH/boundTriples/df_ks0_K1_L1_l0_7')   # 1e-7
    # KL = pd.read_pickle('./dataframes/3BH/boundTriples/df2_ks0_K1_L1_l0')  # 1e-9
    # KLl = pd.read_pickle('./dataframes/3BH/boundTriples/df3_ks0_K1_L1_l0') # 1e-11

    """ manually setup 3 BH sets """
    # K = pd.read_pickle('./dataframes/manualSetup/singleSeed/dataframe_manInp_3BH_ks0_K1_L0_l0')
    # KL = pd.read_pickle('./dataframes/manualSetup/singleSeed/dataframe_manInp_3BH_ks0_K1_L1_l0')

    # K = pd.read_pickle('./dataframes/manualSetup/fiveSeeds/dataframe_manImp_3BH_ks0_K1_L0_l0')
    # KL = pd.read_pickle('./dataframes/manualSetup/fiveSeeds/dataframe_manImp_3BH_ks0_K1_L1_l0')

    # KN = pd.read_pickle('./dataframes/manualSetup/fiveSeeds/newestTsunami/dataframe_lowV_newTsunami_ks0_K1_L0_l0')
    # KLN = pd.read_pickle('./dataframes/manualSetup/fiveSeeds/newestTsunami/dataframe_lowV_newTsunami_ks0_K1_L1_l0')


    # K = pd.read_pickle('./dataframes/manualSetup/fiveSeeds/lowerA/dataframe_manInp_lowV_ks0_K1_L0_l0')      # lower a
    # KL = pd.read_pickle('./dataframes/manualSetup/fiveSeeds/lowerA/dataframe_manInp_lowV_ks0_K1_L1_l0')     # lower a

    # K = pd.read_pickle('./dataframes/manualSetup/fiveSeeds/higherR/previousVersion/dataframe_manImp_lowV_ks0_K1_L0_l0')      # higher R
    # KL = pd.read_pickle('./dataframes/manualSetup/fiveSeeds/higherR/previousVersion/dataframe_manImp_lowV_ks0_K1_L1_l0')     # higher R
    # KN = pd.read_pickle('./dataframes/manualSetup/fiveSeeds/higherR/dataframe_manImp_lowV_ks0_K1_L0_l0')      # higher R
    # KLN = pd.read_pickle('./dataframes/manualSetup/fiveSeeds/higherR/dataframe_manImp_lowV_ks0_K1_L1_l0')     # higher R

    """ select 3BH interactions (a < 10 AU, e > 0.95) """
    # K = pd.read_pickle('./dataframes/3BH/highE_master/df_BH7_ks0_K1_L0_l0')
    # KL = pd.read_pickle('./dataframes/3BH/highE_master/df_BH6_ks0_K1_L1_l0')


    """ manual phi angle """
    # K0 = pd.read_pickle('./dataframes/phi/dataframe_manInp_phi0_ks0_K1_L0_l0')
    # K60 = pd.read_pickle('./dataframes/phi/dataframe_manInp_phi60_ks0_K1_L0_l0')
    # K120 = pd.read_pickle('./dataframes/phi/dataframe_manInp_phi120_ks0_K1_L0_l0')
    # K180 = pd.read_pickle('./dataframes/phi/dataframe_manInp_phi180_ks0_K1_L0_l0')
    # K240 = pd.read_pickle('./dataframes/phi/dataframe_manInp_phi240_ks0_K1_L0_l0')
    # K300 = pd.read_pickle('./dataframes/phi/dataframe_manInp_phi300_ks0_K1_L0_l0')
    # K360 = pd.read_pickle('./dataframes/phi/dataframe_manInp_phi360_ks0_K1_L0_l0')

    # KL0 = pd.read_pickle('./dataframes/phi/dataframe_manInp_phi0_ks0_K1_L1_l0')
    # KL60 = pd.read_pickle('./dataframes/phi/dataframe_manInp_phi60_ks0_K1_L1_l0')
    # KL120 = pd.read_pickle('./dataframes/phi/dataframe_manInp_phi120_ks0_K1_L1_l0')
    # KL180 = pd.read_pickle('./dataframes/phi/dataframe_manInp_phi180_ks0_K1_L1_l0')
    # KL240 = pd.read_pickle('./dataframes/phi/dataframe_manInp_phi240_ks0_K1_L1_l0')
    # KL300 = pd.read_pickle('./dataframes/phi/dataframe_manInp_phi300_ks0_K1_L1_l0')
    # KL360 = pd.read_pickle('./dataframes/phi/dataframe_manInp_phi360_ks0_K1_L1_l0')

    """ 3 BH no seed new Tsunami """
    # KNS = pd.read_pickle('./dataframes/3BH/noSeed/df2_3_noSeed_ks0_K1_L0_l0')
    # KLNS = pd.read_pickle('./dataframes/3BH/noSeed/df1_3_noSeed_ks0_K1_L1_l0')

    """ 3 BH Johan set """
    # KN = pd.read_pickle('./dataframes/johan/higherR/dataframe_johanSet_ks0_K1_L0_l0')    # latest version
    # KLN = pd.read_pickle('./dataframes/johan/higherR/dataframe_johanSet_ks0_K1_L1_l0')   # latest version

    # K = pd.read_pickle('./dataframes/johan/higherR/notLatest/dataframe_johanSet_ks0_K1_L0_l0')
    # KL = pd.read_pickle('./dataframes/johan/higherR/notLatest/dataframe_johanSet_ks0_K1_L1_l0')

    # KN = pd.read_pickle('./dataframes/johan/logA/dataframe_johanSet_ks0_K1_L0_l0')    # log a
    # KLN = pd.read_pickle('./dataframes/johan/logA/dataframe_johanSet_ks0_K1_L1_l0')   # log a

    # K = pd.read_pickle('./dataframes/johan/logA/previous/dataframe_johanSet_ks0_K1_L0_l0')  # log a
    # KL = pd.read_pickle('./dataframes/johan/logA/previous/dataframe_johanSet_ks0_K1_L1_l0') # log a

    # KL5 = pd.read_pickle('./dataframes/johan/tidalTol/dataframe_johanSet_ks0_K1_L1_l0_tidal-5')     # 1e-5 tidal tol
    # KL9 = pd.read_pickle('./dataframes/johan/tidalTol/dataframe_johanSet_ks0_K1_L1_l0_tidal-9')     # 1e-9 tidal tol

    # KN = pd.read_pickle('./dataframes/johan/rad40/dataframe_johanSet_ks0_K1_L0_l0_40Rad')    # 40 rad
    # KLN = pd.read_pickle('./dataframes/johan/rad40/dataframe_johanSet_ks0_K1_L1_l0_40Rad')   # 40 rad

    """ 2 BH MOCCA (post-PN fix)"""
    # KAll = pd.read_pickle('./dataframes/2BH/PNFix/df2_2_ks0_K1_L0_l0')
    # KLAll = pd.read_pickle('./dataframes/2BH/PNFix/df1_2_ks0_K1_L1_l0')
    KLNAll = pd.read_pickle('./dataframes/2BH/PNFix/df1_2_ks0_K1_L1_l1')

    """ 2 BH MOCCA bound triples BH inner """
    # K = pd.read_pickle("./dataframes/2BH/PNFix/boundTriples/boundTriplesBHBinary_ARC")
    # KL = pd.read_pickle("./dataframes/2BH/PNFix/boundTriples/boundTriplesBHBinary_PN")
    # KLN = pd.read_pickle("./dataframes/2BH/PNFix/boundTriples/boundTriplesBHBinary_PNTides")

    """ 2 BH MOCCA bound triples star inner """
    # K = pd.read_pickle("./dataframes/2BH/PNFix/boundTriples/boundTriplesBHStar_ARC")
    # KL = pd.read_pickle("./dataframes/2BH/PNFix/boundTriples/boundTriplesBHStar_PN")
    # KLN = pd.read_pickle("./dataframes/2BH/PNFix/boundTriples/boundTriplesBHStar_PNTides")





""" find index of non-finished interactions """
# nonFinishedIndex = KLN[KLN['t cpu'].astype('float') >= 90 ]
# np.savetxt('pnTidesNonFinished.txt', nonFinishedIndex)

# outcomesks0, outnames = isolateOutcome(ks0, conf)


# confSimple('Outcomes for 2a (all)', [ks0, ks0H, ks0L, ks1, ks1H, ks1L], ['No reg', 'No reg (high acc)', 'No reg (low acc)', 'KS reg', 'KS reg (high acc)', 'KS reg (low acc)'])

""" exclude non-finished runs """
# ks0 = ks0[ks0['t cpu'].astype('float') < 90 ]
# ks1 = ks1[ks1['t cpu'].astype('float') < 90 ]
# # K2 = K2[K2['t cpu'].astype('float') < 90 ]
# K = K[K['t cpu'].astype('float') < 90 ]
KL = KL[KL['t cpu'].astype('float') < 90 ]
# KLC = KLC[KLC['t cpu'].astype('float') < 90 ]
# KLl = KLl[KLl['t cpu'].astype('float') < 90 ]
# Kl = Kl[Kl['t cpu'].astype('float') < 90 ]
# noReg = noReg[noReg['t cpu'].astype('float') < 90 ]

# KN = KN[KN['t cpu'].astype('float') < 90 ]
KLN = KLN[KLN['t cpu'].astype('float') < 90 ]
KLNAll = KLNAll[KLNAll['t cpu'].astype('float') < 90 ]

# KL5 = KL5[KL5['t cpu'].astype('float') < 90 ]
# KL9 = KL9[KL9['t cpu'].astype('float') < 90 ]

# KO = KO[KO['t cpu'].astype('float') < 90]
# KLO = KLO[KLO['t cpu'].astype('float') < 90]
# KLlO = KLlO[KLlO['t cpu'].astype('float') < 90]
# KlO = KlO[KlO['t cpu'].astype('float') < 90]

# K0 = K0[K0['t cpu'].astype('float') < 90 ]
# K60 = K60[K60['t cpu'].astype('float') < 90 ]
# K120 = K120[K120['t cpu'].astype('float') < 90 ]
# K180 = K180[K180['t cpu'].astype('float') < 90 ]
# K240 = K240[K240['t cpu'].astype('float') < 90 ]
# K300 = K300[K300['t cpu'].astype('float') < 90 ]
# K360 = K360[K360['t cpu'].astype('float') < 90 ]

# KL0 = KL0[KL0['t cpu'].astype('float') < 90 ]
# KL60 = KL60[KL60['t cpu'].astype('float') < 90 ]
# KL120 = KL120[KL120['t cpu'].astype('float') < 90 ]
# KL180 = KL180[KL180['t cpu'].astype('float') < 90 ]
# KL240 = KL240[KL240['t cpu'].astype('float') < 90 ]
# KL300 = KL300[KL300['t cpu'].astype('float') < 90 ]
# KL360 = KL360[KL360['t cpu'].astype('float') < 90 ]




# KLDiff = KLDiff[KLDiff['t cpu'].astype('float') < 90 ]

# KT = KT[KT['t cpu'].astype('float') < 90 ]
# KLT = KLT[KLT['t cpu'].astype('float') < 90 ]

# ks0b = ks0b[ks0b['t cpu'].astype('float') < 90 ]
# ks1b = ks1b[ks1b['t cpu'].astype('float') < 90 ]
# # K2 = K2[K2['t cpu'].astype('float') < 90 ]
# Kb = Kb[Kb['t cpu'].astype('float') < 90 ]

# ks0NS = ks0NS[ks0NS['t cpu'].astype('float') < 90 ]
# ks1NS = ks1NS[ks1NS['t cpu'].astype('float') < 90 ]
# KNS = KNS[KNS['t cpu'].astype('float') < 90 ]
# KLNS = KLNS[KLNS['t cpu'].astype('float') < 90 ]
# KLTNS = KLTNS[KLTNS['t cpu'].astype('float') < 90 ]

#ks0O = ks0O[ks0O['t cpu'] != '100']
#ks1O = ks1O[ks1O['t cpu'] != '100']
#KO = KO[KO['t cpu'] != '100']

# ks0H = ks0H[ks0H['t cpu'].astype('float') < 100 ]
# ks1H = ks1H[ks1H['t cpu'].astype('float') < 100 ]
# # # # KH = KH[KH['t cpu'].astype('float') < 100 ]

# ks0L = ks0L[ks0L['t cpu'].astype('float') < 100 ]
# ks1L = ks1L[ks1L['t cpu'].astype('float') < 100 ]
# KL = KL[KL['t cpu'].astype('float') < 100 ]

# KN = KN[KN['t cpu'].astype('float') < 90 ]
# KLN = KLN[KLN['t cpu'].astype('float') < 90 ]
# KTN = KTN[KTN['t cpu'].astype('float') < 90 ]
# KLTN = KLTN[KLTN['t cpu'].astype('float') < 90 ]

# ks0PM = ks0PM[ks0PM['t cpu'].astype('float') < 90 ]
# ks1PM = ks1PM[ks1PM['t cpu'].astype('float') < 90 ]
# KPM = KPM[KPM['t cpu'].astype('float') < 90 ]

# confSimple('Outcomes for 2a (completed)', [ks0, ks0H, ks0L, ks1, ks1H, ks1L], ['No reg', 'No reg (high acc)', 'No reg (low acc)', 'KS reg', 'KS reg (high acc)', 'KS reg (low acc)'])

conf = ['Bound merger', 'Unbound merger', 'Mergers']

outcomesk0, outnames = isolateOutcome(KLNAll, conf)
# outcomesk1, outnames = isolateOutcome(KL, conf)
# outcomesK, outnames = isolateOutcome(KLN, conf)

# outcomesk0b, outnames = isolateOutcome(ks0b, conf)
# outcomesk1b, outnames = isolateOutcome(ks1b, conf)
# outcomesKb, outnames = isolateOutcome(Kb, conf)

""" initial params for BHs """
# initialParamPlots([outcomesk0, outcomesk1, outcomesK], 'a1', ['ARChain', 'PN', 'PN + tides'], 'e$_{initial}$', 0, 1, 0.05)
# initialParamPlots([K], 'e_f1', ['ARChain'], 'e$_{final}$', -5, 0, 0.5)
# initialParamPlots([KL], 'e_f1', ['PN'], 'e$_{final}$', -5, 0, 0.5)
# initialParamPlots([KLN], 'e_f1', ['PN + tides'], 'e$_{final}$', -5, 0, 0.5)

# initialParamPlots([K, KL, KLN], 'a_f1', ['ARChain', 'PN', 'PN + tides'], 'a$_{final}$ [AU]', 0, 2, 0.05)
# initialParamPlots([ks0, ks1, K], 'vInf', ['1 BH', '2BH', '3 BH'], 'v$_{\infty}$ [km/s]', 0, 100, 0.1)
# initialParamPlots([ks0, ks1, K], 'bImp', ['1 BH', '2BH', '3 BH'], 'b/a', 0, 40, 0.1)
# initialParamPlots([ks0, ks1, K], 'm0', ['1 BH', '2BH', '3 BH'], 'Mass [M$_{\odot}$]', 0, 30, 0.3)

# initialParamPlots([outcomesK], 'a_f1', ['1 BH'], 'a$_f$ [AU]', 0, 10, 0.1)
# initialParamPlots([outcomesK], 'e_f1', ['1 BH'], 'e$_f$', 0, 1, 0.01)

# initialParamPlots([outcomesK], 'a1', ['1 BH'], 'a [AU]', 0, 20, 0.1)
# initialParamPlots([outcomesK], 'vInf', ['1 BH'], 'v$_{\infty}$ [km/s]', 0, 100, 1)
# initialParamPlots([outcomesK], 'bImp', ['1 BH'], 'b/a', 0, 40, 1)
# initialParamPlots([outcomesK], 't final', ['1 BH'], 't$_{final}$ [yr]', 0, 250, 5)
# initialParamPlots([outcomesK], 'Nosc', ['1 BH'], '# oscillations', 0, 20, 1)

# initialParamPlots([K], 'a_f1', ['ARChain'], 'a$_f$ [AU]', 0, 10, 0.1)
# initialParamPlots([K, KL, KN, KLN], 'rMinSun', ['ARChain', 'PN', 'ARChain (latest)', 'PN (latest)'], 'Rmin [AU]', 0, 10, 0.1)
# verySmallRmin([KN, KLN])

# bTrip = paramConf([KN, KLN], ['ARChain', 'PN'], 'rMinSun', 'R$_{min}$ [R$_{\odot}$]')

# tooHighEcc = KLN[KLN['e_f1_O'] > 1]
# mergerTimes = timeMergeGW(tooHighEcc, 'Bound triple', 'After')
# mergerTimesK = timeMergeGW(KN, 'Bound triple', 'After')

# # # mergerTimes5 = timeMergeGW(KL5, 'Bound triple', 'After')
# # # mergerTimes9 = timeMergeGW(KL9, 'Bound triple', 'After')

# # mergerTimeSeconds = mergerTimes * 31.5e12
# # # mergerTimeSeconds5 = mergerTimes5 * 31.5e12
# # # mergerTimeSeconds9 = mergerTimes9 * 31.5e12

# binLog = np.logspace(-18, -1, 50)
# # # binLin = np.linspace(0, 1, 50)
# # # periDist = bTrip['a_f1'] * (1- bTrip['e_f1'])
# # # plt.hist(mergerTimeSeconds5, bins=binLog, label='1e-5', histtype='step', linewidth=2)
# plt.hist(mergerTimesK, bins=binLog, label='ARChain', histtype='step', linewidth=2)
# plt.hist(mergerTimes, bins=binLog, label='PN', linewidth=2)

# # # plt.hist(mergerTimeSeconds9, bins=binLog, label='1e-9', histtype='step', linewidth=2)
# plt.xscale('log')
# plt.xlabel('t$_{GW}$ [yr]')
# plt.ylabel('Count')
# plt.legend()
# # plt.title('Bound triples')
# # plt.axvline(x=0.00000019746, label='Schwarzschild radius', c='black')
# plt.legend(loc='upper right', title='$\delta=$')

# initialParamPlots([K], 'e_f1', ['ARChain'], 'e$_f$', 0, 1, 0.01)


""" Find which systems change with different flags """
# ks0Confs = ks0['conf']
# ks1Confs = ks1['conf']
# kConfs = K['conf']
# kLConfs = KL['conf']

# Difference between ks = 0 and ks = 1
#indDiff1 = np.where(ks0Confs.ne(ks1Confs))[0]
#ksChanges = ks1.iloc[indDiff1]
#ks0Changes = ks0.iloc[indDiff1]

### Difference between K = 0 and K = 1 (where ks = 0)
#indDiff2 = np.where(ks0Confs.ne(kConfs))[0]
#kChanges = K.iloc[indDiff2]
#ks0Changes = ks0.iloc[indDiff2]

## Difference between L = 0 and L = 1 (where ks = 0 and K = 1)
# indDiff3 = np.where(kConfs.ne(kLConfs))[0]
# kLChanges = KL.iloc[indDiff3]
# ks0Changes = ks0.iloc[indDiff3]
#
# dfFirst100KL = KL[0:100]
# dfFirst100K = K[0:100]


""" find which mergers were BH-BH mergers in 2BH set """
# bhbhMergers = findBHMergers(ks0)

""" Star mass distribution """
# starMassDistribution([K, KL, KLN], ['ARChain', 'PN', 'PN+tides'])
# starMassDistribution([K], [''])

def initConfBoundTriples(df):
    masses = df[['m0', 'm10', 'm11']].astype(float).to_numpy()
    minMassInd = np.argmin(masses, axis=1)
    minMass = np.amin(masses, axis=1)

    # binMembers = df['conf'].apply(lambda st: st[st.find("[")+1:st.find("]")]).str.split(' ', expand=True)
    # BM1 = binMembers[0]
    # BM2 = binMembers[1]

    print('# single star init: ' + str(np.sum(minMassInd == 0)))
    print('# single BH init: ' + str(np.sum(minMassInd != 0)))

    print('# neutron stars candidates: ' + str(np.sum(minMass >= 8)) + '\n')


""" bound triples, start as BHBH or BHStar binary """
# initConfBoundTriples(K)
# initConfBoundTriples(KL)
# initConfBoundTriples(KLN)

""" Find system that merge with PN but not without """
# kConfs = K['conf']
# kLConfs = ks0['conf']

# indDiff3 = np.where(kConfs.ne(kLConfs))[0]
# kLChanges = ks0.iloc[indDiff3]['conf']
# kChanges = K.iloc[indDiff3]['conf']

# subString = ':'
# mergeBH = kLChanges.str.contains(subString)
# mergedInd = kLChanges[mergeBH].index.values.astype(int)

# periDist = KLN['a_f1']  * (1 - KLN['e_f1'])
# smallPeriDist = KLN[periDist < 0.0000019746]

# np.savetxt('smallPeriDist_python.txt', smallPeriDist.index.to_numpy(), delimiter=',')


# iniConf = kChanges[mergedInd]
# iniConf = KL.iloc[mergedInd]

def kozaiTimescale(df):
    G = 39.478      # AU3 * yr-2 * Msun-1

    binMembers = pd.Series(df['conf']).apply(lambda st: st[st.find("[")+1:st.find("]")]).str.split(' ', expand=True)
    BM1 = binMembers[0].str[1].astype(int)
    BM2 = binMembers[1].astype(int)

    outerInd = df['conf'].str.split(']', expand=True)[1].astype(int)


    masses = np.array([df['m0'].astype(float), df['m10'].astype(float), df['m11'].astype(float)]).T

    aI = df['a_f1'].astype(float)
    aO = df['a_f1_O'].astype(float)
    eO = df['e_f1_O'].astype(float)

    binaryMass = np.zeros(len(BM1))
    outerMass = np.zeros(len(BM1))

    for i in range(len(BM1)):
        binaryMass[i] = masses[i][BM1[i]] + masses[i][BM2[i]]
        outerMass[i] = masses[i][outerInd[i]]

    T = 2 * np.pi * np.sqrt(G * binaryMass) / (G * outerMass) * aO**3/( aI**(3/2) ) * (1 - eO**2)**(3/2)

    return T

""" Kozai timescale """
# timescaleK = kozaiTimescale(K)
# timescaleKL = kozaiTimescale(KL)
# timescaleKLl = kozaiTimescale(KLN)

# # y = np.ones(len(timescaleK))
# plt.figure()
# weightsK = np.ones(len(timescaleK)) / len(timescaleK)
# weightsKL = np.ones(len(timescaleKL)) / len(timescaleKL)
# weightsKLl = np.ones(len(timescaleKLl)) / len(timescaleKLl)
# bins = np.logspace(-3, 5, 50)
# plt.hist(timescaleK, label='ARChain', bins=bins, histtype='step', linewidth=5, weights = weightsK)
# plt.hist(timescaleKL, label='PN', bins=bins, histtype='step', linewidth=5, weights = weightsKL)
# plt.hist(timescaleKLl, label='PN+tides', bins=bins, histtype='step', linewidth=5, weights = weightsKLl)
# plt.xlabel('Kozai timescale [yr]')
# plt.ylabel('Count')
# plt.xscale('log')
# plt.title('Bound triples (BH-S)')
# plt.legend()


""" Plot histograms for different parameters """
includeIMBH = 1
IMBHMassLimit = 1e3
onlyLowMass = 0
lowMass = 1e-2
onlyHighE = 0

""" cross-section plots """
# sigmaKS0 = crossSec(ks0, 'No reg')
# sigmaKS1 = crossSec(ks1, 'KS reg')
# sigmaK = crossSec(K, 'ARChain')
# sigmaKLT = crossSec(KLT, 'PN + tides')
# crossSecConf([ks0, ks1, K], ['No reg', 'KS reg', 'ARChain'], ['Exchange', 'Ionization'])

# crossSecMult([ks0, ks0O], 'ks0', [0, 1])
# crossSecMult([ks1, ks1O], 'ks1', [0, 1])

# crossSecMult([KN, KLN], 'New', [0, 1])
# crossSecMult([ks1, ks1O], 'ks1', [0, 1])

# crossSecConfSemiMajorMOCCA([KAll, KLAll, KLNAll], ['ARChain', 'PN', 'PN + tides'], 'Exchange')
# crossSecConfSemiMajor([KN, KLN], ['ARChain', 'PN'], 'Exchange')



""" filter out bad energy conservation """
# ks0Bad = energyFilter(ks0)
# ks1Bad = energyFilter(ks1)
# K = energyFilter(K)

# ks0HBad = energyFilter(ks0H)
# ks1HBad = energyFilter(ks1H)
# # # KH = energyFilter(KH)

# ks0LBad = energyFilter(ks0L)
# ks1LBad = energyFilter(ks1L)
# KL = energyFilter(KL)

""" extract interactions with specific index """
conf = ['Bound merger', 'Unbound merger', 'Triple merger']
# outcomesK, outnames = isolateOutcome(K, conf)

# seed = np.loadtxt('seed.txt')


# boundTriplesNewM7 = KLC.iloc[outcomesK.index.to_numpy()]

# np.savetxt('mergers_3BH_manSetup_5seeds_K.txt', outcomesK.index.to_numpy(), delimiter=',')

def seedChangedConf(df):
    confs = df['conf']
    confsTemp = []
    numChanged = []
    for i in range(len(df)):
        # print(i != 0 and (df['a1'].iloc[i] == df['a1'].iloc[i-1]) & (df['e1'].iloc[i] == df['e1'].iloc[i-1]))
        if i != 0 and (df['a1'].iloc[i] == df['a1'].iloc[i-1]) & (df['e1'].iloc[i] == df['e1'].iloc[i-1]):
            confsTemp.append(confs.iloc[i])

        elif i == 0:
            confsTemp.append(confs.iloc[i])
        else:
            # print('now')
            if len(confsTemp) > 0:
                numChanged.append(len(np.unique(confsTemp))-1)
                confsTemp = []

    return numChanged

def confSimpleModified(pltTitle, dfs, colNames):
    df = pd.DataFrame()
    for i in range(len(dfs)):
        data, outcomes = outcomeInteger(dfs[i], dfs[i]['conf'])

        numI = np.array([])
        confs = np.array(outcomes)
        confs = np.append(confs, 'Mergers')



        for j in range(len(data)):
            if outcomes[j] == 'Bound triple':
                c = 63198       # AU/yr
                G = 39.478      # AU3 * yr-2 * Msun-1

                t = (5/256 * c**5 * data[j]['a_f1']**4 * (1-data[j]['e_f1']**2)**(7/2)/(G**3 * 10 * 10 * (10 + 10))) * 31556926

                mergers = data[j][t <= 1e4]
                triples = data[j][t > 1e4]

                numI = np.append(numI, len(triples.index))
                numI = np.append(numI, len(mergers.index))

                # print('t')

            else:
                numI = np.append(numI, len(data[j].index))

        dfSort = pd.DataFrame(numI, columns=[colNames[i]])
        dfSort = dfSort.set_index(confs)

        df = pd.concat([df, dfSort], axis=1, sort=False)


    df = df.fillna(0)
    df = df.loc[(df != 0).any(axis=1), :]
    dfNum = df
    df /= np.sum(df)

    sns.set_style("ticks")
    sortInd = ['Exchange', 'Flyby', 'Bound triple', 'Mergers']

    df.loc[sortInd].plot(kind='bar',  legend=False, rot=0, title=pltTitle)
    plt.legend(prop={'size': 30}, bbox_to_anchor=(1, 1))
    plt.title(pltTitle, size=30)
    plt.ylabel('Fraction', size=25)
    plt.xticks(fontsize=25, rotation=20)
    plt.tight_layout()

""" check where seed changed conf """
conf =  ['Flyby', 'Exchange', 'Ionization', 'Bound merger', 'Unbound merger', 'Triple merger',  'Bound triple']
# outcomesK, outnames = isolateOutcome(KLN, conf)
# boundTriplesNewM7 = KL.iloc[outcomesK.index.to_numpy()]

# timesChanged = seedChangedConf(K)

# plt.hist(timesChanged, bins=[0,1,2,3,4])
# plt.title('Times outcome changed with seed')
# plt.xlabel('Times changed')
# plt.ylabel('Count')
# plt.xticks([0.5,1.5, 2.5, 3.5], ['0', '1', '2', 3])

""" Reclassify bound triples as mergers """
# confSimpleModified('', [K, KL, KN, KLN], ['ARChain', 'PN', 'ARChain (new)', 'PN (new)'])
# paramScatterConfMinus(KLN, '', 'a_f1', 'e_f1', 'a$_{final}$', 'e$_{final}$', 'Bound triple')

""" Configurations """
#compChanged('Distribution of final configurations for changed systems')
# confSimple('Distribution of outcomes for 2a', [ks0, ks1, K], ['No reg', 'KS reg', 'ARChain', 'PN + Tides'])
# confSimple('Distribution of outcomes for 2b', [ks0, ks1, K], ['No reg', 'KS reg', 'ARChain', 'PN + Tides'])
# confSimple('Distribution of outcomes for 1BH', [ks0, ks1, K, KL, KT, KLT], ['No reg', 'KS reg', 'ARChain', 'PN terms', 'Tides', 'PN + Tides'])
# confSimple('Distribution of outcomes for 2BH', [ks0, ks1, K, KL, KT, KLT], ['No reg', 'KS reg', 'ARChain', 'PN terms', 'Tides', 'PN + Tides'])
# confSimple('Distribution of outcomes for 1BH', [ks0, ks1, K, KL], ['No regularisation', 'KS regularisation', 'ARChain', 'PN terms', 'PN + Tides'])

# confSimple('Distribution of outcomes for 3BH', [ks0, ks1, K, KL], ['No reg', 'KS reg', 'ARChain', 'PN terms'])
# confSimple('Distribution of outcomes for 3BH', [K, KL], ['ARChain', 'PN terms', 'PN + Tides'])

# confSimple('1 BH (New)', [K, KL, KLl], ['ARChain', 'PN terms', 'PN + tides'])
# confSimple('1 BH (old vs new)', [KO, KLO, KLlO, K, KL, KLl], ['ARChain (old)', 'PN terms (old)', 'PN + tides (old)', 'ARChain (new)', 'PN terms (new)', 'PN + tides (new)'])
# confSimple('2 BH (New)', [K, KL, Kl, KLl], ['ARChain', 'PN terms', 'Tides', 'PN + tides'])
# confSimple('2 BH (old vs new)', [KO, KLO, KlO, KLlO, K, KL, Kl, KLl], ['ARChain (old)', 'PN terms (old)', 'Tides (old)', 'PN + tides (old)', 'ARChain (new)', 'PN terms (new)', 'Tides (new)', 'PN + tides (new)'])
# confSimple('3 BH (old vs new)', [KO, KLO, K, KL], ['ARChain (old)', 'PN terms (old)', 'ARChain (new)', 'PN terms (new)'])
# confSimple('3 BH (a < 1 AU)', [noReg, K, KL], ['No reg', 'ARChain', 'PN terms'])
# confSimple('Bound triples in old', [boundTriplesInOld], [''])
# confSimple('Bound triples in old Tsunami', [K, boundTriplesNewM7, KL, KLl], ['1e-5', '1e-7', '1e-9', '1e-11'])

# confSimple('5 seeds', [K, KL, KN, KLN], ['ARChain', 'PN', 'ARChain (new)', 'PN (new)'])
# confSimple('', [KN, KLN], ['ARChain', 'PN'])

# confSimple('ARChain', [K0, K60, K120, K180, K240, K300, K360], ['0', '60', '120', '180', '240', '300', '360'])
# confSimple('PN', [KL0, KL60, KL120, KL180, KL240, KL300, KL360], ['0', '60', '120', '180', '240', '300', '360'])

# confSimple('Low $\phi$', [K0, K60, K120, KL0, KL60, KL120], ['ARChain (0)', 'ARChain (60)', 'ARChain (120)', 'PN (0)', 'PN (60)', 'PN (120)'])
# confSimple('High $\phi$', [K180, K240, K300, K360, KL180, KL240, KL300, KL360], ['ARChain (180)', 'ARChain (240)', 'ARChain (300)', 'ARChain (360)', 'PN (180)', 'PN (240)', 'PN (300)', 'PN (360)'])

# confSimple('', [K, KL, KNS, KLNS], ['ARChain', 'PN', 'ARChain (no seed)', 'PN (no seed)'])
# confSimple('2 BH', [K, KL, KLN], ['ARChain', 'PN', 'PN + tides'])


# confSimple('', [KL5, KLN, KL9], ['1e-5', '1e-7', '1e-9'])


# confSimple('1 BH', [K, KLDiff], ['PN', 'PN (rerun)'])

def paramHist(dfs, dfNames, param, bins, xlab, title):
    plt.figure()
    for i in range(len(dfs)):
        dfT = dfs[i]
        if param == 'a_f1' or param =='e_f1':
            confs = dfT['conf']
            dfT = dfT[confs.str.contains("\[")]
        weigths = np.ones(len(dfT[param])) / len(dfT[param])
        if param == 'e_f1':
            (dfT[param]**2).hist(bins=bins, label=dfNames[i], histtype='step', weights=weigths, linewidth=2)
        else:
            dfT[param].hist(bins=bins, label=dfNames[i], histtype='step', weights=weigths, linewidth=2)

    if param == 'a_f1':
        weightsInit = np.ones(len(dfT['a1'])) / len(dfT['a1'])
        dfT['a1'].astype(float).hist(bins=bins, label='Initial', histtype='step', weights=weightsInit, linewidth=2)
        plt.xscale('log')
    elif param == 'e_f1':
        weightsInit = np.ones(len(dfT['e1'])) / len(dfT['e1'])
        (dfT['e1'].astype(float)**2).hist(bins=bins, label='Initial', histtype='step', weights=weightsInit, linewidth=2)


    plt.xlabel(xlab, fontsize=40)
    plt.legend(loc='upper left', fontsize=40)
    plt.ylabel('Frac', fontsize=40)
    plt.grid(None)

def numLowE(df, lim):
    e = 1-df['e_f1'].astype(float)  # 1-e
    return np.sum(e < lim)

def aeRad(df):
    ae = df['a1'].astype(float)*(1-df['e1'].astype(float)) * 214.93946938
    rad0 = df['r0'].astype(float)
    rad10 = df['r10'].astype(float)
    rad11 = df['r11'].astype(float)

    dfAll = pd.concat([ae, rad0, rad10, rad11], names=['Pericenter distance', 'Rad single', 'Rad bin1', 'Rad bin2'], axis=1)

    print('t')

def diffInitFinal(df, param):
    mask = df[param].astype(float) != 0
    init = df['a1'].astype(float)[mask]
    final = df[param].astype(float)[mask]

    diff = abs(init-final)

    largestDiff = K.loc[diff.idxmax()]

    print('t')

def countMergersSemiAxis(df):
    data, outcomes = outcomeInteger(df, df['conf'])
    aUni = df['a1'].unique().astype(float)
    numMerge = np.array([])

    for i in range(len(data)):
        if outcomes[i] == 'Bound triple':
            for j in range(len(aUni)):
                numTemp = len(data[i][data[i]['a1'].astype(float) == aUni[j]])
                numMerge = np.append(numMerge, numTemp)


    plt.plot(aUni, numMerge, '--o')
    plt.xscale('log')
    plt.ylabel('# mergers')
    plt.xlabel('a$_{initial}$ [AU]')

def rocheLobe(df):
    binMembers = pd.Series(df['conf']).apply(lambda st: st[st.find("[")+1:st.find("]")]).str.split(' ', expand=True)
    masses = df[['m0', 'm10', 'm11']].astype(float)
    radii = df[['r0', 'r10', 'r11']].astype(float)
    masses.columns = ['0', '1', '2']

    semiAxis = df['a_f1'].astype(float)*214.93946938
    ecc = df['e_f1']
    periDist = semiAxis * (1-ecc)
    rocheLobe = np.zeros(len(binMembers))
    maxRad = radii.max(axis=1)

    for i in range(len(binMembers)):
        mass1 = masses.iloc[i][binMembers.iloc[i][0][1]]
        mass2 = masses.iloc[i][binMembers.iloc[i][1]]

        if mass1 > mass2:
            q = mass2 / mass1
        else:
            q = mass1 / mass2

        rocheLobe[i] = 0.49*q**(2/3) / (0.6*q**(2/3) + np.log(1 + q**(1/3))) * periDist.iloc[i]
    return rocheLobe, maxRad



""" roche lobe triples with star-BH binary """
rocheLobeK, maxRadK = rocheLobe(K)
rocheLobeKL, maxRadKL = rocheLobe(KL)
rocheLobeKLl, maxRadKLl = rocheLobe(KLN)

fracRLK = rocheLobeK/maxRadK
fracRLKL = rocheLobeKL/maxRadKL
fracRLKLl = rocheLobeKLl/maxRadKLl


# plt.figure()
# plt.plot(maxRadK, rocheLobeK, 'o', label='ARChain')
# plt.plot(maxRadKL, rocheLobeKL, 'o', label='PN')
# plt.plot(maxRadKLl, rocheLobeKLl, 'o', label='PN+tides', zorder=0)
# plt.xlabel('R$_{\star}$ [R$_{\odot}$]')
# plt.ylabel('R$_{Roche Lobe}$ [R$_{\odot}$]')
# plt.xscale('log')
# plt.yscale('log')
# ax = plt.gca()
# lims = [
#     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
# ]
# ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
# plt.legend()
# plt.title('A = semi-major axis')


""" Final params """
# paramHist([K, KL], ['ARChain', 'PN'], 'a_f1', np.logspace(-2,4, 500), 'a$_{final}$ [AU]', '3  BH')
# paramHist([K, KL], ['ARChain', 'PN'], 'e_f1', np.arange(0,1, 0.01), 'e$_{final}^2$', '3  BH')

# paramHist([K, KL], ['ARChain', 'PN'], 'a_f1', np.logspace(-2,6, 300), 'a$_{final}$ [AU]', '3  BH')
# paramHist([K, KL], ['ARChain', 'PN'], 'e_f1', np.arange(0,1, 0.01), 'e$_{final}$', '3  BH')

# aInit = K['a1'].astype(float)

# plt.hist(aInit, bins = np.logspace(-2,4, 500))
# plt.xscale('log')

# print(numLowE(K, 1e-2))
# print(numLowE(KL, 1e-2))

# aeRad(K)

# diffInitFinal(K, 'a_f1')

# compRuns('dEE0', 'dE/E0 distribution', 1e-15, 1, 0.01, 'dE/E0', [K, KL, KN, KLN], ['ARChain', 'PN', 'ARChain (latest)', 'PN (latest)'])
# compRuns('dEE0', 'dE/E0 distribution for KS reg', 1e-15, 1, 0.01, 'dE/E0', [ks1H, ks1, ks1L], ['1e-15', '1e-9', '1e-5'])

""" mergers against a """
# countMergersSemiAxis(KLN)


""" b prime & rMin"""
# bPrime([K, KL], ['ARChain', 'PN'], np.logspace(-5, 2, 100))


def exchangeInnerBinaryComp(df):
    masses = df[['m0', 'm10', 'm11']].astype(float)
    minMass = masses.idxmin(axis=1)

    # BHBH = np.array([])
    # BHStar = np.array([])

    initSingleStar = (minMass == 'm0')

    BHBH = df[~initSingleStar]
    BHStar = df[initSingleStar]
    # for i in range(len(df)):
    #     if minMass.iloc[i] == 'm0':
    #         BHStar = BHStar.append(df.iloc[i])
    #     else:
    #         BHBH = BHBH.append(df.iloc[i])


    return BHStar, BHBH


""" check exchanges for 2 BH run """
# exchangeK, outnames = isolateOutcome(K, 'Flyby')
# exchangeKL, outnames = isolateOutcome(KL, 'Flyby')
# exchangeKLl, outnames = isolateOutcome(KLN, 'Flyby')

# BHStarK, BHBHK = exchangeInnerBinaryComp(exchangeK)
# BHStarKL, BHBHKL = exchangeInnerBinaryComp(exchangeKL)
# BHStarKLl, BHBHKLl = exchangeInnerBinaryComp(exchangeKLl)



""" no seed vs seeded confs """
# confSimple('Distribution of outcomes for 2BH', [ks0, ks1, K, KL,  KLT, ks0NS, ks1NS, KNS, KLNS, KLTNS], ['No reg', 'KS reg', 'ARChain','PN terms', 'PN + Tides', 'No reg (no seed)','KS reg (no seed)', 'ARChain (no seed)', 'PN terms (no seed)', 'PN + Tides (no seed)'])
# confSimple('Distribution of outcomes for 2BH', [K, KNS, KL, KLNS, KLT, KLTNS], ['ARChain', 'ARChain (no seed)', 'PN', 'PN (no seed)', 'PN + Tides', 'PN + Tides (no seed)'])
# confSimple('Distribution of outcomes for 2BH', [ks0, ks0NS, ks1, ks1NS, K, KNS], ['No reg', 'No reg (no seed)', 'KS reg', 'KS reg (no seed)', 'ARChain', 'ARChain (no seed)'])
# confSimple('Distribution of outcomes for 2BH', [K, KL, KLT, KNS, KLNS,KLTNS], ['ARChain', 'PN', 'PN + Tides', 'ARChain (no seed)', 'PN (no seed)', 'PN + Tides (no seed)'])
# confSimple('Distribution of outcomes for 2BH', [ks0, ks1, K, KL, ks0NS, ks1NS, KNS, KLNS], ['No reg', 'KS reg', 'ARChain', 'PN terms', 'No reg (no seed)', 'KS reg (no seed)', 'ARChain (no seed)', 'PN terms (no seed)'])

# confSimple('Distribution of outcomes for 3BH', [ks0, ks1, K, KL, ks0NS, ks1NS, KNS, KLNS], ['No reg', 'KS reg', 'ARChain','PN terms', 'No reg (no seed)','KS reg (no seed)', 'ARChain (no seed)', 'PN terms (no seed)'])

""" accuracy outcomes """
# confSimple('Distribution of outcomes for 2b', [ks0, ks1, K, ks0H, ks1H, KH, ks0L, ks1L, KL], ['No reg', 'KS reg', 'ARChain', 'No reg (high acc)', 'KS reg (high acc)', 'ARChain (high acc)', 'No reg (low acc)', 'KS reg (low acc)', 'ARChain (low acc)'])
# confSimple('Outcomes for 2a', [ks0, ks0H, ks0L, ks1, ks1H, ks1L], ['No reg', 'No reg (high acc)', 'No reg (low acc)', 'KS reg', 'KS reg (high acc)', 'KS reg (low acc)', 'ARChain',   'ARChain (high acc)',   'ARChain (low acc)'])
# confSimple('Outcomes for 2b', [ks1, ks1H, ks1L], ['KS reg', 'KS reg (high acc)', 'KS reg (low acc)', 'ARChain',   'ARChain (high acc)',   'ARChain (low acc)'])
# confSimple('Outcomes for 2b', [ks0, ks0H, ks0L], ['No reg', 'No reg (high acc)', 'No reg (low acc)', 'ARChain',   'ARChain (high acc)',   'ARChain (low acc)'])
# confSimple('Outcomes for 2b', [ks0, ks0H, ks0L, ks1, ks1H, ks1L], ['No reg', 'No reg (high acc)', 'No reg (low acc)', 'KS reg', 'KS reg (high acc)', 'KS reg (low acc)'])
# confSimple('Outcomes for 2a ($dE/E0 < 0.1$)', [ks0Bad, ks0HBad, ks0LBad, ks1Bad, ks1HBad, ks1LBad], ['No reg', 'No reg (high acc)', 'No reg (low acc)', 'KS reg', 'KS reg (high acc)', 'KS reg (low acc)'])
# confSimple('Outcomes where $dE/E0 > 0.1$', [ks0LBad, ks1LBad], ['No reg (low acc)', 'KS reg (low acc)'])

""" outcomes PM vs normal """
# confSimple('Outcomes for 2b (PM)', [ks0, ks1, K, ks0PM, ks1PM, KPM], ['No reg', 'KS reg', 'ARChain', 'No reg (PM)', 'KS reg (PM)', 'ARChain (PM)'])
# confSimple('Outcomes for 2b (PM)', [ks0, ks0H, ks0L, ks1, ks1H, ks1L, K ], ['No reg (default acc)', 'No reg (high acc)', 'No reg (low acc)', 'KS reg (default acc)', 'KS reg (high acc)', 'KS reg (low acc)', 'ARChain'])

""" outcomes new tsunami """
# confSimple('Outcomes - old vs new', [KT, KLT, KTN, KLTN], ['Tides (old)', 'PN + tides (old)', 'Tides (new)', 'PN + tides (new)'])
# confSimple('Outcomes - old vs new', [K, KL, KN, KLN], ['ARChain (old)', 'PN (old)', 'ARChain (new)', 'PN (new)'])
# confSimple('Outcomes - old vs new', [K, KL, KT, KLT, KN, KLN, KTN, KLTN], ['ARChain (old)', 'PN (old)', 'Tides (old)', 'PN + tides (old)', 'ARChain (new)', 'PN (new)', 'Tides (new)', 'PN + tides (new)'])

""" outcomes handmade BH set """
# confSimple('Manually setup 2 BH set', [K, KL, KLT], ['ARChain', 'PN', 'PN + tides'])

""" different ways of illustrating outcomes """
#confSimpleMult('Final distribution of configurations', [[ks0, ks1, K], [ks0L, ks1L, KL], [ks0H, ks1H, KH]], ['', '(Low acc)', '(high acc)'])
#confSimplePie('Final distribution of configurations')
# confSankey([ks0, ks1, K], ['No regularisation', 'KS regularisation', 'ARChain'])
# outcomes, outnames = isolateOutcome(K, 'Ionization')

# tripMergeAnalyse([ks0, ks1, K, KL, KLT], ['No regularisation', 'KS regularisation', 'ARChain', 'PN terms', 'Tides'])

""" write index of outcomes with specific outcomes to file """
# conf = ['Exchange', 'Flyby', 'Bound merger', 'Unbound merger', 'Triple merger', 'Bound triple']
# conf = ['Ionization', 'Exchange', 'Flyby', 'Bound merger', 'Unbound merger', 'Triple merger', 'Bound triple']
conf = ['Unbound merger', 'Bound merger', 'Triple merger']
# conf = ['Bound triple']

# outcomes, outnames = isolateOutcome(ks0, conf)
# outcomesks1, outnames = isolateOutcome(ks1, conf)
# outcomesK, outnames = isolateOutcome(KLlO, conf)


# outcomesKL, outnames = isolateOutcome(KLO, conf)
# outcomesKLl, outnames = isolateOutcome(KLl, conf)
# outcomesK, outnames = isolateOutcome(KO, conf)
# outcomesKLl, outnames = isolateOutcome(KLlO, conf)


# np.savetxt('mergers_1BH_KLl_old.txt', outcomesK.index.to_numpy(), delimiter=',')


# outcomesks0H, outnames = isolateOutcome(KLDiff, conf)

# np.sum(KLDiff['a1'].in(outcomesKLOld['a1']))

# oldA = outcomesKLOld['a1']
# newA = KLDiff['a1']

# outcomesks0L, outnames = isolateOutcome(ks0L, conf)
# outcomesks1H, outnames = isolateOutcome(ks1H, conf)
# outcomesks1L, outnames = isolateOutcome(ks1L, conf)

# printParams(outcomesks0, 'dEE0', 'ks0')
# printParams(outcomesks0H, 'dEE0', 'ks0 (1e-15)')
# printParams(outcomesks0L, 'dEE0', 'ks0 (1e-5)')

# printParams(outcomesks1, 'dEE0', '\nks1')
# printParams(outcomesks1H, 'dEE0', 'ks1 (1e-15)')
# printParams(outcomesks1L, 'dEE0', 'ks1 (1e-5)')

# printParams(outcomesK, 'dEE0', '\nARChain')


# printParams(outcomesks0, 'dLL0', 'ks0 (1e-8)')
# printParams(outcomesks0H, 'dLL0', 'ks0 (1e-15)')
# printParams(outcomesks0L, 'dLL0', 'ks0 (1e-5)')

# printParams(outcomesks1, 'dLL0', '\nks1 (1e-8)')
# printParams(outcomesks1H, 'dLL0', 'ks1 (1e-15)')
# printParams(outcomesks1L, 'dLL0', 'ks1 (1e-5)')

# printParams(outcomesK, 'dLL0', '\nARChain')

# outcomesKN, outnames = isolateOutcome(KN, conf)
# outcomesKLN, outnames = isolateOutcome(KLN, conf)
# outcomesKTN, outnames = isolateOutcome(KTN, conf)
# outcomesKLTN, outnames = isolateOutcome(KLTN, conf)

# print('K: Mean = ' + str(np.mean(outcomesKN['dEE0'].astype(float))) + ', Median = ' + str(np.median(outcomesKN['dEE0'].astype(float))))
# print('KL: Mean = ' + str(np.mean(outcomesKLN['dEE0'].astype(float))) + ', Median = ' + str(np.median(outcomesKLN['dEE0'].astype(float))))
# print('KT: Mean = ' + str(np.mean(outcomesKTN['dEE0'].astype(float))) + ', Median = ' + str(np.median(outcomesKTN['dEE0'].astype(float))))
# print('KLT: Mean = ' + str(np.mean(outcomesKLTN['dEE0'].astype(float))) + ', Median = ' + str(np.median(outcomesKLTN['dEE0'].astype(float))))


# print('K: Mean = ' + str(np.mean(outcomesKN['dLL0'].astype(float))) + ', Median = ' + str(np.median(outcomesKN['dLL0'].astype(float))))
# print('KL: Mean = ' + str(np.mean(outcomesKLN['dLL0'].astype(float))) + ', Median = ' + str(np.median(outcomesKLN['dLL0'].astype(float))))
# print('KT: Mean = ' + str(np.mean(outcomesKTN['dLL0'].astype(float))) + ', Median = ' + str(np.median(outcomesKTN['dLL0'].astype(float))))
# print('KLT: Mean = ' + str(np.mean(outcomesKLTN['dLL0'].astype(float))) + ', Median = ' + str(np.median(outcomesKLTN['dLL0'].astype(float))))
# vInf = outcomes['vInf'].astype('float')
# outLowV = outcomes[vInf < 1]

# plt.hist(vInf)

# np.savetxt('mergers_index_1BH_K_newTsunami.txt', outcomesK.index.to_numpy(), delimiter=',')
# np.savetxt(conf + '_index_LowV_2BH.txt', outLowV.index.to_numpy(), delimiter=',')


# crashedInteractions = KLN[KLN['t cpu'].astype(float) > 90]
# np.savetxt('indexCrashed_JohanSet_higherR.txt', crashedInteractions.index.to_numpy(), delimiter=',')

def initParamAdd(df, param):
    plt.figure()
    bins = np.logspace(-5, 1, 100)
    if param == 'radius':
        radii = pd.concat([df['r0'], df['r10'], df['r11']]).astype(float)
    elif param == 'mass':
        radii = pd.concat([df['m0'], df['m10'], df['m11']]).astype(float)


    radii.hist(bins=bins)
    plt.xscale('log')
    plt.xlabel('radius [R$_{\odot}$]')
    plt.ylabel('Count')
    plt.title('2BH')
    plt.grid(False)

def paramAddInnerBinary(df, param, dfName, orgConf):
    # masses = pd.concat([df['m0'], df['m10'], df['m11']]).astype(float)
    masses = np.array([df['m0'], df['m10'], df['m11']]).astype(float)

    binMembers = pd.Series(orgConf).apply(lambda st: st[st.find("[")+1:st.find("]")]).str.split(' ', expand=True)
    BM1 = binMembers[0]
    BM2 = binMembers[1]
    radiiInner1 = np.zeros(len(BM1))
    radiiInner2 = np.zeros(len(BM1))

    minMass = np.argmin(masses, axis=0)


    for i in range(len(BM1)):
        # if '[' in BM1[i]:
        #     BM1[i] = BM1[i][1]
        # radiiInner1[i] = masses[int(BM1[i]), i]
        # radiiInner2[i] = masses[int(BM2[i]), i]
        if minMass[i] == 0:
            radiiInner1[i] = masses[1, i]
            radiiInner2[i] = masses[2, i]
        if minMass[i] == 1:
            radiiInner1[i] = masses[0, i]
            radiiInner2[i] = masses[2, i]
        if minMass[i] == 2:
            radiiInner1[i] = masses[0, i]
            radiiInner2[i] = masses[1, i]

    massesAdded = pd.Series(np.concatenate((radiiInner1, radiiInner2)))

    bins = np.logspace(np.log10(np.amin(massesAdded)), np.log10(np.amax(massesAdded)), 50)


    # plt.figure()
    # massesAdded.hist(bins=bins)
    # plt.xscale('log')
    # plt.xlabel('Mass [M$_{\odot}$]')
    # plt.ylabel('Count')
    # plt.title(dfName + ' (Inspirals)')
    # plt.grid(False)

    plt.figure()
    plt.scatter(radiiInner1, radiiInner2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('M$_1$ [M$_{\odot}$]')
    plt.ylabel('M$_2$ [M$_{\odot}$]')
    plt.title(dfName + ' (Inspirals)')
    plt.grid(False)

def BHStarExchange(df):
    masses = pd.DataFrame([df['m0'].astype(float), df['m10'].astype(float), df['m11'].astype(float)]).T
    minMass = np.argmin([df['m0'].astype(float), df['m10'].astype(float), df['m11'].astype(float)], axis=0)
    starInBinary = (minMass != 0)

    return df[starInBinary], df[~starInBinary]

def reclassify2BH(df):
    t = df['mergerTime'].astype(float)
    periDist = df['a_f1'] * (1 - df['e_f1'])
    periDist[(df['a_f1'] < 0) | (df['e_f1'] > 1)] = 1e7
    outerPeriod = 2 * np.pi * np.sqrt(df['a_f1_O']**3 / ( G * 10))
    dfCopy = df.copy()
    merger = (periDist < df['mergeDist'])

    mergersDF = ((periDist < df['mergeDist']) & (df['conf'] != '[[0 1] 2]') & (df['conf'] != '[[0 2] 1]') & (df['conf'] != '[[1 2] 0]') & (df['conf'] != '0 1 2')
                 & (df['conf'] != '0:1:2') & (df['conf'] != '0:2:1') & (df['conf'] != '1:0:2') & (df['conf'] != '1:2:0') & (df['conf'] != '2:1:0') & (df['conf'] != '0 1 2'))


    dfCopy['conf'][mergersDF] = 'Merger'
    # dfTemp['conf'][(t <= 1) & (dfTemp['conf'] != '[[0 1] 2]') & (dfTemp['conf'] != '[[0 2] 1]') &
    #                 (dfTemp['conf'] != '[[1 2] 0]') & (dfTemp['conf'] != 'Merger')] = ['Inspiral']
    # dfCopy['conf'][(~np.isnan(df['mergeDist'])) & (periDist > 0) & (periDist < df['mergeDist']) & ((df['conf'] == '[[0 2] 1]') | (df['conf'] == '[[0 1] 2]') | (df['conf'] == '[[1 2] 0]')) & (~df['BHBin'])] = 'Inspiral star'

    # inspiralsOrgConf = dfCopy['conf'][(~np.isnan(df['mergeDist'])) & (periDist > 0) & (periDist < df['mergeDist']) & ((df['conf'] == '[[0 2] 1]') | (df['conf'] == '[[0 1] 2]') | (df['conf'] == '[[1 2] 0]'))]

    # dfCopy['conf'][(~np.isnan(df['mergeDist'])) & (periDist > 0) & (periDist < df['mergeDist']) & ((df['conf'] == '[[0 2] 1]') | (df['conf'] == '[[0 1] 2]') | (df['conf'] == '[[1 2] 0]'))] = 'Inspiral'
    # inspiralsOrgConf = np.concatenate((inspiralsOrgConf, dfCopy['conf'][(~np.isnan(t)) & (t < outerPeriod) & (outerPeriod > 0)  & (dfCopy['conf'] != 'Merger') & (dfCopy['conf'] != 'Inspiral')]))

    # dfCopy['conf'][(~np.isnan(t)) & (t < outerPeriod) & (outerPeriod > 0)  & (dfCopy['conf'] != 'Merger')] = ['Inspiral']
    # inspiralsOrgConf = np.concatenate((inspiralsOrgConf, dfCopy['conf'][(~np.isnan(t)) & (t < np.amax(outerPeriod)) & (df['e_f1_O'] > 1)  & (dfCopy['conf'] != 'Merger') & (dfCopy['conf'] != 'Inspiral')]))

    # dfCopy['conf'][(~np.isnan(t)) & (t < np.amax(outerPeriod)) & (df['e_f1_O'] > 1)  & (df['conf'] != 'Merger')] = ['Inspiral']
    dfCopy['conf'][(t < 14e9) & (t != 0) & (df['conf'] != 'Inspiral') & (df['conf'] != '[[0 1] 2]') & (df['conf'] != '[[0 2] 1]') &
                    (df['conf'] != '[[1 2] 0]') & (df['conf'] != 'Merger') & (df['BHBin'])] = ['Post-interaction merger']

    out, outcomes = outcomeInteger2BH(dfCopy, dfCopy['conf'])

    return out[0]




""" find initial parameter space """
# conf = ['Bound merger']
conf = ['Flyby']
# conf = ['Unbound merger']
# conf =  ['Flyby', 'Exchange', 'Ionization', 'Bound merger', 'Unbound merger', 'Triple merger',  'Bound triple']
# outcomesK, outnames = isolateOutcome(K, conf)
# outcomesKL, outnames = isolateOutcome(KL, conf)
# outcomesKLl, outnames = isolateOutcome(KLN, conf)
# paramSpaceK = initParamSpace(K, conf)
# paramSpaceKL = initParamSpace(KL, conf)
# paramSpaceKLN = initParamSpace(KLN, conf)
# paramSpaceKLl = initParamSpace(KLl, conf)

### reclassiy as mergers, inspirals etc ###
# outcomesK = reclassify2BH(outcomesK)
# outcomesKL = reclassify2BH(outcomesKL)
# outcomesKLl = reclassify2BH(outcomesKLl)

# include only BH-Star
starInBinaryK, starSingleK = BHStarExchange(K)
starInBinaryKL, starSingleKL = BHStarExchange(KL)
starInBinaryKLl, starSingleKLl = BHStarExchange(KLN)

paramScatterMult([starInBinaryK, starSingleK], ['Star in initial binary','Star initial single', ],  'vInf', 'bImp', 'v$_{\infty}$ [km/s]', 'b [AU]', 'ARC')
paramScatterMult([starInBinaryKL, starSingleKL], ['Star in initial binary','Star initial single', ],  'vInf', 'bImp', 'v$_{\infty}$ [km/s]', 'b [AU]', 'PN')
paramScatterMult([starInBinaryKLl, starSingleKLl], ['Star in initial binary','Star initial single', ],  'vInf', 'bImp', 'v$_{\infty}$ [km/s]', 'b [AU]', 'PN + tides')


# outcomesK = BHStarExchange(K)
# outcomesKL = BHStarExchange(KL)
# outcomesKLl = BHStarExchange(KLN)

# outcomesK, outnames = isolateOutcome(K, conf)
# outcomesKL, outnames = isolateOutcome(KL, conf)
# outcomesKLN, outnames = isolateOutcome(KLN, conf)
# outcomesKLNAll, outnames = isolateOutcome(KLN, conf)

# paramScatter(K, 'ARChain',  'a_f1', 'e_f1', 'a$_{\mathrm{final}}$ [AU]', '1-e$_{\mathrm{final}}$')
# paramScatter(KL, 'PN terms',  'a_f1', 'e_f1', 'a$_{\mathrm{final}}$ [AU]', '1-e$_{\mathrm{final}}$')
# paramScatter(KLN, 'PN + tides',  'a_f1', 'e_f1', 'a$_{\mathrm{final}}$ [AU]', '1-e$_{\mathrm{final}}$')

# paramScatterMult([K,KL,KLN], ['ARChain','PN', 'PN+tides'],  'a_f1', 'e_f1_O', 'a$_{\mathrm{inner}}$ / a$_{\mathrm{outer}}$', '1-e$_{\mathrm{outer}}^2$')
# paramScatterMult([K,KL,KLN], ['ARChain','PN', 'PN+tides'],  'a1', 'a_f1', 'a$_{\mathrm{inital}}$ [AU]', 'a$_{\mathrm{final}}$ [AU]')
# paramScatterMult([K,KL,KLN], ['ARChain','PN', 'PN+tides'],  'e1', 'e_f1', 'e$_{\mathrm{initial}}$', 'e$_{\mathrm{final}}$')
# paramScatterMult([outcomesK,outcomesKL,outcomesKLl], ['ARChain','PN', 'PN+tides'],  'a1', 'e1', 'a$_{\mathrm{initial}}$', 'e$_{\mathrm{initial}}$')

def indexBoundTripleBHS(df, dfAll):
    a = df['a1']
    e = df['e1']
    aAll = dfAll['a1']
    eAll = dfAll['e1']

    index = np.zeros(len(a))

    for i in range(len(a)):
        maskA = (a[i] == aAll)
        maskE = (e[i] == eAll)
        # maskM = (df['m0'][i] == dfAll['m0'][i])
        # maskB = (df['bImp'][i] == dfAll['bImp'][i])
        index[i] = aAll[maskA & maskE].index.to_numpy()[0]

    return index


""" find index of bound triple BH-S inner """
# indexK = indexBoundTripleBHS(K, KAll)
# indexKL = indexBoundTripleBHS(KL, KLAll)
# indexKLl = indexBoundTripleBHS(KLN, KLNAll)

# smallestTimescaleK = timescaleK.nsmallest(10).index.to_numpy(dtype=int)
# smallestTimescaleKL = timescaleKL.nsmallest(10).index.to_numpy(dtype=int)
# smallestTimescaleKLl = timescaleKLl.nsmallest(10).index.to_numpy(dtype=int)


# indexSmallestKozaiK = indexK[smallestTimescaleK]
# indexSmallestKozaiKL = indexKL[smallestTimescaleKL]
# indexSmallestKozaiKLl = indexKLl[smallestTimescaleKLl]

# np.savetxt('index_boundTriple_BH-Star_KLl.txt', indexKLl, delimiter=',')



# plt.figure()
# bins = np.logspace(-4, 0, 50)
# plt.hist(1-KLN['e_f1']**2, label='inner', bins=bins, histtype='step', linewidth=5)
# plt.hist(1-KLN['e_f1_O']**2, label='outer', bins=bins, histtype='step', linewidth=5)
# plt.legend(loc='upper left')
# plt.xscale('log')
# plt.ylabel('Count')
# plt.xlabel('1-e$^2$')
# plt.title('Bound triple (BH-S)')

# paramScatterConf(K, 'ARChain', 'a1', 'e1', 'a$_0$ [AU]', 'e$_0$', 'Exchange')
# paramScatterConf(KL, 'PN', 'a1', 'e1', 'a$_0$ [AU]', 'e$_0$', 'Exchange')
# paramScatterConf(KLN, 'PN + tides', 'a1', 'e1', 'a$_0$ [AU]', 'e$_0$', 'Exchange')

# triplesBHInnerK = paramScatterConf(K, 'ARChain', 'a1', 'bImp', 'a$_0$ [AU]', 'b [AU]', 'all')
# triplesBHInnerKL = paramScatterConf(KL, 'PN', 'a1', 'bImp', 'a$_0$ [AU]', 'b [AU]', 'all')
# triplesBHInnerKLl = paramScatterConf(KLN, 'PN + tides', 'a1', 'bImp', 'a$_0$ [AU]', 'b [AU]', 'all')

# BHBHMergeK = paramScatterConf(K, 'ARChain', 'a1', 'bImp', 'a$_0$ [AU]', 'b [AU]', 'all')
# BHBHMergeKL = paramScatterConf(KL, 'PN', 'a1', 'bImp', 'a$_0$ [AU]', 'b [AU]', 'all')
# BHBHMergeKLl = paramScatterConf(KLN, 'PN + tides', 'a1', 'bImp', 'a$_0$ [AU]', 'b [AU]', 'all')

# initConfMergeK = K.iloc[BHBHMergeK.index]
# initConfMergeK = initConfMergeK[initConfMergeK['conf'] != '0 1 2']

# initConfMergeKL = KL.iloc[BHBHMergeKL.index]
# initConfMergeKL = initConfMergeKL[initConfMergeKL['conf'] != '0 1 2']

# initConfMergeKLl = KLN.iloc[BHBHMergeKLl.index]
# initConfMergeKLl = initConfMergeKLl[initConfMergeKLl['conf'] != '0 1 2']


# conf = ['Bound triple', 'Flyby']
# outcomesK, outnames = isolateOutcome(initConfMergeK, 'Exchange')
# outcomesKL, outnames = isolateOutcome(initConfMergeKL, 'Exchange')
# outcomesKLl, outnames = isolateOutcome(initConfMergeKLl, conf)

# indexK = triplesBHInnerK.index.to_numpy()
# indexKL = triplesBHInnerKL.index.to_numpy()
# indexKLl = triplesBHInnerKLl.index.to_numpy()

# intersectARC = np.intersect1d(indexKLl, indexK)
# intersectPN = np.intersect1d(indexKLl, indexKL)

# np.savetxt('boundTriplesBHStar_overlap_ARC.txt', intersectARC, delimiter=',')




# # triplesBHInnerKLl.to_pickle('boundTriplesBHStar_PNTides')

# bins = np.logspace(0, 17, 10)
# triplesBHInnerK['mergerTime'].hist(bins=bins, histtype='bar', label='ARChain')
# triplesBHInnerKL['mergerTime'].hist(bins=bins, histtype='bar', label='PN')
# triplesBHInnerKLl['mergerTime'].hist(bins=bins, histtype='bar', label='PN + tides')
# plt.xscale('log')
# plt.xlabel('Merger time [yr]')
# plt.ylabel('count')
# plt.grid(None)
# plt.legend()

# yValsK = np.full(len(triplesBHInnerK['mergerTime']), 1)
# yValsKL = np.full(len(triplesBHInnerKL['mergerTime']), 2)
# yValsKLl = np.full(len(triplesBHInnerKLl['mergerTime']), 3)

# plt.figure()
# plt.scatter(triplesBHInnerKLl['mergerTime'], yValsKLl)
# plt.scatter(triplesBHInnerK['mergerTime'], yValsK)
# plt.scatter(triplesBHInnerKL['mergerTime'], yValsKL)

# plt.xlim(1e-5, 5e21)
# plt.xscale('log')
# plt.yticks([1, 2, 3], ['ARChain', 'PN', 'PN + tides'])
# plt.xlabel('Merger time [yr]')
# plt.title('Bound triples (inner BH binary)')



# initParamAdd(K, 'radius')

# radii = np.array([KLN['r0'], KLN['r10'], KLN['r11']])
# locMaxObj = np.argmax(radii, axis=0)

# BHBH = KLN[locMaxObj == 0]
# BHStar = KLN[locMaxObj != 0]
# plt.figure()
# bins = np.logspace(-2, 5, 50)
# bins = np.linspace(0, 1, 20)
# weigthsK = np.ones(len(outcomesK)) / len(outcomesK)
# weigthsKL = np.ones(len(outcomesKL)) / len(outcomesKL)
# weigthsKLN = np.ones(len(outcomesKLN)) / len(outcomesKLN)

# weigthsBH = np.ones(len(BHBH)) / len(BHBH)
# weigthsStar = np.ones(len(BHStar)) / len(BHStar)

# weigthsAll = np.ones(len(outcomesKLNAll)) / len(outcomesKLNAll)
# plt.hist(outcomesK['e1'].astype(float), bins=bins, histtype='step', linewidth=2, label='ARChain', weights=weigthsK)
# plt.hist(outcomesKL['e1'].astype(float), bins=bins, histtype='step', linewidth=2, label='PN', weights=weigthsKL)
# plt.hist(outcomesKLN['e1'].astype(float), bins=bins, histtype='step', linewidth=2, label='PN + tides', weights=weigthsKLN)
# plt.hist(BHBH['a1'].astype(float), bins=bins, histtype='step', linewidth=2, label='BH-BH', weights=weigthsBH)
# plt.hist(BHStar['a1'].astype(float), bins=bins, histtype='step', linewidth=2, label='BH-Star', weights=weigthsStar)
# plt.legend(loc='upper left')
# plt.xlabel('a$_{initial}$ [AU]')
# plt.ylabel('Frac')
# plt.xscale('log')
# plt.title('Bound triples')

# paramScatter(outcomesK, 'ARChain', 'vInf', 'bImp', 'v$_{\infty}$/v$_c$', 'b/a')
# paramScatter(outcomesKL, 'PN terms', 'vInf', 'bImp', 'v$_{\infty}$/v$_c$', 'b/a')
# paramScatter(outcomesKLl, 'PN + tides', 'vInf', 'bImp', 'v$_{\infty}$/v$_c$', 'b/a')

# outcomesOldMergeNewK = KO.iloc[outcomesK.index]['conf']
# outcomesOldMergeNewKL = KLO.iloc[outcomesKL.index]['conf']
# outcomesOldMergeNewKLl = KLlO.iloc[outcomesKLl.index]['conf']

# np.savetxt('mergers_1BH_KLl_fromNew.txt', outcomesKLl.index.to_numpy(), delimiter=',')


conf = ['Bound merger']
# paramSpaceKTNew = initParamSpace(KL, conf)
# paramSpaceKLNewT = initParamSpace(KLTN, conf)

""" plot stuff of specific outcome"""
# massesOutcome(outcomes)
# paramHistSingle(outLowV, 'a1', 'a_{initial} [AU]', [0, 100, 100], 'Ionization')
# paramHistSingle(KLT, 'e1', 'e$_{initial}$', [0, 1, 100], 'Ionization')
# paramHistSingle(outcomes, 'vInf', 'v$_{\infty}$/v$_c$', [0, 6, 100])
# paramHistSingle(outcomes, 'dEE0', '$\Delta$E/E0', [0, 1e-3, 50])


def boundTripleExcOrCap(df):
    conf = df['conf']

    bin1 = conf.str.split(' ', expand=True)[0].str[-1]
    bin2 = conf.str.split(' ', expand=True)[1].str[0]

    numExch = np.sum((bin1 == '0'))

    print(numExch, len(df))

""" check if bound triples were exchange or capture """

# boundTripleExcOrCap(K)
# boundTripleExcOrCap(KL)
# boundTripleExcOrCap(KLN)


""" investigate triple mergers """
# conf = ['Triple merger']
# outcomes, outnames = isolateOutcome(ks0, conf)

# whichObjectsMergedFirst(outcomes)


""" find initial conf (BH vs star) """
# initialConf(outcomes)

""" find initial params """
# initialParams()

""" find which objects merge """
# mergerAna('Mergers', [ks0, ks1, K, KL, KLT], ['No regularisation', 'KS regularisation', 'ARChain', 'PN terms', 'Tides'])

""" Number of oscillations """
# compRuns('Nosc', 'Number of oscillations distribution', 0, 5, 1, 'Nosc', [ks0, ks1, K], ['No reg', 'KS reg', 'ARChain'])
#compRunsChanged('Nosc', 'Number of oscillations distribution', 0, 50, 1, 'Nosc')

""" Semi-major axis """
#compAll('a1', 'Normalized semi major axis distribution')
#compOutcome('a_f1', 'a distribution for different outcomes', 0, 4, 0.03, 'a')
# paramVS(K[K['a_f1'] != 0],'a1', 'a_f1', '')
# paramDiff([ks0, ks1], 'a_f1', 'a1', 10, 0.05, ['ks0', 'ks1'], '$\Delta$a [AU]', 'Difference in a')
#paramDiffConf([ks0, ks1, K], 'a_f1', 'a1', 5, 0.01, ['ks0', 'ks1', 'K'], '$\Delta$a [AU]', 'Difference in a')
#negDeltaA([ks0, ks1, K], ['ks0', 'ks1', 'K'])

#paramScatterConf(ks0, 'No reg', 'a1', 'a_f1', 'a$_{\mathrm{initial}}$ [AU]', 'a$_{\mathrm{final}}$ [AU]', 'Exchange')
#paramScatterConf(ks1, 'KS reg', 'a1', 'a_f1', 'a$_{\mathrm{initial}}$ [AU]', 'a$_{\mathrm{final}}$ [AU]', 'Exchange')
#paramScatterConf(K, 'ARChain', 'a1', 'a_f1', 'a$_{\mathrm{initial}}$ [AU]', 'a$_{\mathrm{final}}$ [AU]', 'Exchange')

# paramScatterConf(ks0, 'No reg', 'a_f1', 'e_f1', 'a$_{\mathrm{final}}$ [AU]', 'e$_{\mathrm{final}}$', 'Exchange')
#paramScatterConf(ks1, 'KS reg', 'a_f1', 'e_f1', 'a$_{\mathrm{final}}$ [AU]', 'e$_{\mathrm{final}}$', 'Exchange')
#paramScatterConf(K, 'ARChain', 'a_f1', 'e_f1', 'a$_{\mathrm{final}}$ [AU]', 'e$_{\mathrm{final}}$', 'Exchange')

""" eccentricity """
# compOutcome('e_f1', 'e distribution for different outcomes', 0, 0.99, 0.03, 'e$_{final}$')

# compInputConfMinusDouble('e1', 'e_f1', 'e distribution for different flags', 0, 0.99, 0.01, 'e', 0)


# paramScatterConf(K, 'ARChain', 'e1', 'e_f1', 'e$_{\mathrm{initial}}$', 'e$_{\mathrm{final}}$', 'Flyby')
# paramScatterConf(KL, 'PN terms', 'e1', 'e_f1', 'e$_{\mathrm{initial}}$', 'e$_{\mathrm{final}}$', 'Flyby')

# paramScatterConfMinus(K, 'ARChain', 'e1', 'e_f1', 'e$_{\mathrm{initial}}$', 'e$_{\mathrm{final}}$', 'Flyby')
# paramScatterConfMinus(KL, 'PN terms', 'e1', 'e_f1', 'e$_{\mathrm{initial}}$', 'e$_{\mathrm{final}}$', 'Flyby')

# paramScatterConfMinus(K, '', 'e1', 'a1', 'e$_{\mathrm{initial}}$', 'a$_{\mathrm{initial}} [AU]$', 'Exchange')

# paramScatterConfMinus(K, 'ARChain', 'e_f1', 'a_f1', 'e$_{\mathrm{final}}$', 'a$_{\mathrm{final}} [AU]$', 'Flyby')
# paramScatterConfMinus(KL, 'PN terms', 'e_f1', 'a_f1', 'e$_{\mathrm{final}}$', 'a$_{\mathrm{final}} [AU]$', 'Flyby')

# paramScatterConfXMinus(K, 'ARChain', 'e1', 'a1', 'e$_{\mathrm{initial}}$', 'a$_{\mathrm{initial}}$ [AU]', 'Flyby')
# paramScatterConfXMinus(KL, 'PN terms', 'e1', 'a1', 'e$_{\mathrm{initial}}$', 'a$_{\mathrm{initial}}$ [AU]', 'Flyby')

# paramVS(K[K['a_f1'] != 0],'e1', 'e_f1', 'ARChain')
# paramVSMinus(K[K['a_f1'] != 0],'e_f1', 'a_f1', 'ARChain')

# paramVSMinus(KL[KL['a_f1'] != 0], 'e1', 'e_f1', 'ARChain', '1 - e$_{\mathrm{initial}}$', '1 - e$_{\mathrm{final}}$')
# paramVSMinus(K[K['a_f1'] != 0], 'e1', 'e_f1', 'ARChain', '1 - e$_{\mathrm{initial}}$', '1 - e$_{\mathrm{final}}$')
# paramVSMinus(K[K['a_f1'] != 0], 'e1', 'a1', '', '1 - e$_{\mathrm{initial}}$', 'a$_{\mathrm{initial}}$ [AU]')
# paramVSMinus(K[K['a_f1'] != 0], 'e_f1', 'a_f1', 'ARChain', '1 - e$_{\mathrm{final}}$', 'a$_{\mathrm{final}}$ [AU]')
# paramVSMinus(KL[KL['a_f1'] != 0], 'e_f1', 'a_f1', 'PN', '1 - e$_{\mathrm{final}}$', 'a$_{\mathrm{final}}$ [AU]')

# paramSpaceK = paramComps('e1', 'e_f1', K, 'ARChain', 'e$_{initial}$', 'e$_{final}$')
# paramSpaceKL = paramComps('e1', 'e_f1', KL, 'PN', 'e$_{initial}$', 'e$_{final}$')

# paramComp('e1', 'e_f1', K, ['ARChain', 'PN'], 'e$_{initial}$', 'e$_{final}$')

# paramCompsMult('e1', 'e_f1', [K, KL], ['ARChain', 'PN'], 'e$_{initial}$', 'e$_{final}$')

# paramScatterConfColor(K, 'ARChain', 'e1', 'bImp', 'rMin', 'e$_{\mathrm{initial}}$', 'Impact parameter [AU]')

# compOutcome('e_f1', '', 0, 1, 0.01, 'b/a',  [ks0, ks1, K], ['No reg', 'KS reg', 'ARChain'])

#%%
# firstks0 = ks0.iloc[0:100]
# firstks1 = ks1.iloc[0:100]
# firstK = K.iloc[0:100]
# firstKT = KT.iloc[0:100]
# firstKL = KL.iloc[0:100]
# firstKLT = KLT.iloc[0:100]

# first=[firstks0.iloc[0], firstks1.iloc[0], firstK.iloc[0], firstKT.iloc[0], firstKL.iloc[0], firstKLT.iloc[0]]





#%%

""" find number of binaries that circularize """
# conf = ['Flyby']
# outcomesK, outnames = isolateOutcome(K, conf)

# higherE = outcomesK[outcomesK['e1'].astype(float) < outcomesK['e_f1'].astype(float)]
# lowerE = outcomesK[outcomesK['e1'].astype(float) > outcomesK['e_f1'].astype(float)]
# equalE = outcomesK[outcomesK['e1'].astype(float) == outcomesK['e_f1'].astype(float)]

# fracHigher = len(higherE)/len(outcomesK)
# fraclower = len(lowerE)/len(outcomesK)




""" filter out BBHs """
# bbhs = bbhFilter(K2)
# BHStar = BHStarFilter(K2)

# """ merger time due to GW radiation """
# conf = ['Flyby', 'Exchange']
# outcomesK2, outnames = isolateOutcome(bbhs, ['Flyby', 'Exchange'])
# outExchange, outnames = isolateOutcome(BHStar, ['Flyby', 'Exchange'])

# outcomesK, outnames = isolateOutcome(K, ['Flyby', 'Exchange'])

# # outExchange, outnames = isolateOutcome(BHStar, 'Exchange')

# outcomesK2 = pd.concat([outcomesK2, outExchange], axis=0)


# # timeKBefore = timeMergeGW(outcomesK, conf, 'Before')
# timeKAfter = timeMergeGW(outcomesK, ['Flyby', 'Exchange'], 'After')
# timeK2After = timeMergeGW(outcomesK2, ['Flyby', 'Exchange'], 'After')


# hubbleMerge = outcomesK[timeKAfter < 14e3]
# hubbleMerge2 = outcomesK2[timeK2After < 14e3]

# indHubbleMerge2 = outcomesK2[timeK2After < 14e3].index.to_numpy()
# indHubbleMerge = outcomesK[timeKAfter < 14e3].index.to_numpy()


# # # # np.savetxt('mergers_index_3BH.txt', indHubbleMerge, delimiter=',')

# def strip_spaces(a_str_with_spaces):
#     return a_str_with_spaces.replace(' ', '')

con# data = pd.read_csv('3BH-merging-match.dat', engine='python', sep='binsingle', names=['0', '1'], converters={'1': strip_spaces})
# dat2 = pd.read_csv('./inputs/mergers_input_3BH.dat', engine='python', sep='binsingle', names=['0', '1'], converters={'1': strip_spaces})

# data_2 = pd.read_csv('2BH-merging-match.dat', engine='python', sep='binsingle', names=['0', '1'], converters={'1': strip_spaces})
# dat2_2 = pd.read_csv('./inputs/mergers_input_2BH.dat', engine='python', sep='binsingle', names=['0', '1'], converters={'1': strip_spaces})


# mergerTime = timeKAfter[indHubbleMerge]
# mergerTime2 = timeK2After[indHubbleMerge2]


# data['1'] = data['1'].str.strip()
# dat2['1'] = dat2['1'].str.strip()

# data_2['1'] = data_2['1'].str.strip()
# dat2_2['1'] = dat2_2['1'].str.strip()

# match = dat2['1'].isin(data['1']).to_numpy()
# match_2 = dat2_2['1'].isin(data_2['1']).to_numpy()


# hubbleMerge = hubbleMerge[match]
# hubbleMerge2 = hubbleMerge2[match_2]

# mergerTime = mergerTime[match]
# mergerTime2 = mergerTime2[match_2]


# times = pd.read_csv('3BH-merging-match.dat', engine='python', sep='\s+', names=['0', '1'], usecols=[0,1])['1']
# times2 = pd.read_csv('2BH-merging-match.dat', engine='python', sep='\s+', names=['0', '1'], usecols=[0,1])['1']


# binaries = hubbleMerge['conf'].str.split('[', expand=True)
# bin1 = binaries[1].str[0]
# bin2 = binaries[1].str[2]

# binaries2 = hubbleMerge2['conf'].str.split('[', expand=True)
# bin1_2 = binaries2[1].str[0]
# bin2_2 = binaries2[1].str[2]


# # def chirpMass(m1, m2):
# #     return (m1*m2)**(3/5) / (m1 + m2)**(1/5)

# # chirpMassVec = np.ones(len(bin1))
# totalMass = np.ones(len(bin1))
# # chirpMassVec_2 = np.ones(len(bin1_2))
# totalMass_2 = np.ones(len(bin1_2))

# m1V_1 = np.ones(len(bin1))
# m2V_1 = np.ones(len(bin1))

# for i in range(len(bin1)):
#     m1T = 0
#     m2T = 0
#     if bin1.iloc[i] == '0':
#         m1T = hubbleMerge['m0'].astype(float).iloc[i]
#     elif bin1.iloc[i] == '1':
#         m1T = hubbleMerge['m10'].astype(float).iloc[i]
#     else:
#         continue
#     if bin2.iloc[i] == '1':
#         m2T = hubbleMerge['m10'].astype(float).iloc[i]
#     elif bin2.iloc[i] == '2':
#         m2T = hubbleMerge['m11'].astype(float).iloc[i]
#     else:
#         continue
#         # m1V_1[i] = m1T
#         # m2V_1[i] = m2T
#     if m1T > m2T:
#         m1V_1[i] = m1T
#         m2V_1[i] = m2T
#     else:
#         m1V_1[i] = m2T
#         m2V_1[i] = m1T

#     # chirpMassVec[i] = chirpMass(m1T, m2T)
#     totalMass[i] = m1T + m2T


# m1V_2 = np.ones(len(bin2_2))
# m2V_2 = np.ones(len(bin2_2))

# for i in range(len(bin1_2)):
#     m1T = 0
#     m2T = 0
#     if bin1_2.iloc[i] == '0':
#         m1T = hubbleMerge2['m0'].astype(float).iloc[i]
#     elif bin1_2.iloc[i] == '1':
#         m1T = hubbleMerge2['m10'].astype(float).iloc[i]
#     else:
#         continue
#     if bin2_2.iloc[i] == '1':
#         m2T = hubbleMerge2['m10'].astype(float).iloc[i]
#     elif bin2_2.iloc[i] == '2':
#         m2T = hubbleMerge2['m11'].astype(float).iloc[i]
#     else:
#         continue

#     # m1V_2[i] = m1T
#     # m2V_2[i] = m2T
#     if m1T > m2T:
#         m1V_2[i] = m1T
#         m2V_2[i] = m2T
#     else:
#         m1V_2[i] = m2T
#         m2V_2[i] = m1T


#     # chirpMassVec_2[i] = chirpMass(m1T, m2T)
#     totalMass_2[i] = m1T + m2T
#%%


# massLIGO = pd.read_csv('bh-mergers-ligo-virgo.dat', sep='\s+', names=['Name', 'm1', 'm2', 'totM'])

# totMass = m1V_1 + m2V_1
# totMass2 = m1V_2 + m2V_2
# totMassLigo = massLIGO['totM']

# # yLigo = np.ones(len(totMassLigo))
# # yLigo[:] =

# yLigo = [0.5, 0.52, 0.52, 0.5, 0.5, 0.52, 0.52, 0.48, 0.46, 0.48, 0.5]

# # y=np.linspace(0,0.4,3)

# import matplotlib.patches as patches
# from matplotlib.collections import PatchCollection

# posErr = [3.3, 9.9, 6.4, 5.2, 3.2, 14.6, 5.2, 3.2, 4.8, 9.4, 4]
# negErr = [3, 3.8, 1.5, 3.5, 0.7, 10.2, 3.7, 2.4, 3.8, 6.6, 3.7]
# massErrors = pd.DataFrame((posErr, negErr)).T
# massErrors.columns =  ['pos', 'neg']


# weigths = np.ones(len(totMass[totMass < 14000])) / len(totMass[totMass < 14000])
# weigths2 = np.ones(len(totMass2[totMass2 < 14000])) / len(totMass2[totMass2 < 14000])

# plt.figure()
# plt.hist(totMass, color='blue', histtype='step', label='3 BH', linewidth=2, weights=weigths)
# plt.hist(totMass2, color='green', histtype='step' , label='2 BH', linewidth=2, weights=weigths2)

# plt.plot([-1,-0.5],[-1,-0.5],  c='black', label='11 LIGO/Virgo Detections')
# plt.vlines(totMassLigo, ymin=0, ymax=1, color='black', zorder=10)
# errBoxes = []
# for i in range(len(totMassLigo)):
#     # rect = patches.Rectangle((totMassLigo.iloc[i]-massErrors['neg'].iloc[i], 0), width=massErrors.iloc[i].sum(), height=1)
#     # errBoxes.append(rect)
#     if i == 0:
#         plt.errorbar(totMassLigo, yLigo, xerr=(negErr, posErr), linestyle='none', c='black', capsize=2, markersize=7, marker='o', label='11 LIGO/Virgo Detections')
#     else:
#         plt.errorbar(totMassLigo, yLigo, xerr=(negErr, posErr), linestyle='none', c='black', capsize=2, markersize=7, marker='o')

# plt.errorbar(totMassLigo, yLigo, xerr=(negErr, posErr), linestyle='none', c='black', capsize=2, markersize=7, marker='o', label='11 LIGO/Virgo Detections')

# ax = plt.gca()
# pc = PatchCollection(errBoxes, facecolor='r', alpha=0.5, edgecolor='None')
# ax.add_collection(pc)

# plt.legend()
# plt.xlabel('Total mass [M$_{\odot}$]')
# plt.ylabel('Fraction')
# plt.ylim(0,0.55)

# m1PErr = [4.8, 14, 8.8, 7.3, 5.3, 16.6, 8.3, 5.7, 7.5, 10, 5.0]
# m1NErr = [3.0, 5.5, 3.2, 5.6, 1.7, 10.2, 6, 3, 4.7, 6.6, 5.3]

# m2PErr = [3.0, 4.1, 2.2, 4.9, 1.3, 9.1, 5.2, 2.9, 4.3, 6.3, 1.8]
# m2NErr = [4.4, 4.8, 2.6, 4.5, 2.1, 10.1, 5.1, 4.1, 5.2, 7.1, 1]


# plt.figure()
# plt.scatter(m1V_1, m2V_1, label='3 BHs', c='blue')
# plt.scatter(m1V_2, m2V_2, label='2 BHs', c='green')
# # plt.axvline(massLIGO['m1'].astype(float) + massLIGO['m2'].astype(float))
# # plt.scatter(massLIGO['m1'], massLIGO['m2'], label='11 LIGO/Virgo Detections', c='red')
# plt.errorbar(massLIGO['m1'], massLIGO['m2'], yerr=(m2NErr, m2PErr), xerr=(m1NErr, m1PErr), linestyle='none', c='red', capsize=2, markersize=7, marker='o', label='11 LIGO/Virgo Detections')

# plt.ylabel('M$_2$ [M$_{\odot}$]')
# plt.xlabel('M$_1$ [M$_{\odot}$]')
# plt.legend()


# timesDelay = times.to_numpy() + mergerTime.to_numpy()
# timesDelay2 = times2.to_numpy() + mergerTime2.to_numpy()[:-1]


# plt.figure()
# plt.scatter(timesDelay, chirpMassVec, s=7, label='3BH', c='blue')
# plt.scatter(timesDelay2, chirpMassVec_2[:-1], s=7, label='2BH', c='green')
# plt.xlabel('Delay time [Myr]')
# plt.ylabel('Chirp mass [$M_{\odot}$]')
# plt.xscale('log')
# plt.xlim([5, 14e3])
# plt.legend()

# bins = np.linspace(0, 14000, 50)
# weigths = np.ones(len(timesDelay[timesDelay < 14000])) / len(timesDelay[timesDelay < 14000])
# weigths2 = np.ones(len(timesDelay2[timesDelay2 < 14000])) / len(timesDelay2[timesDelay2 < 14000])
# plt.figure()
# plt.hist(timesDelay[timesDelay < 14000], histtype='step', linewidth=2, label='3 BH', bins=bins, color='blue', weights=weigths)
# plt.hist(timesDelay2[timesDelay2 < 14000], histtype='step', linewidth=2, label='2 BH', bins=bins, color='green', weights=weigths2)
# plt.legend()
# plt.xlabel('Delay time [Myr]')
# plt.ylabel('Fraction')
# plt.title('3 BH')

# plt.figure()
# plt.scatter(timesDelay, totalMass, s=0.6)
# plt.xlabel('Delay time [Myr]')
# plt.ylabel('Total binary mass [$M_{\odot}$]')
# plt.title('3 BH')









# frame = { 'Before': timeKBefore, 'After': timeKAfter}

# timeDF = pd.DataFrame(frame, columns=['Before', 'After'])

# outcomesKL, outnames = isolateOutcome(KL, conf)

# timeKBefore = timeMergeGW(outcomesKL, conf, 'Before')
# timeKAfter = timeMergeGW(outcomesKL, conf, 'After')



# frame = { 'Before': timeKBefore, 'After': timeKAfter}

# timeDFKL = pd.DataFrame(frame, columns=['Before', 'After'])

# reducedT = outcomesK[timeDF['Before'] > timeDF['After']]
# increasedT = outcomesK[timeDF['Before'] < timeDF['After']]

# numIncreasedEK = np.sum(outcomesK['e1'].astype(float) < outcomesK['e_f1'].astype(float))/len(outcomesK)
# numReducedEK = np.sum(outcomesK['e1'].astype(float) > outcomesK['e_f1'].astype(float))/len(outcomesK)

# numIncreasedEKL = np.sum(outcomesKL['e1'].astype(float) < outcomesKL['e_f1'].astype(float))/len(outcomesKL)
# numReducedEK = np.sum(outcomesKL['e1'].astype(float) > outcomesKL['e_f1'].astype(float))/len(outcomesKL)


# fracRedT10 = outcomesK[timeDF['Before'] / timeDF['After'] < 0.9]
# fracIncT10 = outcomesK[timeDF['Before'] / timeDF['After'] > 1.1]


# compRuns('rMin', 'ARChain', 0, 100, 1, 'R$_{\mathrm{min}}$ [R$_{\odot}]$', [reducedT, increasedT], ['Decreased t', 'Increased t'])

# beforeNumMergers = numMergeTime(timeDF, 'Before')
# afterNumMergers = numMergeTime(timeDF, 'After')

# print(afterNumMergers)

# hubbleTime = 14e9/1e6

# numMergeHubbleAfter = len(timeDF[(timeDF['After'] < hubbleTime)])
# numMergeHubbleBefore = len(timeDF[(timeDF['Before'] < hubbleTime)])

# numMergeHubbleAfterKL = len(timeDFKL[(timeDFKL['After'] < hubbleTime)])
# numMergeHubbleBeforeKL = len(timeDFKL[(timeDFKL['Before'] < hubbleTime)])



# numMergeHubbleAfter = len(timeDF[(timeDF['After'] < hubbleTime) & (timeDF['Before'] > hubbleTime)])
# numMergeHubbleBefore = len(timeDF[(timeDF['After'] > hubbleTime) & (timeDF['Before'] < hubbleTime)])


# plt.figure()
# periDistanceI = outcomesK['a1'].astype(float) * (1 - outcomesK['e1'].astype(float))
# periDistance = outcomesK['a_f1'].astype(float) * (1 - outcomesK['e_f1'].astype(float))
# plt.scatter(periDistanceI, timeDF['Before'], s=0.6, alpha=0.3, label='Initial')
# plt.scatter(periDistance, timeDF['After'], s=0.6, alpha=0.3, label='Final')
# plt.axhline(y=hubbleTime/1e6, c='black', linestyle='--')
# plt.legend(markerscale=20)
# plt.yscale('log')
# plt.xlabel('Pericenter distance [AU]')
# plt.ylabel('Merger time [Myr]')
# plt.ylim(1e-17, 5e11)
# plt.xlim(-0.5, 8)
# plt.title('ARChain')


# plt.figure()
# binsLog = np.logspace(0, 12, 150)
# plt.hist(timeDF['Before'], bins=binsLog, histtype='step', label='Initial')
# plt.hist(timeDF['After'], bins=binsLog, histtype='step', label='ARChain')
# plt.hist(timeDFKL['After'], bins=binsLog, histtype='step', label='PN')
# plt.xscale('log')
# plt.ylabel('Count')
# plt.xlabel('Merger time [Myr]')
# plt.axvline(x=hubbleTime/1e6, zorder=2, c='black', linestyle='--', label='Hubble time')
# plt.legend(loc='upper right')
# plt.title('$e_{initial}$ > 0.95')

# plt.figure()
# binsLog = np.logspace(-1,4, 200)

# histo1 = np.histogram(K['vInf'].astype(float)*K['vCrit Unit'].astype(float), bins=binsLog)
# plt.hist(K['vInf'].astype(float)*K['vCrit Unit'].astype(float), bins=binsLog)
# plt.xscale('log')
# plt.ylabel('Count')
# plt.xlabel('Impact parameter [AU]')


# paramScatterConfMinus(reducedT, 'ARChain', 'e_f1', 'a_f1', 'e$_{\mathrm{final}}$', 'a$_{\mathrm{final}} [AU]$', conf[0])
# paramScatterConfMinus(K, 'ARChain', 'e_f1', 'a_f1', 'e$_{\mathrm{final}}$', 'a$_{\mathrm{final}} [AU]$', conf[0])

# paramScatterConfMinusDouble([reducedT, increasedT], ['ARChain','ARChain'], 'e_f1', 'a_f1', 'e$_{\mathrm{final}}$', 'a$_{\mathrm{final}} [AU]$', conf[0])
# paramScatterConfMinusInitialAfter(outcomesK, ['ARChain'], 'e1', 'a1', 'e_f1', 'a_f1', 'Initial', 'Final', 'e', 'a [AU]', conf[0])
# paramScatterConfMinusInitialOrAfter(outcomesK, ['Initial'], 'e1', 'a1', 'Initial', 'Final', 'e', 'a [AU]', conf[0])
# paramScatterConfMinusInitialOrAfter(outcomesK, ['ARChain'], 'e_f1', 'a_f1', 'Final', 'Initial', 'e', 'a [AU]', conf[0])
# paramScatterConfMinusInitialOrAfter(outcomesKL, ['PN'], 'e_f1', 'a_f1', 'Final', 'Initial', 'e', 'a [AU]', conf[0])

def mergesHubbleTime(df, conf):
    bbhs = bbhFilter(df)
    outcomesK, outnames = isolateOutcome(bbhs, conf)

    timeKAfter = timeMergeGW(outcomesK, conf, 'After')

    hubbleTime = 14e9/1e6
    mergeHubble = timeKAfter[(timeKAfter < hubbleTime)]
    return outcomesK.loc[mergeHubble.index]

""" parameters of black hole that merger within Hubble time """
# mergersK = mergesHubbleTime(K, 'Flyby')
# mergersKL = mergesHubbleTime(KL, 'Flyby')
# mergersKLT = mergesHubbleTime(KLT, 'Flyby')



# plt.figure()
# plt.hist(mergersK['m10'].astype(float), histtype='step', label='Component 1')
# plt.hist(mergersK['m11'].astype(float), histtype='step', label='Component 2')
# plt.ylabel('Count')
# plt.xlabel('Mass $M_{\odot}$')
# plt.legend()

# plt.figure()
# plt.hist(mergersK['e_f1'].astype(float), bins=50, histtype='step', label='ARChain')
# plt.hist(mergersK['e_f1'].astype(float), bins=50, histtype='step', label='PN')
# plt.hist(mergersK['e_f1'].astype(float), bins=50, histtype='step', label='PN + tides')
# plt.ylabel('Count')
# plt.xlabel('Eccentricity')
# plt.legend(loc='upper left')

# plt.figure()
# bins=np.linspace(0,0.5, 25)
# plt.hist(mergersK['a_f1'].astype(float), bins=bins, histtype='step', label='ARChain')
# plt.hist(mergersK['a_f1'].astype(float), bins=bins, histtype='step', label='PN')
# plt.hist(mergersK['a_f1'].astype(float), bins=bins, histtype='step', label='PN + tides')
# plt.ylabel('Count')
# plt.xlabel('a [AU]')
# plt.legend()





""" accuracy """
# compRuns('e_f1', '$e_{final}$ distribution for ks1', 0, 0.99, 0.01, '$e_{final}$', [ks1, ks1H, ks1L], ['1e-9', '1e-15', '1e-5'])
# compRuns('e_f1', '$e_{final}$ distribution for ks1', 0, 0.99, 0.01, '$e_{final}$', [ks0, ks0H, ks0L], ['1e-9', '1e-15', '1e-5'])
# compRuns('a_f1', '$a_{final}$ distribution for ks0', 0, 2.5, 0.01, '$a_{final}$', [ks1, ks1H, ks1L], ['1e-9', '1e-15', '1e-5'])
# compRuns('a_f1', '$a_{final}$ distribution for ks0', 0, 2.5, 0.01, '$a_{final}$', [ks0, ks0H, ks0L], ['1e-9', '1e-15', '1e-5'])

# compRuns('dEE0', 'dE/E0 distribution for KS reg', 1e-15, 1, 0.01, 'dE/E0', [ks1H, ks1, ks1L], ['1e-15', '1e-9', '1e-5'])
# compRuns('dLL0', 'dL/L0 distribution for KS reg', 1e-15, 1, 0.01, 'dL/L0', [ks1H, ks1, ks1L], ['1e-15', '1e-9', '1e-5'])
# compRuns('t cpu', 'Computation time for KS reg', 0, 20, 0.1, 't cpu', [ks1, ks1H, ks1L], ['Default', 'High', 'Low'])
# compRuns('t final', 'Computation time for KS reg', 0, 1000, 10, 't cpu', [ks1, ks1H, ks1L], ['Default', 'High', 'Low'])

# conf = ['Exchange']
# outcomesDefault, outnames = isolateOutcome(ks0, conf)
# outcomesHigh, outnames = isolateOutcome(ks0H, conf)
# outcomesLow, outnames = isolateOutcome(ks0L, conf)

# plt.figure()
# bins = np.linspace(0,1000, 100)
# # bins = np.logspace(-2, np.log10(5), 20)
# plt.hist(ks1H['t final'].astype(float), label='1e-15', bins=bins, histtype='step', linewidth=2)
# plt.hist(ks1['t final'].astype(float), label='1e-9', bins=bins, histtype='step', linewidth=2)
# plt.hist(ks1L['t final'].astype(float), label='1e-5', bins=bins, histtype='step', linewidth=2)
# # plt.hist(K['t cpu'].astype(float), label='ARChain', bins=bins, histtype='step')
# plt.legend(title='Accuracy =')
# # plt.xscale('log')
# plt.title('KS regularization')
# plt.ylabel('Count')
# plt.xlabel('Physical time [yr]')

# print('# interactions where t_cpu > 5: \nks0: \nDefault: ' + str(len(ks0[ks0['t cpu'].astype(float) > 5]))
#       + '\nHigh: ' + str(len(ks0H[ks0H['t cpu'].astype(float) > 5])) + '\nLow: ' + str(len(ks0L[ks0L['t cpu'].astype(float) > 5])))

# print('# interactions where t_cpu > 5: \nks1: \nDefault: ' + str(len(ks1[ks1['t cpu'].astype(float) > 5]))
#       + '\nHigh: ' + str(len(ks1H[ks1H['t cpu'].astype(float) > 5])) + '\nLow: ' + str(len(ks1L[ks1L['t cpu'].astype(float) > 5])))

# ks0HTime = ks0H['t cpu'][ks0H['t cpu'].astype(float) < 5].astype(float)
# ks0LTime = ks0L['t cpu'][ks0L['t cpu'].astype(float) < 5].astype(float)
# ks1HTime = ks1H['t cpu'][ks1H['t cpu'].astype(float) < 5].astype(float)
# ks1LTime = ks1L['t cpu'][ks1L['t cpu'].astype(float) < 5].astype(float)


# print('High: ' + str(np.mean(ks0HTime)) + ', ' + str(np.median(ks0HTime)))
# print('Low: ' + str(np.mean(ks0LTime)) + ', ' + str(np.median(ks0LTime)) + '\n')

# print('High: ' + str(np.mean(ks1HTime)) + ', ' + str(np.median(ks1HTime)))
# print('Low: ' + str(np.mean(ks1LTime)) + ', ' + str(np.median(ks1LTime)) + '\n')

# compRuns('e_f1', '$e_{final}$ distribution for handmade', 0, 0.99, 0.01, '$e_{final}$', [K, KL, KLT], ['ARChain', 'PN', 'PN + tides'])
# compRuns('a_f1', '$a_{final}$ distribution for ks0', 0, 5, 0.01, '$a_{final}$', [ks0, ks0H, ks0L], ['Default', 'High', 'Low'])
# compRuns('e_f1', '$e_{final}$ distribution', 0.9, 1, 0.001, '$e_{final}$', [K, KL], ['ARChain', 'PN', 'PN + tides'])


#compRunsChanged('e1', 'Normalized eccentricity distribution', 0, 1, 0.01, 'e')
#compOutcome('e_f1', 'e distribution for different outcomes', 0, 0.99, 0.03, 'e$_f$')
# compInputConf('e_f1', 'e distribution for different flags', 0, 0.99, 0.01, 'e$_{final}^2$')
# compInputConfFrac('e_f1', 'e distribution for different flags', 0, 0.99, 0.01, 'e$_{final}^2$')


# paramVS(ks0[ks0['a_f1'] != 0],'e1', 'e_f1', 'No reg')
# paramVS(ks1[ks1['a_f1'] != 0],'e1', 'e_f1', 'KS reg')
# paramVS(K[K['a_f1'] != 0],'e1', 'e_f1', 'ARChain')

# paramVS(ks0[ks0['a_f1'] != 0],'e1', 'e_f1', 'No reg')
# paramVS(ks1[ks1['a_f1'] != 0],'e1', 'e_f1', 'KS reg')

# paramVS(ks0H[ks0H['a_f1'] != 0],'e1', 'e_f1', 'ks0 (high acc)')
# paramVS(ks1H[ks1H['a_f1'] != 0],'e1', 'e_f1', 'ks1 (high acc)')

# paramVS(ks0L[ks0L['a_f1'] != 0],'e1', 'e_f1', 'ks0 (low acc)')
# paramVS(ks1L[ks1L['a_f1'] != 0],'e1', 'e_f1', 'ks1 (low acc)')
# paramVS(K[K['a_f1'] != 0],'e1', 'e_f1', 'K')

# paramVS(outLowV,'e1', 'e_f1', 'KLT Ion Vinf < 1')

#paramDiff([ks0, ks1, K], 'e_f1', 'e1', 1, 0.01, ['ks0', 'ks1', 'K'], '$\Delta$e', 'Difference in e')
#paramDiffConf([ks0, ks1, K], 'e_f1', 'e1', 1, 0.05, ['ks0', 'ks1', 'K'], '$\Delta$e', 'Difference in e')

# paramScatterConf(ks0, 'No reg', 'a_f1', 'e_f1', 'a$_{\mathrm{final}}$ [AU]', 'e$_{\mathrm{final}}$', 'Exchange')
# paramScatterConf(ks1, 'KS reg', 'a_f1', 'e_f1', 'a$_{\mathrm{final}}$ [AU]', 'e$_{\mathrm{final}}$', 'Exchange')
# paramScatterConf(K, 'ARChain', 'a_f1', 'e_f1', 'a$_{\mathrm{final}}$ [AU]', 'e$_{\mathrm{final}}$', 'Exchange')

# paramScatterConf(ks0H, 'Acc = 1e-15', 'e1', 'e_f1', 'e$_{\mathrm{initial}}^2$', 'e$_{\mathrm{final}}^2$', 'Exchange')
# paramScatterConf(ks0L, 'Acc = 1e-5', 'e1', 'e_f1', 'e$_{\mathrm{initial}}^2$', 'e$_{\mathrm{final}}^2$', 'Exchange')

# compRuns('dEE0', 'dE/E0 distribution for 2BH', 1e-15, 1, 0.01, 'dE/E0', [ks0, ks1, K, KL, KT, KLT], ['No reg', 'KS reg', 'ARChain', 'PN terms', 'Tides', 'PN + tides'])
# compRuns('dLL0', 'dL/L0 distribution for 2BH', 1e-15, 1, 0.01, 'dE/E0', [ks0, ks1, K, KL, KT, KLT], ['No reg', 'KS reg', 'ARChain', 'PN terms', 'Tides', 'PN + tides'])

# compRuns('dEE0', 'dE/E0 distribution for 3BH', 1e-15, 1, 0.01, 'dE/E0', [ks0, ks1, K, KL], ['No reg', 'KS reg', 'ARChain', 'PN terms'])
# compRuns('dLL0', 'dL/L0 distribution for 3BH', 1e-15, 1, 0.01, 'dE/E0', [ks0, ks1, K, KL], ['No reg', 'KS reg', 'ARChain', 'PN terms'])


""" rMin """
# compRuns('rMin', 'R$_{min}$ for 2b KS reg', 0, 100, 0.1, 'R$_{min}$ [R$_{\odot}$]', [ks1, ks1H, ks1L, K], ['1e-8', '1e-15', '1e-5', 'ARChain'])
# compRuns('rMin', 'R$_{min}$ for 2b No reg', 0, 100, 0.1, 'R$_{min}$ [R$_{\odot}$]', [ks0, ks0H, ks0L, K], ['1e-8', '1e-15', '1e-5', 'ARChain'])
# compRuns('rMin', 'R$_{min}$ for 3BH select', 0, 100, 0.1, 'R$_{min}$ [R$_{\odot}$]', [K, KL], ['ARChain', 'PN'])

# compRuns('rMin', 'R$_{min}$ for 2b', 0, 100, 0.1, 'R$_{min}$ [R$_{\odot}$]', [ks0, ks1, K], ['No reg','KS reg', 'ARChain'])


""" investigate peaks in rMin (2b) """
# peaksks1 = rMinPeaks(ks0, 10)
# peaksks1 = rMinPeaks(ks1, 0.5)
# peaksks0H = rMinPeaks(ks1H, 0.5)
# peaksks0L = rMinPeaks(ks1L, 0.5)

# confSimple('Outcomes for Rmin peaks (KS reg)', [peaksks0, peaksks0H, peaksks0L], ['1e-8', '1e-15', '1e-5'])
# confSimple('Outcomes for 2b', [ks0, peaksks1], ['No reg (all)', 'No reg (peaks in R$_{\min}$)'])
# confSimple('Outcomes for 2b', [ks1, peaksks1], ['KS reg (all)', 'KS reg (peaks in R$_{\min}$)'])

# compRuns('e_f1', '$e_{final}^2$ distribution for 2b', 0, 0.99, 0.01, '$e_{final}^2$', [ks0, peaksks1], ['No reg (all)', 'No reg (peaks in R$_{\min}$)'])
# compRuns('e_f1', '$e_{final}^2$ distribution for 2b', 0, 0.99, 0.01, '$e_{final}^2$', [ks1, peaksks1], ['KS reg (all)', 'KS reg (peaks in R$_{\min}$)'])

# paramScatter(ks0, 'No reg', 'e1', 'bImp', 'e$_{\mathrm{initial}}$', 'Impact parameter [1/a]')
# paramScatter(peaksks1, 'No reg', 'e1', 'bImp', 'e$_{\mathrm{initial}}$', 'Impact parameter [1/a]')

# paramScatterConfColor(peaksks1, 'No reg', 'e1', 'bImp', 'rMin', 'e$_{\mathrm{initial}}$', 'Impact parameter [AU]')

# bins = np.arange(0,100,0.1)
# periDist = ks0['a1'].astype(float)*215 * (1-ks0['e1'].astype(float))
# weightsPeri = np.ones(np.shape(periDist)) / len(periDist)
# weightsPeaks = np.ones(np.shape(peaksks1['rMin'].astype(float))) / len(peaksks1['rMin'].astype(float))

# plt.figure()
# plt.hist(periDist, histtype='step', label='Pericenter distance', bins=bins)
# plt.hist(ks0['rMin'].astype(float)*215, histtype='step', label='Rmin', bins=bins, rwidth=1)
# plt.legend()

# conf = ['Flyby']
# outcomesF, outnames = isolateOutcome(ks0, conf)
# conf = ['Exchange']
# outcomesE, outnames = isolateOutcome(ks0, conf)


# compRuns('rMin', '', 0, 100, 0.1, 'R$_{min}$ [R$_{\odot}$]', [ks0, outcomesF, outcomesE], ['All', 'Flyby', 'Exchange'])



# compRuns('Nosc', 'Number of oscillations (No reg)', 0, 50, 1, 'Nosc', [ks0, peaksks0], ['All', 'Rmin peaks', 'ARChain'])
# compRuns('Nosc', 'Number of oscillations (KS reg)', 0, 50, 1, 'Nosc', [ks1, peaksks1], ['All', 'Rmin peaks', 'ARChain'])

# print('ks0 (all): ' + str(np.mean(ks0['Nosc'].astype(float))))
# print('ks0 (peaks): ' + str(np.mean(peaksks1['Nosc'].astype(float))))

# print('\nks1 (all): ' + str(np.mean(ks1['Nosc'].astype(float))))
# print('ks1 (peaks): ' + str(np.mean(peaksks1['Nosc'].astype(float))))


""" physical time """
#compRuns('t final', '$t_{final}$ distribution', 0, 1e5, 1, 't [yr]')
#compRunsDouble('t final', '$t_{final}$ distribution', 0, 3e3, 10, 't [yr]')

#compRunsChanged('t final', 'Integration time distribution for changed systems', 0, 1e5, 1, 't [yr]')

""" CPU time """
#compRuns('t cpu', '$t_{cpu}$ distribution', 0, 100, 0.05, 't [s]', ks0, ks1, K)
#compRuns('t cpu', '$t_{cpu}$ distribution', 0, 100, 0.05, 't [s]', ks0O, ks1O, KO)

#compRunsDouble('t cpu', '$t_{cpu}$ distribution', 0, 20, 0.1, 't [s]')

""" impact parameter """
# compRuns('bImp', 'Normalized impact parameter distribution', 10e-5, 10, 1, 'b', ks0, ks1, K)
# compOutcome('bImp', '', 0.5, 10, 0.1, 'b/a')
# compOutcome('bImp', '', 0.5, 20, 0.1, 'b/a',  [ks0, ks0H, ks0L], ['No reg (Acc = 1e-8)', 'No reg (Acc = 1e-15)', 'No reg (Acc = 1e-5)'])
# compOutcome('bImp', '', 0.5, 20, 0.1, 'b/a',  [ks1, ks1H, ks1L], ['KS reg (Acc = 1e-8)', 'KS reg (Acc = 1e-15)', 'KS reg (Acc = 1e-5)'])
# compOutcome('bImp', '', 0.5, 20, 0.1, 'b/a',  [ks0, ks1, K], ['No reg', 'KS reg', 'ARChain'])

# compOutcomeSamePlot('bImp', '', 0.5, 20, 0.1, 'b/a',  [ks0, ks1, K], ['No reg', 'KS reg', 'ARChain'])
# compOutcomeSamePlot('bImp', '', 0.5, 20, 0.1, 'b/a',  [ks1H, ks1, ks1L], ['KS reg'])



# compInputConf('bImp', 'b distribution for different flags', 0, 10, 0.1, 'b/a', 0)
# compInputConfFrac('bImp', 'b distribution for different flags', 0, 10, 0.1, 'b/a')
# compInputConfFracMult('bImp', 'b distribution for different flags', 0, 10, 0.1, 'b/a', ['Flyby', 'Exchange'])


""" E0 """
#compRuns('E0', 'Normalized E0 distribution', -1, 1, 0.01, 'E0')
#compRunsChanged('E0', 'Normalized E0 distribution', -1, 1, 0.01, 'E0')

""" mass ratio """
#compAll('mass', 'Normalized mass ratio distribution (M$_{lower}$/M$_{higher}$)')

""" L0 """
#compRuns('L0', 'Normalized L0 distribution', 0, 5, 0.05, 'L0')
#compRunsChanged('L0', 'Normalized L0 distribution', 0, 5, 0.05, 'L0')

""" rmin """
#compRuns('rMin', 'Rmin distribution', 1, 2.3, 0.1, 'r$_{min}$ [R$_{\odot}$]')
#compRunsChanged('rMinSun', 'Normalized Rmin distribution', 0, 1000, 5, 'Rmin [R$_{\odot}$]')
#compRunsOutcome('rMin', 'Rmin distribution for unbound mergers', 1, 2.3, 0.1, 'r$_{min}$ [R$_{\odot}$]', 'Unbound merger', [ks0, ks1, K], ['ks0', 'ks1', 'K'])
#compInputConf('rMin', 'b distribution for different flags', 1, 2.3, 0.1, 'b/a')

""" number of black holes """
#numBH('r', 'Number of black holes')

""" v infinity """
#compRunsChanged('vInf', 'Normalized v infinity distribution', 0, 2.5, 0.025, 'V$_{\mathrm{inf}}$')
#compOutcome('vInf', 'V$_{\infty}$ distribution for different outcomes', 0.1, 1.2, 0.1, 'V$_{\infty}$/V$_c$')
#compInputConf('vInf', '$v_{\infty}$ distribution for different flags', 0, 1.3, 0.03, '$v_{\infty}/v_{crit}$')

""" vCrit """
#compInputConf('vCrit Unit', '$v_{crit}$ distribution for different flags', 0, 100, 1, '$v_{crit}$')


""" Vinit vs Vfinal """
#compV([ks0, ks1, K, KL], ['ks0', 'ks1', 'K', 'KL'])


""" delta E """
xMax = 1e-5
#if onlyLowMass == 1:
#    compRuns('dEE0', 'Distribution of energy conservation (Mass ratio < ' + str(lowMass) + ')', -xMax, xMax, xMax/50, 'dE', ks0, ks1, K)
#else:
#    compRuns('dEE0', 'Distribution of energy conservation',  -xMax, xMax, xMax/50, 'dE', ks0, ks1, K)


# compRunsDouble('dEE0', 'Distribution of energy conservation',  -xMax, xMax, xMax/50, 'dE')
# compRunsMult('dEE0', 'Distribution of energy conservation',  -xMax, xMax, xMax/50, 'dE', [])
#compRunsChanged('dE', 'Distribution of energy conservation for changed systems', -xMax, xMax, xMax/50, 'dE')

""" delta E over E0 """
xMax = 1e-5
# compRuns('dEE0', 'Distribution of dE over E0', -xMax, xMax, xMax/1000, 'dE/E0')
#compRunsChanged('dEE0', 'Distribution of dE over E0 for changed systems', -xMax, xMax, xMax/100, 'dE/E0')

""" delta L """
xMax = 0.5
#if onlyLowMass == 1:
#   compRuns('dL', 'Distribution of angular momentum conservation (Mass ratio < ' + str(lowMass) + ')', 0, xMax, xMax/100, 'dL')
#else:
#   compRuns('dL', 'Distribution of angular momentum conservation', 0, xMax, xMax/100, 'dL')
#
#compRunsChanged('dL', 'Distribution of angular momentum conservation for changed systems', 0, xMax, xMax/100, 'dL')

""" delta L over L0 """
xMax = 2e-7
#compRuns('dLL0', 'Distribution of dL over L0', 0, xMax, xMax/100, 'dL/L0')
#compRunsChanged('dLL0', 'Distribution of dL over L0 for changed systems', 0, xMax, xMax/100, 'dL/L0')

""" log a vs log e """
#logComp('a1', 'e1', 'log a', 'log e')

""" log a vs log b """
#logComp('a1', 'bImp', 'log a', 'log b')

""" log a vs dE """
energyLimit = 10e-19
#semiAxisdE(KL, 'Semi-major axis vs Energy conservation')

""" find interactions with high e1 """
# highEks0 = ks0[ks0['e1'] == '0.99']

""" fraction of collisions """
#fracCol(KL)

""" find confs of low mass ratio """
#compLowMassRat(1e-5)

""" find confs of low semi-major axis """
#confLowA(1)


""" find number of stars with mass above 100 Msun """
#masses0 = ks0['m0'].astype('float')
#masses10 = ks0['m10'].astype('float')
#masses11 = ks0['m11'].astype('float')
#
#singMass = np.where((masses0 > 100))[0]
#bin1Mass =  np.where((masses10 > 100))[0]
#bin2Mass = np.where((masses11 > 100))[0]
#total =  (masses0 > 100).sum() + (masses10 > 100).sum() + (masses11 > 100).sum()

""" Find interactions with certain vInf """
#vInf = K['vInf'].to_numpy(dtype='float')
#
#vinfLarge = K[vInf < 1]['conf']


""" Find systems which timed out """
#zeroConfs = ks0Confs[ks0Confs == '0'].index[:]
#zeroC = ks1.iloc[zeroConfs]

""" Indices of large stars """
#rLim = 1e-3
#radiiLarge = ks0.iloc[0:,24:27].astype('float')
#maskR = (radiiLarge > rLim)
#indexL = radiiLarge[maskR].dropna(thresh=2)
#ind = indexL.index[indexL == True].tolist()

#dEK = K['dE'].astype('float')
#deKSys = K[dEK > 1]

#cpuT = KL['r0'].astype('float')
#cpuSys = KL[cpuT >= 1]
#numOBj = cpuSys['nObj'].astype('float')
#confCpu = cpuSys[numOBj < 2]

#r11 = K['r11'].astype('float')
#r10 = K['r10'].astype('float')
#a1 = K['a1'].astype('float')

#sysR = K[(r10 > 1)]
#sysR2 = sysR[(r10 > a1)]

#sysR3 = K[(r11 > 100)]
#sysR3 = np.append(sysR3, K[(r11 > 100)])

#df = pd.concat([r11, r10, a1])

#IMBHInd = np.append(singMass, bin1Mass)
#IMBHInd = np.append(IMBHInd, bin2Mass)

#KIndex = ksChanges.index.tolist()
#np.savetxt('ksIndex.txt', KIndex, delimiter='\n')

#IMBHInd = np.unique(np.sort(IMBHInd))
#np.savetxt('IMBH_index.txt', IMBHInd, delimiter='\n')

#ks0.to_csv('./csvs/ks0.csv')
#ks1.to_csv('./csvs/ks1.csv')
#K.to_csv('./csvs/K1.csv')

""" find systems with outcome exchange """
#flag = (ks0['conf'] == ['[0 1] 2'])



