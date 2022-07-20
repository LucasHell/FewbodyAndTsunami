#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 19:36:32 2021

@author: lucas

"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from investigateBinaryAfterMerge import investigateBinary
# from compareT2PeriMergerTime import comparePeriTimeMergerTime
import scipy.integrate as integrate

def countOutcomes(dfs, dfNames, plotTitle, plotResults, fractionFlag):
    outcomes = pd.DataFrame()
    flybysArr = []
    exchangesArr = []
    mergersArr = []
    ionArr = []
    tripleArr = []
    outcomesColumns = ['Flyby', 'Exchange', 'Merger']
    outcomesColumns = ['Flyby','Exchange', 'Merger', 'Ionization', 'triples']
    # outcomesColumns = ['Flyby','Exchange', 'Merger', 'Ionization', 'Stable triples']
    for i in range(len(dfs)):
        df = dfs[i]

        escId = df.escId.astype(float)
        encounterComplete = df.encounterComplete.astype(float)
        ionization = df.ionization.astype(float)
        aFin = df.aFin.astype(float)
        eFin = df.eFin.astype(float)
        colInd1 = df.colInd1.astype(float)
        colInd2 = df.colInd2.astype(float)

        exchanges = df[(escId != 0) & (encounterComplete == 1) & (ionization != 1)]
        incomplete = df[encounterComplete == 0]
        breakup = df[(encounterComplete == 1) & (ionization == 1)]
        triple = df[(encounterComplete == 1) & (aFin == 0) &
                        (eFin == 0) & (ionization != 1)]
        merger = df[(colInd1 != 0) | (colInd2 != 0)]

        mask = ((escId == 0) & (encounterComplete == 1) & (ionization != 1) &
        (colInd1 == 0) & (colInd2 == 0) & (aFin != 0) & (eFin != 0))

        flybys = df[mask]

        outcomesTemp = pd.Series([len(flybys), len(exchanges), len(merger), len(breakup), len(triple)],
                                    index=outcomesColumns, name=dfNames[i])

        # outcomesTemp = pd.Series([len(flybys), len(exchanges), len(merger), len(breakup), len(triple)],
        #                             index=outcomesColumns, name=dfNames[i])

        # outcomesTemp = pd.Series([len(flybys), len(exchanges), len(merger)],
        #                           index=outcomesColumns, name=dfNames[i])
        if fractionFlag:
            outcomesTemp = outcomesTemp
        outcomes = outcomes.append(outcomesTemp.T, ignore_index=True)
        flybysArr.append(flybys)
        exchangesArr.append(exchanges)
        mergersArr.append(merger)
        ionArr.append(breakup)
        tripleArr.append(triple)

        # if plotResults:
        #     weights = np.ones(len(outcomesTemp)) / len(outcomesTemp)
        #     outcomesTemp.sort_values(ascending=False).plot(kind='bar', legend=True)


    outcomes = outcomes.T
    outcomes.columns = dfNames

    if plotResults:
        outcomes.sort_values(dfNames[i], ascending=False).plot(kind='bar', rot=0, legend=True)

        if fractionFlag:
            plt.ylabel('Fraction')
        else:
            plt.ylabel('Count')

        plt.title(plotTitle)


        # a1 = plt.axes([.33, .25, .56, .2])
        # outcomes = outcomes.drop('Flyby')
        # outcomes.sort_values(dfNames[i], ascending=False).plot(kind='bar',legend=False, rot=0, ax=a1)
        # a1.set_xlabel('')
        # a1.set_ylabel('')
        # a1.set_xticks([])
        # a1.set_yticks([])


    return outcomes.sort_values(dfNames[i], ascending=False), flybysArr, exchangesArr, mergersArr, ionArr, tripleArr


def initialParameters(df, param):
    if param == 'bImp':
        paramMin = np.amin(df[param].astype(float) * df['a1'].astype(float))
        paramMax = np.amax(df[param].astype(float) * df['a1'].astype(float))
        param = df[param].astype(float) * df['a1'].astype(float)
    else:
        paramMin = np.amin(df[param].astype(float))
        paramMax = np.amax(df[param].astype(float))
        param = df[param].astype(float)

    return [paramMin, paramMax]

def initialParametersBHBHBHStar(dfsBHBH, dfsBHStar, param, xLabel, bins, title, log):
    plt.figure()

    dfBHBH = dfsBHBH[0]
    dfBHStar = dfsBHStar[1]

    if param == 'bImp':
        paramBHBH = dfBHBH['bImp'].astype(float)
        paramBHStar = dfBHStar['bImp'].astype(float)

    elif param == 'mass':
        paramBHBH = dfBHBH[['m0', 'm10', 'm11']].astype(float)
        paramBHStar = dfBHStar[['m0', 'm10', 'm11']].astype(float)


        starMassBHBH = paramBHBH['m0']
        BHMassBHBH = pd.concat([paramBHBH['m10'], paramBHBH['m11']])

        minMass = paramBHStar.idxmin(axis=1)
        starMassBHStar = []
        BHMassBHStar = []
        for i in range(len(minMass)):
            if minMass.iloc[i] == 'm0':
                starMassBHStar.append(paramBHStar.iloc[i]['m0'])
                BHMassBHStar.append(paramBHStar.iloc[i]['m10'])
                BHMassBHStar.append(paramBHStar.iloc[i]['m11'])
            elif minMass.iloc[i] == 'm10':
                starMassBHStar.append(paramBHStar.iloc[i]['m10'])
                BHMassBHStar.append(paramBHStar.iloc[i]['m0'])
                BHMassBHStar.append(paramBHStar.iloc[i]['m11'])
            else:
                starMassBHStar.append(paramBHStar.iloc[i]['m11'])
                BHMassBHStar.append(paramBHStar.iloc[i]['m10'])
                BHMassBHStar.append(paramBHStar.iloc[i]['m0'])


        # plt.hist(starMassBHBH, label='BH-BH binary (stars)', bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)
        plt.hist(BHMassBHStar, label='Setup 1', bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)
        plt.hist(BHMassBHBH, label='Setup 2', bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)
        # plt.hist(starMassBHStar, label='BH-Star binary (stars)', bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)

        plt.ylabel('Fraction')
        plt.xlabel(xLabel)
        plt.legend(loc='upper left')
        if log:
            plt.xscale('log')
        plt.title(title)

        plt.figure()
        plt.hist(starMassBHStar, label='Setup 1', bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)
        plt.hist(starMassBHBH, label='Setup 2', bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)
        plt.ylabel('Fraction')
        plt.xlabel(xLabel)
        plt.legend(loc='upper left')
        if log:
            plt.xscale('log')
        plt.title(title)

        return 0

    elif param == 'vInf':
        paramBHBH = dfBHBH['vInfInit'].astype(float)
        paramBHStar = dfBHStar['vInfInit'].astype(float)

        G = 887.3515302300001    # AU, km/s, solar mass

        vCritBHBH = calcVCrit(dfBHBH['m10'].astype(float), dfBHBH['m11'].astype(float), dfBHBH['m0'].astype(float), dfBHBH['a1'].astype(float), G)
        vCritBHStar = calcVCrit(dfBHStar['m10'].astype(float), dfBHStar['m11'].astype(float), dfBHStar['m0'].astype(float), dfBHStar['a1'].astype(float), G)

        paramBHBH *= vCritBHBH
        paramBHStar *= vCritBHStar
    elif param == 'e1':
        paramBHBH = dfBHBH[param].astype(float)
        paramBHStar = dfBHStar[param].astype(float)
    else:
        paramBHBH = dfBHBH[param].astype(float)
        paramBHStar = dfBHStar[param].astype(float)

    cumPlot = False

    plt.hist(paramBHStar, label='Setup 1', bins=bins, density=cumPlot, cumulative=cumPlot, histtype='step', linewidth=3)
    plt.hist(paramBHBH, label='Setup 2', bins=bins, density=cumPlot, cumulative=cumPlot, histtype='step', linewidth=3)
    plt.ylabel('Fraction')
    plt.xlabel(xLabel)
    plt.legend(loc='upper left')
    if log:
        plt.xscale('log')
    plt.title(title)

#%%

def initialParametersAll(dfsBHBH, dfsBHStar):
    figs, axs = plt.subplots(ncols=3, nrows=2, sharey=True)

    dfBHBH = dfsBHBH[0]
    dfBHStar = dfsBHStar[0]

    """ bImp """
    bins = np.logspace(-2, 3.3, 100)
    paramBHBH = dfBHBH['bImp'].astype(float)
    paramBHStar = dfBHStar['bImp'].astype(float)

    axs[0,2].hist(paramBHStar, label='Setup 1', bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)
    axs[0,2].hist(paramBHBH, label='Setup 2', bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)

    # axs[0,2].set_xlabel('b/a')
    axs[0,2].set(xlabel ='b/a', xscale='log')

    """ mass BH """
    bins = np.linspace(0, 40, 100)
    paramBHBH = dfBHBH[['m0', 'm10', 'm11']].astype(float)
    paramBHStar = dfBHStar[['m0', 'm10', 'm11']].astype(float)

    starMassBHBH = paramBHBH['m0']
    BHMassBHBH = pd.concat([paramBHBH['m10'], paramBHBH['m11']])

    minMass = paramBHStar.idxmin(axis=1)
    starMassBHStar = []
    BHMassBHStar = []
    for i in range(len(minMass)):
        if minMass.iloc[i] == 'm0':
            starMassBHStar.append(paramBHStar.iloc[i]['m0'])
            BHMassBHStar.append(paramBHStar.iloc[i]['m10'])
            BHMassBHStar.append(paramBHStar.iloc[i]['m11'])
        elif minMass.iloc[i] == 'm10':
            starMassBHStar.append(paramBHStar.iloc[i]['m10'])
            BHMassBHStar.append(paramBHStar.iloc[i]['m0'])
            BHMassBHStar.append(paramBHStar.iloc[i]['m11'])
        else:
            starMassBHStar.append(paramBHStar.iloc[i]['m11'])
            BHMassBHStar.append(paramBHStar.iloc[i]['m10'])
            BHMassBHStar.append(paramBHStar.iloc[i]['m0'])


    axs[1,2].hist(BHMassBHStar, bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)
    axs[1,2].hist(BHMassBHBH, bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)

    axs[1,2].set(xlabel ='M$_{\mathrm{BH}}$')
    # axs[1,1].set_)

    """ mass star """
    bins = np.linspace(0,3, 100)
    axs[1,1].hist(starMassBHStar, bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)
    axs[1,1].hist(starMassBHBH, bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)

    # axs[1,2].set_ylabel('Fraction')
    # axs[1,2].set_xlabel('M$_{\mathtext{star}}$')
    axs[1,1].set(xlabel ='M$_{\mathrm{star}}$')


    """ vInf """
    bins = np.logspace(-1,2.5,100)
    paramBHBH = dfBHBH['vInfInit'].astype(float)
    paramBHStar = dfBHStar['vInfInit'].astype(float)

    G = 887.3515302300001    # AU, km/s, solar mass

    vCritBHBH = calcVCrit(dfBHBH['m10'].astype(float), dfBHBH['m11'].astype(float), dfBHBH['m0'].astype(float), dfBHBH['a1'].astype(float), G)
    vCritBHStar = calcVCrit(dfBHStar['m10'].astype(float), dfBHStar['m11'].astype(float), dfBHStar['m0'].astype(float), dfBHStar['a1'].astype(float), G)

    paramBHBH *= vCritBHBH
    paramBHStar *= vCritBHStar

    axs[1,0].hist(paramBHBH, bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)
    axs[1,0].hist(paramBHStar, bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)

    # axs[1,0].set_ylabel('Fraction')
    # axs[1,0].set_xlabel('v$_{\infty}$')
    axs[1,0].set(ylabel='Fraction',xlabel ='v$_{\infty}$', xscale='log')


    """ e """
    bins = np.linspace(0, 1, 100)
    paramBHBH = dfBHBH['e1'].astype(float)
    paramBHStar = dfBHStar['e1'].astype(float)

    axs[0,1].hist(paramBHBH, bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)
    axs[0,1].hist(paramBHStar, bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)

    # axs[0,1].set_ylabel('Fraction')
    # axs[0,1].set_xlabel('e')
    axs[0,1].set(xlabel ='e')


    """ a """
    bins = np.logspace(-2, 4, 100)
    paramBHBH = dfBHBH['a1'].astype(float)
    paramBHStar = dfBHStar['a1'].astype(float)

    axs[0,0].hist(paramBHBH, bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)
    axs[0,0].hist(paramBHStar, bins=bins, density=True, cumulative=True, histtype='step', linewidth=3)

    # axs[0,1].set_ylabel('Fraction')
    # axs[0,1].set_xlabel('a [AU]')
    axs[0,0].set(ylabel='Fraction',xlabel ='a [AU]', xscale='log')
    plt.subplots_adjust(wspace=0)
    plt.subplots_adjust(hspace=0.35)

    lines_labels = [ax.get_legend_handles_labels() for ax in figs.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    figs.legend(lines, labels, loc='upper center')


    # plt.legend(loc='upper left')
    # plt.xscale('log')
    # figs.tight_layout()


#%%
# initialParametersBHBHBHStar(BHBHData, BHStarData, 'mass', 'M$_{ \mathrm{star}}$ [M$_{\odot}$]', np.linspace(0, 3, 100), pltTitle, False)
# initialParametersBHBHBHStar(BHBHData, BHStarData, 'a1', 'a [AU]', np.logspace(-2, 4, 100), pltTitle, True)
# initialParametersBHBHBHStar(BHBHData, BHStarData, 'e1', 'e$^2$', np.linspace(0, 1, 100), pltTitle, False)
# initialParametersBHBHBHStar(BHBHData, BHStarData, 'bImp', 'b/a', np.logspace(-2, 3.3, 100), pltTitle, True)
# initialParametersBHBHBHStar(BHBHData, BHStarData, 'vInf', 'V$_{\infty}$ [km/s]', np.logspace(-1, 2.5, 100), pltTitle, True)

def calcVCrit(mb1, mb2, ms, a, G):
    totalMass = mb1 + mb2 + ms
    redMass = ((mb1 + mb2) * ms) / totalMass
    vCrit = np.sqrt((G/redMass) * (mb1 * mb2) / a)
    return vCrit

def investIncomplete(dfs, dfNames, param):
    dfResults = pd.DataFrame()
    index = ['# total', '# unique']
    for i in range(len(dfs)):
        df = dfs[i]
        incomplete = df[(df.colInd1 != '0') & (df.colInd2 != '0')]
        uniqueParam = np.unique(incomplete[param])
        series = pd.Series([len(incomplete), len(uniqueParam)], index=index, name=dfNames[i])
        dfResults = dfResults.append(series)

    return dfResults.T

def finalParamPlots(dfs, dfNames, param, bins, xLab, logFlag, title):
    plt.figure()
    for i in range(len(dfs)):
        df = dfs[i][1]
        # print(dfNames[i] + ': ' + str(np.mean(df[param].astype(float)[df[param].astype(float) != 0])) + '\n')
        plt.hist(df[param].astype(float)[df[param].astype(float) != 0] * 2 * np.pi, bins=bins, histtype='step', label=dfNames[i], linewidth=4,density=True, cumulative=True)

    if (logFlag):
        plt.xscale('log')
    plt.xlabel(xLab)
    plt.ylabel('Count')
    plt.legend(loc='upper left')
    plt.title(title)

def finalBHBHBinary(dfs):
    finalBHBHArr = []
    for i in range(len(dfs)):
        df = dfs[i]
        exchanges = df[(df.escId != '0') & (df.encounterComplete == '1')]

        exchangeMass = exchanges[['m0', 'm10', 'm11']]
        exchangeMass.columns = [0,1,2]
        minMassInd = exchangeMass.astype(float).idxmin(axis=1)
        finalBHBHExch = exchanges[(minMassInd != 0) & (minMassInd == exchanges.escId.astype(int))]


        mask = ((df.escId == '0') & (df.encounterComplete == '1') & (df.ionization != '1') &
        (df.colInd1 == '0') & (df.colInd2 == '0') & (df.aFin != '0') & (df.eFin != '0'))

        flybys = df[mask]
        flybyMass = flybys[['m0', 'm10', 'm11']]
        minMassIndFlyby = flybyMass.astype(float).idxmin(axis=1)
        finalBHBHFlyby = flybys[minMassIndFlyby == 'm0']

        allBHBHBins = pd.concat([finalBHBHExch, finalBHBHFlyby])
        finalBHBHArr.append(allBHBHBins)

    return finalBHBHArr


def investigateMergers(dfs, dfNames, pltTitle):
    mergerDF = pd.DataFrame()
    mergersArray = []
    for i in range(len(dfs)):
        df = dfs[i]

        merger = df[(df.colInd1.astype(float) != 0.0) | (df.colInd2.astype(float) != 0.0)]

        mergerMass = merger[['m0', 'm10', 'm11']]
        mergerMass.columns = [0,1,2]
        minMassMergers = np.array(mergerMass.astype(float).idxmin(axis=1))

        colInd1 = np.array(merger.colInd1.astype(float))
        colInd2 = np.array(merger.colInd2.astype(float))
        # maskBHBH = []
        # maskBHStar = []

        # for j in range(len(merger)):
        #     if ((minMassMergers[i] != colInd1[i]) & (minMassMergers[i] != merger.colInd2[i])):
        #         maskBHBH.append(True)
        #         maskBHStar.append(False)
        #     if ((minMassMergers[i] != colInd1[i]) & (minMassMergers[i] != merger.colInd2[i])):
        #         maskBHBH.append(False)
        #         maskBHStar.append(True)

        BHBHMergers = merger[(minMassMergers != colInd1) & (minMassMergers != colInd2)]
        BHStarMergers = merger[(minMassMergers == colInd1) | (minMassMergers == colInd2)]

        singleBHStarMergers = BHStarMergers[(BHStarMergers['colInd1'] == '0.0') | (BHStarMergers['colInd2'] == '0.0')]
        binBHStarMergers = BHStarMergers[(BHStarMergers['colInd1'] != '0.0') & (BHStarMergers['colInd2'] != '0.0')]

        # singleMergers = df[(df['colInd1'] == '0.0') | (df['colInd2'] == '0.0')]
        # binMergers = df[(df['colInd1'] != '0.0') & (df['colInd2'] != '0.0')]

        # dfTemp = pd.Series([len(BHBHMergers)/len(merger), len(BHStarMergers)/len(merger)], name=dfNames[i])
        dfTemp = pd.Series([len(BHBHMergers), len(BHStarMergers)], name=dfNames[i])
        dfTemp.index = ['BH-BH', 'BH-Star']
        # dfTemp = pd.Series([len(BHBHMergers)/len(merger), len(singleBHStarMergers)/len(merger), len(binBHStarMergers)/len(merger)], name=dfNames[i])
        # dfTemp.index = ['BH-BH', 'Single BH-Star', 'Binary BH-Star']
        mergerDF = mergerDF.append(dfTemp, ignore_index=True)
        mergersArray.append([BHBHMergers, BHStarMergers])

    mergerDF = mergerDF.T
    mergerDF.columns = dfNames

    mergerDF.plot(kind='bar', legend=True)
    plt.xticks(rotation=0)
    plt.title(pltTitle)
    plt.ylabel('Count')

    # a1 = plt.axes([.15, .25, .25, .2])
    # mergerDF = mergerDF.drop('BH-Star')
    # # mergerDF = mergerDF.drop('Binary BH-Star')
    # # mergerDF = mergerDF.T
    # # a1.bar([0,1,2,3], mergerDF)
    # mergerDF.plot(kind='bar',legend=False, rot=0, ax=a1)
    # a1.set_xlabel('')
    # a1.set_ylabel('')
    # a1.set_xticks([])
    # a1.tick_params(axis='y', rotation=90)

    # print(mergerDF)

    return mergersArray



def splitInitialConfs(dfs):
    BHBH = []
    BHStar = []
    for df in dfs:
        masses = df[['m0', 'm10', 'm11']]
        minMass  = masses.astype(float).idxmin(axis=1)

        BHBH.append(df[minMass == 'm0'])
        BHStar.append(df[minMass != 'm0'])

    return BHBH, BHStar

def mergerParamsScatterPlots(dfs, dfNames, param1, param2, title, xlab, ylab, binSinSplitFlag):
    legendName = ['BH-BH merger', 'BH-Star merger']
    plt.figure()
    arrMergerLessThanPeriod = []
    # legendName = dfNames
    G = 39.478           # AU3 * yr-2 * Msun-1
    for i in range(len(dfs)):
        df2 = dfs[i]
        # plt.figure()
        # plt.title(title + '(' + dfNames[i] + ')')
        # for j in range(len(df2)):
        if binSinSplitFlag:
            df = df2[1]
        else:
            df = df2
        # p1 = df[param1].astype(float)
        p1 =  (df['a1'].astype(float)**3*(4*np.pi**2) / (G * (df['m10'].astype(float) + df['m11'].astype(float))))**(1/2)


        p2 = df[param2].astype(float) * 2 * np.pi

        p1 = p1[p1 != 0]
        p2 = p2[p2 != 0]

        plt.scatter(p1, p2, label=dfNames[i])

        arrMergerLessThanPeriod.append(df[p2 < p1])

    x = np.linspace(0, 1e6)
    y = x
    plt.plot(x, y)
    plt.legend()
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.yscale('log')
    plt.xscale('log')
    plt.title(title)

    return arrMergerLessThanPeriod


def paramsScatterPlots(dfs, dfNames, param1, param2, title, xlab, ylab):
    plt.figure()
    for i in range(len(dfs)):
        df = dfs[i]
        p1 = df[param1].astype(float)
        if param2 == 'binMass':
            p2 = df['m10'].astype(float) + df['m11'].astype(float)
        else:
            p2 = df[param2].astype(float)

        plt.scatter(p1, p2, label=dfNames[i], alpha=0.5, s=10)

    # plt.legend()
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)

def params2dHisto(dfs, dfNames, param1, param2, title, xlab, ylab):
    bins = np.linspace(0,1, 100)

    # for i in range(len(dfs)):
    #     df = dfs[i]
    #     p1 = df[param1].astype(float)
    #     p2 = df[param2].astype(float)

    #     plt.figure()
    #     plt.hist2d(p1, p2, bins=bins, norm=mpl.colors.LogNorm())
    #     plt.title(title + '(' + dfNames[i] + ')')
    #     # plt.legend()
    #     # plt.xscale('log')
    #     # plt.yscale('log')
    #     plt.xlabel(xlab)
    #     plt.ylabel(ylab)
    #     # plt.title(title)

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    plt.gca().set_title(title)
    df = dfs[0]
    p1 = df[param1].astype(float)
    p2 = df[param2].astype(float)
    axs[0,0].hist2d(p1, p2, bins=bins, norm=mpl.colors.LogNorm())
    axs[0,0].set_title(dfNames[0])
    axs[0,0].set_ylabel(ylab)

    df = dfs[1]
    p1 = df[param1].astype(float)
    p2 = df[param2].astype(float)
    axs[0,1].hist2d(p1, p2, bins=bins, norm=mpl.colors.LogNorm())
    axs[0,1].set_title(dfNames[1])

    df = dfs[2]
    p1 = df[param1].astype(float)
    p2 = df[param2].astype(float)
    axs[1,0].hist2d(p1, p2, bins=bins, norm=mpl.colors.LogNorm())
    axs[1,0].set_title(dfNames[2])
    axs[1,0].set_ylabel(ylab)
    axs[1,0].set_xlabel(xlab)

    df = dfs[3]
    p1 = df[param1].astype(float)
    p2 = df[param2].astype(float)
    axs[1,1].hist2d(p1, p2, bins=bins, norm=mpl.colors.LogNorm())
    axs[1,1].set_title(dfNames[3])
    axs[1,1].set_xlabel(xlab)

def paramsHisto(dfs, dfNames, param1, param2, title, xlab):
    bins = np.linspace(0,1, 100)
    for i in range(len(dfs)):
      df = dfs[i]
      p1 = df[param1].astype(float)
      p2 = df[param2].astype(float)

      plt.figure()
      # plt.hist(p1, bins=bins, histtype='step', label='Initial')
      plt.hist(p2, bins=bins, histtype='step', label='Final')
      plt.title(title + '(' + dfNames[i] + ')')
      plt.legend()
      # plt.xscale('log')
      # plt.yscale('log')
      plt.xlabel(xlab)
      plt.ylabel('Count')
      # plt.title(title)



def mergerTimesGW(dfs, dfNames, bins, pltTitle, cumulativeFlag):
    def calcTGW(a, e, m1, m2):
        def integrand(x):
            return x**(29/19) * (1+(121/304)*x**2)**(1181/2299) / (1-x**2)**(3/2)

        fArr = []
        for ecc in e:
            # print(ecc)
            if (ecc == 0.0):
                # print('stop')
                fArr.append(1)
            else:
                f = (1-ecc**2)**4 / (ecc**(48/19) * (ecc**2 + 304/121)**(3480/2299)) * integrate.quad(integrand, 0, ecc)[0]
                fArr.append(f)

        tGW = 5 * c**5 / (304 * G**3) * (a**4 / (m1 * m2 * (m1 + m2))) * np.array(f)

        return tGW


    mergerTimeDF = pd.DataFrame()
    mergerTimeDFInit = pd.DataFrame()
    fig = plt.plot()
    for i in range(len(dfs)):
        df = dfs[i]
        c = 63239.7263       # AU/yr
        G = 39.478           # AU3 * yr-2 * Msun-1
        a = df['aFin'].astype(float)
        e = df['eFin'].astype(float)

        aInit = df['a1'].astype(float)
        eInit = df['e1'].astype(float)

        masses = df[['m0', 'm10', 'm11']].astype(float)
        m1 = []
        m2 = []
        minMass = masses.idxmin(axis=1)
        for j in range(len(masses)):
            if minMass.iloc[j] == 0:
                m1.append(masses.iloc[j]['m10'])
                m2.append(masses.iloc[j]['m11'])
            elif minMass.iloc[j] == 1:
                m1.append(masses.iloc[j]['m0'])
                m2.append(masses.iloc[j]['m11'])
            else:
                m1.append(masses.iloc[j]['m0'])
                m2.append(masses.iloc[j]['m10'])

        m1 = np.array(m1)
        m2 = np.array(m2)
        tGW = 5/256 * ((c**5 * a**4 * (1-e**2)**(7/2)) / (G**3 * m1 * m2 * (m1 + m2)))
        tGWInit = 5/256 * ((c**5 * aInit**4 * (1-eInit**2)**(7/2)) / (G**3 * m1 * m2 * (m1 + m2)))

        try:
            tGW2 = calcTGW(a, e, m1, m2)
            tGWInit2 = calcTGW(aInit, eInit, m1, m2)
        except ZeroDivisionError as err:
            print(e)

        # print('Old:\n ' + tGW)
        # print(tGWInit)

        # print('\nNew: ' + tGW2)
        # print('tGWInit2')
        if cumulativeFlag:
            plt.hist(tGW, histtype='step', bins=bins, label=dfNames[i], cumulative=True, density=True, linewidth=2)
        else:
            plt.hist(tGW, histtype='step', bins=bins, label=dfNames[i], cumulative=False, density=False, linewidth=2)
        mergerTimeDF = mergerTimeDF.append(tGW, ignore_index=True)
        mergerTimeDFInit = mergerTimeDFInit.append(tGWInit, ignore_index=True)
        print(dfNames[i] + ' done')


    plt.xscale('log')
    if cumulativeFlag:
        plt.ylabel('Fraction')
    else:
        plt.ylabel('Count')
    plt.xlabel('T$_{GW}$ [yr]')
    plt.title(pltTitle)
    plt.legend(loc='upper left')
    mergerTimeDF = mergerTimeDF.T
    mergerTimeDF.columns = dfNames

    mergerTimeDFInit = mergerTimeDFInit.T
    mergerTimeDFInit.columns = dfNames

    return mergerTimeDF, mergerTimeDFInit

def mergerTimesGWAfterMerge(dfs, dfNames, bins, pltTitle, cumulativeFlag):
    mergerTimeDF = pd.DataFrame()
    plt.figure()
    for i in range(len(dfs)):
        df = dfs[i]
        c = 63239.7263       # AU/yr
        G = 39.478           # AU3 * yr-2 * Msun-1

        a = df['a'].astype(float)
        e = df['e'].astype(float)
        mask = (df['a'] != 0) & (df['e'] != 1) &  (df['e'] != 0)

        a = a[mask]
        e = e[mask]

        masses = df[['m0', 'm10', 'm11']].astype(float)
        masses = masses[mask]
        m1 = []
        m2 = []
        colInd1 = df['colInd1'][mask]
        colInd2 = df['colInd2'][mask]
        for j in range(len(colInd1)):
            massesInd = masses.iloc[j]
            if colInd1.iloc[j] == 0.0 and colInd2.iloc[j] == 1.0:
                m1.append(massesInd['m0'] + massesInd['m10'])
                m2.append(massesInd['m11'])
            elif colInd1.iloc[j] == 0.0 and colInd2.iloc[j] == 2.0:
                m1.append(massesInd['m0'] + massesInd['m11'])
                m2.append(massesInd['m10'])
            else:
                m1.append(massesInd['m0'])
                m2.append(massesInd['m10'] + massesInd['m11'])


        m1 = np.array(m1)
        m2 = np.array(m2)
        tGW = 5/256 * ((c**5 * a**4 * (1-e**2)**(7/2)) / (G**3 * m1 * m2 * (m1 + m2)))
        # plt.figure()
        weights = np.ones_like(tGW) / len(tGW)
        if cumulativeFlag:
            plt.hist(tGW, histtype='step', bins=bins, label=dfNames[i], cumulative=True, density=True, linewidth=3, weights=weights)
            # plt.hist(m1, histtype='step', bins=bins, label=dfNames[i] + ' (m1)', cumulative=False, density=False, linewidth=3, weights=weights)
            # plt.hist(m2, histtype='step', bins=bins, label=dfNames[i] + ' (m2)', cumulative=False, density=False, linewidth=3, weights=weights)
        else:
            plt.hist(tGW, histtype='step', bins=bins, label=dfNames[i], cumulative=False, density=False, linewidth=3)
        mergerTimeDF = mergerTimeDF.append(tGW, ignore_index=True)

        # aBins = np.logspace(-5, 5, 50)
        # weights = np.ones_like(a) / len(a)
        # plt.hist(a, histtype='step', bins=aBins, linewidth=2,label=dfNames[i], cumulative=False, density=False, weights=weights)

        # eBins = np.logspace(-5,-2,50)
        # weights = np.ones_like(e) / len(e)
        # plt.hist(1-e, histtype='step', linewidth=2, bins=eBins, label=dfNames[i], cumulative=False, density=False, weights=weights)


        print(dfNames[i] + ' done')


    plt.xscale('log')
    if cumulativeFlag:
        plt.ylabel('Fraction')
    else:
        plt.ylabel('Count')
    plt.xlabel('T$_{GW}$ [yr]')
    # plt.xlabel('Semi-major axis [AU]')
    # plt.xlabel('1-e')
    plt.title(pltTitle)
    plt.legend(loc='upper left')
    mergerTimeDF = mergerTimeDF.T
    mergerTimeDF.columns = dfNames

    return mergerTimeDF

def investigateBinaryAfterMerge(dfs, dfNames):
    # flags = ['', ' -PN', ' -T', ' -PN -T']
    flags = [' -PN -T']
    # flags = ['', ' -PN']
    f1, ax1 = plt.subplots()
    f2, ax2 = plt.subplots()
    bins1 = np.logspace(-5, 0)
    bins2 = np.logspace(-10, 0)
    dataArr = []
    for i in range(len(dfs)):
        kepData = investigateBinary(dfs[i][1], flags[i])
        dataArr.append(kepData)

        ax1.hist(kepData['a'], label=dfNames[i], bins=bins1, histtype='step')
        ax2.hist(1-kepData['e'], label=dfNames[i], bins=bins2, histtype='step')

        print(dfNames[i] + ' done')


    ax1.set_xlabel('a [AU]')
    ax1.set_ylabel('count')
    ax1.set_xscale('log')
    ax1.legend()

    ax2.set_xlabel('1 - e')
    ax2.set_ylabel('count')
    ax2.set_xscale('log')
    ax2.legend()

    return dataArr

def plotKepData(dfs, dfNames, pltTitle):
    f1, ax1 = plt.subplots()
    f2, ax2 = plt.subplots()
    f3, ax3 = plt.subplots()
    bins1 = np.logspace(-2, 4)
    bins2 = np.logspace(-5, -2.5)
    bins3 = np.logspace(-5, 0)
    for i in range(len(dfs)):
        df = dfs[i]
        # ax1.hist(df['a'][df['e'] < 1], label=dfNames[i], bins=bins1, histtype='step', linewidth=4)
        # ax2.hist(1-df['e'][df['e'] < 1], label=dfNames[i], bins=bins2, histtype='step', linewidth=4)

        ax1.hist(df['a'], label=dfNames[i], bins=bins1, histtype='step', linewidth=4, cumulative=True, density=True)
        ax2.hist(1-df['e'], label=dfNames[i], bins=bins2, histtype='step', linewidth=4, cumulative=True, density=True)
        ax3.hist(df['a']* (1-df['e']), label=dfNames[i], bins=bins3, histtype='step', linewidth=4, cumulative=True, density=True)

    ax1.set_xlabel('a$_{final}$ [AU]')
    ax1.set_ylabel('Fraction')
    ax1.set_xscale('log')
    ax1.legend(loc='upper left')

    ax2.set_xlabel('1 - e$_{final}$')
    ax2.set_ylabel('Fraction')
    ax2.set_xscale('log')
    ax2.legend(loc='upper left')

    ax3.set_xlabel('pericenter distance [AU]')
    ax3.set_ylabel('Fraction')
    ax3.set_xscale('log')
    ax3.legend(loc='upper left')

    f1.suptitle(pltTitle)
    f2.suptitle(pltTitle)
    f3.suptitle(pltTitle)

def splitSingleBinaryMergers(dfs, dfNames, pltTitle,):

    sinMergersArr = []
    binMergersArr = []

    lenSinArr = []
    lenBinArr = []


    for i in range(len(dfs)):
        df = dfs[i][1]
        # df = df[(df['flybyFlag'] == 0) & (df['timeoutFlag'] == 0)]

        singleMergers = df[(df['colInd1'] == 0.0) | (df['colInd2'] == 0.0)]
        binMergers = df[(df['colInd1'] != 0.0) & (df['colInd2'] != 0.0)]

        sinMergersArr.append(singleMergers)
        binMergersArr.append(binMergers)

        lenSinArr.append(len(singleMergers))
        lenBinArr.append(len(binMergers))



        # print('t')
    mergerDF = pd.DataFrame(data=[lenSinArr, lenBinArr], index=['Single BH', 'Binary BH'], columns=dfNames)
    mergerDF.plot(kind='bar', rot=0).legend(loc="best")
    # plt.ylim(0,1400)
    plt.ylabel('Count')
    plt.xlabel('Merger component')
    plt.title(pltTitle)

    return sinMergersArr, binMergersArr



def countUniqueInteractions(dfs, dfNames):
    counts = []
    for i in range(len(dfs)):
        df = dfs[i]
        initParams = df[['a1', 'e1', 'bImp', 'vInfInit']]

        uniques = initParams.drop_duplicates()

        counts.append(len(uniques))

    return pd.Series(counts, index=dfNames)

def countUniqueInteractions2(outcomes, dfNames):
    # counts = []
    # outcomes = [flybys, exchanges, mergers, ions]
    outNames = ['Flyby', 'Exchange', 'Mergers', 'ions']
    for i in range(len(outcomes)):
        print(outNames[i])
        for j in range(len(outcomes[i])):
            df = outcomes[i][j]
            initParams = df[['a1', 'e1', 'bImp', 'vInfInit']]

            uniques = initParams.drop_duplicates()

            print(dfNames[j] + ': ' + str(len(uniques)))
            # counts.append(len(uniques))
        print('')

def countUniqueInteractions3(outcomes, dfNames):
    # counts = []
    # outcomes = [flybys, exchanges, mergers, ions]
    outNames = ['Flyby', 'Exchange', 'Mergers', 'ions']
    for i in range(len(outcomes)):
        print(outNames[i])
        for j in range(len(outcomes[i])):
            df = outcomes[i][j]
            initParams = df[['a1', 'e1', 'bImp', 'vInfInit']]

            uniques = initParams.drop_duplicates()

            print(dfNames[j] + ': ' + str(len(uniques)))
            # counts.append(len(uniques))
        print('')

    # return pd.Series(counts, index=dfNames)

def compareTGWBeforeAfter(dfs, dfNames, pltTitle):
    # plt.figure()
    bins = np.logspace(-3, 30)
    for i in range(len(dfs)):
        df = dfs[i]
        aInit = df['a1'].astype(float)
        eInit = df['e1'].astype(float)

        aFin = df['aFin'].astype(float)
        eFin = df['eFin'].astype(float)

        m1 = df['m10'].astype(float)
        m2 = df['m11'].astype(float)

        tGWInit = calcMergerTime(aInit, eInit, m1, m2)
        tGWFin = calcMergerTime(aFin, eFin, m1, m2)

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005


        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]


        # start with a rectangular Figure
        plt.figure(figsize=(8, 8))

        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.tick_params(direction='in', top=True, right=True)
        ax_histx = plt.axes(rect_histx)
        ax_histx.tick_params(direction='in', labelbottom=False)
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False)

        # the scatter plot:
        x = tGWInit
        y = tGWFin
        ax_scatter.scatter(x, y)

        # now determine nice limits by hand:
        binwidth = 0.25
        lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
        xlim = [np.amin(x), np.amax(x)]
        # ylim = [np.amin(y), np.amax(y)]
        ylim = [1e7, 1e30]
        ax_scatter.set_xlim(ylim)
        ax_scatter.set_ylim(ylim)

        ax_scatter.set_xscale('log')
        ax_scatter.set_yscale('log')

        ax_scatter.set_xlabel('T$_{GW, before}$')
        ax_scatter.set_ylabel('T$_{GW, after}$')

        bins = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 50)
        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation='horizontal')

        ax_histx.set_xlim(ylim)
        ax_histy.set_ylim(ylim)

        ax_histx.set_xscale('log')
        ax_histy.set_yscale('log')

        ax_histx.set_yticks([], [])
        ax_histy.set_xticks([], [])

        ax_histx.set_title(pltTitle + ' (' + dfNames[i] + ')')





        # tGWDiff = tGWInit - tGWFin

        # plt.hist(tGWDiff, bins=bins, histtype='step', label=dfNames[i], linewidth=4)

    # plt.xscale('log')
    # plt.xlabel('t$_{GW, init}$ - t$_{GW, fin}$ [yr]')
    # plt.ylabel('Count')
    # plt.title(pltTitle)
    # plt.legend()

def compareTGWBeforeAfterExchanges(dfs, dfNames, pltTitle):
    # plt.figure()
    bins = np.logspace(-3, 30)
    for i in range(len(dfs)):
        df = dfs[i]
        aInit = df['a1'].astype(float)
        eInit = df['e1'].astype(float)

        aFin = df['aFin'].astype(float)
        eFin = df['eFin'].astype(float)

        m1 = df['m0'].astype(float)
        m2 = []
        for j in range(len(df)):
            if df['escId'].iloc[j] == '2':
                m2.append(float(df['m10'].iloc[j]))
            else:
                m2.append(float(df['m11'].iloc[j]))

        tGWInit = calcMergerTime(aInit, eInit, m1, m2)
        tGWFin = calcMergerTime(aFin, eFin, m1, m2)

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005


        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]


        # start with a rectangular Figure
        plt.figure(figsize=(8, 8))

        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.tick_params(direction='in', top=True, right=True)
        ax_histx = plt.axes(rect_histx)
        ax_histx.tick_params(direction='in', labelbottom=False)
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False)

        # the scatter plot:
        x = tGWInit
        y = tGWFin
        ax_scatter.scatter(x, y)

        # now determine nice limits by hand:
        binwidth = 0.25
        lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
        xlim = [np.amin(x), np.amax(x)]
        # ylim = [np.amin(y), np.amax(y)]
        ylim = [9e5, 4e30]
        ax_scatter.set_xlim(ylim)
        ax_scatter.set_ylim(ylim)

        ax_scatter.set_xscale('log')
        ax_scatter.set_yscale('log')

        ax_scatter.set_xlabel('T$_{GW, before}$')
        ax_scatter.set_ylabel('T$_{GW, after}$')

        bins = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 50)
        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation='horizontal')

        ax_histx.set_xlim(ylim)
        ax_histy.set_ylim(ylim)

        ax_histx.set_xscale('log')
        ax_histy.set_yscale('log')

        ax_histx.set_yticks([], [])
        ax_histy.set_xticks([], [])

        ax_histx.set_title(pltTitle + ' (' + dfNames[i] + ')')

    # plt.xscale('log')
    # plt.xlabel('t$_{GW, init}$ - t$_{GW, fin}$ [yr]')
    # plt.ylabel('Count')
    # plt.title(pltTitle)
    # plt.legend()



def calcMergerTime(a, e, m1, m2):
    c = 63239.7263       # AU/yr
    G = 39.478           # AU3 * yr-2 * Msun-1

    tGW = 5/256 * ((c**5 * a**4 * (1-e**2)**(7/2)) / (G**3 * m1 * m2 * (m1 + m2)))

    return tGW

def initParamsWithMergerOnTop(dfs, mergers, dfNames, pltTitle):
    for i in range(len(dfs)):
        df = dfs[i]
        BHStarMerger = mergers[i][1]
        BHBHMerger = mergers[i][0]
        plt.figure()
        plt.scatter(df['a1'].astype(float), df['e1'].astype(float), label='All', s=25)
        plt.scatter(BHStarMerger['a1'].astype(float), BHStarMerger['e1'].astype(float), label='BH-Star mergers', s=50)
        plt.scatter(BHBHMerger['a1'].astype(float), BHBHMerger['e1'].astype(float), label='BH-BH mergers', s=50)
        plt.xlabel('a$_{init}$ [AU]')
        plt.ylabel('e$_{init}$')
        plt.xscale('log')
        plt.title(pltTitle + ' - ' + dfNames[i])
        plt.legend()


def countNumUniqueIncomplete(dfs, names):
    for i in range(len(dfs)):
        df = dfs[i]
        numSeeds = 0
        totalSeeds = 0
        for j in range(len(df) - 1):
            if (df.iloc[j]['a1'] != df.iloc[j+1]['a1']) or (df.iloc[j]['bImp'] != df.iloc[j+1]['bImp']):
                totalSeeds += numSeeds+1
                numSeeds = 0
            else:
                numSeeds += 1

        print(names[i] + ': ' + str(totalSeeds))


def checkVRadAfterMerge(dfs, dataNames, pltTitle, xlim, ylim):
    plt.figure()
    for i in range(len(dfs)):
        df = dfs[i]
        mask = (df['vRad'] != 0) & (df['flybyFlag'] == 0) & (df['timeoutFlag'] == 0)

        vRad = df['vRad'].astype(float)

        negativeVRad = vRad[(vRad < 0) & (df['flybyFlag'] == 0) & (df['timeoutFlag'] == 0)]
        positiveVRad = vRad[(vRad > 0) & (df['flybyFlag'] == 0) & (df['timeoutFlag'] == 0)]

        print('\n' + dataNames[i] + ':')
        print('Negative: ' + str(len(negativeVRad)))
        print('Positive: ' + str(len(positiveVRad)))

        # negativeVRad.hist()
        # positiveVRad.hist()
        # vRad.hist(histtype='step')

        plt.scatter(dfs[i]['tMerger'].astype(float)[mask], vRad[mask], label=dataNames[i], s=35)
    plt.axhline(0)
    plt.legend()
    plt.xlabel('t$_{merger}$ [yr]')
    plt.ylabel('V$_{rad}$ [km/s]')
    plt.xscale('log')
    plt.title(pltTitle)
    plt.xlim(xlim)
    plt.ylim(ylim)
    # plt.yscale('log')




def readBinsAfterMergeData(dataSet):
    if (dataSet == 'MOCCA Filtered'):
        noFlagsBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHBH_filtered_noFlags')
        noFlagsBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHStar_filtered_noFlags')

        PNBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHBH_filtered_PN')
        PNBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHStar_filtered_PN')

        TidesBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHBH_filtered_Tides')
        TidesBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHStar_filtered_Tides')

        PNTidesBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHBH_filtered_PNTides')
        PNTidesBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHStar_filtered_PNTides')

        # noFlagsBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/nonFlybys/binsAfterMerge/2-bh_noFlags_BHBH')
        # noFlagsBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/nonFlybys/binsAfterMerge/2-bh_noFlags_BHStar')

        # PNBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/nonFlybys/binsAfterMerge/2-bh_PN_BHBH')
        # PNBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/nonFlybys/binsAfterMerge/2-bh_PN_BHStar')

        # TidesBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/nonFlybys/binsAfterMerge/2-bh_Tides_BHBH')
        # TidesBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/nonFlybys/binsAfterMerge/2-bh_Tides_BHStar')

        # PNTidesBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/nonFlybys/binsAfterMerge/2-bh_PNTides_BHBH')
        # PNTidesBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/nonFlybys/binsAfterMerge/2-bh_PNTides_BHStar')
    if (dataSet == 'MOCCA Complete'):
        noFlagsBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHBH_noFlags')
        noFlagsBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHStar_noFlags')

        PNBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHBH_PN')
        PNBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHStar_PN')

        TidesBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHBH_Tides')
        TidesBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHStar_Tides')

        PNTidesBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHBH_PNTides')
        PNTidesBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/2BH_BHStar_PNTides')

        # noFlagsBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/binsAfterMerge/2bh_complete_BHBH_noFlags')
        # noFlagsBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/binsAfterMerge/2bh_complete_BHStar_noFlags')

        # PNBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/binsAfterMerge/2bh_complete_BHBH_PN')
        # PNBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/binsAfterMerge/2bh_complete_BHStar_PN')

        # TidesBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/binsAfterMerge/2bh_complete_BHBH_Tides')
        # TidesBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/binsAfterMerge/2bh_complete_BHStar_Tides')

        # PNTidesBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/binsAfterMerge/2bh_complete_BHBH_PNTides')
        # PNTidesBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/new/binsAfterMerge/2bh_complete_BHStar_PNTides')
    if (dataSet == 'Manual set'):
        noFlagsBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/manSetup_BHBH_noFlags')
        noFlagsBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/manSetup_BHStar_noFlags')

        PNBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/manSetup_BHBH_PN')
        PNBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/manSetup_BHStar_PN')

        TidesBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/manSetup_BHBH_Tides')
        TidesBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/manSetup_BHStar_Tides')

        PNTidesBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/manSetup_BHBH_PNTides')
        PNTidesBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/manSetup_BHStar_PNTides')

        # noFlagsBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/manSetup/higherParams/BHBH/binsAfterMerge/newTsunami/manSet_higherParams_BHBH_noFlags')
        # noFlagsBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/manSetup/higherParams/BHStar/binsAfterMerge/newTsunami/manSet_higherParams_BHStar_noFlags')

        # PNBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/manSetup/higherParams/BHBH/binsAfterMerge/newTsunami/manSet_higherParams_BHBH_PN')
        # PNBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/manSetup/higherParams/BHStar/binsAfterMerge/newTsunami/manSet_higherParams_BHStar_PN')

        # TidesBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/manSetup/higherParams/BHBH/binsAfterMerge/newTsunami/manSet_higherParams_BHBH_Tides')
        # TidesBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/manSetup/higherParams/BHStar/binsAfterMerge/newTsunami/manSet_higherParams_BHStar_Tides')

        # PNTidesBHBH = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/manSetup/higherParams/BHBH/binsAfterMerge/newTsunami/manSet_higherParams_BHBH_PNTides')
        # PNTidesBHStar = pd.read_pickle('/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/manSetup/higherParams/BHStar/binsAfterMerge/newTsunami/manSet_higherParams_BHStar_PNTides')

    binsBHBH = [noFlagsBHBH, PNBHBH, TidesBHBH, PNTidesBHBH]
    binsBHStar = [noFlagsBHStar, PNBHStar, TidesBHStar, PNTidesBHStar]

    return binsBHBH, binsBHStar


def splitMergersBinSin(binsAfterMerge, sinMergers, binMergers):
    afterMSin = []
    afterMBin = []
    for i in range(len(binsAfterMerge)):
        afterM = binsAfterMerge[i]
        afterM = afterM
        afterMIndex = afterM.index.values

        sinM = sinMergers[i]
        sinM = sinM
        sinMIndex = sinM.ind.astype(int)
        binM = binMergers[i]
        binM = binM
        binMIndex = binM.ind.astype(int)

        sinIntersect = np.intersect1d(afterMIndex, sinMIndex)
        binIntersect = np.intersect1d(afterMIndex, binMIndex)
        afterMergerBHSin = afterM.loc[sinIntersect]
        afterMergerBHBin = afterM.loc[binIntersect]

        afterMSin.append(afterMergerBHSin)
        afterMBin.append(afterMergerBHBin)

    return afterMSin, afterMBin

def combineInitAfter(afterAll, initAll, names):
    # plt.figure()
    # bins = np.logspace(-3, 2, 100)
    # for i in range(len(afterAll)):
    #     after = afterAll[i]
    #     init = initAll[i][1]

    #     afterIndex = after.index.values
    #     initIndex = init.ind.astype(float)

    #     intersect = np.intersect1d(afterIndex, initIndex)

    #     initData = init[init.ind.astype(int).isin(afterIndex)]
    #     afterData = after.loc[intersect]

    #     indMaxVrad = np.argmax(after.loc[intersect]['vRad'])
    #     vRad = after.loc[intersect]['vRad']

    #     # plt.hist(initData['a1'].astype(float) * (1 - initData['e1'].astype(float)), histtype='step', label=names[i], linewidth=5, bins=bins, cumulative=True, density=True)

    #     plt.scatter(initData['a1'].astype(float) * (1-initData['e1'].astype(float)), after.loc[intersect]['tMerger'], label=names[i], s=70)

    # plt.legend(loc='upper left')
    # plt.xlabel('t$_{merger}$ [yr]')
    # plt.ylabel('P$_{dist, init}$ [AU]')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('MOCCA set (BH-star init)')
    # plt.xlim(0.7e-2, 1.4e1)
    # plt.ylim(0.9e-2, 1.1e6)


    if (len(initAll) == 2):
        sinInitAll = initAll[0]
        binInitAll = initAll[1]

    for i in range(len(afterAll)):
        plt.figure()
        after = afterAll[i]
        if (len(initAll) != 2):
            init = initAll[i][1]
        else:
            sinInit = sinInitAll[i]
            binInit = binInitAll[i]


        afterIndex = after.index.values
        if (len(initAll) != 2):
            initIndex = init.ind.astype(float)
            intersect = np.intersect1d(afterIndex, initIndex)
            initData = init[init.ind.astype(int).isin(afterIndex)]
        else:
            sinIndex = sinInit.ind.astype(float)
            binIndex = binInit.ind.astype(float)

            intersectSin = np.intersect1d(afterIndex, sinIndex)
            intersectBin = np.intersect1d(afterIndex, binIndex)

            sinData = sinInit[sinInit.ind.astype(int).isin(afterIndex)]
            binData = binInit[binInit.ind.astype(int).isin(afterIndex)]

        G = 887.3515302300001    # AU, km/s, solar mass

        if (len(initAll) != 2):
            totalMass = initData['m10'].astype(float) + initData['m11'].astype(float) + initData['m0'].astype(float)
            redMass = ((initData['m10'].astype(float) + initData['m11'].astype(float)) * initData['m0'].astype(float))/totalMass
            vCrit = np.sqrt((G/redMass) * (initData['m10'].astype(float) * initData['m11'].astype(float)) / initData['a1'].astype(float))


            plt.scatter(initData['bImp'].astype(float)*initData['a1'].astype(float), initData['vInfInit'].astype(float)*vCrit, c=after.loc[intersect]['tMerger'], norm=mpl.colors.LogNorm(), s=70)
        else:
            totalMassSin = sinData['m10'].astype(float) + sinData['m11'].astype(float) + sinData['m0'].astype(float)
            redMassSin = ((sinData['m10'].astype(float) + sinData['m11'].astype(float)) * sinData['m0'].astype(float))/totalMassSin
            vCritSin = np.sqrt((G/redMassSin) * (sinData['m10'].astype(float) * sinData['m11'].astype(float)) / sinData['a1'].astype(float))

            totalMassBin = binData['m10'].astype(float) + binData['m11'].astype(float) + binData['m0'].astype(float)
            redMassBin = ((binData['m10'].astype(float) + binData['m11'].astype(float)) * binData['m0'].astype(float))/totalMassBin
            vCritBin = np.sqrt((G/redMassBin) * (binData['m10'].astype(float) * binData['m11'].astype(float)) / binData['a1'].astype(float))

            plt.scatter(sinData['bImp'].astype(float)*sinData['a1'].astype(float), sinData['vInfInit'].astype(float)*vCritSin, c=after.loc[intersectSin]['tMerger'], norm=mpl.colors.LogNorm(), s=70, marker='o', label='Single BH + star')
            plt.scatter(binData['bImp'].astype(float)*binData['a1'].astype(float), binData['vInfInit'].astype(float)*vCritBin, c=after.loc[intersectBin]['tMerger'], norm=mpl.colors.LogNorm(), s=70, marker='^', label='Binary BH + star')




            # plt.scatter(sinData['a1'].astype(float), 1-sinData['e1'].astype(float), c=after.loc[intersectSin]['tMerger'], norm=mpl.colors.LogNorm(), s=70, marker='o', label='Single BH + star')
            # plt.scatter(binData['a1'].astype(float), 1-binData['e1'].astype(float), c=after.loc[intersectBin]['tMerger'], norm=mpl.colors.LogNorm(), s=70, marker='^', label='Binary BH + star')
        plt.colorbar(label='t$_{merger}$ [yr]')

        plt.xlabel('a$_{init}$ [AU]')
        plt.ylabel('1-e$_{init}$')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('MOCCA Set - BHStar init (' + names[i] + ')')
        # plt.xlim(1e-2, 1e3)
        # plt.ylim(1e-4, 1.1e0)
        plt.legend()

def countFlybysAndTimeout(dfs, dfNames):
    for i in range(len(dfs)):
        df = dfs[i]
        numFlybys = np.sum(df['flybyFlag'][df['flybyFlag'] == 1])
        numTimeout = np.sum(df['timeoutFlag'][df['timeoutFlag'] == 1])

        print(dfNames[i])
        print('Flybys: ' + str(numFlybys))
        print('Timeout: ' + str(numTimeout))
        print('Total: ' + str(len(df['flybyFlag']))  + '\n\n')

def findMergerOverlap(dfs):
    first = dfs[0]
    second = dfs[1]

    indexFirst = first['ind']
    indexSecond = second['ind']

    overlap = np.intersect1d(indexFirst, indexSecond)

    first = first.set_index(indexFirst)
    second = second.set_index(indexSecond)

    overlapFirst = first.loc[overlap]
    overlapSecond = second.loc[overlap]

    return [overlapFirst,overlapSecond]


def checkMergerTypes(dfs, dfNames):
    for i in range(len(dfs)):
        df = dfs[i]
        BHBH = df[0]
        BHStar = df[1]

        BHBHCandidate = np.sum(BHBH['mergerType'] == 1)
        BHBHReal = np.sum(BHBH['mergerType'] == 2)
        BHBHTidal = np.sum(BHBH['mergerType'] == 3)

        BHStarCandidate = np.sum(BHStar['mergerType'] == 1)
        BHStarReal = np.sum(BHStar['mergerType'] == 2)
        BHStarTidal = np.sum(BHStar['mergerType'] == 3)

        print(dfNames[i])
        print('BHBH:')
        print('Candidate ' + str(BHBHCandidate) + ', Real: ' + str(BHBHReal) + ', Tidal: ' + str(BHBHTidal))
        print('BHStar:')
        print('Candidate ' + str(BHStarCandidate) + ', Real: ' + str(BHStarReal) + ', Tidal: ' + str(BHStarTidal) + '\n')

def removeGiantsFromDFs(dfs, dataNames):
    DFsWithoutGiants = []
    onlyGiants = []
    for i in range(len(dfs)):
        df = dfs[i]

        masses = df[['m0', 'm10', 'm11']]
        radii = df[['r0', 'r10', 'r11']]*214.9394693836

        minMass = masses.min(axis=1)
        maxRadii = radii.max(axis=1)

        giants = maxRadii > 1.5 * minMass**0.8
        notGiants = maxRadii < 1.5 * minMass**0.8

        # BHBHGiantsOld = maxRadiiBHBH > 2 * minMassBHBH
        # BHStarGiantsOld = maxRadiiBHStar > 2 * minMassBHStar

        gaints2 = df.loc[giants[giants].index]
        notGiants2 = df.loc[notGiants[notGiants].index]

        # BHBHGiants2Old = BHBH.loc[BHBHGiantsOld[BHBHGiantsOld].index]
        # BHStarGiants2Old = BHStar.loc[BHStarGiantsOld[BHStarGiantsOld].index]

        print(dataNames[i] + '\n')
        print('Total: ' + str(len(df)))
        print('Giants: ' + str(len(gaints2)))
        print('Not giants: ' + str(len(notGiants2)) + '\n')

        # BHBHIndex.append(BHBHGiants2['ind'])
        # BHStarIndex.append(BHStarGiants2['ind'])

        # np.savetxt('M2GiantsAll_' + str(i) + '.txt', BHBHGiants2['ind'].append(BHStarGiants2['ind']))

        DFsWithoutGiants.append(notGiants2)
        onlyGiants.append(gaints2)

        # BHBHIndexOld.append(BHBHGiants2Old)
        # BHStarIndexOld.append(BHStarGiants2Old)


    return DFsWithoutGiants, onlyGiants




""" Read data """
# Non-flybys (stars)
# dataNoFlags = pd.read_pickle('./dataframes/2BH_santai/new/nonFlybys/2-bh_noFlags_nonFlyby')
# dataPN = pd.read_pickle('./dataframes/2BH_santai/new/nonFlybys/2-bh_PN_nonFlyby')
# dataTides = pd.read_pickle('./dataframes/2BH_santai/new/nonFlybys/2-bh_Tides_nonFlyby')
# dataPNTides = pd.read_pickle('./dataframes/2BH_santai/new/nonFlybys/2-bh_PNTides_nonFlyby')

# dataNoFlags2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/MOCCA1_noFlags')
# dataPN2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/MOCCA1_PN')
# dataTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/MOCCA1_Tides')
# dataPNTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/MOCCA1_PNTides')

# dataNoFlags2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/MOCCA1_noFlags')
# dataPN2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/MOCCA1_PN')
# dataTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/MOCCA1_Tides')
# dataPNTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/MOCCA1_PNTides')

# dataNoFlags2.to_pickle('~/testCompress', compression='bz2')

dataNoFlags2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/MOCCA2_noFlags')
dataPN2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/MOCCA2_PN')
dataTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/MOCCA2_Tides')
dataPNTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/MOCCA2_PNTides')

# dataNoFlags2G = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/M1GiantRerun_noFlags_All')
# dataPN2G = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/M1GiantRerun_PN_All')
# dataTides2G = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/M1GiantRerun_Tides_All')
# dataPNTides2G = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/M1GiantRerun_PNTides_All')

# dataNoFlags2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/M2GiantRerun_noFlags_All')
# dataPN2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/M2GiantRerun_PN_All')
# dataTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/M2GiantRerun_Tides_All')
# dataPNTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/M2GiantRerun_PNTides_All')

# dataNoFlags2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/MOCCA2_noFlags')
# dataPN2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/MOCCA2_PN')
# dataTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/MOCCA2_Tides')
# dataPNTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/MOCCA2_PNTides')

# dataNoFlags = pd.read_pickle('./dataframes/2BH_santai/new/2-bh_noFlags')
# dataPN = pd.read_pickle('./dataframes/2BH_santai/new/2-bh_PN')
# dataTides = pd.read_pickle('./dataframes/2BH_santai/new/2-bh_tides')
# dataPNTides = pd.read_pickle('./dataframes/2BH_santai/new/2-bh_PNTides')

# dataNoFlags2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/MOCCA1_noFlags')
# dataPN2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/MOCCA1_PN')
# dataTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/MOCCA1_Tides')
# dataPNTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/MOCCA1_PNTides')

# noFlagsIncomplete = pd.read_pickle('./dataframes/2BH_santai/nonFlybys/incomplete/2bh_incomplete_noFlags')
# PNIncomplete = pd.read_pickle('./dataframes/2BH_santai/nonFlybys/incomplete/2bh_incomplete_PN')
# TidesIncomplete = pd.read_pickle('./dataframes/2BH_santai/nonFlybys/incomplete/2bh_incomplete_Tides')
# PNTidesIncomplete = pd.read_pickle('./dataframes/2BH_santai/nonFlybys/incomplete/2bh_incomplete_PNTides')



# 2BH + wd
# dataNoFlags = pd.read_pickle('./dataframes/2BH_santai/wd/2-bhs-wd-noFlags')
# dataPN = pd.read_pickle('./dataframes/2BH_santai/wd/2-bhs-wd-PN')
# dataTides = pd.read_pickle('./dataframes/2BH_santai/wd/2-bhs-wd-Tides')
# dataPNTides = pd.read_pickle('./dataframes/2BH_santai/wd/2-bhs-wd-PNTides')

# manual setup
# dataNoFlags = pd.read_pickle('./dataframes/2BH_santai/manSetup/manSet_2BH_noFlags')
# dataPN = pd.read_pickle('./dataframes/2BH_santai/manSetup/manSet_2BH_PN')
# dataTides = pd.read_pickle('./dataframes/2BH_santai/manSetup/manSet_2BH_Tides')
# dataPNTides = pd.read_pickle('./dataframes/2BH_santai/manSetup/manSet_2BH_PNTides')

# dataNoFlags = pd.read_pickle('./dataframes/2BH_santai/manSetup/manSet_2BH_noFlags_higherBMin')
# dataPN = pd.read_pickle('./dataframes/2BH_santai/manSetup/manSet_2BH_PN_higherBMin')
# dataTides = pd.read_pickle('./dataframes/2BH_santai/manSetup/manSet_2BH_Tides_higherBMin')
# dataPNTides = pd.read_pickle('./dataframes/2BH_santai/manSetup/manSet_2BH_PNTides_higherBMin')

# dataNoFlags = pd.read_pickle('./dataframes/2BH_santai/manSetup/BHStarBin/manSet_2BH_noFlags_BHStarBin')
# dataPN = pd.read_pickle('./dataframes/2BH_santai/manSetup/BHStarBin/manSet_2BH_PN_BHStarBin')
# dataTides = pd.read_pickle('./dataframes/2BH_santai/manSetup/BHStarBin/manSet_2BH_Tides_BHStarBin')
# dataPNTides = pd.read_pickle('./dataframes/2BH_santai/manSetup/BHStarBin/manSet_2BH_PNTides_BHStarBin')

# Higher params set (larger a and b)
# dataNoFlags = pd.read_pickle('./dataframes/2BH_santai/manSetup/higherParams/BHStar/manSet_higherParams_BHStar_noFlags')
# dataPN = pd.read_pickle('./dataframes/2BH_santai/manSetup/higherParams/BHStar/manSet_higherParams_BHStar_PN')
# dataTides = pd.read_pickle('./dataframes/2BH_santai/manSetup/higherParams/BHStar/manSet_higherParams_BHStar_Tides')
# dataPNTides = pd.read_pickle('./dataframes/2BH_santai/manSetup/higherParams/BHStar/manSet_higherParams_BHStar_PNTides')

# dataNoFlags = pd.read_pickle('./dataframes/2BH_santai/manSetup/higherParams/BHBH/manSet_higherParams_BHBH_noFlags')
# dataPN = pd.read_pickle('./dataframes/2BH_santai/manSetup/higherParams/BHBH/manSet_higherParams_BHBH_PN')
# dataTides = pd.read_pickle('./dataframes/2BH_santai/manSetup/higherParams/BHBH/manSet_higherParams_BHBH_Tides')
# dataPNTides = pd.read_pickle('./dataframes/2BH_santai/manSetup/higherParams/BHBH/manSet_higherParams_BHBH_PNTides')

# dataNoFlags = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/manualSet_BHStar_noFlags')
# dataPN = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/manualSet_BHStar_PN')
# dataTides = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/manualSet_BHStar_Tides')
# dataPNTides = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/manualSet_BHStar_PNTides')

# dataNoFlags = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/manualSet_BHBH_noFlags')
# dataPN = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/manualSet_BHBH_PN')
# dataTides = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/manualSet_BHBH_Tides')
# dataPNTides = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/manualSet_BHBH_PNTides')

# dataNoFlags2 = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/newTsunami/manualSet_BHStar_new_noFlags')
# dataPN2 = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/newTsunami/manualSet_BHStar_new_PN')
# dataTides2 = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/newTsunami/manualSet_BHStar_new_Tides')
# dataPNTides2 = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/newTsunami/manualSet_BHStar_new_PNTides')

# dataNoFlags2 = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/newTsunami/manualSet_BHBH_new_noFlags')
# dataPN2 = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/newTsunami/manualSet_BHBH_new_PN')
# dataTides2 = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/newTsunami/manualSet_BHBH_new_Tides')
# dataPNTides2 = pd.read_pickle('./dataframes/2BH_santai/manSetup/bImpSquared/newTsunami/manualSet_BHBH_new_PNTides')

# dataNoFlags2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211018/manualSet_BHStar_noFlags')
# dataPN2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211018/manualSet_BHStar_PN')
# dataTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211018/manualSet_BHStar_Tides')
# dataPNTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211018/manualSet_BHStar_PNTides')

# dataNoFlags2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211018/manualSet_BHBH_noFlags')
# dataPN2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211018/manualSet_BHBH_PN')
# dataTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211018/manualSet_BHBH_Tides')
# dataPNTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211018/manualSet_BHBH_PNTides')

# dataNoFlags2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/manualSet_BHStar_noFlags')
# dataPN2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/manualSet_BHStar_PN')
# dataTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/manualSet_BHStar_Tides')
# dataPNTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/manualSet_BHStar_PNTides')

# dataNoFlags2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/manualSet_BHBH_noFlags')
# dataPN2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/manualSet_BHBH_PN')
# dataTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/manualSet_BHBH_Tides')
# dataPNTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211020/manualSet_BHBH_PNTides')

# dataNoFlags2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/testSetup_BHStar_noFlags')
# dataPN2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/testSetup_BHStar_PN')
# dataTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/testSetup_BHStar_Tides')
# dataPNTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/testSetup_BHStar_PNTides')

# dataNoFlags2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/testSetup_BHBH_noFlags')
# dataPN2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/testSetup_BHBH_PN')
# dataTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/testSetup_BHBH_Tides')
# dataPNTides2 = pd.read_pickle('./dataframes/2BH_santai/tsunami211101/testSetup_BHBH_PNTides')


# data = [pd.concat([dataNoFlags,noFlagsIncomplete]), pd.concat([dataPN,PNIncomplete]), pd.concat([dataTides,TidesIncomplete]), pd.concat([dataPNTides,PNTidesIncomplete])]
# data = [noFlagsIncomplete, PNIncomplete, TidesIncomplete, PNTidesIncomplete]
# data = [dataNoFlags, dataPN, dataTides, dataPNTides]
data = [dataNoFlags2, dataPN2, dataTides2, dataPNTides2]
# data = [dataNoFlags2G, dataPN2G, dataTides2G, dataPNTides2G]
# data = [dataNoFlags, dataNoFlags2, dataPN, dataPN2, dataTides, dataTides2, dataPNTides, dataPNTides2]
# data = [dataNoFlags, dataPN, dataTides, dataPNTides, dataNoFlags2, dataPN2, dataTides2, dataPNTides2]
# data = [dataPN, dataPNNew]
# data = [dataNoFlags, dataPN]
# data = [dataPNNew]
# dataNames = ['No add. proc.', 'PN']
dataNames = ['No flags', 'PN', 'Tides', 'PN + Tides']
# dataNames = ['No flags', 'No flags (new)', 'PN', 'PN (new)', 'Tides', 'Tides (new)', 'PN + Tides', 'PN + Tides (new)']
# dataNames = ['No flags', 'PN', 'Tides', 'PN + Tides', 'No flags (new)', 'PN (new)', 'Tides (new)', 'PN + Tides (new)']
# dataNames = ['PN','PN (new)']
# dataNames = ['PN (new)']

data2, giants = removeGiantsFromDFs(data, dataNames)
# data = giants
# data = data2


# data = mergers1
""" Plot parameters """
sns.set_style("ticks")
# sns.set_palette("deep")
# sns.set(font_scale=3)
# plt.rcParams['xtick.minor.visible'], plt.rcParams['xtick.top'] = True,True
# plt.rcParams['xtick.minor.visible'], plt.rcParams['xtick.bottom'] = True,True
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.left'] = True,True
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.right'] = True,True
plt.rcParams['xtick.direction'], plt.rcParams['ytick.direction'] = 'in','in'
# plt.rcParams['axes.grid'] = True
plt.rcParams.update({'font.size': 35})
# plt.rcParams['savefig.transparent'] = True
plt.style.use('seaborn-deep')

""" Split initial BHBH and BHStar """
BHBHData, BHStarData = splitInitialConfs(data)
data = BHStarData
# data = BHBHData

""" Get number of outcomes """
# numOutcomes, flybys, exchanges, mergers1, ion, triple = countOutcomes(data, dataNames, '', True, False)
# numOutcomes, flybys, exchanges, mergers1, ion = countOutcomes(allGiants, dataNames, 'Test setup (initial BH-BH)', True, False)
# numOutcomesBHBH = countOutcomes(BHBHData, dataNames, 'Initial BHBH binary', False, False)
# numOutcomesBHStar = countOutcomes(BHStarData, dataNames, 'Initial BHStar binary', False, False)
# numOutcomes = countOutcomes(data, dataNames, 'Incomplete (rerun)', True, True)


# numOutcomes = numOutcomes.reindex( index = ['Flyby', 'Merger', 'Exchange', 'Ionization'])
""" Find initial parameter distribution """
# aDist = initialParameters(dataNoFlags, 'r')
# eDist = initialParameters(dataNoFlags, 'e1')
# bDist = initialParameters(dataNoFlags, 'bImp')

pltTitle = 'M1'
# initialParametersBHBHBHStar(BHBHData, BHStarData, 'mass', 'M$_{ \mathrm{star}}$ [M$_{\odot}$]', np.linspace(0, 3, 100), pltTitle, False)
# initialParametersBHBHBHStar(BHBHData, BHStarData, 'a1', 'a [AU]', np.logspace(-2, 4, 100), pltTitle, True)
# initialParametersBHBHBHStar(BHBHData, BHStarData, 'e1', 'e', np.linspace(0, 1, 100), pltTitle, False)
# initialParametersBHBHBHStar(BHBHData, BHStarData, 'bImp', 'b/a', np.logspace(-2, 3.3, 100), pltTitle, True)
# initialParametersBHBHBHStar(BHBHData, BHStarData, 'vInf', 'V$_{\infty}$ [km/s]', np.logspace(-1, 2.5, 100), pltTitle, True)
# initialParametersAll(BHBHData, BHStarData)

# plt.figure()
# plt.hist(dataPNTides['a1'].astype(float), bins=np.logspace(-2, 5, 200))
# plt.xscale('log')
# plt.ylabel('Count')
# plt.xlabel('a$_0$')

""" Get final BH-BH binary"""
# finalBHBHs = finalBHBHBinary(data)

""" Investigate incomplete interactions """
# uniqueMerger = investIncomplete(BHStarData, dataNames, 'a1')
# np.savetxt('./data/2bhs-nonFlyby_incomplete_PNTides.dat', dataPNTides[dataPNTides.encounterComplete == '0'].ind, fmt='%s')

""" investigate mergers """
# binsAfterMergeBHBH, binsAfterMergeBHStar = readBinsAfterMergeData('MOCCA Complete')
# binsAfterMergeBHBH, binsAfterMergeBHStar = readBinsAfterMergeData('MOCCA Filtered')
# binsAfterMergeBHBH, binsAfterMergeBHStar = readBinsAfterMergeData('Manual set')

# overlappingMergers = findMergerOverlap([binsAfterMergeBHBH[1], binsAfterMergeBHBH[3]])

# countFlybysAndTimeout(binsAfterMergeBHStar, dataNames)

# mergerTimesGWAfterMerge(overlappingMergers, ['PN', 'PN+tides'], np.logspace(-8,12,20), 'MOCCA set 1 (BH-BH, overlap)', True)

# mergers = investigateMergers(data, dataNames, 'Test setup (initial BH-BH)')
# BHStarMergers = investigateMergers(BHStarData, dataNames, 'Initial BHStar binary')
# BHStarMergers = investigateMergers(data, dataNames, 'BH + star binary')

# checkMergerTypes(mergers, dataNames)

# isIndexInMergers(195, mergers, dataNames)

# binsAfterMerge = investigateBinaryAfterMerge([mergers[3]], dataNames)

# testNoFlags = pd.read_pickle('./dataframes/2BH_santai/manSetup/higherParams/BHStar/binsAfterMerge/withDiffFCol/manSet_higherParams_BHBH_PNTides')

# binsAfterMerge[0].to_pickle('./dataframes/2BH_santai/manSetup/higherParams/BHStar/binsAfterMerge/newTsunami/manSet_higherParams_BHStar_PNTides')

# plotKepData(binsAfterMergeBHStar, dataNames, 'MOCCA set (filtered)')

# single, binary = splitSingleBinaryMergers(mergers, dataNames, 'Test setup (initial BH-BH)')

# afterMSin, afterMBin = splitMergersBinSin(binsAfterMergeBHBH, single, binary)

# combineInitAfter(binsAfterMerge, mergers, dataNames)
# combineInitAfter(binsAfterMerge, [single, binary], dataNames)

# binsAfterMerge = investigateBinaryAfterMerge(single, dataNames)
# plotKepData(binsAfterMerge, dataNames, 'Manual setup (init BHBH)')

# finalParamPlots(binary, dataNames, 'a1', np.logspace(-2, 1, int(1e3)), 'a$_{initial}$ [AU]', True, 'Manual setup (Binary BH-star)')
# finalParamPlots(binary, dataNames, 'bImp', np.logspace(-0.5, 2.5, int(1e2)), 'b$_{initial}/a$', True, 'Manual setup (Binary BH-star)')

def plotNHomo(dfs, dataNames):
    plt.figure()
    bins = np.linspace(0, 30)
    for i in range(len(dfs)):
        df = dfs[i]

        nHomo = df['Nhomo'].astype('int')
        plt.hist(nHomo, histtype='step', label=dataNames[i], linewidth=5, bins=bins)

        print('\n' + dataNames[i])
        print('Max: ' + str(np.amax(nHomo)))
        print('Mean: ' + str(np.mean(nHomo)))
        print('Median: ' + str(np.median(nHomo)))

    plt.xlabel('N$_{homo}$')
    plt.ylabel('Count')
    plt.legend()


""" Investigate how resonant encounters are """
# plotNHomo(flybys, dataNames)


def countMergersWhereAllSeedsMerge(dfs, dataNames):
    dfArr = []
    indsAllSeeds = []
    for i in range(len(dfs)):
        df = dfs[i]
        initialParams = df[['a1', 'e1', 'bImp', 'vInfInit']]

        newDF = initialParams.value_counts(dropna=False).reset_index(name='count')
        dfArr.append(newDF)

        numAllSeedsMerge = np.sum(newDF['count'] == 5)
        print(dataNames[i])
        print(str(numAllSeedsMerge) + '\n')

        allSeedsMerge = newDF[newDF['count'] == 5]
        allSeedsTemp = np.array([])
        for j in range(len(allSeedsMerge)):
            mask = (df[['a1', 'e1', 'bImp']] == allSeedsMerge[['a1', 'e1', 'bImp']].iloc[j]).all(axis=1)
            intWithThisSetup = df[mask]

            if 27860 in intWithThisSetup['ind'].values.astype(int): continue
            # print(intWithThisSetup.iloc[0]['ind'])
            # print(len(intWithThisSetup))

            allSeedsTemp = np.append(allSeedsTemp,intWithThisSetup['ind'].to_numpy())
        indsAllSeeds.append(allSeedsTemp)

    return dfArr, indsAllSeeds


# M1Flag = False
# giantsFlag = False
# mergerTimes = comparePeriTimeMergerTime(binary, M1Flag, giantsFlag)
# mergerInds = [mergerTimes[0][0]['ind'][mergerTimes[0][0]['mergerBeforeInteractionFlag'] == True],
#               mergerTimes[1][0]['ind'][mergerTimes[1][0]['mergerBeforeInteractionFlag'] == True],
#               mergerTimes[2][0]['ind'][mergerTimes[2][0]['mergerBeforeInteractionFlag'] == True],
#                 mergerTimes[3][0]['ind'][mergerTimes[3][0]['mergerBeforeInteractionFlag'] == True]]

# initialParamsWithCount, allSeedsMergeInd = countMergersWhereAllSeedsMerge(mergers1, dataNames)

def removeInstantMergers(dfToRemoveFrom, allSeedsMergeInd, dataNames):
    for i in range(len(dfToRemoveFrom)):
        df = dfToRemoveFrom[i]
        indsToRemove = allSeedsMergeInd[i]
        if len(indsToRemove) == 0:
            continue

        dfInd = df['ind']
        dropMask = dfInd.isin(indsToRemove)
        df = df.loc[~dropMask]

        dfToRemoveFrom[i] = df
    return dfToRemoveFrom

def plotTPeriTMerge(mergerTimes):
    plt.figure()
    names = ['No flags', 'PN', 'Tides', 'PN + tides']
    maxTMerge = 0
    for i in range(len(mergerTimes)):
        df = mergerTimes[i][0]

        plt.scatter(df['t2peri'], df['tMerge'], label=names[i])

        maxTMergeTemp = np.amax(df['tMerge'])
        if maxTMergeTemp> 0:
            maxTMerge = maxTMergeTemp


    plt.plot([0, maxTMerge], [0,maxTMerge], linestyle='dashed', label='T2peri = TMerge')
    plt.plot([0, maxTMerge], [0,0.5*maxTMerge], linestyle='dotted', label='T2peri = 0.5TMerge')
    plt.legend(loc='upper left', fontsize=21)
    plt.xlabel('T$_{peri}$ [yr]')
    plt.ylabel('T$_{merger}$ [yr]')
    plt.yscale('log')
    plt.xscale('log')

# plotTPeriTMerge(mergerTimes)

#%%
""" compare distance at merge to semi-major axis of binary """
def filterMergersDistance(mergerTimes, mergers1):
    # import pdb; pdb.set_trace()
    plt.figure()
    names = ['No flags', 'PN', 'Tides', 'PN + tides']
    for i in [0,1,2,3]:
        df = mergers1[i]
        times = mergerTimes[i][0][['t2peri', 'tMerge']]

        lastSnapshot = df['finalSnapshot']

        sepAtMerge = []
        aArr = []
        maxSep = 0
        for j in range(len(lastSnapshot)):
            if (times['t2peri'].iloc[j] * 0.5) < times['tMerge'].iloc[j]:
                ### continue if merger happens after interaction
                continue

            snapshot = lastSnapshot.iloc[j]

            mergerO1Index = df['colInd1'].iloc[j]
            mergerO2Index = df['colInd2'].iloc[j]

            mergerO1 = snapshot[mergerO1Index]
            mergerO2 = snapshot[mergerO2Index]

            COMPos = (mergerO1['m']*mergerO1[['rx', 'ry', 'rz']] + mergerO2['m']*mergerO2[['rx', 'ry', 'rz']])/(mergerO1['m'] + mergerO2['m'])

            if mergerO1Index == 0 and mergerO2Index == 1:
                otherObjIndex = 2
            elif mergerO1Index == 0 and mergerO2Index == 2:
                otherObjIndex = 1
            elif mergerO1Index == 1 and mergerO2Index == 2:
                otherObjIndex = 0

            otherObj = snapshot[otherObjIndex]

            pos = COMPos - otherObj[['rx', 'ry', 'rz']]
            posAbs = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)

            sepAtMerge.append(posAbs)
            aArr.append(df['a1'].iloc[j])

        if len(sepAtMerge) == 0:
            continue
        maxSepTemp = np.amax(sepAtMerge)
        if maxSepTemp > maxSep:
            maxSep = maxSepTemp

        plt.scatter(aArr, sepAtMerge, label=names[i])

        sepAtMerge = np.array(sepAtMerge).astype(float)
        aArr = np.array(aArr).astype(float)

        sepLessThanA = np.greater(sepAtMerge, aArr)
        sepLessThan3A = np.greater(sepAtMerge, 3*aArr)
        sepLessThan5A = np.greater(sepAtMerge, 5*aArr)
        sepLessThan10A = np.greater(sepAtMerge, 10*aArr)
        sepLessThan100A = np.greater(sepAtMerge, 100*aArr)

        print('\n' + names[i])
        print('Sep greater than a: ' + str(np.sum(sepLessThanA)))
        print('Sep greater than 3*a: ' + str(np.sum(sepLessThan3A)))
        print('Sep greater than 5*a: ' + str(np.sum(sepLessThan5A)))
        print('Sep greater than 10*a: ' + str(np.sum(sepLessThan10A)))
        print('Sep greater than 100*a: ' + str(np.sum(sepLessThan100A)))


    plt.plot([0, 1.4e2], [0,1.4e2], linestyle='dashed')
    plt.legend()
    plt.xlabel('a [AU]')
    plt.ylabel('Sep at merge [AU]')
    plt.xscale('log')
    plt.yscale('log')


# filterMergersDistance(mergerTimes, binary)

#%%
# mergers1 = removeInstantMergers(mergers1, mergerInds, dataNames)



""" Plots to summarize results """
def plotSummarize(dataNames):
    figs, axs = plt.subplots(nrows=3, ncols=2, sharex='col', squeeze=True)
    
    # Test setup
    # with BH:BH
    # dataBHStar = np.array([[0.073, 0.060, 0], [0.074, 0.066, 4.2e-4], [0.101, 0.116, 0], [0.0955, 0.109, 4.2e-4]]).T
    # dataBHBH = np.array([[0.188, 0], [0.179, 0], [0.243, 0], [0.248, 0]]).T

    # dfBHStarT = pd.DataFrame(data=dataBHStar, columns=dataNames, index=['BH$_1$:S', 'BH$_2$:S', 'BH:BH'])
    # dfBHBHT = pd.DataFrame(data=dataBHBH, columns=dataNames, index=['BH$_{1,2}$:S', 'BH:BH'])

    # without BH:BH
    dataBHStar = np.array([[0.073, 0.060], [0.074, 0.066], [0.101, 0.116], [0.0955, 0.109]]).T
    dataBHBH = np.array([[0.188], [0.179], [0.243], [0.248]]).T
    

    dfBHStarT = pd.DataFrame(data=dataBHStar, columns=dataNames, index=['BH$_1$:S', 'BH$_2$:S'])
    dfBHBHT = pd.DataFrame(data=dataBHBH, columns=dataNames, index=['BH$_{1,2}$:S'])
    
    errorsBHStar = np.sqrt(dfBHStarT*1000)/1000
    errorsBHBH = np.sqrt(dfBHBHT*1000)/1000
    
    import pdb; pdb.set_trace()
    
    dfBHStarT.plot(kind='bar', rot=0, legend=False, ax=axs[0,0], yerr=errorsBHStar, capsize=4,)
    dfBHBHT.plot(kind='bar', rot=0, legend=False, ax=axs[0,1], yerr=errorsBHBH, capsize=4,)


    # M1

    # with BH:BH
    # dataBHStar = np.array([[0.0064, 0.01185, 0], [0.00622, 0.01303, 5.8e-4], [0.012249, 0.01497, 0], [0.014777, 0.015166, 3.8887e-4]]).T
    # dataBHBH = np.array([[0.0098644, 0], [0.00965668, 2.6e-5],  [0.0234665, 0], [0.02336692, 2.6e-5]]).T

    # dfBHStarM1 = pd.DataFrame(data=dataBHStar, columns=dataNames, index=['BH$_1$:S', 'BH$_2$:S', 'BH:BH'])
    # dfBHBHM2 = pd.DataFrame(data=dataBHBH, columns=dataNames, index=['BH$_{1,2}$:S', 'BH:BH'])

    # without BH:BH
    dataBHStar = np.array([[0.0064, 0.01185], [0.00622, 0.01303], [0.012249, 0.01497], [0.014777, 0.015166]]).T
    dataBHBH = np.array([[0.0098644], [0.00965668],  [0.0234665], [0.02336692]]).T
    

    dfBHStarM1 = pd.DataFrame(data=dataBHStar, columns=dataNames, index=['BH$_1$:S', 'BH$_2$:S'])
    dfBHBHM2 = pd.DataFrame(data=dataBHBH, columns=dataNames, index=['BH$_{1,2}$:S'])
    
    errorsBHStar = np.sqrt(dfBHStarM1*1000)/1000
    errorsBHBH = np.sqrt(dfBHBHM2*1000)/1000

    dfBHStarM1.plot(kind='bar', rot=0, legend=False, ax=axs[1,0], yerr=errorsBHStar, capsize=4)
    dfBHBHM2.plot(kind='bar', rot=0, legend=False, ax=axs[1,1], yerr=errorsBHBH, capsize=4)

    # M2

    # with BH:BH
    # dataBHStar = np.array([[0.00113367, 0.008966299, 5.153e-4], [0.0011348396, 0.00856288, 6.190034e-4], [0.0022614689, 0.01604566, 5.3844497e-4], [0.00226123, 0.0160439324, 6.4606e-4]]).T
    # dataBHBH = np.array([[0.01865097, 1.65e-4], [0.0186633, 3.30e-4], [0.042089, 1.65e-4], [0.04317656, 3.86e-4]]).T

    # dfBHStarM1 = pd.DataFrame(data=dataBHStar, columns=dataNames, index=['BH$_1$:S', 'BH$_2$:S', 'BH:BH'])
    # dfBHBHM2 = pd.DataFrame(data=dataBHBH, columns=dataNames, index=['BH$_{1,2}$:S', 'BH:BH'])

    # without BH:BH
    dataBHStar = np.array([[0.00113367, 0.008966299], [0.0011348396, 0.00856288], [0.0022614689, 0.01604566], [0.00226123, 0.0160439324]]).T
    dataBHBH = np.array([[0.01865097], [0.0186633], [0.042089], [0.04317656]]).T
    
    dfBHStarM1 = pd.DataFrame(data=dataBHStar, columns=dataNames, index=['BH$_1$:S', 'BH$_2$:S'])
    dfBHBHM2 = pd.DataFrame(data=dataBHBH, columns=dataNames, index=['BH$_{1,2}$:S'])
    
    errorsBHStar = np.sqrt(dfBHStarM1*1000)/1000
    errorsBHBH = np.sqrt(dfBHBHM2*1000)/1000
    
    # import pdb; pdb.set_trace()

    dfBHStarM1.plot(kind='bar', rot=0, legend=False, ax=axs[2,0], yerr=errorsBHStar, capsize=4)
    dfBHBHM2.plot(kind='bar', rot=0, legend=False, ax=axs[2,1], yerr=errorsBHBH, capsize=4)


    # plt.yscale('log')

    # axs[0,0].set(ylabel='Fraction',xlabel ='a [AU]', xscale='log')
    plt.subplots_adjust(wspace=0.1, hspace=0)

    axs[0,0].set(title='Setup 1')
    axs[0,1].set(title='Setup 2')


    axs[0,0].set_ylabel('Fraction')
    axs[1,0].set_ylabel('Fraction')
    axs[2,0].set_ylabel('Fraction')
    
    axs[0,1].set_yticks([0.2])
    axs[1,1].set_yticks([0.02])
    axs[2,1].set_yticks([0.025])

    plt.text(-0.55,0.13, 'Test\nsetup', ha='center', va='center')
    plt.text(-0.6,0.07, 'M1')
    plt.text(-0.6,0.02, 'M2')

    for i in range(3):
        for j in range(2):
            yticks = axs[i,j].yaxis.get_major_ticks()
            yticks[0].label1.set_visible(False)

        axs[i,1].yaxis.tick_right()
        yticks = axs[i,1].yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)
        print(yticks[0].label1)
        axs[i,1].yaxis.tick_right()





    plt.legend(bbox_to_anchor=(0.7, 3.5), fontsize=22, ncol=len(dataNames))


plotSummarize(dataNames)





#%%


def plotOutcomes(flybys, exchanges, mergers, ion, triple, dataNames):
    # outcomesColumns = ['Flyby', 'Exchange', 'Merger']
    outcomesColumns = ['Flyby','Exchange', 'Merger', 'Ionization']
    # outcomesColumns = ['Flyby','Exchange', 'Merger', 'Ionization', 'Stable triples']
    outcomes = pd.DataFrame(columns = outcomesColumns)
    for i in range(len(flybys)):
        outcomesSeries = pd.Series([len(flybys[i]), len(exchanges[i]), len(mergers[i]), len(ion[i])],
                                   name=dataNames[i], index = outcomesColumns)
        # outcomesSeries = pd.Series([len(flybys[i]), len(exchanges[i]), len(mergers[i]), len(ion[i]), len(triple[i])],
        #                            name=dataNames[i], index = outcomesColumns)
        outcomes = outcomes.append(outcomesSeries.T, ignore_index=True)

    outcomes = outcomes.T
    outcomes.columns = dataNames

    outcomes.sort_values(dataNames[i], ascending=False).plot(kind='bar', rot=0, legend=True)

    # outcomes.plot(kind='bar')

""" plot num outcomes after filtering out instant mergers """
plt.rcParams.update({'font.size': 70})
plt.rcParams.update({'font.size': 35})

# plotOutcomes(flybys, exchanges, mergers1, ion, triple, dataNames)
# plotOutcomesnumOutcomes, flybys, exchanges, mergers1, ion, triple = countOutcomes(data, dataNames, '', True, False)


name = 'MOCCA set 1'

xLim = (0.0006, 1.7*10**6)
yLim = (-40,40)

# checkVRadAfterMerge(single, dataNames, name + ' (BH-Star init, sin BH merge)', xLim, yLim)
# checkVRadAfterMerge(binary, dataNames, name + ' (BH-Star init, bin BH merge)', xLim, yLim)

# checkVRadAfterMerge(mergers1, dataNames, name + ' (BH-BH init)', xLim, yLim)

def afterMergeBoundOrNot(dfs, dataNames):
    unbound_arr = list()
    bound_arr = list()

    for i in range(len(dfs)):
        df = dfs[i]
        # aAfterMerge = df[['aFin', 'eFin']]
        a = df['a']
        e = df['e']
        unbound = (a < 0) | (e < 0) | (e > 1)

        print('\n' + dataNames[i])
        print('Num bound: ' + str(np.sum(~unbound)))
        print('Num unbound: ' + str(np.sum(unbound)))

        unbound_arr.append(df[unbound])
        bound_arr.append(df[~unbound])

    return bound_arr, unbound_arr





""" find if system after merge is bound or not """
# mergersBound, mergersUnbound = afterMergeBoundOrNot(mergers1, dataNames)

""" plot final parameter distributions """
# finalParamPlots(mergers, dataNames, 'a1', np.logspace(-3, 5, int(1e2)), 'a$_{initial}$ [AU]', True, 'BH:star mergers')
# finalParamPlots(mergers, dataNames, 'e1', np.logspace(-3, 0, int(1e2)), 'e$_{final}$', True, 'BH:star mergers')
# finalParamPlots(mergers, dataNames, 'e1', np.linspace(0, 1, int(1e2)), 'e$_{final}$', False, 'BH:star mergers')
# finalParamPlots(mergers, dataNames, 'bImp', np.logspace(-3, 2, int(1e2)), 'b/a', True, 'BH:star mergers')
# finalParamPlots(mergers, dataNames, 'vInfInit', np.logspace(-3, 2, int(1e2)), 'V$_{\infty}$/V$_C', True, 'BH:star mergers')
# finalParamPlots(mergers, dataNames, 'tMerger', np.logspace(-3, 8, int(1e2)), 'T_merger [yr]', True, 'BH:star mergers')

def checkIndexOfIncomplete(dfs, dataNames):
    for i in range(len(dfs)):
        df = dfs[i]
        index = df['ind'].astype(int)
        allIndexes = np.arange(0,2400)

        missingIndexes = []
        for ind in allIndexes:
            if not np.isin(ind, index):
                missingIndexes.append(ind)

        print(dataNames[i] + '\n')
        print(missingIndexes)
        print('\n')

# checkIndexOfIncomplete(data, dataNames)

def isIndexInMergers(index, mergers, names):
    for i in range(len(mergers)):
        BHBH = mergers[i][0]
        BHStar = mergers[i][1]

        if np.isin(index, BHBH.ind.astype(int)):
            print(names[i] + ' BHBH')
        elif np.isin(index, BHStar.ind.astype(int)):
            print(names[i] + ' BHStar')
        else:
            print(names[i] + ' no match')

def checkPDistAfterMerge(dfs, dataNames, pltTitle):
    plt.figure()
    for i in range(len(dfs)):
        dfFiltered = dfs[i][dfs[i] != 0]
        pDist = dfFiltered['a'] * (1-dfFiltered['e'])
        # vRad = dfs[i]['vRad'].astype(float)

        # negativeVRad = vRad[vRad < 0]
        # positiveVRad = vRad[vRad > 0]

        # print('\n' + dataNames[i] + ':')
        # print('Negative: ' + str(len(negativeVRad)))
        # print('Positive: ' + str(len(positiveVRad)))


        plt.scatter(dfFiltered['tMerger'].astype(float), pDist, label=dataNames[i], s=55)
    plt.axhline(0)
    plt.legend()
    plt.xlabel('t$_{merger}$ [yr]')
    plt.ylabel('P$_{dist}$ [AU]')
    plt.xscale('log')
    plt.title(pltTitle)
    plt.yscale('log')
    plt.xlim(1e0, 10e3)
    plt.ylim(1e-6, 1e-3)

# checkPDistAfterMerge(binsAfterMergeBHStar, dataNames, 'MOCCA Filtered set (BH-Star init)')

def checkNumberOfGiants(dfs, dataNames):
    BHBHIndex = []
    BHStarIndex = []

    BHBHIndexOld = []
    BHStarIndexOld = []

    allGiants = []
    for i in range(4):
        # df = dfs[i]
        BHBH = dfs[0][i]
        BHStar = dfs[1][i]

        BHBHMasses = BHBH[['m0', 'm10', 'm11']]
        BHStarMasses = BHStar[['m0', 'm10', 'm11']]

        BHBHRadii = BHBH[['r0', 'r10', 'r11']]*214.9394693836
        BHStarRadii = BHStar[['r0', 'r10', 'r11']]*214.9394693836

        minMassBHBH = BHBHMasses.min(axis=1)
        minMassBHStar = BHStarMasses.min(axis=1)

        maxRadiiBHBH = BHBHRadii.max(axis=1)
        maxRadiiBHStar = BHStarRadii.max(axis=1)

        BHBHGiants = maxRadiiBHBH > 1.5 * minMassBHBH**0.8
        BHStarGiants = maxRadiiBHStar > 1.5 * minMassBHStar**0.8

        # BHBHGiantsOld = maxRadiiBHBH > 2 * minMassBHBH
        # BHStarGiantsOld = maxRadiiBHStar > 2 * minMassBHStar

        BHBHGiants2 = BHBH.loc[BHBHGiants[BHBHGiants].index]
        BHStarGiants2 = BHStar.loc[BHStarGiants[BHStarGiants].index]

        # BHBHGiants2Old = BHBH.loc[BHBHGiantsOld[BHBHGiantsOld].index]
        # BHStarGiants2Old = BHStar.loc[BHStarGiantsOld[BHStarGiantsOld].index]

        print(dataNames[i] + '\n')
        print('BHBH: ' + str(len(BHBHGiants2)) + ' out of ' + str(len(BHBH)))
        print('BHStar: ' + str(len(BHStarGiants2)) + ' out of ' + str(len(BHStar)) + '\n')

        BHBHIndex.append(BHBHGiants2['ind'])
        BHStarIndex.append(BHStarGiants2['ind'])

        np.savetxt('M1GiantsAll_' + str(i) + '.txt', BHBHGiants2['ind'].append(BHStarGiants2['ind']))

        allGiants.append(BHBHGiants2.append(BHStarGiants2))

        # BHBHIndexOld.append(BHBHGiants2Old)
        # BHStarIndexOld.append(BHStarGiants2Old)


    return BHBHIndex, BHStarIndex, allGiants



# BHBHIndex, BHStarIndex, allGiants = checkNumberOfGiants([BHBHData, BHStarData], dataNames)
# BHBHIndex, BHStarIndex, BHBHOld, BHStarOld = checkNumberOfGiants(mergers, dataNames)


""" get num unique interactions """
# numUniqueInteractions = countUniqueInteractions(data, dataNames)
# countUniqueInteractions2([flybys, exchanges, mergers1, ion], dataNames)
# countUniqueInteractions2(mergers, ['BH-BH', 'BH-Star'])
# countUniqueInteractions2([single, binary], dataNames)
# countUniqueInteractions2([mergersBound, mergersUnbound], dataNames)

# flybys, exchanges, mergers1, ion

""" investigate incomplete """
# countNumUniqueIncomplete(BHStarData, dataNames)


def plotMergerTimes(initTimes, finalTimes, dataNames):
    for i in range(len(dataNames)):
        fig, ax = plt.subplots()
        plt.scatter(initTimes[dataNames[i]], finalTimes[dataNames[i]])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('t$_{\mathrm{GW, i}}$ [yr]')
        plt.ylabel('t$_{\mathrm{GW}}$ [yr]')
        plt.title(dataNames[i])

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')


        print('\n' + dataNames[i])
        print('Mean before: ' + str(np.median(initTimes[dataNames[i]])))
        print('Mean after: ' + str(np.median(finalTimes[dataNames[i]])))


""" investigate GW merger times """
bins = np.logspace(4, 30, 100)
# mergerTimes, mergerTimesInit = mergerTimesGW(flybys, dataNames, bins, 'Flybys', True)
# mergerTimes, mergerTimesInit = mergerTimesGW(flybys, dataNames, bins, 'Flybys', False, True)
# plotMergerTimes(mergerTimes, mergerTimesInit, dataNames)



""" find mergers in PN+tides which are not in any other"""
# mergersNoFlags = mergers1[0]['ind']
# mergersPN = mergers1[1]['ind']
# mergersTides = mergers1[2]['ind']
# mergersPNTides = mergers1[3]['ind']

# allExceptPNTides = pd.concat([mergersTides,mergersPN,mergersNoFlags]).drop_duplicates(keep='first')

# diff = mergersPNTides[~mergersPNTides.isin(allExceptPNTides)]

# diff = pd.concat([mergersTides,mergersPN]).drop_duplicates(keep=False)

""" params scatter mergers plots """
# tMergerPeriodAll = mergerParamsScatterPlots(mergers, dataNames, 'a1', 'tMerger', 'M1 (BH:star - BH-BH initial)', 'period [yr]', 't$_{merger}$ [yr]', True)
# tMergerPeriodSin = mergerParamsScatterPlots(single, dataNames, 'a1', 'tMerger', 'M1 (BH1:star - BH-Star initial)', 'period [yr]', 't$_{merger}$ [yr]', False)
# tMergerPeriodBin = mergerParamsScatterPlots(binary, dataNames, 'a1', 'tMerger', 'M1 (BH2:star - BH-Star initial)', 'period [yr]', 't$_{merger}$ [yr]', False)
# mergerParamsScatterPlots(BHBHMergers, dataNames, 'bImp', 'vInfInit', 'Init BHBH', 'b$_{initial}$ [a]', 'v$_{\infty}$')

# plt.figure()
# plt.scatter(BHBHMergers[0][1]['a1'].astype(float), BHBHMergers[0][1]['e1'].astype(float))
# plt.ylabel('e')
# plt.xlabel('a [AU]')

# initParamsWithMergerOnTop(data, mergers, dataNames, 'MOCCA set (filtered)')

""" compare tGW init final """
# compareTGWBeforeAfter(flybys, dataNames, 'MOCCA set 1 - flybys')
# compareTGWBeforeAfterExchanges(exchanges, dataNames, 'MOCCA set 1 - exchanges')

""" params plots """
# paramsScatterPlots(BHBHData, dataNames, 'a1', 'aFin', 'MOCCA set 1 (BH-BH)', 'a$_{initial}$ [AU]', 'a$_{final}$ [AU]')
# params2dHisto(BHBHData, dataNames, 'e1', 'eFin', 'MOCCA set 1', 'e$_{init}$', 'e$_{fin}$')
# paramsHisto(finalBHBHs, dataNames, 'e1', 'eFin', 'Manual set (Final BH-BH bin - flyby)', 'e')
# paramsHisto(finalBHBHs, dataNames, 'a1', 'aFin', 'Manual set (Final BH-BH bin - flyby)', 'a [AU]')
# paramsScatterPlots(BHStarData, dataNames, 'a1', 'binMass', 'MOCCA set 2 (BH [BH S])', 'a$_{initial}$ [AU]', 'Binary mass [M$_{\odot}$]')
# paramsScatterPlots(BHBHData, dataNames, 'a1', 'binMass', 'MOCCA set 2 (S [BH BH])', 'a$_{initial}$ [AU]', 'Binary mass [M$_{\odot}$]')

# weightsSeed = np.ones(len( data2BH[data2BH.encounterComplete == '1'] )) / len(data2BH[data2BH.encounterComplete == '1'])
# weightsNoSeed = np.ones(len(data2BHNoSeed[data2BHNoSeed.encounterComplete == '1'])) / len(data2BHNoSeed[data2BHNoSeed.encounterComplete == '1'])

# bins = [0,1,2]
# confsSeed = data2BH[data2BH.encounterComplete == '1'].escId.astype(float).rename('No seed')
# confsNoSeed = data2BHNoSeed[data2BHNoSeed.encounterComplete == '1'].escId.astype(float).rename('Seed')
# df = pd.concat([confsNoSeed, confsSeed], axis=1)

# # plt.hist([confsNoSeed/weightsNoSeed, confsSeed/weightsSeed])
# arr = np.histogram(data2BH[data2BH.encounterComplete == '1'].escId.astype(float), 3, weights=weightsSeed) #log=True,
# arrNoSeed = np.histogram(data2BHNoSeed[data2BHNoSeed.encounterComplete == '1'].escId.astype(float), 3, weights=weightsNoSeed) #log=True,

# """ kolla så att endast flybys / exchanges tas med """

# fracsSeed = pd.Series(arr[0], name='Seed')
# fracsNoSeed = pd.Series(arrNoSeed[0], name='No seed')

# df = pd.concat([fracsSeed, fracsNoSeed], axis=1)

# df.plot(kind='bar', legend=True)

# # plt.hist([fracsSeed, fracsNoSeed])


# plt.xticks([0.125,1.125,2.125], ['0', '1', '2'], rotation=0)
# plt.xlabel('Escaper id')
# plt.ylabel('Fraction')
# plt.title('2-BH (PN + tides)')


# numMergers = data2BH[(data2BH.colInd1 != '0') & (data2BH.colInd2 != '0')]
# numMergersNoSeed = data2BHNoSeed[(data2BHNoSeed.colInd1 != '0') & (data2BHNoSeed.colInd2 != '0')]
# for i in range(len(bins)):
#     plt.text(arr[1][i],arr[0][i],str(arr[0][i]))




def fixDataframes():
    # df = binsAfterMergeBHBH[0]
    path = '/home/lucas/Documents/Astronomi/article/dataframes/2BH_santai/afterMergerData/'
    names = ['2BH_BHBH_filtered_noFlags', '2BH_BHBH_filtered_PN', '2BH_BHBH_filtered_PNTides','2BH_BHBH_filtered_Tides',
              '2BH_BHBH_noFlags', '2BH_BHBH_PN', '2BH_BHBH_PNTides','2BH_BHBH_Tides',
              '2BH_BHStar_filtered_noFlags', '2BH_BHStar_filtered_PN', '2BH_BHStar_filtered_PNTides','2BH_BHStar_filtered_Tides',
              '2BH_BHStar_noFlags', '2BH_BHStar_PN', '2BH_BHStar_PNTides','2BH_BHStar_Tides',
              'manSetup_BHBH_noFlags', 'manSetup_BHBH_PN', 'manSetup_BHBH_PNTides','manSetup_BHBH_Tides',
              'manSetup_BHStar_noFlags', 'manSetup_BHStar_PN', 'manSetup_BHStar_PNTides','manSetup_BHStar_Tides']

    colFirst = ['ind', 'a1', 'e1', 'vInfInit', 'bImp', 'm0', 'm10', 'm11', 'r0', 'r10',
        'r11', 'pType0', 'pType10', 'pType11', 'encounterComplete', 'aFin',
        'eFin', 'vFin', 'escId', 'escM', 'Nhomo', 'Nhomo_fex', 'longest_ex',
        'longest_begin', 'first_ex', 'tex_cumulative', 'last_ex',
        'last_ex_begin', 'Nex', 'originalbin', 'hangle', 'status', 'ahyp',
        'ehyp', 'omegabin', 'ionization', 'colInd1', 'colInd2']
    colSecond = ['a', 'e', 'i','omega_AP', 'omega_LAN', 'T', 'EA',
                  'vRad', 'tMerger', 'fcol','flybyFlag', 'timeoutFlag','finalSnapshot']

    for i in range(len(names)):
        print(names[i])
        df = pd.read_pickle(path + names[i])

        dfFirst = df.iloc[::2][colFirst].reset_index(drop=True)
        dfSecond = df.iloc[1::2][colSecond].reset_index(drop=True)
        dfComb = dfFirst.join(dfSecond)

        pd.to_pickle(dfComb, path + names[i])
        print(len(dfComb))

# fixDataframes()


# plt.rcParams.update({'font.size': 75})

def createSeriesOfMergers(dfs, dfNames):
    BHBHArr = []
    BHStarArr = []

    splitMergersInNum = numOutcomes.drop('Merger')
    for i in range(len(dfs)):
        df = dfs[i]
        BHBHArr.append(len(df[0]))
        BHStarArr.append(len(df[1]))


    BHBHSeries = pd.Series(BHBHArr, index=dfNames, name='BH-BH mergers')
    BHStarSeries = pd.Series(BHStarArr, index=dfNames, name='BH-star mergers')

    print(BHBHSeries)
    print(BHStarSeries)

    splitMergersInNum = splitMergersInNum.append(BHBHSeries)
    splitMergersInNum = splitMergersInNum.append(BHStarSeries)

    fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # ax2 = plt.axes([.65, .6, .2, .2])
    splitMergersInNum.plot(kind='bar', rot=0, legend=False)
    # plt.title('MOCCA 2')
    # plt.legend()
    # plt.yticks([0, 4, 8, 12], ['0', '4', '8', '12'])
    # splitMergersInNum.loc['BH-BH mergers'].plot(kind='bar', rot=0, legend=True)
    # ax2.hist(splitMergersInNum.loc['BH-BH mergers'])
    # print(splitMergersInNum.loc['BH-BH mergers'])

# createSeriesOfMergers(mergers, dataNames)


def TGWCalc(a, e, m1, m2):
    c = 63239.7263       # AU/yr
    G = 39.478           # AU3 * yr-2 * Msun-1
    tGW = 5/256 * ((c**5 * a**4 * (1-e**2)**(7/2)) / (G**3 * m1 * m2 * (m1 + m2)))

    return tGW

""" manual Tgw calcs for example """
# n = 6056
# df1 = dataTides2
# df2 = dataPNTides2

# run1 = df1[df1['ind'] == n]
# run2 = df2[df2['ind'] == n]

# binPropsRun1 = { 'a': float(run1['aFin']), 'e': float(run1['eFin']), 'm1': run1['m0'], 'm2': run1['m10']}
# binPropsRun2 = { 'a': run2['a'], 'e': run2['e'], 'm1': run2['m0'], 'm2': run2['m10'] + run2['m11']}


# tGWRun1 = TGWCalc(binPropsRun1['a'], binPropsRun1['e'], binPropsRun1['m1'], binPropsRun1['m2']).iloc[0]
# tGWRun2 = TGWCalc(binPropsRun2['a'], binPropsRun2['e'], binPropsRun2['m1'], binPropsRun2['m2']).iloc[0]


def findRealIndexOfGiants(dfRerun, dfFirstRun, index):
    rerun = dfRerun.iloc[index]

    rerunInitParams = rerun[['a1', 'e1', 'bImp', 'vInfInit', 'm0', 'm10', 'm11']]
    firstRunInitParams = dfFirstRun[['a1', 'e1', 'bImp', 'vInfInit', 'm0', 'm10', 'm11']]

    print('t')


""" find real index for giants """
index = 1762

# findRealIndexOfGiants(dataPNTides2G, dataPNTides2, index)




