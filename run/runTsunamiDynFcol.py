#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:33:15 2021

@author: lucas
"""

import pandas as pd
import numpy as np
import subprocess
from subprocess import PIPE
from cart_kep import cart_2_kep
import re
import signal
import os

def readOutputData():
    outputData = pd.read_csv('./temp/tsunami_ic2_info.dat', sep='    ', names=["data"], engine='python')
    lastRow = outputData.iloc[-1]["data"]
    #secondLastRow = outputData.iloc[-2]["data"]

    if ("incomplete" in lastRow):
        encounterComplete = 0
        aFin = 0
        eFin = 0
        vFin = 0
        escId = 3
        escM = 0

        Nhomo = 0
        Nhomo_fex = 0
        longest_ex = 0
        longest_begin = 0
        first_ex = 0
        tex_cumulative = 0
        last_ex = 0
        last_ex_begin = 0
        Nex = 0
        originalbin = 0
        hangle = 0
        status = 0
        ahyp = 0
        ehyp = 0
        omegabin = 0
        ionization = 0
    else:
        secondLastRow = outputData.iloc[-2]["data"]

        if ("breakup" in secondLastRow):
            encounterComplete = 1
            aFin = 0
            eFin = 0
            vFin = 0
            escId = 3
            escM = 0

            Nhomo = 0
            Nhomo_fex = 0
            longest_ex = 0
            longest_begin = 0
            first_ex = 0
            tex_cumulative = 0
            last_ex = 0
            last_ex_begin = 0
            Nex = 0
            originalbin = 0
            hangle = 0
            status = 0
            ahyp = 0
            ehyp = 0
            omegabin = 0
            ionization = 1
        elif ("triple formation" in secondLastRow):
            encounterComplete = 1
            aFin = 0
            eFin = 0
            vFin = 0
            escId = 3
            escM = 0

            Nhomo = 0
            Nhomo_fex = 0
            longest_ex = 0
            longest_begin = 0
            first_ex = 0
            tex_cumulative = 0
            last_ex = 0
            last_ex_begin = 0
            Nex = 0
            originalbin = 0
            hangle = 0
            status = 0
            ahyp = 0
            ehyp = 0
            omegabin = 0
            ionization = 0

        else:
            encounterComplete = 1
            aFin = re.search('abin=(.*) au', secondLastRow).group(1)                    # au
            eFin = re.search('ebin=(.*)', secondLastRow).group(1)[:-1]
            vFin = re.search('vesc=(.*) km/s', secondLastRow).group(1)                  # km/s
            escId = re.search('id=(.*),', secondLastRow).group(1)[0]
            escM = re.search('m=(.*) MSun', secondLastRow).group(1)

            Nhomo = re.search(r'Nhomo=(.+?);', lastRow).group(1)
            Nhomo_fex = re.search('Nhomo_fex=(.+?) ', lastRow).group(1)
            longest_ex = re.search('longest_ex=(.+?) yr;', lastRow).group(1)            # yr
            longest_begin = re.search('longest_begin=(.+?) yr;', lastRow).group(1)      # yr
            first_ex = re.search('first_ex=(.+?) yr;', lastRow).group(1)                # yr
            tex_cumulative = re.search('tex_cumulative=(.+?) yr;', lastRow).group(1)    # yr
            last_ex = re.search('last_ex=(.+?) yr;', lastRow).group(1)                  # yr
            last_ex_begin = re.search('last_ex_begin=(.+?) yr;', lastRow).group(1)      # yr
            Nex = re.search('Nex=(.+?);', lastRow).group(1)
            originalbin = re.search('originalbin=(.+?);', lastRow).group(1)
            hangle = re.search('hangle=(.+?);', lastRow).group(1)
            status = re.search('status=(.+?);', lastRow).group(1)
            ahyp = re.search('ahyp=(.+?);', lastRow).group(1)
            ehyp = re.search('ehyp=(.+?);', lastRow).group(1)
            omegabin = lastRow.split('omegabin=')[1]
            ionization = 0


    return [encounterComplete, aFin, eFin, vFin, escId, escM, Nhomo, Nhomo_fex, longest_ex, longest_begin, first_ex, tex_cumulative,
            last_ex, last_ex_begin, Nex, originalbin, hangle, status, ahyp, ehyp, omegabin, ionization]

def readCollisionData():
    collisionData = pd.read_csv('./temp/tsunami_ic2_collision.dat', sep='   ', names=collisionColumns, engine='python')
    typeOfMerger = collisionData['rx'].iloc[0] # 2 for merger, 3 for tidal disruption

    mergerO1Index = collisionData["index"].iloc[1].astype(int)
    mergerO2Index = collisionData["index"].iloc[2].astype(int)

    trajData = pd.read_csv('./temp/tsunami_ic2_output.dat', sep='   ', names=['rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'm'], engine='python')
    obj1 = trajData[::3].iloc[-1]
    obj2 = trajData[1::3].iloc[-1]
    obj3 = trajData[2::3].iloc[-1]
    objs = [obj1, obj2, obj3]

    try:
        mergerO1 = objs[mergerO1Index].astype(float)
        mergerO2 = objs[mergerO2Index].astype(float)
    except ValueError:
        # print('continue - cannot convert trajectory to float - index: ' + str(index))
        return pd.Series([0,0,0,0,0,0,0,0,0,0,0,fcol,0,0], index=['colInd1', 'coldInd2','a', 'e', 'i', 'omega_AP', 'omega_LAN', 'T', 'EA', 'vRad', 'tMerger', 'fcol', 'flybyFlag', 'timeoutFlag'])

    COMPos = (mergerO1['m']*mergerO1[['rx', 'ry', 'rz']] + mergerO2['m']*mergerO2[['rx', 'ry', 'rz']])/(mergerO1['m'] + mergerO2['m'])
    COMVel = (mergerO1['m']*mergerO1[['vx', 'vy', 'vz']] + mergerO2['m']*mergerO2[['vx', 'vy', 'vz']])/(mergerO1['m'] + mergerO2['m'])


    if mergerO1Index == 0 and mergerO2Index == 1:
        otherObjIndex = 2
    elif mergerO1Index == 0 and mergerO2Index == 2:
        otherObjIndex = 1
    elif mergerO1Index == 1 and mergerO2Index == 2:
        otherObjIndex = 0

    otherObj = objs[otherObjIndex].astype(float)

    pos = COMPos - otherObj[['rx', 'ry', 'rz']]
    vel = (COMVel - otherObj[['vx', 'vy', 'vz']])/(2*np.pi) * 4.74057172

    posAbs = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    posWithAbs = pos/posAbs
    vRad = np.dot(vel, posWithAbs)


    totMass = obj1['m'] + obj2['m'] + obj3['m']
    kepCords = cart_2_kep(pos.to_numpy(), vel.to_numpy(), totMass)
    subprocess.run('rm ./temp/tsunami_ic2_collision.dat', shell=True)          # remove collision file

    subprocess.run('rm ./temp/tsunami_ic2_output.dat', shell=True)          # remove output file
    return pd.Series((mergerO1Index, mergerO2Index) + kepCords + (vRad,collisionData["index"].iloc[0], fcol,0,typeOfMerger,0, objs), index=['colInd1', 'coldInd2','a', 'e', 'i', 'omega_AP', 'omega_LAN', 'T', 'EA', 'vRad', 'tMerger', 'fcol', 'flybyFlag', 'mergerType', 'timeoutFlag', 'finalSnapshot'])

flags = ''
# pathFewbody = '~/tsunami-santai/fewbodyToTsunami/fewbody-dev-tsunami/'
# pathTsunami = '~/tsunami-santai/tsunami-santai_230821/build/bin/'

pathFewbody = '~/Documents/Astronomi/article/tsunami_python_new/fewbody/'
# pathTsunami = '~/Documents/Astronomi/article/tsunami-santai_290620/build/bin/'
pathTsunami = '~/Documents/Astronomi/article/tsunami-santai_240921/build/bin/'
pathTsunami = '~/Documents/Astronomi/article/tsunami-santai_211020/build/bin/'

""" read fewbody data """
data = pd.read_csv('~/Documents/Astronomi/article/data/2bhs-input-stars-radFix.dat', sep='./', names=['ID', 'rest'], engine='python')
# data = pd.read_csv('~/article/data/manInteractions_2BH_tsunami_BHBH_fixedImpB.dat', sep='./', names=['ID', 'rest'], engine='python')[140:150]
# data = pd.read_csv('../inputs/manInteractions_2BH_tsunami_BHBH_fixedImpB.dat', sep='./', names=['ID', 'rest'], engine='python')
rest = data['rest'].iloc[10374:10376].reset_index(drop=True)
#seed = np.loadtxt('seed.txt')

colNames = ['rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'm', 't', 'r', 'pType']
collisionColumns = ["index", "rx", "ry", "rz", "vx", "vy", "vz", "m", "r"]
resultsColumns = ["ind", "a1", "e1", "vInfInit", "bImp", "m0", "m10", "m11", "r0", "r10", "r11", "pType0", "pType10", "pType11",
                  "encounterComplete", "aFin", "eFin", "vFin", "escId", "escM", "Nhomo", "Nhomo_fex", "longest_ex",
                  "longest_begin", "first_ex", "tex_cumulative", "last_ex", "last_ex_begin", "Nex", "originalbin",
                  "hangle", "status", "ahyp", "ehyp", "omegabin", "ionization", "colInd1", "colInd2",'a', 'e', 'i', 'omega_AP',
                  'omega_LAN', 'T', 'EA', 'vRad', 'tMerger', 'fcol', 'flybyFlag', 'mergerType','timeoutFlag', 'finalSnapshot']
resultsDF = pd.DataFrame(columns=resultsColumns)

def runInteraction():
        timeout = 5
        tsunamiProcess = subprocess.Popen('timeout 30 ' + pathTsunami + 'tsunami.x ./temp/tsunami_ic2.dat -N 3' + tsunamiFlags,shell=True, bufsize=0, stdin=None,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        def _handler(signum, frame):
            # print('Timeout of %s sec reached, stopping execution' % timeout)
            tsunamiProcess.kill()
            raise RuntimeError('Timeout')

        signal.signal(signal.SIGALRM, _handler)
        try:
            while True:
                signal.alarm(int(timeout))
                inline = tsunamiProcess.stdout.readline()
                if not inline:
                    break
                signal.alarm(0)
        except RuntimeError:
            return 0

        return 1

for i in range(len(rest)):
    """ save index to file for debugging """
    indexOut = open('lastIndex.txt', 'w')
    indexOut.write(str(i) + '\n')
    indexOut.close()

    """ create Tsunami input data """
    createInputData = subprocess.run(pathFewbody + rest[i] + ' -F ./temp/tsunami_ic.dat', shell=True, stdout=PIPE, stderr=PIPE)
    vInfInit = re.search('vinf=(.+?) ', rest[i]).group(1)
    bImp = re.search('b=(.+?) ', rest[i]).group(1)
    aInit = re.search('a1=(.+?) ', rest[i]).group(1)
    eInit = re.search('e1=(.+?) ', rest[i]).group(1)

    inputData = pd.read_csv('./temp/tsunami_ic.dat', sep='   ', names=colNames, engine='python')

    """ get star mass and change pType if neccessary """
    if (0 not in inputData["pType"].to_numpy()):
        tripleBHOut = open('tripleBHIndex.txt', 'a')
        tripleBHOut.write(str(i) + '\n')
        tripleBHOut.close()
        continue

    star = np.where(inputData["pType"] == 0)[0][0]
    starMass = inputData['m'].iloc[star]

    if float(starMass) > 0.7:
        inputData["pType"].iloc[star] = 4

    np.savetxt("./temp/tsunami_ic2.dat", inputData)


    inputDataForSave = [i, aInit, eInit, vInfInit, bImp,
                        inputData['m'].iloc[0], inputData['m'].iloc[1], inputData['m'].iloc[2],
                        inputData['r'].iloc[0], inputData['r'].iloc[1], inputData['r'].iloc[2],
                        inputData['pType'].iloc[0], inputData['pType'].iloc[1], inputData['pType'].iloc[2]]

    """ run tsunami """
    fcolArr = [2, 5, 10, 20, 40, 80]
    for fcol in fcolArr:
        if (fcol == 80):
            # print('interaction not complete with fcol 40, skip \n')
            outputData = [1]
            outputData.extend(np.zeros(32).tolist())
            outputData.extend([fcol,0,1,0])
            inputDataForSave = pd.Series(np.append(inputDataForSave, outputData), index=resultsColumns)
            resultsDF = resultsDF.append(inputDataForSave, ignore_index=True)
            break

        tsunamiFlags = ' -ft 20000000000 -L au -fcol ' + str(fcol) + ' ' + flags
        tsunamiResults = runInteraction()
        if (tsunamiResults == 0):
            # print('continue - tsunami timedout with fcol ' + str(fcol) + '\n')
            continue
        elif not os.path.getsize('./temp/tsunami_ic2_info.dat'):
            # print('info file empty')
            continue

        """ read tsunami output and save data """
        try:
            outputData = pd.read_csv('./temp/tsunami_ic2_info.dat', sep='    ', names=["data"], engine='python')
        except IOError:
            continue
        try:
            # check if collision file exists, if it does: save collision data
            outputData = readOutputData()
            collisionData = readCollisionData()
            outputData.extend(collisionData)
            inputDataForSave = pd.Series(np.append(inputDataForSave, outputData), index=resultsColumns)
            # subprocess.call('rm ./temp/tsunami_ic2_collision.dat', shell=True)          # remove collision file

        except IOError:
            # save non-collison data
            outputData = readOutputData()
            outputData.extend([0,0,0,0,0,0,0,0,0,0,0,fcol,0,0,0,0])
            inputDataForSave = pd.Series(np.append(inputDataForSave, outputData), index=resultsColumns)

        resultsDF = resultsDF.append(inputDataForSave, ignore_index=True)
        break

# resultsDF.to_pickle('manSet_run')








