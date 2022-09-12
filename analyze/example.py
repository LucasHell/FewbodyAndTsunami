# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:12:10 2022

@author: Lucas
"""

#%% Initialize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("ticks")
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.left'] = True,True
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.right'] = True,True
plt.rcParams['xtick.direction'], plt.rcParams['ytick.direction'] = 'in','in'
plt.rcParams.update({'font.size': 35})
plt.style.use('seaborn-deep')


data = pd.read_pickle('./results/herbert_noFlags')

escId = data.escId.astype(float)
encounterComplete = data.encounterComplete.astype(float)
ionization = data.ionization.astype(float)
aFin = data.aFin.astype(float)
eFin = data.eFin.astype(float)
colInd1 = data.colInd1.astype(float)
colInd2 = data.colInd2.astype(float)

mask = ((escId == 0) & (encounterComplete == 1) & (ionization != 1) &
(colInd1 == 0) & (colInd2 == 0) & (aFin != 0) & (eFin != 0))

flybys = data[mask]

#%% Analyze
def calcAngMom(snap):
    p1 = snap.iloc[1][['x','y','z']]
    p2 = snap.iloc[2][['x','y','z']]
    
    v1 = snap.iloc[1][['vx','vy','vz']]
    v2 = snap.iloc[2][['vx','vy','vz']]
    
    dp = p1 - p2
    dv = v1 - v2
    
    angMom = np.cross(dp, dv)
    angMom_mag = np.sqrt((angMom**2).sum())
    
    return angMom_mag

angMomIni = []
angMomFin = []

for i in range(len(flybys)):
    interaction = flybys.iloc[i]
    iniSnap = pd.DataFrame(interaction['initialSnapshot'])
    finSnap = pd.DataFrame(interaction['finalSnapshot'])
    
    angMom_ini = calcAngMom(iniSnap)
    angMom_fin = calcAngMom(finSnap)
    
    angMomIni.append(angMom_ini)
    angMomFin.append(angMom_fin)
    
    
#%% Plots
fig, ax = plt.subplots()
plt.scatter(angMomIni, angMomFin)
plt.xlabel('AngMom_ini')
plt.ylabel('AngMom_fin')

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

ax.plot(lims, lims, 'k-', alpha=0.75, zorder=3)
# ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)


#%%
diff = np.array(angMomIni) - np.array(angMomFin)

plt.hist(diff, bins=np.linspace(-10, 10, 100))
# plt.xscale('log')











