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
import pdb
from cart_kep import cart_2_kep


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
    m1 = snap.iloc[1]['mass']
    m2 = snap.iloc[2]['mass']
    
    p1 = snap.iloc[1][['x','y','z']]
    p2 = snap.iloc[2][['x','y','z']]
    
    v1 = snap.iloc[1][['vx','vy','vz']]
    v2 = snap.iloc[2][['vx','vy','vz']]
    
    dp = p1 - p2
    dv = v1 - v2
    
    angMom = np.cross(dp, dv)
    angMom_mag = np.sqrt((angMom**2).sum())
    
    alpha = np.arccos(angMom[0]/angMom_mag)     
    beta = np.arccos(angMom[1]/angMom_mag)
    gamma = np.arccos(angMom[2]/angMom_mag)
    
    # print(alpha,beta,gamma)
    
    COMPos = (m1*p1 + m2*p2)/(m1 + m2)
    COMVel = (m1*v1 + m2*v2)/(m1 + m2)
    
    pS = snap.iloc[0][['x','y','z']]
    vS = snap.iloc[0][['vx','vy','vz']]
    mS = snap.iloc[0]['mass']
    
    posVec = COMPos - pS
    velVec = COMVel - vS
    
    a,e,i,omega_AP,omega_LAN,T, EA = cart_2_kep(posVec, velVec, m1 + m2 + mS)
    
    rMin = -a*(e-1)
    
    # pdb.set_trace() 
    
    return angMom, angMom_mag, alpha, beta, gamma, rMin

angMomIni = []
angMomFin = []

angMomMagIni = []
angMomMagFin = []

anglesIni = []
anglesFin = []

rMin = []

df = pd.DataFrame(columns=['angMom_ini','angMomMag_ini','angMom_fin','angMomMag_fin', 'alpha_ini', 'alpha_fin', 'beta_ini', 'beta_fin', 'gamma_ini', 'gamma_fin', 'rMin'])

for i in range(len(flybys)):
    interaction = flybys.iloc[i]
    iniSnap = pd.DataFrame(interaction['initialSnapshot'])
    finSnap = pd.DataFrame(interaction['finalSnapshot'])
    
    angMom_ini, angMom_mag_ini, alphaI, betaI, gammaI, rMinI = calcAngMom(iniSnap)
    angMom_fin, angMom_mag_fin, alphaF, betaF, gammaF, rMinF = calcAngMom(finSnap)
    
    seriesTemp = pd.Series(data=[angMom_ini, angMom_mag_ini, angMom_fin, angMom_mag_fin, alphaI, alphaF, betaI, betaF, gammaI, gammaF, rMinI, interaction['a1'], interaction['e1']], index=['angMom_ini','angMomMag_ini','angMom_fin','angMomMag_fin', 'alpha_ini', 'alpha_fin', 'beta_ini', 'beta_fin', 'gamma_ini', 'gamma_fin', 'rMin', 'a_ini', 'e_ini'], name=i)
    
    df = df.append(seriesTemp, ignore_index=True)
    
    # pdb.set_trace() 
    
    
    
#%% Filter 
between = df[df['a_ini'].astype(float) > df['rMin']]

close = df[(df['a_ini'].astype(float) > 2*df['rMin']) & (df['a_ini'].astype(float) < df['rMin'])]

far = df[(df['a_ini'].astype(float) < 2*df['rMin']) & (df['a_ini'].astype(float) < df['rMin'])]
    
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











