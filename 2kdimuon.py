# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:57:31 2023

@author: QuarkNetPM

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read data and convert from DataFrame to Array of object
df = pd.read_csv('2kdimuon-Jpsi.csv')
data = df.to_numpy()
Nevents = len(data)

# create physics arrays
type = np.empty((Nevents, 1)).astype(str)
type[:, 0] = data[:, 0]
Q1 = np.zeros((Nevents, 1)).astype(int)
Q2 = np.zeros((Nevents, 1)).astype(int)
Q1[:, 0] = data[:, 10]
Q2[:, 0] = data[:, 18]
p4Mu1 = np.matrix(data[:, 3:7]).astype(float)
p4Mu2 = np.matrix(data[:, 11:15]).astype(float)

# initialize matrices
pDiMu = np.empty((Nevents, 1))
DiMuMass = np.empty((Nevents, 1))
DiMuMassOpp = np.zeros((Nevents, 1))
DiMuMass1G = np.zeros((Nevents, 1))
DiMuMass2G = np.zeros((Nevents, 1))
B3 = np.zeros((1, 3))
Ball = np.zeros((Nevents, 1))
BGG = np.zeros((Nevents, 1))
p4Mu1RF = np.matrix(np.zeros((Nevents, 4)))
p4Mu2RF = np.matrix(np.zeros((Nevents, 4)))
pDiMuRF = np.empty((Nevents, 1))
DiMuMassRF = np.empty((Nevents, 1))
DiMuMassOppRF = np.zeros((Nevents, 1))
DiMuMass1GRF = np.zeros((Nevents, 1))
DiMuMass2GRF = np.zeros((Nevents, 1))

# create diagnostic histograms
plt.figure(1, figsize=[20, 40])
plt.subplots(3,4,figsize=[15, 12])
plt.subplot(3, 4, 1)
plt.hist(type)
plt.xlabel('Muon Quality')
plt.subplot(3, 4, 3)
plt.title('Diagnotic Histograms of Data in Lab Frame')
plt.hist(Q1)
plt.xlabel('Charge of Mu1')
plt.subplot(3, 4, 4)
plt.hist(Q2)
plt.xlabel('Charge of Mu2')
plt.subplot(3, 4, 5)
plt.hist(p4Mu1[:, 0], bins=np.arange(0, 40, 0.5))
plt.xlabel('E1 (GeV)')
plt.subplot(3, 4, 6)
plt.hist(p4Mu1[:, 1], bins=np.arange(-10, 10, 0.5))
plt.xlabel('px1 (GeV/c)')
plt.subplot(3, 4, 7)
plt.hist(p4Mu1[:, 2], bins=np.arange(-10, 10, 0.5))
plt.xlabel('py1 (GeV/c)')
plt.subplot(3, 4, 8)
plt.hist(p4Mu1[:, 3], bins=np.arange(-25, 25, 0.5))
plt.xlabel('pz1 (GeV/c)')
plt.subplot(3, 4, 9)
plt.hist(p4Mu2[:, 0], bins=np.arange(0, 40, 0.5))
plt.xlabel('E2 (GeV)')
plt.subplot(3, 4, 10)
plt.hist(p4Mu2[:, 1], bins=np.arange(-10, 10, 0.5))
plt.xlabel('px2 (GeV/c)')
plt.subplot(3, 4, 11)
plt.hist(p4Mu2[:, 2], bins=np.arange(-10, 10, 0.5))
plt.xlabel('py2 (GeV/c)')
plt.subplot(3, 4, 12)
plt.hist(p4Mu2[:, 3], bins=np.arange(-25, 25, 0.5))
plt.xlabel('pz2 (GeV/c)')
plt.show()

# calculate the dimuon mass
p4DiMu = p4Mu1 + p4Mu2
sig = np.matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

for ie in range(0, Nevents):
    pDiMu[ie, 0] = np.sqrt(p4DiMu[ie, 1:4] @ np.transpose(p4DiMu[ie, 1:4]))
    DiMuMass[ie, 0] = np.sqrt(p4DiMu[ie, :] @ sig @ np.transpose(p4DiMu[ie, :]))
    
# plot diMuon energy, momentum and mass
plt.figure(2)
plt.subplots(1, 3, figsize=[30, 10])
plt.subplot(1, 3, 1)
plt.hist(p4DiMu[:, 0], bins=np.arange(0, 40, 0.5))
plt.xlabel('EDiMuon (GeV)')
plt.subplot(1, 3, 2)
plt.hist(pDiMu, bins=np.arange(0, 40, 0.5))
plt.xlabel('pDiMu (GeV/c)')
plt.title('DiMuon Energy, Momentum and Mass in Lab Frame')
plt.subplot(1, 3, 3)
plt.hist(DiMuMass, bins=np.arange(1.75, 5.25, 0.04))
plt.xlabel('DiMuon Mass (GeV/c$^2$)')
plt.ylabel('Number per 0.04 GeV/c$^2$')
plt.show()

# create DiMuon mass arrays with quality of muon selections
DiMuMassAll = DiMuMass

for ie in range(0, Nevents):
    if( Q1[ie, 0] * Q2[ie, 0] < 0 ):
        DiMuMassOpp[ie, 0] = DiMuMass[ie, 0]
        if ( (type[ie, 0] == 'GT') or (type[ie, 0] == 'GG') ):
            DiMuMass1G[ie, 0] = DiMuMass[ie, 0]
            if ( (type[ie, 0] == 'GG') ):
                DiMuMass2G[ie, 0] = DiMuMass[ie, 0]
                
# plot DiMuon mass with quality of muon selections
plt.figure(3)
plt.subplots(2, 2, figsize=[30, 14])
plt.subplot(2, 2, 1)
plt.hist(DiMuMassAll, bins=np.arange(1.75, 5.25, 0.04))
plt.xlabel('DiMuon Mass (GeV/c$^2$)')
plt.ylabel('Number per 0.04 GeV/c$^2$')
plt.title('All DiMuon Events in Lab Frame')
plt.subplot(2, 2, 2)
plt.hist(DiMuMassOpp, bins=np.arange(1.75, 5.25, 0.04))
plt.xlabel('DiMuon Mass (GeV/c$^2$)')
plt.ylabel('Number per 0.04 GeV/c$^2$')
plt.title('All Oppositely Charged DiMuon Events in Lab Frame')
plt.subplot(2, 2, 3)
plt.hist(DiMuMass1G, bins=np.arange(1.75, 5.25, 0.04))
plt.xlabel('DiMuon Mass (GeV/c$^2$)')
plt.ylabel('Number per 0.04 GeV/c$^2$')
plt.title('Oppositely Charged DiMuons Events with at Least 1 Global Muon in Lab Frame')
plt.subplot(2, 2, 4)
plt.hist(DiMuMass2G, bins=np.arange(1.75, 5.25, 0.04))
plt.xlabel('DiMuon Mass (GeV/c$^2$)')
plt.ylabel('Number per 0.04 GeV/c$^2$')
plt.title('Oppositely Charged DiMuons Events with 2 Global Muons in Lab Frame')
plt.show()


# The above analysis is performed with 4-momenta measured in the Lab Frame.
# Below we will use the Lorentz Transformation to transform p4Mu1 and P4Mu2
# to the DiMuon Rest Frame.

# Calculate the 4-velocity of the DiMuon, and from this 4-velocity calculate
# other quantitities needed for the Lorentz Transformation from the Lab Frame
# to the DiMuon Rest Frame.

for ie in range(0, Nevents):
    p4DiMu[0, :] = p4Mu1[ie, :] + p4Mu2[ie, :]
    B3[0, :] = [p4DiMu[0, 1], p4DiMu[0, 2], p4DiMu[0, 3]]/p4DiMu[0, 0]
    Bx = B3[0, 0]
    By = B3[0, 1]
    Bz = B3[0, 2]
    B = np.sqrt(Bx**2 + By**2 + Bz**2)
    Ball[ie, 0] = B
    gam = 1/np.sqrt(1 - B**2)
    
    # Create Lorentz Transformation Matrix for this event
    L = np.matrix([[gam,            -gam*Bx,            -gam*By,               -gam*Bz     ],
                   [-gam*Bx, 1+(gam-1)*Bx**2/B**2, (gam-1)*Bx*By/B**2,  (gam-1)*Bx*Bz/B**2 ],
                   [-gam*By,  (gam-1)*By*Bx/B**2, 1+(gam-1)*By*By/B**2, (gam-1)*By*Bz/B**2 ],
                   [-gam*Bz,  (gam-1)*Bz*Bx/B**2,  (gam-1)*Bz*By/B**2, 1+(gam-1)*Bz*Bz/B**2]])

    # Transform p4Mu1 and p4Mu2 to Rest Frame of the DiMuon
    p4Mu1RF[ie, :] = np.transpose(L @ np.transpose(p4Mu1[ie, :]))
    p4Mu2RF[ie, :] = np.transpose(L @ np.transpose(p4Mu2[ie, :]))
    
# create diagnostic histograms in DiMuon Rest Frame
plt.figure(4, figsize=[20, 40])
plt.subplots(3,4,figsize=[15, 12])
plt.subplot(3, 4, 1)
plt.hist(type)
plt.xlabel('Muon Quality')
plt.subplot(3, 4, 3)
plt.title('Diagnotic Histograms of Data in DiMuon Rest Frame')
plt.hist(Q1)
plt.xlabel('Charge of Mu1')
plt.subplot(3, 4, 4)
plt.hist(Q2)
plt.xlabel('Charge of Mu2')
plt.subplot(3, 4, 5)
plt.hist(p4Mu1RF[:, 0], bins=np.arange(0, 3, 0.1))
plt.xlabel('E1 (GeV)')
plt.subplot(3, 4, 6)
plt.hist(p4Mu1RF[:, 1], bins=np.arange(-4, 4, 0.1))
plt.xlabel('px1 (GeV/c)')
plt.subplot(3, 4, 7)
plt.hist(p4Mu1RF[:, 2], bins=np.arange(-4, 4, 0.1))
plt.xlabel('py1 (GeV/c)')
plt.subplot(3, 4, 8)
plt.hist(p4Mu1RF[:, 3], bins=np.arange(-4, 4, 0.1))
plt.xlabel('pz1 (GeV/c)')
plt.subplot(3, 4, 9)
plt.hist(p4Mu2RF[:, 0], bins=np.arange(0, 3, 0.1))
plt.xlabel('E2 (GeV)')
plt.subplot(3, 4, 10)
plt.hist(p4Mu2RF[:, 1], bins=np.arange(-4, 4, 0.1))
plt.xlabel('px2 (GeV/c)')
plt.subplot(3, 4, 11)
plt.hist(p4Mu2RF[:, 2], bins=np.arange(-4, 4, 0.1))
plt.xlabel('py2 (GeV/c)')
plt.subplot(3, 4, 12)
plt.hist(p4Mu2RF[:, 3], bins=np.arange(-4, 4, 0.1))
plt.xlabel('pz2 (GeV/c)')
plt.show()

# calculate the dimuon mass in DiMuon Rest Frame
p4DiMuRF = p4Mu1RF + p4Mu2RF
sig = np.matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

for ie in range(0, Nevents):
    pDiMuRF[ie, 0] = np.sqrt(p4DiMuRF[ie, 1:4] @ np.transpose(p4DiMuRF[ie, 1:4]))
    DiMuMassRF[ie, 0] = np.sqrt(p4DiMuRF[ie, :] @ sig @ np.transpose(p4DiMuRF[ie, :]))
    
# plot diMuon energy, momentum and mass in the DiMuon Rest Frame
plt.figure(5)
plt.subplots(1, 3, figsize=[30, 10])
plt.subplot(1, 3, 1)
plt.hist(p4DiMuRF[:, 0], bins=np.arange(1.75, 5.25, 0.04))
plt.xlabel('EDiMuon (GeV)')
plt.subplot(1, 3, 2)
plt.hist(pDiMuRF, bins=np.arange(-1e-10, 1e-10, 1e-12))
plt.xlabel('pDiMu (GeV/c)')
plt.title('DiMuon Energy, Momentum and Mass in DiMuon Rest Frame')
plt.subplot(1, 3, 3)
plt.hist(DiMuMassRF, bins=np.arange(1.75, 5.25, 0.04))
plt.xlabel('DiMuon Mass (GeV/c$^2$)')
plt.ylabel('Number per 0.04 GeV/c$^2$')
plt.show()

# create DiMuon mass arrays with quality of muon selections in the DiMuon Rest Frame
DiMuMassAllRF = DiMuMassRF

for ie in range(0, Nevents):
    if( Q1[ie, 0] * Q2[ie, 0] < 0 ):
        DiMuMassOppRF[ie, 0] = DiMuMassRF[ie, 0]
        if ( (type[ie, 0] == 'GT') or (type[ie, 0] == 'GG') ):
            DiMuMass1GRF[ie, 0] = DiMuMassRF[ie, 0]
            if (type[ie, 0] == 'GG'):
                DiMuMass2GRF[ie, 0] = DiMuMassRF[ie, 0]
                BGG[ie, 0] = Ball[ie, 0]
                
# plot DiMuon mass with quality of muon selections in the DiMuon Rest Frame
plt.figure(6)
plt.subplots(2, 2, figsize=[30, 14])
plt.subplot(2, 2, 1)
plt.hist(DiMuMassAllRF, bins=np.arange(1.75, 5.25, 0.04))
plt.xlabel('DiMuon Mass (GeV/c$^2$)')
plt.ylabel('Number per 0.04 GeV/c$^2$')
plt.title('All DiMuon Events in DiMuon Rest Frame')
plt.subplot(2, 2, 2)
plt.hist(DiMuMassOppRF, bins=np.arange(1.75, 5.25, 0.04))
plt.xlabel('DiMuon Mass (GeV/c$^2$)')
plt.ylabel('Number per 0.04 GeV/c$^2$')
plt.title('All Oppositely Charged DiMuon Events in DiMuon Rest Frame')
plt.subplot(2, 2, 3)
plt.hist(DiMuMass1GRF, bins=np.arange(1.75, 5.25, 0.04))
plt.xlabel('DiMuon Mass (GeV/c$^2$)')
plt.ylabel('Number per 0.04 GeV/c$^2$')
plt.title('Oppositely Charged DiMuons Events with at Least 1 Global Muon in DiMuon Rest Frame')
plt.subplot(2, 2, 4)
plt.hist(DiMuMass2GRF, bins=np.arange(1.75, 5.25, 0.04))
plt.xlabel('DiMuon Mass (GeV/c$^2$)')
plt.ylabel('Number per 0.04 GeV/c$^2$')
plt.title('Oppositely Charged DiMuons Events with 2 Global Muons in DiMuon Rest Frame')
plt.show()

plt.figure(7)
plt.hist(BGG, bins=np.arange(0.7, 1, 0.001))
plt.show()
