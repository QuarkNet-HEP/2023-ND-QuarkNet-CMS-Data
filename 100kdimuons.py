
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def colors():
    ax = plt.gca()
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for _, value in ax.spines.items():
        value.set_color('white')


# read data and create data array

df = pd.read_csv('100kdimuon.csv')
data = df.to_numpy()
Nevents = len(data)

# create useful physics arrays

Type = np.empty((Nevents, 1)).astype(str)
Type[:, 0] = data[:, 0]
Q1 = np.zeros((Nevents, 1)).astype(int)
Q2 = np.zeros((Nevents, 1)).astype(int)
Q1[:, 0] = np.array(data[:, 10].astype(int))
Q2[:, 0] = np.array(data[:, 18].astype(int))
p4Mu1 = np.matrix(data[:, 3:7]).astype(float)
p4Mu2 = np.matrix(data[:, 11:15]).astype(float)
sig = np.matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

# initialize matrices

DiMuMass = np.empty((Nevents, 1))
pDiMu = np.empty((Nevents, 1))
DiMuMassOpp = np.zeros((Nevents, 1))
DiMuMass1G = np.zeros((Nevents, 1))
DiMuMass2G = np.zeros((Nevents, 1))
B3 = np.zeros((Nevents, 3))
Ball = np.zeros((Nevents, 1))
BGG = np.zeros((Nevents, 1))
p4Mu1RF = np.matrix(np.zeros((Nevents, 4)))
p4Mu2RF = np.matrix(np.zeros((Nevents, 4)))
DiMuMassRF = np.zeros((Nevents, 1))
pDiMuRF = np.empty((Nevents, 1))
DiMuMassOppRF = np.zeros((Nevents, 1))
DiMuMass1GRF = np.zeros((Nevents, 1))
DiMuMass2GRF = np.zeros((Nevents, 1))
BallX = np.zeros((Nevents, 1))
DiMuMassX = np.zeros((Nevents, 1))

# create diagnostic plots

plt.figure(1)
plt.subplots(3, 4, figsize=[15, 12])
plt.subplot(3, 4, 1)
plt.xlabel('Muon Quality', color='white')
colors()
plt.hist(Type, color='mediumseagreen')
plt.subplot(3, 4, 2)
plt.title('Diagnostic Histograms of Data in Lab Frame', color='white')
colors()
plt.subplot(3, 4, 3)
plt.xlabel('Charge of Mu1', color='white')
colors()
plt.hist(Q1, color='mediumseagreen')
plt.subplot(3, 4, 4)
plt.xlabel('Charge of Mu2', color='white')
colors()
plt.hist(Q2, color='mediumseagreen')
plt.subplot(3, 4, 5)
plt.xlabel('E1 (GeV)', color="white")
colors()
plt.hist(p4Mu1[:, 0], bins=np.arange(0, 60, 0.3), color='mediumseagreen')
plt.subplot(3, 4, 6)
plt.xlabel('px1 (GeV/c)', color='white')
colors()
plt.hist(p4Mu1[:, 1], bins=np.arange(-20, 20, 0.2), color='mediumseagreen')
plt.subplot(3, 4, 7)
plt.xlabel('py1 (GeV/c)', color='white')
colors()
plt.hist(p4Mu1[:, 2], bins=np.arange(-20, 20, 0.2), color='mediumseagreen')
plt.subplot(3, 4, 8)
plt.xlabel('pz1 (GeV/c)', color='white')
colors()
plt.hist(p4Mu1[:, 3], bins=np.arange(-50, 50, 0.5), color='mediumseagreen')
plt.subplot(3, 4, 9)
plt.xlabel('E2 (GeV)', color='white')
ax = plt.gca()
colors()
plt.hist(p4Mu2[:, 0], bins=np.arange(0, 60, 0.3), color='mediumseagreen')
plt.subplot(3, 4, 10)
plt.xlabel('px2 (GeV/c)', color='white')
colors()
plt.hist(p4Mu2[:, 1], bins=np.arange(-20, 20, 0.2), color='mediumseagreen')
plt.subplot(3, 4, 11)
plt.xlabel('py2 (GeV/c)', color='white')
colors()
plt.hist(p4Mu2[:, 2], bins=np.arange(-20, 20, 0.2), color='mediumseagreen')
plt.subplot(3, 4, 12)
plt.xlabel('pz2 (GeV/c)', color='white')
colors()
plt.hist(p4Mu2[:, 3], bins=np.arange(-50, 50, 0.5), color='mediumseagreen')

plt.tight_layout(pad=1.0)
plt.show()

# calculating diMuon mass

p4DiMu = p4Mu1 + p4Mu2
for ie in range(0, Nevents):
    pDiMu[ie, 0] = np.sqrt(p4DiMu[ie, 1:4] @ np.transpose(p4DiMu[ie, 1:4]))
    if p4DiMu[ie, :] @ sig @ np.transpose(p4DiMu[ie, :]) >= 0:
        DiMuMass[ie, 0] = np.sqrt(p4DiMu[ie, :] @
                                  sig @ np.transpose(p4DiMu[ie, :]))
    else:
        DiMuMass[ie, 0] = np.nan
    if p4DiMu[ie, :] @ sig @ np.transpose(p4DiMu[ie, :]) < 0:
        DiMuMassX[ie, 0] = p4DiMu[ie, :] @ sig @ np.transpose(p4DiMu[ie, :])

condition = DiMuMassX[:, 0] < 0

indices_greater_than_one = np.where(condition)[0]

print("Indices of elements >= 1:", indices_greater_than_one)

elements_greater_than_one = DiMuMassX[indices_greater_than_one]

print("Elements >= 1:", elements_greater_than_one)

# plot diMuon energy, momentum, and mass

plt.figure(2)
plt.subplots(1, 3, figsize=[30, 10])
plt.subplot(1, 3, 1)
plt.hist(p4DiMu[:, 0], bins=np.arange(0, 80, 0.5), color='mediumseagreen')
plt.xlabel('DiMuon Energy (GeV)', color='white')
colors()
plt.subplot(1, 3, 2)
plt.hist(pDiMu, bins=np.arange(0, 80, 0.5), color='mediumseagreen')
plt.xlabel('DiMuon Momentum (GeV/c)', color='white')
colors()
plt.subplot(1, 3, 3)
plt.hist(DiMuMass, bins=np.arange(1.75, 5.25, 0.0075), color='mediumseagreen')
plt.xlabel('DiMuon Mass (GeV/c$^2$)', color='white')
plt.ylabel('Number per 0.0075 GeV/c$^2$', color='white')
plt.title('DiMuon Mass')
colors()
plt.show()

# Find DiMuon mass with quality of muon selections

DiMuMassAll = DiMuMass
for ie in range(0, Nevents):
    if Q1[ie, 0] * Q2[ie, 0] < 0:
        DiMuMassOpp[ie, 0] = DiMuMass[ie, 0]
        if Type[ie, :] == "GT" or Type[ie, :] == "GG":
            DiMuMass1G[ie, 0] = DiMuMass[ie, 0]
            if Type[ie, :] == "GG":
                DiMuMass2G[ie, 0] = DiMuMass[ie, 0]

# Plot DiMuon Mass with Quality of Muon Selections

plt.figure(3)
plt.subplots(2, 2, figsize=[30, 14])
plt.subplot(2, 2, 1)
plt.hist(DiMuMassAll, bins=np.arange(1.75, 110, 0.25), color='mediumseagreen')
plt.title('All DiMuon Events', color='white')
plt.xlabel('DiMuon Mass (GeV/c$^2$)', color='white')
colors()
plt.subplot(2, 2, 2)
plt.hist(DiMuMassOpp, bins=np.arange(1.75, 110, 0.065), color='mediumseagreen')
plt.title('Oppositely Charged DiMuon Events', color='white')
plt.xlabel('DiMuon Mass (GeV/c$^2$)', color='white')
plt.yscale("log")
colors()
plt.subplot(2, 2, 3)
plt.hist(DiMuMass1G, bins=np.arange(1.75, 110, 0.065), color='mediumseagreen')
plt.title('Oppositely Charged DiMuon Events with GT or GG', color='white')
plt.xlabel('DiMuon Mass (GeV/c$^2$)', color='white')
plt.yscale("log")
colors()
plt.subplot(2, 2, 4)
plt.hist(DiMuMass2G, bins=np.arange(1.75, 110, 0.25), color='mediumseagreen')
plt.title('Oppositely Charged DiMuon Events with GG', color='white')
plt.xlabel('DiMuon Mass (GeV/c$^2$)', color='white')
colors()
plt.show()

# The above analysis is performed with 4-momenta measured in the Lab Frame.
# The following will use the Lorentz Transoformation to transform p4Mu1 and
# p4Mu2 to the DiMuon Rest Frame.

# Calculate the 4-velocity of the DiMuon, and from this 4-velocity, calculate
# other quantities needed for the Lorentz Transformation from the Lab Frame
# to the DiMuon Rest Frame.

for ie in range(0, Nevents):
    p4DiMu[ie, :] = p4Mu1[ie, :] + p4Mu2[ie, :]
    B3[ie, :] = [p4DiMu[ie, 1], p4DiMu[ie, 2], p4DiMu[ie, 3]]/p4DiMu[ie, 0]
    Bx = B3[ie, 0]
    By = B3[ie, 1]
    Bz = B3[ie, 2]
    B = np.sqrt((Bx ** 2) + (By ** 2) + (Bz ** 2))
    if B < 1:
        Ball[ie, 0] = B
        gam = 1/np.sqrt(1 - B**2)
        L = np.matrix([[gam, -gam * Bx, -gam * By, -gam * Bz],
                       [-gam*Bx, 1+(gam-1)*(Bx**2/B**2),
                        (gam-1)*(Bx*By)/(B**2), (gam-1)*(Bx*Bz)/(B**2)],
                       [-gam*By, (gam-1)*(By*Bx)/(B**2),
                        1+(gam-1)*(By**2/B**2), (gam-1)*(By*Bz)/(B**2)],
                       [-gam*Bz, (gam-1)*(Bz*Bx)/(B**2),
                        (gam-1)*(Bz*By)/(B**2), 1+(gam-1)*(Bz**2/B**2)]])
        p4Mu1RF[ie, 0:4] = np.transpose(L @ np.transpose(p4Mu1[ie, 0:4]))
        p4Mu2RF[ie, 0:4] = np.transpose(L @ np.transpose(p4Mu2[ie, 0:4]))
    else:
        Ball[ie, 0] = np.nan
    if B**2 >= 1:
        BallX[ie, 0] = B

condition = BallX[:, 0] >= 1

indices_greater_than_one = np.where(condition)[0]

print("Indices of elements >= 1:", indices_greater_than_one)

elements_greater_than_one = BallX[indices_greater_than_one]
print("Elements >= 1:", elements_greater_than_one)

# Create Diagnostic Histograms in DiMuon Rest Frame
plt.figure(4)
plt.subplots(3, 4, figsize=[15, 12])
plt.subplot(3, 4, 1)
plt.xlabel('Muon Quality', color='white')
colors()
plt.hist(Type, color='mediumseagreen')
plt.subplot(3, 4, 2)
plt.title('Diagnostic Histograms of Data in DiMuon Rest Frame', loc='left',
          color='white')
plt.subplot(3, 4, 3)
plt.xlabel('Charge of Mu1', color='white')
colors()
plt.hist(Q1, color='mediumseagreen')
plt.subplot(3, 4, 4)
plt.xlabel('Charge of Mu2', color='white')
colors()
plt.hist(Q2, color='mediumseagreen')
plt.subplot(3, 4, 5)
plt.xlabel('E1 (GeV)', color='white')
colors()
plt.hist(p4Mu1RF[:, 0], bins=np.arange(0, 15, 0.05), color='mediumseagreen')
plt.subplot(3, 4, 6)
plt.xlabel('px1 (GeV/c)', color='white')
colors()
plt.hist(p4Mu1RF[:, 1], bins=np.arange(-10, 10, 0.1), color='mediumseagreen')
plt.subplot(3, 4, 7)
plt.xlabel('py1 (GeV/c)', color='white')
colors()
plt.hist(p4Mu1RF[:, 2], bins=np.arange(-10, 10, 0.1), color='mediumseagreen')
plt.subplot(3, 4, 8)
plt.xlabel('pz1 (GeV/c)', color='white')
colors()
plt.hist(p4Mu1RF[:, 3], bins=np.arange(-15, 15, 0.1), color='mediumseagreen')
plt.subplot(3, 4, 9)
plt.xlabel('E2 (GeV)', color='white')
colors()
plt.hist(p4Mu2RF[:, 0], bins=np.arange(0, 15, 0.05), color='mediumseagreen')
plt.subplot(3, 4, 10)
plt.xlabel('px2 (GeV/c)', color='white')
colors()
plt.hist(p4Mu2RF[:, 1], bins=np.arange(-10, 10, 0.1), color='mediumseagreen')
plt.subplot(3, 4, 11)
plt.xlabel('py2 (GeV/c)', color='white')
colors()
plt.hist(p4Mu2RF[:, 2], bins=np.arange(-10, 10, 0.1), color='mediumseagreen')
plt.subplot(3, 4, 12)
plt.xlabel('pz2 (GeV/c)', color='white')
colors()
plt.hist(p4Mu2RF[:, 3], bins=np.arange(-15, 15, 0.1), color='mediumseagreen')

plt.tight_layout(pad=1.0)
plt.show()

# calculating diMuon mass in Rest Frame

p4DiMuRF = p4Mu1RF + p4Mu2RF
DiMuMassRF = p4Mu1RF[:, 0] + p4Mu2RF[:, 0]
for ie in range(0, Nevents):
    pDiMuRF[ie, 0] = np.sqrt(p4DiMuRF[ie, 1:4] @
                             np.transpose(p4DiMuRF[ie, 1:4]))
    DiMuMassRF[ie, 0] = np.sqrt(p4DiMuRF[ie, :] @
                                sig @ np.transpose(p4DiMuRF[ie, :]))

# plot diMuon energy, momentum, and mass in DiMuon Rest Fram

plt.figure(5)
plt.subplots(1, 3, figsize=[30, 10])
plt.subplot(1, 3, 1)
plt.hist(p4DiMuRF[:, 0], bins=np.arange(1.75, 5.25, 0.0075),
         color='mediumseagreen')
plt.xlabel('DiMuon Energy in Rest Frame (GeV)', color='white')
colors()
plt.subplot(1, 3, 2)
plt.hist(pDiMuRF, bins=np.arange(-1e-10, 1e-10, 1e-12), color='mediumseagreen')
plt.xlabel('DiMuon Momentum in Rest Frame (GeV/c)', color='white')
colors()
plt.subplot(1, 3, 3)
plt.hist(DiMuMassRF, bins=np.arange(1.75, 5.25, 0.0075),
         color='mediumseagreen')
plt.xlabel('DiMuon Mass (GeV/c$^2$)', color='white')
plt.ylabel('Number per 0.0075 GeV/c$^2$', color='white')
plt.title('DiMuon Mass in DiMuon Rest Frame', color='white')
colors()
plt.show()

# Find DiMuon mass with quality of muon selections in Rest Frame

DiMuMassAllRF = DiMuMassRF
for ie in range(0, Nevents):
    if Q1[ie, 0] * Q2[ie, 0] < 0:
        DiMuMassOppRF[ie, 0] = DiMuMassRF[ie, 0]
        if Type[ie, :] == "GT" or Type[ie, :] == "GG":
            DiMuMass1GRF[ie, 0] = DiMuMassRF[ie, 0]
            if Type[ie, :] == "GG":
                DiMuMass2GRF[ie, 0] = DiMuMassRF[ie, 0]
                BGG[ie, 0] = Ball[ie, 0]

# Plot DiMuon Mass with Quality of Muon Selections in DiMuon Rest Frame

plt.figure(6)
plt.subplots(2, 2, figsize=[30, 14])
plt.subplot(2, 2, 1)
plt.hist(DiMuMassAllRF, bins=np.arange(1.75, 5.25, 0.001),
         color="mediumseagreen")
plt.title('All DiMuon Events in Rest Frame', color='white')
plt.xlabel('DiMuon Mass (GeV/c$^2$)', color='white')
colors()
plt.subplot(2, 2, 2)
plt.hist(DiMuMassOppRF, bins=np.arange(1.75, 5.25, 0.001),
         color='mediumseagreen')
plt.title('Oppositely Charged DiMuon Events in Rest Frame', color='white')
plt.xlabel('DiMuon Mass (GeV/c$^2$)', color='white')
colors()
plt.subplot(2, 2, 3)
plt.hist(DiMuMass1GRF, bins=np.arange(1.75, 5.25, 0.001),
         color='mediumseagreen')
plt.title('Oppositely Charged DiMuon Events with GT or GG in Rest Frame',
          color='white')
plt.xlabel('DiMuon Mass (GeV/c$^2$)', color='white')
colors()
plt.subplot(2, 2, 4)
plt.hist(DiMuMass2GRF, bins=np.arange(1.75, 5.25, 0.001),
         color="mediumseagreen")
plt.title('Oppositely Charged DiMuon Events with GG in Rest Frame',
          color='white')
plt.xlabel('DiMuon Mass (GeV/c$^2$)', color='white')
colors()
plt.show()

# Plotting beta of the particle in the lab frame

plt.figure(7)
plt.hist(BGG, bins=np.arange(0.01, 1, 0.001), color='mediumseagreen')
plt.xlabel('Beta of the GG DiMuons', color='white')
colors()
plt.show()

# Creating plot for symposium

plt.figure(8)
plt.subplots(3, 4, figsize=[15, 12])
plt.subplot(3, 4, 1)
plt.xlabel('E1 (GeV)', color='white')
plt.hist(p4Mu1RF[:, 0], bins=np.arange(0, 15, 0.05), color='mediumseagreen')
colors()
plt.subplot(3, 4, 2)
plt.xlabel('px1 (GeV/c)', color='white')
plt.hist(p4Mu1RF[:, 1], bins=np.arange(-10, 10, 0.1), color='mediumseagreen')
colors()
plt.subplot(3, 4, 3)
plt.xlabel('py1 (GeV/c)', color='white')
plt.hist(p4Mu1RF[:, 2], bins=np.arange(-10, 10, 0.1), color='mediumseagreen')
colors()
plt.subplot(3, 4, 4)
plt.xlabel('pz1 (GeV/c)', color='white')
plt.hist(p4Mu1RF[:, 3], bins=np.arange(-15, 15, 0.1), color='mediumseagreen')
colors()
plt.subplot(3, 4, 5)
plt.xlabel('E2 (GeV)', color='white')
plt.hist(p4Mu2RF[:, 0], bins=np.arange(0, 15, 0.05), color='mediumseagreen')
colors()
plt.subplot(3, 4, 6)
plt.xlabel('px2 (GeV/c)', color='white')
plt.hist(p4Mu2RF[:, 1], bins=np.arange(-10, 10, 0.1), color='mediumseagreen')
colors()
plt.subplot(3, 4, 7)
plt.xlabel('py2 (GeV/c)', color='white')
plt.hist(p4Mu2RF[:, 2], bins=np.arange(-10, 10, 0.1), color='mediumseagreen')
colors()
plt.subplot(3, 4, 8)
plt.xlabel('pz2 (GeV/c)', color='white')
plt.hist(p4Mu2RF[:, 3], bins=np.arange(-15, 15, 0.1), color='mediumseagreen')
colors()

plt.delaxes(plt.subplot(3, 4, 12))

plt.subplot(3, 4, 9)
plt.xlabel('DiMuon Energy (GeV)', color='white')
plt.hist(p4DiMuRF[:, 0], bins=np.arange(1.75, 5.25, 0.0175),
         color='mediumseagreen')
colors()
plt.subplot(3, 4, 10)
plt.xlabel('DiMuon Momentum (GeV/c)', color='white')
plt.hist(pDiMuRF, bins=np.arange(-1e-10, 1e-10, 1e-12), color='mediumseagreen')
colors()
plt.subplot(3, 4, 11)
plt.xlabel('DiMuon Mass (GeV/c$^2$)', color='white')
plt.hist(DiMuMassRF, bins=np.arange(1.75, 5.25, 0.0175),
         color='mediumseagreen')
colors()
plt.tight_layout(pad=1)
plt.show()
