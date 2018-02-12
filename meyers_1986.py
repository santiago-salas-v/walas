from itertools import combinations
import numpy as np
from scipy import optimize
import scipy

p = 0.101325  # MPa
temp = 1478.  # °K
t0_ref = 298.15  # K
r = 8.314  # J/(mol K)

namen = ['CO2', 'SO2', 'CO', 'H2O', 'COS', 'CS2', 'H2S', 'H2', 'S2']
elemente = ['C', 'O', 'S', 'H']


c = len(namen)
e = len(elemente)

atom_m = np.array([
    [1, 0, 1, 0, 1, 1, 0, 0, 0],
    [2, 2, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 2, 1, 0, 2],
    [0, 0, 0, 2, 0, 0, 2, 2, 0]
    ])

rho = np.linalg.matrix_rank(atom_m)

print('mögliche Gruppen mit Rang>rho:')
for comb in combinations(range(9), 4):
    mat = atom_m[:,comb]
    rank = np.linalg.matrix_rank(mat)
    if rank < rho:
        print('rank: ' + str(rank))
        print(np.array(namen)[[comb]])
        print(mat.tolist())

print(np.linalg.matrix_rank(atom_m))

print(scipy.linalg.lu(atom_m))

nuij = np.array([
    [-1, +0.25, +1, ]
    ])

ne = np.array([
    113.15,
    0.0,
    -81.25,
    0,
    0,
    0,
    0,
    89.1,
    18.73]) # mol

g_t =r * temp * np.array([
    -32.2528,
    -20.4331,
    -19.7219,
    -13.4872,
    -18.0753,
    -1.9250,
    -1.4522,
    0.,
    0.]) # J/mol



k_t = nuij.T.dot(g_t)
print(k_t)