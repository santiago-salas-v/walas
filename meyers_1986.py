from itertools import combinations
import numpy as np
from scipy import optimize
import scipy
import itertools
from numerik import lrpd, rref, gauss_elimination

# REF:
# MYERS, Andrea K.; MYERS, Alan L. 
# Numerical solution of chemical equilibria with simultaneous reactions. 
# The Journal of chemical physics, 1986, 84. Jg., Nr. 10, S. 5787-5795.

p = 0.101325  # MPa
temp = 1478.  # °K
t0_ref = 298.15  # K
r = 8.314  # J/(mol K)

namen = ['CO2', 'SO2', 'H2O', 'S2', 'CO', 'COS', 'CS2', 'H2S', 'H2']
elemente = ['C', 'O', 'S', 'H']


c = len(namen)
e = len(elemente)

atom_m = np.array([
    [1, 0, 0, 0, 1, 1, 1, 0, 0],
    [2, 2, 1, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 2, 0, 1, 2, 1, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 2]
    ])

rho = np.linalg.matrix_rank(atom_m)

print('mögliche Gruppen mit Rang>=rho:')
for comb in combinations(range(9), 4):
    mat = atom_m[:,comb]
    rank = np.linalg.matrix_rank(mat)
    if rank >= rho:
        print('Rang: ' + str(rank))
        print(np.array(namen)[[comb]])
        print(mat.tolist())

print('Atomische Matrix: A')
print(atom_m)

print('Rang(A) (rho):')
print(rho)

print('rho >= E ? :')
print(rho >= e)

_, r_atom_m, _, _, _ = lrpd(atom_m)

rref_atom = rref(r_atom_m)

print('rref(A):')
print(rref_atom)

b = rref_atom[:e, c-rho-1:]

stoech_m = np.concatenate([
    -b.T, np.eye( c - rho, dtype=float)
    ], axis=1)

print('Stöchiometriche Matrix ((C-rho) X C): N')
print(stoech_m)

print('A N^T = 0')
print(atom_m.dot(stoech_m.T))

nuij = np.array([
    [-1, +0.25, +1, ]
    ])

ne = np.array([
    113.15,
    0.0,
    0.0,
    18.73,
    -81.25,
    0,
    0,
    0,
    89.1,
    ]) # mol

g_t =r * temp * np.array([
    -32.2528,
    -20.4331,
    -13.4872,
    0.,
    -19.7219,
    -18.0753,
    -1.9250,
    -1.4522,
    0.
    ]) # J/mol

k_t = np.exp(-stoech_m.dot(g_t/(r*temp)))  
print('Kj')
print(k_t)
print('sum(stoech_m)')
print(np.sum(stoech_m[:, :(c-rho)-1]))

nach_g_sortieren = np.argsort(g_t)
atom_m = atom_m[:, nach_g_sortieren]
rho = np.linalg.matrix_rank(atom_m)
_, r_atom_m, _, _, _ = lrpd(atom_m)
rref_atom = rref(r_atom_m)
b = rref_atom[:e, c-rho-1:]
stoech_m = np.concatenate([
    -b.T, np.eye( c - rho, dtype=float)
    ], axis=1)
k_t = np.exp(-stoech_m.dot(g_t/(r*temp)))

print('Namen, nach g0 sortiert:')
print([namen[i] for i in nach_g_sortieren])
print('A, nach g0 sortiert:')
print(atom_m)
print('rho:')
print(rho)
print('Kj')
print(k_t)
print('sum(stoech_m)')
print(np.sum(stoech_m[:, :(c-rho)-1]))
print('Atomischer Vektor m:')
print(np.sum(atom_m*(ne[nach_g_sortieren]), axis=1))


def comb(c, rho, lst=None, i=0):
    if lst==None:
        lst=[]
    j=i
    while j in range(i, c):
        lst.append(j)
        j+=1
        if len(lst) > 0 and divmod(len(lst),rho)[1] ==0:
            return lst
        else:
            comb(c, rho, lst, j)


comb = itertools.combinations(range(c), rho)
i = 0
for item in comb:
    i+=1
    indexes = np.concatenate([
        np.array(item),
        np.array([index for index in range(c) if index not in item])
    ])
    rho = np.linalg.matrix_rank(atom_m[:, indexes])
    _, r_atom_m, _, _, _ = lrpd(atom_m[:, indexes])
    rref_atom = rref(r_atom_m)
    b = rref_atom[:e, c - rho - 1:]

    stoech_m = np.concatenate([
        -b.T, np.eye(c - rho, dtype=float)
    ], axis=1)
    k_t = np.exp(-stoech_m.dot(g_t / (r * temp)))
    if np.all(k_t<1):
        break
print('')
print('Bewertete Zusammenstellung: ' + str(i))
print('Kj')
print(k_t)
print('Alle Kj<0 ?')
print(np.all(k_t<1))
print('Kombination, der nach g0 sortierten Indexes:')
print(nach_g_sortieren[indexes]+1)
print('Namen, sortiert:')
print(np.array(namen)[nach_g_sortieren][indexes])
print('A, nach g0 sortiert:')
print(atom_m)
print('rho:')
print(rho)
print('Stöchiometrische Matrix:')
print(stoech_m)
print('Atomischer Vektor')
print(np.sum(atom_m[:, indexes]*(ne[nach_g_sortieren][indexes]), axis=1))
print('sum(stoech_m) (für die (c-rho) Hauptkomponenten)')
print(np.sum(stoech_m[:, :(c-rho)-1]))
print('sum(stoech_m)<0 ?')
print(np.sum(stoech_m[:, :(c-rho)-1])<0)
print((ne[nach_g_sortieren][indexes][:(c-rho)-1]))
print(sum(atom_m[:, indexes][:, :(c-rho)-1]*(ne[nach_g_sortieren][indexes][:(c-rho)-1])))
print(np.matrix(atom_m[:, indexes][:, :(c-rho)-1], dtype=float)**-1*np.matrix(ne[nach_g_sortieren][indexes][:(c-rho)-1]).T)
print(gauss_elimination(
    atom_m[:, indexes][:, :(c-rho)-1],
    ne[nach_g_sortieren][indexes][:(c-rho)-1].reshape([rho,1]))
)