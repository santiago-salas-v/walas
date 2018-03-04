from itertools import combinations
import numpy as np
from scipy import optimize
import scipy
import itertools
from numerik import lrpd, rref, ref, gauss_elimination

np.set_printoptions(linewidth=200)

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

rho = np.linalg.matrix_rank(atom_m)

print('(1) Eingangsdaten')
print('Namen:' + str(namen))
print('Elemente: ' + str(elemente))
print('g_t^\circ: ' + str(g_t))
print('n_e: ' + str(ne))
print('(2) Atomische Matrix \n A = \n' + str(atom_m))
print('E = ' + str(e) + '; C = ' + str(c))
print('rho = Rang(A)  = ' + str(rho))
print('R = (C-rho) = ' + str(c-rho))
print('rho >= E ? : ' + str(rho >= e))

_, r_atom_m, _, _, _ = lrpd(atom_m)

rref_atom = rref(r_atom_m)

print('')
print('Reduzierte Stufenform(A): \n rref(A) = \n' + str(rref_atom))

b = rref_atom[:e, c-rho-1:]

stoech_m = np.concatenate([
    -b.T, np.eye( c - rho, dtype=float)
    ], axis=1)
k_t = np.exp(-stoech_m.dot(g_t/(r*temp)))  

print('Stöchiometriche Matrix ((C-rho) X C): \n N = \n' + str(stoech_m))

print('A N^T = 0')
print(atom_m.dot(stoech_m.T))
print('Kj = ' + str(k_t))
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


print('(4) Schlüsselkomponente')
print('Mögliche Gruppen mit Rang>=rho:')
for comb in combinations(range(9), 4):
    mat = atom_m[:,comb]
    rank = np.linalg.matrix_rank(mat)
    if rank >= rho:
        print('Rang: ' + str(rank))
        print(np.array(namen)[[comb]])
        print(mat.tolist())

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
    rho_gruppe = np.linalg.matrix_rank(atom_m[:,item])
    if rho_gruppe >= rho:
        r_atom_m = ref(atom_m[:, indexes])[0]
        rref_atom = rref(r_atom_m)
        b = rref_atom[:e, -(c - rho_gruppe):]

        stoech_m = np.concatenate([
            -b.T, np.eye(c - rho_gruppe, dtype=float)
        ], axis=1)
        k_t = np.exp(-stoech_m.dot(g_t / (r * temp)))

        sortierte_namen = np.array(namen)[nach_g_sortieren][list(indexes)]
        a_m_mit_namen = np.concatenate(
            [[sortierte_namen],
             [*[np.array(row, dtype=object) for row in atom_m]]]
        )
        stoech_m_mit_namen =np.concatenate(
            [[sortierte_namen],
             [*[np.array(row, dtype=object) for row in np.array(stoech_m)]]]
        )
        ind_sek = [i for i in indexes if i not in item]
        a_p = atom_m[:, item]
        a_s = atom_m[:, ind_sek]

        print('Gruppe:' + str(sortierte_namen))
        print('Primär:' + str(sortierte_namen[:rho]))
        print('Sekundär:' + str(sortierte_namen[rho:]))
        print('Ap')
        print(a_p)
        print('[Ip, b]')
        print(rref_atom)
        print('[-b.T, I]')
        print(stoech_m)
        print('A N^T')
        print(rref_atom.dot(stoech_m.T))
        for row in np.array(stoech_m):
            lhs = '+ '.join([str(abs(row[r])) + ' ' +
                sortierte_namen[r] for r in np.where(row<0)[0]])
            rhs = '+'.join([str(abs(row[r])) + ' ' +
                sortierte_namen[r] for r in np.where(row>0)[0]])
            print(lhs + ' <<==>> ' + rhs)
        print('Kj(T)')
        print(k_t)

        if np.all(k_t<1):# and np.linalg.matrix_rank(b.T)>= c-rho_gruppe:
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
print('Wiederberechnetes Gemisch:')
# m = A n
n = gauss_elimination(
    atom_m[:, indexes][:, :(c-rho)-1],
    np.sum(atom_m*(ne[nach_g_sortieren]), axis=1).reshape([rho,1]))
print(n)

print(ne)

xi_j = np.sum(ne)/(c-rho)*(k_t/(1+k_t))*7.10309/2.47820182

n0 = np.matrix(ne[nach_g_sortieren][indexes][0:(c-rho)-1]).T
s_xi = stoech_m[:, :(c-rho)-1].T*xi_j.T

n1 = n0 + s_xi

xi_j_k_m_1 = np.copy(xi_j)

for i, n_k in enumerate(n1):
    for j, item in enumerate(xi_j.T):
        if n_k < 0 and stoech_m[j, i] != 0:
            s = np.sum([nuij for nuij in stoech_m[:, i] if nuij != 0])
            xi_j_k_m_1[0, j] = xi_j[0, j] + (0.01 * np.sum(n0) - (n_k)) / s
            print((0.01 * np.sum(n0) - (n_k)) / s)


stoech_m[abs(stoech_m)<np.finfo(float).eps] = 0
print(stoech_m)
print(0**-0)

print(np.sum(ne)**sum(stoech_m))
print(np.power(ne, -stoech_m))
for x, item in enumerate(stoech_m):
    print(np.power(ne, -item))
print(np.product(np.power(ne, -stoech_m), axis=1))
print(np.multiply(k_t,
    np.multiply(
            np.sum(ne)**sum(stoech_m) ,
            np.product(np.power(ne, -stoech_m), axis=0),
            )))