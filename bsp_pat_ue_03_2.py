import numpy as np
from scipy import integrate
from scipy import optimize
import z_l_v
import logging
from numerik import nr_ls
from numerik import gauss_elimination, lrpd, rref, ref, sdm
import itertools

eps = np.finfo(float).eps
np.set_printoptions(linewidth=200)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console)
logging.getLogger().setLevel(logging.DEBUG)

# Modell feststellen
z_l_v.use_pr_eos()

p = 35.  # bar
temp = 273.15 + 220.  # K
t_flash = 273.15 + 60  # K
t0_ref = 298.15  # K
r = 8.314  # J/(mol K)

namen = ['CO', 'H2', 'CO2', 'H2O', 'CH4', 'NH3', 'AR', 'O2', 'N2']
elemente = ['C', 'O', 'N', 'H', 'AR']

atom_m = np.array([
    [1, 0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 2, 1, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 2],
    [0, 2, 0, 2, 4, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0]
], dtype=float)

red_atom_m = rref(ref(atom_m)[0])

rho = int(np.linalg.matrix_rank(red_atom_m))
n_c = len(namen)
n_e = len(elemente)
n_r = n_c - rho

ne_dampf = np.array([
    0, 0, 0, 60000, 0, 0, 0, 0, 0
], dtype=float)  # kmol/h
ne_rohgas = np.array([
    0, 0, 0, 0, 20000, 0, 0, 0, 0
], dtype=float)  # kmol/h
ne_luft = np.array([
    0, 0, 0, 0, 0, 0,
    0.01 * 15000,
    0.21 * 15000,
    0.78 * 15000
], dtype=float)  # kmol/h

te_dampf = 500 + 273.15  # °K
te_rohgas = 20 + 273.15  # °K
te_luft = 20 + 273.15  # °K

# Thermochemische Daten

# Barin, Ihsan: Thermochemical Data of Pure Substances.
# Weinheim, New York: VCH, 1993.

h_298 = np.array(
    [-110.541, 0., -393.505,
     -241.826, -74.873, -45.940,
     0., 0., 0.]) * 1000  # J/mol

g_298 = np.array(
    [-169.474, -38.962, -457.240,
     -298.164, -130.393, -103.417,
     -46.167, -61.165, -57.128
     ]) * 1000  # J/mol

# Kritische Parameter Tc, Pc, omega(azentrischer Faktor)

# e.V., VDI: VDI-Wärmeatlas. Wiesbaden: Springer Berlin Heidelberg, 2013.

tc = np.array([
    132.86, 33.19, 304.13,
    647.10, 190.56, 405.50,
    150.69, 154.60, 126.19
])  # K

pc = np.array([
    34.98, 13.15, 73.77,
    220.64, 45.99, 113.59,
    48.63, 50.46, 33.96
])  # bar

omega_af = np.array([
    0.050, -0.219, 0.224,
    0.344, 0.011, 0.256,
    -0.002, 0.022, 0.037
])
# umformen (reshape), um direkte Division zu ermöglichen
mm = np.array([
    28.01, 2.02, 44.01,
    18.02, 16.04, 17.03,
    39.95, 32.00, 28.01
]).reshape([len(namen), 1])

# Koeffizienten für Cp(T)/R = B+(C-B)(T/(A+T))^2*(
# 1-A/(A+T)*(D+E*T/(A+T)+F*(T/(A+T))^2+G*(T/(A+T))^3))
# Nach rechts hin: A, B, C, D

# e.V., VDI: VDI-Wärmeatlas. Wiesbaden: Springer Berlin Heidelberg, 2013.

cp_coefs = np.array([z for z in [
    [
        y.replace(',', '.').replace('–', '-') for y in x.split('  ')
    ] for x in """
407,9796  3,5028  2,8524  –2,3018  32,9055  –100,1815  106,1141
392,8422  2,4906  –3,6262  –1,9624  35,6197  –81,3691  62,6668
514,5073  3,4923  –0,9306  –6,0861  54,1586  –97,5157  70,9687
706,3032  5,1703  –6,0865  –6,6011  36,2723  –63,0965  46,2085
1530,8043  4,2038  –16,6150  –3,5668  43,0563  –86,5507  65,5986
931,6298  4,8468  –7,1757  –7,6727  51,3877  –93,4217  67,9515
0,0000  2,5000  2,5000  0,0000  0,0000  0,0000  0,0000
2122,2098  3,5302  –7,1076  –1,4542  30,6057  –83,6696  79,4375
432,2027  3,5160  2,8021  –4,1924  42,0153  –114,2500  111,1019
""".split('\n') if len(x) > 0] if len(z) > 1], dtype=float)


def cp_durch_r(t, component=-1):
    if component != -1:
        cp_c_temp = cp_coefs[component, :]
        a, b, c, d, e, f, g = np.split(cp_c_temp, len(cp_c_temp), axis=0)
    else:
        a, b, c, d, e, f, g = np.split(cp_coefs, cp_coefs.shape[1], axis=1)
    return b + (c - b) * (t / (a + t))**2 * (
        1 - a / (a + t) * (
            d + e * t / (a + t) + f * (t / (a + t))**2 + g * (t / (a + t))**3
        ))  # dimensionslos

# Berechne H(T), G(T) und K(T) mit Cp(T)


def h(t):
    enthalpien = np.empty_like(h_298)
    for i in range(len(enthalpien)):
        int_cp_durch_r = integrate.quad(
            lambda temp: cp_durch_r(temp, i), 298.15, t)[0]
        enthalpien[i] = h_298[i] + r * int_cp_durch_r
    return enthalpien  # J/mol


def g(t, h_t):
    freie_energien = np.empty_like(h_298)
    for i in range(len(freie_energien)):
        int_cp_durch_rt = integrate.quad(
            lambda temp: cp_durch_r(temp, i) / temp, 298.15, t)[0]
        freie_energien[i] = \
            h_t[i] - \
            t / t0_ref * (h_298[i] - g_298[i]) - r * t * int_cp_durch_rt
    return freie_energien  # J/mol


def k(t, g_t, nuij):
    delta_g_t = nuij.T.dot(g_t)
    return np.exp(-delta_g_t / (r * t))


n_0 = (ne_dampf + ne_rohgas + ne_luft) * 1000  # mol/h
h_dampf_ein = h(te_dampf)
h_rohgas_ein = h(te_rohgas)
h_luft_ein = h(te_luft)
# Adiabatische Vermischung sum(n_i h_i(T)-n_i h_i(T0))=0
t_ein = optimize.root(lambda temp:
                      sum(
                          ne_dampf * 1000 * (h(temp) - h_dampf_ein) +
                          ne_rohgas * 1000 * (h(temp) - h_rohgas_ein) +
                          ne_luft * 1000 * (h(temp) - h_luft_ein)
                      ),
                      (te_luft + te_dampf + te_rohgas) / 3
                      ).x

h_t_ein = h(t_ein)
cp_t_ein = r * cp_durch_r(t_ein)
g_t_ein = g(t_ein, h_t_ein)  # J/mol

t_aus_rdampfr = 995 + 273.15  # °K
h_0 = h_t_ein
cp_0 = cp_t_ein
g_0 = g_t_ein

h_1 = h(t_aus_rdampfr)
cp_1 = r * cp_durch_r(t_aus_rdampfr)
g_1 = g(t_aus_rdampfr, h_1)

# Hauptreaktionen nach Meyers 1986
# REF:
# MYERS, Andrea K.; MYERS, Alan L.
# Numerical solution of chemical equilibria with simultaneous reactions.
# The Journal of chemical physics, 1986, 84. Jg., Nr. 10, S. 5787-5795.
nach_g_sortieren = np.argsort(g_1)
# Hauptkomponente festlegen (bei Dampfrerormierung, CH4)
festgelegte_komponente = [4]
# in nach g sortierten Koordinaten
festgelegte_komponente_sortiert = sorted(
    nach_g_sortieren.argsort()[festgelegte_komponente])
# pot_gruppen = itertools.permutations(range(n_c), rho)
pot_gruppen = itertools.combinations(range(n_c), rho)
i = 0
for komb in pot_gruppen:
    i += 1
    indexes = np.concatenate([
        np.array(komb),
        np.array([index for index in range(n_c) if index not in komb])
    ])
    sortierte_namen = np.array(namen)[nach_g_sortieren][indexes]
    ind_sek = [i for i in indexes if i not in komb]
    # namen = np.array(namen)[nach_g_sortieren][indexes][np.argsort(indexes)][np.argsort(nach_g_sortieren)]
    # A = [A_p, A_s]
    a_nach_g_sortiert = atom_m[:, nach_g_sortieren]
    a_p = a_nach_g_sortiert[:, komb]
    a_s = a_nach_g_sortiert[:, ind_sek]
    rho_gruppe = np.linalg.matrix_rank(a_p)
    if rho_gruppe >= rho and all(
            [x in komb for x in festgelegte_komponente_sortiert]):
        ref_atom_m = ref(a_nach_g_sortiert[:, indexes])[0]
        rref_atom_m = rref(ref_atom_m)
        b = rref_atom_m[:n_e, -(n_c - rho_gruppe):]

        stoech_m = np.concatenate([
            -b.T, np.eye(n_c - rho_gruppe, dtype=float)
        ], axis=1)
        k_t = np.exp(-stoech_m.dot(g_1[nach_g_sortieren]
                                   [indexes] / (r * t_aus_rdampfr)))
        print('Gruppe:' + str(sortierte_namen))
        print('Primär:' + str(sortierte_namen[:rho]))
        print('Sekundär:' + str(sortierte_namen[rho:]))
        print('Ap')
        print(a_p)
        print('[Ip, b]')
        print(rref_atom_m)
        print('[-b.T, I]')
        print(stoech_m)
        print('A N^T')
        print(rref_atom_m.dot(stoech_m.T))
        for row in np.array(stoech_m):
            lhs = '+ '.join([str(abs(row[r])) + ' ' +
                             sortierte_namen[r] for r in np.where(row < 0)[0]])
            rhs = '+'.join([str(abs(row[r])) + ' ' +
                            sortierte_namen[r] for r in np.where(row > 0)[0]])
            print(lhs + ' <<==>> ' + rhs)
        print('Kj(T)')
        print(k_t)

        if np.all(k_t < 1):
            break


def r_isoterm(x_vec, k, n_0):
    n_aus = x_vec[:len(n_0)]
    xi_aus = x_vec[len(n_0):]

    n_t = np.sum(n_aus)
    delta_h_1 = nuij.T.dot(h_1)  # J/mol

    f = np.empty_like(x_vec)

    # Stoffbilanzen
    f[:len(n_0)] = -n_aus + n_0 + nuij.dot(xi_aus)

    # Gleichgewicht-Verhältnisse
    # 0 = - Kj * n_T^sum_i(nuij) *  prod_i(ni^(nuij*deltaij)) +
    #                               prod_i(ni^(nuij*(1-deltaij)))
    # deltaij = 1 , wenn nuij < 0 ; 0 sonst
    pir = np.ones_like(k)
    pip = np.ones_like(k)

    for i in range(nuij.shape[0]):  # Komponente i
        for j in range(nuij.shape[1]):  # Reaktion j
            if nuij[i, j] < 0:
                pir[j] = pir[j] * np.power(n_aus[i], abs(nuij[i, j]))
            elif nuij[i, j] > 0:
                pip[j] = pip[j] * np.power(n_aus[i], abs(nuij[i, j]))
            elif nuij[i, j] == 0:
                # Mit ni^0 multiplizieren
                pass

    # Substrate, inklusive totale Mengen je Molenbruch
    k_pir_nt = n_t**sum(+ nuij) * k * pir
    # f_val = k_pir_nt - pip
    # Abstand zum Gleichgewicht
    for i in range(len(f[len(n_0):])):
        if k_pir_nt[i] < pip[i]:
            f[len(n_0) + i] = k_pir_nt[i] / pip[i] - 1
        elif k_pir_nt[i] > pip[i]:
            f[len(n_0) + i] = 1 - pip[i] / k_pir_nt[i]
        elif k_pir_nt[i] == pip[i]:
            f[len(n_0) + i] = 0
    # Energiebilanz
    # q = np.sum(
    #    np.multiply(n_ein, (h_0 - h_298)) -
    #    np.multiply(n_aus, (h_1 - h_298))
    #) + np.dot(xi_aus, -delta_h_1)

    f[len(n_0):] = k_pir_nt - pip
    return f


def gg_abstand(k, nuij, n, xi, full_output=False):
    pip = 1.0
    pir = 1.0
    n_t = sum(n)
    nu_t = sum(nuij)
    for i in range(nuij.shape[0]):  # Komponente i
        if nuij[i] < 0:
            pir = pir * np.power(n[i] + xi * nuij[i],
                                 abs(nuij[i]))
        elif nuij[i] > 0:
            pip = pip * np.power(n[i] + xi * nuij[i],
                                 abs(nuij[i]))
        elif nuij[i] == 0:
            # Mit ni^0 multiplizieren
            pass
    k_pir_nt = k * (n_t + xi * nu_t)**nu_t * pir
    if full_output:
        return (k_pir_nt, pip, k_pir_nt - pip)
    else:
        return k_pir_nt - pip


def r_entspannung(k_j, n_0, x_mal, temp_0, betrieb='isotherm'):
    """ Entspannungs-Methode (Gmehling Chem. Therm.).
    """
    n = np.copy(n_0)
    xi_j = np.array([0 for j in range(nuij.shape[1])], dtype=float)
    xi_j_accum = np.array([0 for j in range(nuij.shape[1])], dtype=float)
    h_temp_0 = h(temp_0)
    temp = temp_0

    for x in range(x_mal):
        for j in range(nuij.shape[1]):
            soln_xi = optimize.newton(lambda xi: gg_abstand(
                k_j[j], nuij[:, j], n, xi), -eps)
            xi_j[j] = soln_xi

            xi_j_accum[j] = xi_j_accum[j] + xi_j[j]
            n = n + nuij[:, j] * xi_j[j]

            if betrieb == 'adiabat':
                temp = optimize.root(
                    lambda temp_var:
                    sum(n * h(temp_var) - n_0 * h_temp_0),
                    temp
                ).x
                k_j = k(temp, g(temp, h(temp)), nuij)
    return (n, xi_j, xi_j_accum, temp)


def notify_status_func(progress_k, stop_value, k,
                       j_it_backtrack, lambda_ls, accum_step,
                       x, diff, f_val, j_val, lambda_ls_y,
                       method_loops):
    g_min = np.nan
    g1 = np.nan
    y = lambda_ls_y
    pr_str = ';k=' + str(k) + \
        ';backtrack=' + str(j_it_backtrack) + \
        ';lambda_ls=' + str(lambda_ls) + \
        ';accum_step=' + str(accum_step) + \
        ';stop=' + str(stop_value) + \
        ';X=' + '[' + ','.join(map(str, x)) + ']' + \
        ';||X(k)-X(k-1)||=' + str(diff.T.dot(diff)) + \
        ';f(X)=' + '[' + ','.join(map(str, f_val)) + ']' + \
        ';||f(X)||=' + str(np.sqrt(f_val.T.dot(f_val))) + \
        ';j(X)=' + str(j_val.tolist()) + \
        ';Y=' + '[' + ','.join(map(str, y)) + ']' + \
        ';||Y||=' + str(np.sqrt(y.T.dot(y))) + \
        ';g=' + str(g_min) + \
        ';|g-g1|=' + str(abs(g_min - g1))
    logging.debug(pr_str)


def fj_0(xi, n_0, k_t):
    n = n_0 + nuij.dot(xi)
    n_t = sum(n_0) + sum(nuij.dot(xi))
    #f = nuij.T.dot(np.log((n) / (n_t))) - np.log(k_1)
    pi = np.product(np.power(n / n_t, nuij.T), axis=1)
    f = pi - k_t
    return f


def jac_fj_0(xi, n_0):
    jac = np.zeros([len(xi), len(xi)])
    n = n_0 + nuij.dot(xi)
    n_t = sum(n)
    n[n == 0] = eps
    pi = np.product(np.power(n / n_t, nuij.T), axis=1)
    for i in range(n_r):
        for j in range(n_r):
            for k in range(n_c):
                jac[i, j] = jac[i, j] + nuij[k, j] * nuij[k, i] / n[k]
            jac[i, j] = pi[i] * (jac[i, j] - sum(nuij[:, j]) *
                                 sum(nuij[:, i]) / n_t)
    return jac


nuij = np.array(stoech_m[:, np.argsort(indexes)]
                [:, np.argsort(nach_g_sortieren)]).T
k_1 = k(t_aus_rdampfr, g_1, nuij)
naus_0 = np.copy(n_0)
# Entspannung
_, _, xi_0, _ = r_entspannung(k_1, naus_0, 1, t_aus_rdampfr)
# Steifster Gradient
xi_sdm = sdm(
    xi_0,
    lambda xi: fj_0(xi, n_0, k_1),
    lambda xi: jac_fj_0(xi, n_0),
    1e-4)
# Newton-Raphson
progress_k, stop, outer_it_k, outer_it_j, \
    lambda_ls, accum_step, x, \
    diff, f_val, lambda_ls_y, \
    method_loops = \
    nr_ls(x0=xi_sdm,
          f=lambda xi: fj_0(xi, n_0, k_1),
          j=lambda xi: jac_fj_0(xi, n_0),
          tol=1e-12,
          max_it=1000,
          inner_loop_condition=lambda x_vec:
          all([item >= 0 for item in
               n_0 + nuij.dot(x_vec)]),  # keine Voraussetzung
          notify_status_func=notify_status_func,
          method_loops=[0, 0],
          process_func_handle=None)


xi_accum_1 = x
n_1 = n_0 + nuij.dot(xi_accum_1)

for komb in n_1:
    print('{0:0.20g}'.format(komb / 1000.).replace('.', ','))

for komb in np.array(xi_accum_1):
    print('{0:0.20g}'.format(komb / 1000.).replace('.', ','))

print('========================================')

fehler = r_isoterm(np.concatenate([n_1, xi_accum_1]), k_1, n_0)
print('Gesamtfehler (Relaxation): ' + str(np.sqrt(fehler.dot(fehler))))

for j in range(5):
    print('')

print('========================================')
print('Dampf-Reformierung: Vorwärmer + Reaktor')
print('========================================')

q = sum(n_1 * h_1 - n_0 * h_0)  # mol/h * J/mol = J/h

# 1h/60^2s * 1kW / 1000W
print(
    'Vormärmer (auf T= ' + str(t_aus_rdampfr) + ' °K), Q: ' +
    '{0:0.20g}'.format(
        sum(n_0 * (h_1 - h_0)) * 1 / 60. ** 2 * 1 / 1000.
    ).replace('.', ',') + ' kW'
)

print(
    'Isothermisch betriebener Reaktor, Q: ' +
    '{0:0.20g}'.format(
        sum(nuij.dot(xi_accum_1) * h_1) * 1 / 60. ** 2 * 1 / 1000.
    ).replace('.', ',') + ' kW'
)

print(
    'Totale zu tauschende Energie, Q: ' +
    '{0:0.20g}'.format(
        q * 1 / 60.**2 * 1 / 1000.
    ).replace('.', ',') + ' kW'
)

print('')

for i, komb in enumerate(np.array(n_1)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum(n_1) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(np.array(n_1 / sum(n_1))):
    print('y_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('T: ' + '{:g}'.format(t_aus_rdampfr) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum(n_1 * h_1) / sum(n_1) * 1000.) + ' J/kmol')

print('========================================')
print('Wassergashift: Vorkühler + Reaktor')
print('========================================')

t_ein_rwgs = 210 + 273.15  # °K
h_2 = h(t_ein_rwgs)
cp_2 = r * cp_durch_r(t_ein_rwgs)
g_2 = g(t_ein_rwgs, h_2)
k_2 = k(t_ein_rwgs, g_2, nuij)

n_2 = n_1

q = sum(n_2 * (h_2 - h_1))  # mol/h * J/mol = J/h

# 4-Mal Relaxation-Methode (Entspannung)
n_2, _, xi_accum_2, temp = r_entspannung(k_2, n_1, 4, t_ein_rwgs, 'isotherm')

x0 = np.concatenate([
    n_1,
    xi_accum_2
])
