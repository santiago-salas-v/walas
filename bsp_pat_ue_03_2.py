import numpy as np
from scipy import optimize
import z_l_v
import logging
from numerik import nr_ls
from numerik import rref, ref
import itertools
from setup_results_log import notify_status_func, setup_log_file
import timeit

eps = np.finfo(float).eps
np.set_printoptions(linewidth=200)
setup_log_file('log_bsp_pat_ue_03_2.log', with_console=False)

# Modell feststellen
alpha_tr, epsilon, sigma, psi, omega = z_l_v.use_pr_eos()

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

mm_el = np.array([
    12.011,
    15.999,
    14.007,
    1.008,
    39.948,
]) / 1000.  # kg/gmol

mm_k = atom_m.T.dot(mm_el)


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


def cp_durch_r(t, component):
    a, b, c, d, e, f, g = cp_coefs[component, :]
    gamma_var = t / (a + t)
    return b + (c - b) * gamma_var**2 * (
        1 + (gamma_var - 1) * (
            d + e * gamma_var + f * gamma_var**2 + g * gamma_var**3
        ))  # dimensionslos


def int_cp_durch_r_dt_minus_const(t):
    a, b, c, d, e, f, g = [
        item.reshape(n_c) for item in np.split(
            cp_coefs, cp_coefs.shape[1], axis=1
        )
    ]
    return b * t + (c - b) * (
        t - (d + e + f + g + 2) * a * np.log(a + t) +
        -(2 * d + 3 * e + 4 * f + 5 * g + 1) * a**2 / (a + t) +
        +(1 * d + 3 * e + 6 * f + 10 * g) * a**3 / 2 / (a + t)**2 +
        -(1 * e + 4 * f + 10 * g) * a**4 / 3 / (a + t)**3 +
        +(1 * f + 5 * g) * a**5 / 4 / (a + t)**4 +
        - g * a**6 / 5 / (a + t)**5
    )


def int_cp_durch_rt_dt_minus_const(t):
    a, b, c, d, e, f, g = [
        item.reshape(n_c) for item in np.split(
            cp_coefs, cp_coefs.shape[1], axis=1
        )
    ]
    return b * np.log(t) + (c - b) * (
        np.log(a + t) + (1 + d + e + f + g) * a / (a + t) +
        -(d / 2 + e + 3 * f / 2 + 2 * g) * a**2 / (a + t)**2 +
        +(e / 3 + f + 2 * g) * a**3 / (a + t)**3 +
        -(f / 4 + g) * a**4 / (a + t)**4 +
        +(g / 5) * a**5 / (a + t)**5
    )


def mcph(x, t0, t):
    return sum(x * (
        int_cp_durch_r_dt_minus_const(t) -
        int_cp_durch_r_dt_minus_const(t0))
    ) / sum(x) / (t - t0)


def mcps(x, t0, t):
    return sum(x * (
        int_cp_durch_rt_dt_minus_const(t) -
        int_cp_durch_rt_dt_minus_const(t0))
    ) / sum(x) / np.log(t / t0)

# Berechne H(T), G(T) und K(T) mit Cp(T)


def h(t):
    enthalpien = h_298 + r * (
        int_cp_durch_r_dt_minus_const(t) -
        int_cp_durch_r_dt_minus_const(298.15)
    )
    return enthalpien  # J/mol


def g(t, h_t):
    freie_energien = h_t - t / t0_ref * (h_298 - g_298) - \
        r * t * (int_cp_durch_rt_dt_minus_const(t) -
                 int_cp_durch_rt_dt_minus_const(t0_ref))
    return freie_energien  # J/mol


def k(t, g_t, nuij):
    delta_g_t = nuij.T.dot(g_t)
    return np.exp(-delta_g_t / (r * t))


n_0 = (ne_dampf + ne_rohgas + ne_luft) * 1000  # mol/h
h_dampf_ein = h(te_dampf)
h_rohgas_ein = h(te_rohgas)
h_luft_ein = h(te_luft)
# Adiabate Vermischung sum(n_i h_i(T)-n_i h_i(T0))=0
t_ein = optimize.root(lambda temp:
                      sum(
                          ne_dampf * 1000 * (h(temp) - h_dampf_ein) +
                          ne_rohgas * 1000 * (h(temp) - h_rohgas_ein) +
                          ne_luft * 1000 * (h(temp) - h_luft_ein)
                      ),
                      (te_luft + te_dampf + te_rohgas) / 3
                      ).x

h_t_ein = h(t_ein)
g_t_ein = g(t_ein, h_t_ein)  # J/mol

t_aus_rdampfr = 995 + 273.15  # °K
h_0 = h_t_ein
g_0 = g_t_ein

h_1 = h(t_aus_rdampfr)
g_1 = g(t_aus_rdampfr, h_1)


def stoech_matrix(atom_m, g_t, p, temp, namen, festgelegte_komponente=None):
    """
    Hauptreaktionen nach Meyers 1986
    REF:
    MYERS, Andrea K.; MYERS, Alan L.
    Numerical solution of chemical equilibria with simultaneous reactions.
    The Journal of chemical physics, 1986, 84. Jg., Nr. 10, S. 5787-5795.

    :param atom_m: atomic matrix. Unsorted. E by C.
    :param g_t: Gibbs free energy of formation of species in atom matrix at T.
    :param p: Operating pressure
    :param namen: names of compounds in atomic matrix.
    :return: stoech_m, indexes, nach_g_sortieren, k_t, nuij
    """
    nach_g_sortieren = np.argsort(g_t)
    # Hauptkomponente festlegen (bei Dampfrerormierung, CH4)
    # festgelegte_komponente = [4]
    # in nach g sortierten Koordinaten
    if festgelegte_komponente is None:
        festgelegte_komponente_sortiert = None
    else:
        festgelegte_komponente_sortiert = sorted(
            nach_g_sortieren.argsort()[festgelegte_komponente])
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
        # A = [A_p, A_s]
        a_nach_g_sortiert = atom_m[:, nach_g_sortieren]
        a_p = a_nach_g_sortiert[:, komb]
        a_s = a_nach_g_sortiert[:, ind_sek]
        rho_gruppe = np.linalg.matrix_rank(a_p)
        cond_1 = rho_gruppe >= rho
        if festgelegte_komponente_sortiert is None:
            cond_2 = True
        else:
            cond_2 = all([x in komb for x in festgelegte_komponente_sortiert])
        if cond_1 and cond_2:
            ref_atom_m = ref(a_nach_g_sortiert[:, indexes])[0]
            rref_atom_m = rref(ref_atom_m)
            b = rref_atom_m[:n_e, -(n_c - rho_gruppe):]

            stoech_m = np.concatenate([
                -b.T, np.eye(n_c - rho_gruppe, dtype=float)
            ], axis=1)
            k_t = np.exp(-stoech_m.dot(g_t[nach_g_sortieren]
                                       [indexes] / (r * temp)))
            k_x_t = np.multiply(k_t, np.power(p / 1., -sum(stoech_m.T)))
            logging.debug('Gruppe:' + str(sortierte_namen))
            logging.debug('Primär:' + str(sortierte_namen[:rho]))
            logging.debug('Sekundär:' + str(sortierte_namen[rho:]))
            logging.debug('Ap')
            logging.debug(str(a_p.tolist()))
            logging.debug('[Ip, b]')
            logging.debug(str(rref_atom_m.tolist()))
            logging.debug('[-b.T, I]')
            logging.debug(str(stoech_m.tolist()))
            logging.debug('A N^T')
            logging.debug(str(rref_atom_m.dot(stoech_m.T).tolist()))
            for row in np.array(stoech_m):
                lhs = '+ '.join([str(abs(row[i])) + ' ' + sortierte_namen[i]
                                 for i in np.where(row < 0)[0]])
                rhs = '+'.join([str(abs(row[i])) + ' ' + sortierte_namen[i]
                                for i in np.where(row > 0)[0]])
                logging.debug(lhs + ' <<==>> ' + rhs)
            logging.debug('Kj(T)')
            logging.debug(str(k_t.tolist()))

            if np.all(k_x_t < 1):
                break
    # rearrange columns to original order (rows remain the same, so does K)
    nuij = np.array(stoech_m[:, np.argsort(indexes)]
                    [:, np.argsort(nach_g_sortieren)]).T
    return stoech_m, indexes, nach_g_sortieren, k_x_t, k_t, nuij


def jac_gg_abstaende(k_x, nuij, n_0, xi_t,
                     h_0=None, betriebsw='isotherm'):
    if betriebsw == 'adiabat' and h_0 is not None:
        xi = xi_t[:-1]
        temp = xi_t[-1]
    elif betriebsw == 'isotherm':
        xi = xi_t
    if np.ndim(nuij) > 1:
        n_r = nuij.shape[1]
        nuji = nuij.T
        jac = np.zeros([len(xi), len(xi)])
        nuij_xi = nuij.dot(xi)
    else:
        n_r = 1
        nuji = nuij.reshape([1, nuij.shape[0]])
        jac = np.zeros([1, 1])
        nuij_xi = nuij * xi
    n = n_0 + nuij_xi
    nu_t = sum(nuij)
    n_t = np.sum(n)
    s_nujknuik_nk_r = np.zeros_like(jac)
    s_nujknuik_nk_p = np.zeros_like(jac)
    if betriebsw == 'adiabat' and h_0 is not None:
        k_x_pir_nt, pir, pip, f_q, h_t = \
            gg_abstaende(k_x, nuij, n_0, xi_t,
                         h_0, betriebsw, full_output=True)
        jac = np.zeros([len(xi_t), len(xi_t)])
    elif betriebsw == 'isotherm':
        k_x_pir_nt, pir, pip = \
            gg_abstaende(k_x, nuij, n_0, xi_t,
                         betriebsw, full_output=True)
    for j in range(n_r):
        for i in range(n_r):
            for k_st, n_k in enumerate(n):
                if nuji[j, k_st] < 0 and abs(n_k) > eps:
                    s_nujknuik_nk_r[j, i] = s_nujknuik_nk_r[j,
                                                            i] + nuji[j, k_st] * nuji[i, k_st] / n_k
                elif nuji[j, k_st] > 0 and abs(n_k) > eps:
                    s_nujknuik_nk_p[j, i] = s_nujknuik_nk_p[j,
                                                            i] + nuji[j, k_st] * nuji[i, k_st] / n_k
                elif nuji[j, k_st] == 0:
                    # add 0
                    pass
            jac[j, i] = - k_x_pir_nt[j] * (
                s_nujknuik_nk_r[j, i] -
                sum(nuji[j, :]) * sum(nuji[i, :]) / n_t
            ) - pip[j] * s_nujknuik_nk_p[j, i]
    if betriebsw == 'adiabat' and h_0 is not None:
        delta_h_r_t = nuji.dot(h_t)
        d_lnk_dt = delta_h_r_t / (r * temp ** 2)
        cp_t = r * np.concatenate([[cp_durch_r(temp, i)]
                                   for i in range(np.size(n))])
        for j in range(n_r):
            jac[j, n_r] = +k_x_pir_nt[j] * d_lnk_dt[j]
            jac[n_r, j] = -delta_h_r_t[j]
        jac[n_r, n_r] = -sum(n * cp_t)
        logging.debug('j=' + str(jac.tolist()))
        return jac
    elif betriebsw == 'isotherm':
        if n_r == 1:
            return jac[0].item()  # avoid embedded array
        else:
            return jac


def gg_abstaende(k_x, nuij, n_0, xi_t,
                 h_0=None, betriebsw='isotherm',
                 full_output=False):
    if betriebsw == 'adiabat' and h_0 is not None:
        xi = xi_t[:-1]
        temp = xi_t[-1]
        h_t = h(temp)
        g_t = g(temp, h_t)
        k_t = k(temp, g_t, nuij)
        k_x = np.multiply(
            k_t, np.power(p / 1., -sum(nuij)))
    elif betriebsw == 'isotherm':
        xi = xi_t
    if np.ndim(nuij) > 1:
        n_r = nuij.shape[1]
        nuji = nuij.T
        n = n_0 + nuij.dot(xi)
    else:
        n_r = 1
        nuji = nuij.reshape([1, nuij.shape[0]])
        n = n_0 + nuij * xi
    nu_t = sum(nuij)
    n_t = np.sum(n)
    pir = np.ones(n_r)
    pip = np.ones(n_r)
    for j in range(n_r):
        for k_st, nujk in enumerate(nuji[j]):
            if nujk < 0:
                pir[j] = pir[j] * n[k_st]**abs(nujk)
            elif nujk > 0:
                pip[j] = pip[j] * n[k_st]**abs(nujk)
            elif nujk == 0:
                # multiply by 1
                pass
    k_x_pir_nt = k_x * pir * n_t ** nu_t
    logging.debug('xi=' + str(xi))
    logging.debug('f(xi)=' + str(k_x_pir_nt - pip))
    if betriebsw == 'adiabat' and h_0 is not None:
        f_q = np.sum(
            np.multiply(n_0, h_0) -
            np.multiply(n, h_t))
        if full_output:
            return (k_x_pir_nt, pir, pip, f_q, h_t)
        else:
            if n_r == 1:
                return np.array([(k_x_pir_nt - pip).item(), f_q])
            else:
                return np.concatenate([k_x_pir_nt - pip, [f_q]])
    elif betriebsw == 'isotherm':
        if full_output:
            return (k_x_pir_nt, pir, pip)
        else:
            if n_r == 1:
                return (k_x_pir_nt - pip).item()
            else:
                return k_x_pir_nt - pip


def r_entspannung_isoth(k_x_j, nuij, n_0, temp_0, p, x_mal=1):
    """ Entspannungs-Methode (Gmehling Chem. Therm.).
    :param nuij:
    """
    n = np.copy(n_0)
    xi_j = np.array([-eps for j in range(nuij.shape[1])], dtype=float)
    xi_j_accum = np.array([-eps for j in range(nuij.shape[1])], dtype=float)
    h_0 = h(temp_0)
    temp = temp_0
    x0 = -eps
    for rel in range(x_mal):
        r_to_relax = [j for j in range(nuij.shape[1])]
        while r_to_relax:
            # nach Spannung sortieren, gespannteste Reaktion zunächst
            # entspannen.
            order = np.argsort(
                [-abs(gg_abstaende(k_x_j[i], nuij[:, i], n, 0)) for i in r_to_relax])
            j = r_to_relax[order[0]]
            soln =  nr_ls(
                x0=x0,
                f=lambda xi_t: gg_abstaende(
                  k_x_j[j], nuij[:, j], n, xi_t, h_0, 'isotherm'),
                j=lambda xi_t: jac_gg_abstaende(
                  k_x_j[j], nuij[:, j], n, xi_t, h_0, 'isotherm'),
                tol=1e-5,
                max_it=1000,
                inner_loop_condition=lambda xi: all(
                  [item >= 0 or abs(item) < eps for item in
                   n + nuij[:, j] * xi]),
                notify_status_func=notify_status_func,
                process_func_handle=lambda: logging.debug('no progress'))

            soln_xi = soln['x']

            xi_j[j] = soln_xi

            xi_j_accum[j] = xi_j_accum[j] + xi_j[j]
            n = n + nuij[:, j] * xi_j[j]

            r_to_relax.pop(order[0])
    return (n, xi_j, xi_j_accum, temp)


def r_entspannung_adiab(k_x_j, nuij, n_0, temp_0, p, x_mal=1):
    xi_j = np.array([-eps for j in range(nuij.shape[1])], dtype=float)
    xi_j_accum_adiab = np.array(
        [-eps for j in range(nuij.shape[1])], dtype=float)
    h_0 = h(temp_0)
    n, xi_j, xi_j_accum_isot, temp = r_entspannung_isoth(
        k_x_j, nuij, n_0, temp_0, p, x_mal=1)
    soln = nr_ls(
        x0=temp,
        f=lambda t: np.sum(
          np.multiply(n_0, h_0) -
          np.multiply(n, h(t))),
        j=lambda t: -sum(n * r * np.concatenate(
          [[cp_durch_r(t, i)] for i in range(np.size(n))])),
        tol=1e-5,
        max_it=1000,
        inner_loop_condition=None,
        notify_status_func=notify_status_func,
        process_func_handle=lambda: logging.debug('no progress'))
    temp = soln['x']
    n = np.copy(n_0)
    for rel in range(x_mal):
        r_to_relax = [j for j in range(nuij.shape[1])]
        while r_to_relax:
            # nach Spannung sortieren, gespannteste Reaktion zunächst
            # entspannen.
            order = np.argsort(
                [-abs(gg_abstaende(k_x_j[i], nuij[:, i], n, 0)) for i in r_to_relax])
            j = r_to_relax[order[0]]
            x0 = np.concatenate([[xi_j_accum_isot[j]], [temp]])
            soln = nr_ls(
                x0=x0,
                f=lambda xi_t: gg_abstaende(
                  k_x_j[j], nuij[:, j], n, xi_t, h_0, 'adiabat'),
                j=lambda xi_t: jac_gg_abstaende(
                  k_x_j[j], nuij[:, j], n, xi_t, h_0, 'adiabat'),
                tol=1e-5,
                max_it=1000,
                inner_loop_condition=lambda xi_t: all(
                  [item >= 0 or abs(item) < eps for item in
                   n + nuij[:, j] * xi_t[:-1]]),
                notify_status_func=notify_status_func,
                process_func_handle=lambda: logging.debug('no progress'))

            soln_xi = soln['x'][:-1]

            temp = soln['x'][-1]

            h_0 = h(temp)

            xi_j[j] = soln_xi

            xi_j_accum_adiab[j] = xi_j_accum_adiab[j] + xi_j[j]
            n = n + nuij[:, j] * xi_j[j]

            r_to_relax.pop(order[0])
    return (n, xi_j, xi_j_accum_adiab, temp)


def fj_0(xi, n_0, k_t, nuij):
    if np.ndim(nuij) > 1:
        n_r = nuij.shape[1]
        n = n_0 + nuij.dot(xi)
    else:
        n_r = 1
        n = n_0 + nuij * xi
    n_t = sum(n)
    pi = np.product(
        np.power(n / n_t, nuij.T),
        axis=nuij.ndim - 1
    )
    if n_r == 1:
        pi = np.array([pi])
    f = pi - k_t
    logging.debug('x=' + str(xi))
    logging.debug('f=' + str(f))
    return f


stoech_m, indexes, nach_g_sortieren, \
    k_x_1, k_1, nuij = stoech_matrix(atom_m, g_1, p, t_aus_rdampfr, namen, [4])
n_1 = np.copy(n_0)
n_1_norm = n_1 / sum(n_0)  # Normalisieren
# Entspannung
_, _, xi_1_norm, _ = r_entspannung_isoth(
    k_x_1, nuij, n_1_norm, t_aus_rdampfr, p, 1)
# Newton Raphson
soln = nr_ls(
    x0=xi_1_norm,
    f=lambda xi: gg_abstaende(k_x_1, nuij, n_1_norm, xi),
    j=lambda xi: jac_gg_abstaende(k_x_1, nuij, n_1_norm, xi),
    tol=1e-12,
    max_it=1000,
    inner_loop_condition=lambda xi:
    all([item >= 0 for item in
       n_1_norm + nuij.dot(xi)]),
    notify_status_func=notify_status_func,
    process_func_handle=lambda: logging.debug('no progress'))

xi_1 = soln['x'] * sum(n_0)
n_1 = (n_0 + nuij.dot(xi_1))
fehler = fj_0(xi_1, n_0, k_x_1, nuij)

for komb in n_1:
    logging.debug('{0:0.20g}'.format(komb / 1000.).replace('.', ','))

for komb in np.array(xi_1):
    logging.debug('{0:0.20g}'.format(komb / 1000.).replace('.', ','))

print('========================================')
print('Dampf-Reformierung: Vorwärmer + Reaktor')
print('========================================')

q = sum(n_1 * h_1 - n_0 * h_0)  # mol/h * J/mol = J/h


print(
    'Vormärmer (auf T= ' + str(t_aus_rdampfr) + ' °K), Q: ' +
    '{0:0.20g}'.format(
        sum(n_0 * (h_1 - h_0)) * 1 / 60. ** 2 *
        1 / 1000.  # 1h/60^2s * 1kW / 1000W
    ).replace('.', ',') + ' kW'
)

print('Berücksichtigte (unabhängige) Reaktionen:')

for j, row in enumerate(np.array(nuij.T)):
    lhs = '+ '.join([str(abs(row[i])) + ' ' + namen[i]
                     for i in np.where(row < 0)[0]])
    rhs = '+'.join([str(abs(row[i])) + ' ' + namen[i]
                    for i in np.where(row > 0)[0]])
    print(lhs + '<<==>>' + rhs + '    K(T)=' + '{:g}'.format(k_1[j]))

print(
    'Isothermisch betriebener Reaktor, Q: ' +
    '{0:0.20g}'.format(
        sum(nuij.dot(xi_1) * h_1) * 1 / 60. ** 2 * 1 / 1000.
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

print('Gesamtfehler: ' + str(np.sqrt(fehler.dot(fehler))))
print('')
print('========================================')
for j in range(2):
    print('')

print('========================================')
print('Wassergashift: Vorkühler + Reaktor')
print('========================================')

t_aus_rwgs = 210 + 273.15  # °K
h_2 = h(t_aus_rwgs)
g_2 = g(t_aus_rwgs, h_2)
# Vorkühler-Leistung
q = sum(n_1 * (h_2 - h_1))  # mol/h * J/mol = J/h

stoech_m, indexes, nach_g_sortieren, \
    k_x_2, k_2, nuij = stoech_matrix(atom_m, g_2, p, t_aus_rwgs, namen, None)
# Force selected reactions
# namen = ['CO', 'H2', 'CO2', 'H2O', 'CH4', 'NH3', 'AR', 'O2', 'N2']
nuij_force = np.array([
    [+1, +2, +0, +0, -1, +0, +0, -1 / 2, +0],  # PO
    [+1, +3, +0, -1, -1, +0, +0, +0, +0],  # DR
    [-1, +1, +1, -1, +0, +0, +0, +0, +0],  # WGS
    [-1, -3, +0, +1, +1, +0, +0, +0, +0],  # Meth1
    [+0, -4, -1, +2, +1, +0, +0, +0, +0],  # Meth2
    [+0, -3 / 2, 0, 0, +0, +1, +0, +0, -1 / 2]  # Amm
]).T
# WGS
nuij = np.array([nuij_force[:, 2]]).T
k_2 = np.exp(-nuij.T.dot(g_2 / (r * t_aus_rwgs)))
k_x_2 = np.multiply(k_2, np.power(p / 1., -sum(nuij)))
for i in range(len(nuij.T)):
    if k_x_2[i] > 1:
        nuij.T[i] = -nuij.T[i]
k_2 = np.exp(-nuij.T.dot(g_2 / (r * t_aus_rwgs)))
k_x_2 = np.multiply(k_2, np.power(p / 1., -sum(nuij)))

n_2 = np.copy(n_1)
n_2_norm = n_2 / sum(n_1)  # Normalisieren
# Entspannung
_, _, xi_2_norm, _ = r_entspannung_isoth(
    k_x_2, nuij, n_2_norm, t_aus_rwgs, p, 1)
# Newton Raphson
soln = nr_ls(
    x0=xi_2_norm,
    f=lambda xi: gg_abstaende(k_x_2, nuij, n_2_norm, xi),
    j=lambda xi: jac_gg_abstaende(k_x_2, nuij, n_2_norm, xi),
    tol=1e-12,
    max_it=1000,
    inner_loop_condition=lambda xi:
    all([item >= 0 for item in
       n_2_norm + nuij.dot(xi)]),
    notify_status_func=notify_status_func,
process_func_handle=lambda: logging.debug('no progress'))
xi_2 = soln['x'] * sum(n_1)
n_2 = (n_1 + nuij.dot(xi_2))
fehler = fj_0(xi_2, n_1, k_x_2, nuij)

print(
    'Vorkühler (auf T= ' + str(t_aus_rwgs) + ' °K), Q: ' +
    '{0:0.20g}'.format(
        q * 1 / 60. ** 2 * 1 / 1000.  # 1h/60^2s * 1kW / 1000W
    ).replace('.', ',') + ' kW'
)

print('Berücksichtigte Reaktionen:')

for j, row in enumerate(np.array(nuij.T)):
    lhs = '+ '.join([str(abs(row[i])) + ' ' + namen[i]
                     for i in np.where(row < 0)[0]])
    rhs = '+'.join([str(abs(row[i])) + ' ' + namen[i]
                    for i in np.where(row > 0)[0]])
    print(lhs + '<<==>>' + rhs + '    K(T)=' + '{:g}'.format(k_2[j]))

q = sum(n_2 * h_2 - n_1 * h_1)  # mol/h * J/mol = J/h

print(
    'Isothermisch betriebener Reaktor, Q: ' +
    '{0:0.20g}'.format(
        sum(nuij.dot(xi_2) * h_2) * 1 / 60. ** 2 * 1 / 1000.
    ).replace('.', ',') + ' kW'
)

print(
    'Totale zu tauschende Energie, Q: ' +
    '{0:0.20g}'.format(
        q * 1 / 60.**2 * 1 / 1000.
    ).replace('.', ',') + ' kW'
)

print('')

for i, komb in enumerate(np.array(n_2)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum(n_2) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(np.array(n_2 / sum(n_2))):
    print('y_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('T: ' + '{:g}'.format(t_aus_rwgs) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum(n_2 * h_2) / sum(n_2) * 1000.) + ' J/kmol')

print('Gesamtfehler: ' + str(np.sqrt(fehler.dot(fehler))))

print('')

print('========================================')
for j in range(2):
    print('')

print('========================================')
print('Trocknung: Vorkühler + Abscheider')
print('========================================')

t_aus_tkuehler = 20 + 273.15  # °K
h_3 = h(t_aus_tkuehler)
g_3 = g(t_aus_tkuehler, h_3)

# Lösung des isothermischen Verdampfers (des Abscheiders)
z_i = n_2 / sum(n_2)
soln = z_l_v.pt_flash(t_aus_tkuehler, p, z_i, tc, pc, omega_af, alpha_tr, epsilon, sigma, psi, omega, tol=1e-10)
y_i = soln['y_i']
x_i = soln['x_i']
v_f = soln['v_f']
k_i_verteilung = soln['k_i']

n_3_l = (1 - v_f) * sum(n_2) * x_i
n_3_v = v_f * sum(n_2) * y_i


# die Verdampfungsenthalpie der Mischung lässt sich anhand der Clausius-Clapeyron
# Gleichung berechnen, wofür man dp_s/dT braucht:
# $\Delta h_v = T(v''-v') \fac{d p_s}{d_T}$
# Bei Berechnung des Gemisches entstehen Probleme, wenn es keinen Siedepunkt
# aufweist, aber für die einzelnen Komponente erfolgt die Berechnung, was
# letztendlich die übliche Praxis für Verdampfungsenthalpie ist (Wärmeatlas D1 -
# Gl 49).
# Das molare Volumen der Flüssigkeit v' ist gegenüber demjenigen des Gases v''
# vernachlässigbar.

p_sat = np.empty_like(n_2)
v_m_l = np.empty_like(n_2)
v_m_v = np.empty_like(n_2)
z_l = np.empty_like(n_2)
z_v = np.empty_like(n_2)
delta_h_v = np.empty_like(n_2)

for i in range(len(p_sat)):
    if not t_aus_tkuehler > tc[i]:
        soln = z_l_v.bubl_p(
            t_aus_tkuehler, 1e-5, 1.0, tc[i], pc[i], omega_af[i],
            alpha_tr, epsilon, sigma, psi, omega, tol=1e-10, y_i_est=1.0)
        p_sat[i] = soln['p']
        z_l[i] = soln['z_l']
        z_v[i] = soln['z_v']
        v_m_l[i] = z_l[i] * r * t_aus_tkuehler / (
            p_sat[i] * 100000.)  # m^3 / mol
        v_m_v[i] = z_v[i] * r * t_aus_tkuehler / (
            p_sat[i] * 100000.)  # m^3 / mol
        t_1, t_2 = t_aus_tkuehler, 1.0001 * t_aus_tkuehler
        ps_2 = z_l_v.bubl_p(
            t_2, p_sat[i], 1.0, tc[i], pc[i], omega_af[i],
            alpha_tr, epsilon, sigma, psi, omega, tol=1e-10,
                     y_i_est=1.0)['p']
        ps_1 = z_l_v.bubl_p(
            t_1, p_sat[i], 1.0, tc[i], pc[i], omega_af[i],
            alpha_tr, epsilon, sigma, psi, omega, tol=1e-10,
            y_i_est=1.0)['p']
        dps_dt = (ps_2 - ps_1) / (t_2 - t_1) # bar / °K

        # Clausius-Clapeyron
        delta_h_v[i] = t_aus_tkuehler * (v_m_v[i] - v_m_l[i]) * dps_dt * \
            100000.  # J / mol
    else:
        p_sat[i] = np.nan
        z_l[i] = np.nan
        z_v[i] = np.nan
        v_m_l[i] = np.nan
        v_m_v[i] = np.nan
        delta_h_v[i] = 0


# Vorkühler-Leistung, inklusive Verflüssigung des entsprechenden
# Anteils bei Trockner-Temperatur (20°C)
q = sum(n_2 * (h_3 - h_2)) + sum(n_3_l * -delta_h_v)  # mol/h * J/mol = J/h

print(
    'Vorkühler (auf T= ' + str(t_aus_tkuehler) + ' °K), Q: ' +
    '{0:0.20g}'.format(
        q * 1 / 60. ** 2 * 1 / 1000.  # 1h/60^2s * 1kW / 1000W
    ).replace('.', ',') + ' kW'
)


print('')

print('Dampfphase:')

for i, komb in enumerate(np.array(n_3_v)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum(n_3_v) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(y_i):
    print('y_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('H2/N2 Verhältnis: ' +
      str(y_i[namen.index('H2')] /
          y_i[namen.index('N2')]))
print('Stoechiometrisches Verhältnis: ' + str(3))
print('')
print('T: ' + '{:g}'.format(t_aus_tkuehler) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum(n_3_v * h_3) / sum(n_3_v) * 1000.) + ' J/kmol')

print('')

print('========================================')

print('Flüssigphase (Abwasser):')

for i, komb in enumerate(np.array(n_3_l)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum(n_3_l) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(x_i):
    print('x_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('T: ' + '{:g}'.format(t_aus_tkuehler) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum(n_3_l * h_3) / sum(n_3_l) * 1000.) + ' J/kmol')

print('')

print('========================================')
for j in range(2):
    print('')

print('========================================')
print('CO2-Abschneider: ')
print('========================================')


h_4 = h_3
abschn_fakt_ab = np.zeros([len(n_3_v), len(n_3_v)])
abschn_fakt_pr = np.eye(len(n_3_v))
co2_index = np.array([n == 'CO2' for n in namen])
abschn_fakt_ab[co2_index, co2_index] = 0.95
abschn_fakt_pr[co2_index, co2_index] = 0.05

n_4_ab = abschn_fakt_ab.dot(n_3_v)
n_4_pr = abschn_fakt_pr.dot(n_3_v)

print('Abtrennung-Faktor (CO2): ' + str(0.05) + ' - Produkt')

for i, komb in enumerate(np.array(n_4_pr)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum(n_4_pr) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(n_4_pr / sum(n_4_pr)):
    print('y_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('T: ' + '{:g}'.format(t_aus_tkuehler) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum(n_4_pr * h_4) / sum(n_4_pr) * 1000.) + ' J/kmol')

print('')

print('========================================')
print('Abtrennung-Faktor (CO2): ' + str(0.95) + ' - Abfluss')

for i, komb in enumerate(np.array(n_4_ab)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum(n_4_ab) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(n_4_ab / sum(n_4_ab)):
    print('y_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('T: ' + '{:g}'.format(t_aus_tkuehler) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum(n_4_ab * h_4) / sum(n_4_ab) * 1000.) + ' J/kmol')
print('')

print('========================================')
for j in range(2):
    print('')

print('========================================')
print('Methanisierung: Aufheizer + Reaktor')
print('========================================')

t_aus_rmeth = 350 + 273.15  # °K
h_5 = h(t_aus_rmeth)
g_5 = g(t_aus_rmeth, h_5)
# Aufheizer-Leistung
q = sum(n_4_pr * (h_5 - h_4))  # mol/h * J/mol = J/h

stoech_m, indexes, nach_g_sortieren, \
    k_x_5, k_5, nuij = stoech_matrix(atom_m, g_5, p, t_aus_rmeth, namen, None)

# Ammoniaksynthese sei künstlich ausgeschlossen
k_5 = k_5[nuij[namen.index('NH3'), :] == 0]
k_x_5 = k_x_5[nuij[namen.index('NH3'), :] == 0]
nuij = nuij[:, nuij[namen.index('NH3'), :] == 0]

n_5 = np.copy(n_4_pr)
n_5_norm = n_5 / sum(n_5)  # Normalisieren
# Entspannung
_, _, xi_5_norm, _ = r_entspannung_isoth(
    k_x_5, nuij, n_5_norm, t_aus_rmeth, p, 1)
# Newton Raphson
soln = nr_ls(
    x0=xi_5_norm,
    f=lambda xi: gg_abstaende(k_x_5, nuij, n_5_norm, xi),
    j=lambda xi: jac_gg_abstaende(k_x_5, nuij, n_5_norm, xi),
    tol=1e-12,
    max_it=1000,
    inner_loop_condition=lambda xi:
    all([item >= 0 for item in
       n_5_norm + nuij.dot(xi)]),
    notify_status_func=notify_status_func,
    process_func_handle=lambda: logging.debug('no progress'))
xi_5 = soln['x'] * sum(n_5)
n_5 = (n_4_pr + nuij.dot(xi_5))
fehler = fj_0(xi_5, n_4_pr, k_x_5, nuij)

print(
    'Aufheizer (auf T= ' + str(t_aus_rmeth) + ' °K), Q: ' +
    '{0:0.20g}'.format(
        q * 1 / 60. ** 2 * 1 / 1000.  # 1h/60^2s * 1kW / 1000W
    ).replace('.', ',') + ' kW'
)

print('Berücksichtigte (unabhängige) Reaktionen:')

for j, row in enumerate(np.array(nuij.T)):
    lhs = '+ '.join([str(abs(row[i])) + ' ' + namen[i]
                     for i in np.where(row < 0)[0]])
    rhs = '+'.join([str(abs(row[i])) + ' ' + namen[i]
                    for i in np.where(row > 0)[0]])
    print(lhs + '<<==>>' + rhs + '    K(T)=' + '{:g}'.format(k_5[j]))

q = sum(n_5 * h_5 - n_4_pr * h_4)  # mol/h * J/mol = J/h

print(
    'Isothermisch betriebener Reaktor, Q: ' +
    '{0:0.20g}'.format(
        sum(nuij.dot(xi_5) * h_5) * 1 / 60. ** 2 * 1 / 1000.
    ).replace('.', ',') + ' kW'
)

print(
    'Totale zu tauschende Energie, Q: ' +
    '{0:0.20g}'.format(
        q * 1 / 60.**2 * 1 / 1000.
    ).replace('.', ',') + ' kW'
)

print('')

for i, komb in enumerate(np.array(n_5)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum(n_5) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(np.array(n_5 / sum(n_5))):
    print('y_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('T: ' + '{:g}'.format(t_aus_rmeth) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum(n_5 * h_5) / sum(n_5) * 1000.) + ' J/kmol')

print('Gesamtfehler: ' + str(np.sqrt(fehler.dot(fehler))))

print('')

print('========================================')
for j in range(2):
    print('')

print('========================================')
print('Verdichtung: Adiabater Verdichter + Abkühler')
print('========================================')

n = n_5
p2 = 180.  # bar
p1 = 35.  # bar
p = p2
t1 = t_aus_rmeth

# isentropisch: $dS = \frac{dQ_rev}{T}=\frac{Cp^{iG}}{T}-R/PdP=0$

t2_rev = optimize.newton(
    lambda t: np.log(p2 / p1) -
    mcps(n / sum(n), t1, t) * np.log(t / t1),
    t1 + 5
)

wg_mech = 1.0

wg_th = 0.72

h1 = h(t1)

h2 = h(t2_rev)

# p12: Verdichter-Leistung. Zunächst reversibel, danach mit Wirkungsgraden.

p12_rev = sum(n * (h2 - h1))

# Alt. Methode (AP, PAT)

# Idealgas: $ gamma \equiv C_p/C_v = C_v + R $
gamma = 1 / (1 - 1 / mcps(n / sum(n), t1, t2_rev))

p12_rev = sum(n) * r * t1 / ((gamma - 1) / gamma) * (
    (p2 / p1)**((gamma - 1) / gamma) - 1)

p12 = p12_rev / wg_th / wg_mech


# Gmehling-Methode
t2 = optimize.newton(
    lambda t2: - sum(n / sum(n) * h(t2)) + (
        p12_rev / wg_th / sum(n) + sum(n / sum(n) * h1)),
    t2_rev
)

# SVN-Methode
t2 = optimize.newton(
    lambda t2: -t2 + t1 + sum(n * (h2 - h1)) / sum(n) / wg_th / (
        r * mcph(n / sum(n), t1, t2)
    ), t2_rev)

h2 = h(t2)

t_aus_verdichter = t2
t_aus_vd_kuehler = t1

q = sum(n * (h1 - h2))

print(
    'Verdichten (auf p= ' + str(p) + ' bar), W: ' +
    '{0:0.20g}'.format(
        p12_rev / wg_th * 1 / 60. ** 2  # 1h/60^2s
    ).replace('.', ',') + ' W'
)

print('gamma=Cp/Cv=: ' + '{:g}'.format(gamma) + '')

print('T: ' + '{:g}'.format(t_aus_verdichter) + ' °K')

print('')

for i, komb in enumerate(np.array(n_5)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum(n_5) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(np.array(n_5 / sum(n_5))):
    print('y_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('T: ' + '{:g}'.format(t_aus_verdichter) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum(n_5 * h2) / sum(n_5) * 1000.) + ' J/kmol')

print('')

print('========================================')

print(
    'Abkühlen (zurück auf T= ' + str(t_aus_vd_kuehler) + ' °K), Q: ' +
    '{0:0.20g}'.format(
        q * 1 / 60. ** 2 * 1 / 1000.  # 1h/60^2s * 1kW / 1000W
    ).replace('.', ',') + ' kW'
)

print('')

for i, komb in enumerate(np.array(n_5)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum(n_5) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(np.array(n_5 / sum(n_5))):
    print('y_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('T: ' + '{:g}'.format(t_aus_vd_kuehler) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum(n_5 * h1) / sum(n_5) * 1000.) + ' J/kmol')

print('')

print('========================================')
for j in range(2):
    print('')

print('========================================')
for j in range(2):
    print('')
it_str = '1.' + ' ' + 'Iteration'
print(
    '=' * int(np.ceil((40 - len(it_str)) / 2.)) +
    it_str +
    '=' * int(np.floor((40 - len(it_str)) / 2.))
)
for j in range(2):
    print('')
print('========================================')
for j in range(2):
    print('')

print('========================================')
print('Mischung: Rücklauf + Zulauf + Vorwärmer')
print('========================================')

rlv = 0

t_mischung = t_aus_vd_kuehler
h_mischung = h_5

n_6 = n_5 + 0

t_ein_nh3_syn = 300 + 273.15  # °K
h_6 = h(t_ein_nh3_syn)
g_6 = g(t_ein_nh3_syn, h_6)
# Aufheizer-Leistung
q = sum(n_6 * (h_6 - h_mischung))  # mol/h * J/mol = J/h

print('Rücklaufverhältnis = nr/nv = ' + '{:g}'.format(rlv) + '')
print('Temperatur der Mischung: ' + '{:g}'.format(t_ein_nh3_syn) + ' °K')
print(
    'Vorwärmer (auf T= ' + str(t_ein_nh3_syn) + ' °K), Q: ' +
    '{0:0.20g}'.format(
        q * 1 / 60. ** 2 * 1 / 1000.  # 1h/60^2s * 1kW / 1000W
    ).replace('.', ',') + ' kW'
)

print('')

for i, komb in enumerate(np.array(n_6)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum(n_6) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(np.array(n_6 / sum(n_6))):
    print('y_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('T: ' + '{:g}'.format(t_ein_nh3_syn) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum(n_6 * h_6) / sum(n_6) * 1000.) + ' J/kmol')

print('')
print('========================================')
for j in range(2):
    print('')


print('========================================')
print('Adiabate Ammoniaksynthese: Reaktor')
print('========================================')


stoech_m, indexes, nach_g_sortieren, \
    k_x_7, k_7, nuij = stoech_matrix(atom_m, g_6, p, t_ein_nh3_syn, namen,
                                     [namen.index('NH3')])

print('Berücksichtigte (unabhängige) Reaktionen:')

for j, row in enumerate(np.array(nuij.T)):
    lhs = '+ '.join([str(abs(row[i])) + ' ' + namen[i]
                     for i in np.where(row < 0)[0]])
    rhs = '+'.join([str(abs(row[i])) + ' ' + namen[i]
                    for i in np.where(row > 0)[0]])
    print(lhs + '<<==>>' + rhs + '    K(T)=' + '{:g}'.format(k_7[j]))

n_6_norm = n_6 / sum(n_6)  # Normalisieren
n_7_norm = n_6_norm
# Entspannung
n_7_norm, _, xi_7_norm, t_aus_nh3_syn = r_entspannung_adiab(
    k_x_7, nuij, n_6_norm, t_ein_nh3_syn, p, 1)
# denormalize
xi_7 = xi_7_norm * sum(n_6)
# Newton Raphson
x0 = np.concatenate([xi_7, [t_aus_nh3_syn]])
soln = nr_ls(
    x0=x0,
    f=lambda xi_t: gg_abstaende(
      k_x_7, nuij, n_6, xi_t, h_6, 'adiabat'),
    j=lambda xi_t: jac_gg_abstaende(
      k_x_7, nuij, n_6, xi_t, h_6, 'adiabat'),
    tol=1e-12,
    max_it=1000,
    inner_loop_condition=lambda xi_t: all(
      [item >= 0 for item in
       n_6 + nuij.dot(xi_t[:-1])]),
    notify_status_func=notify_status_func,
process_func_handle=lambda: logging.debug('no progress'))
xi_7 = soln['x'][:-1]
t_aus_nh3_syn = soln['x'][-1]
n_7 = (n_6 + nuij.dot(xi_7))
h_7 = h(t_aus_nh3_syn)
g_7 = g(t_aus_nh3_syn, h_7)
k_7 = k(t_aus_nh3_syn, g_7, nuij)
k_x_7 = np.multiply(k_7, np.power(p / 1., -sum(nuij)))
fehler = gg_abstaende(k_x_7, nuij, n_6,
                      np.concatenate([xi_7, [t_aus_nh3_syn]]),
                      h_6, 'adiabat')

h_7 = h(t_aus_nh3_syn)

q = sum(n_7 * h_7 - n_6 * h_6)  # mol/h * J/mol = J/h

print(
    'Totale zu tauschende Energie, Q: ' +
    '{0:0.20g}'.format(
        q * 1 / 60.**2 * 1 / 1000.
    ).replace('.', ',') + ' kW (adiabat)'
)

print('')

for i, komb in enumerate(np.array(n_7)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum(n_7) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(np.array(n_7 / sum(n_7))):
    print('y_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('T: ' + '{:g}'.format(t_aus_nh3_syn) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum(n_7 * h_7) / sum(n_7) * 1000.) + ' J/kmol')

print('Gesamtfehler: ' + str(np.sqrt(fehler.dot(fehler))))

print('')

print('========================================')
for j in range(2):
    print('')

print('========================================')
print('Abkühler + Produkt-Abscheider')
print('========================================')

t_aus_nh3_fl = 21 + 273.15  # °K
h_8 = h(t_aus_nh3_fl)
g_8 = g(t_aus_nh3_fl, h_8)

# Lösung des isothermischen Verdampfers (des Abscheiders)
z_i = n_7 / sum(n_7)
x_i = 1 / len(n_7) * np.ones(len(n_7))
y_i = z_i
v_f = 1.
# make a better estimate of p
soln = z_l_v.p_i_sat_ceos(t_aus_tkuehler, p, tc, pc, omega_af, alpha_tr, epsilon, sigma, psi, omega, tol=1e-10)
pisat = np.array([soln['p'][i] for i in range(len(z_i)) if soln['success'][i]])
zisat = np.array([z_i[i] for i in range(len(z_i)) if soln['success'][i]])
p_est = sum(pisat * zisat / sum(zisat))
soln = z_l_v.pt_flash(t_aus_tkuehler, p, z_i, tc, pc, omega_af,
                      alpha_tr, epsilon, sigma, psi, omega, tol=1e-10, p_est_0=p_est)
v_f = soln['v_f']
x_i = soln['x_i']
y_i = soln['y_i']
k_i_verteilung = soln['k_i']
n_8_l = (1 - v_f) * sum(n_7) * x_i
n_8_v = v_f * sum(n_7) * y_i

# Verdampfungsenthalpie-Berechnung (s. oben).
p_sat = np.empty_like(n_7)
v_m_l = np.empty_like(n_7)
v_m_v = np.empty_like(n_7)
z_l = np.empty_like(n_7)
z_v = np.empty_like(n_7)
delta_h_v = np.empty_like(n_7)

for i in range(len(p_sat)):
    if not t_aus_tkuehler > tc[i]:
        soln = z_l_v.bubl_p(
            t_aus_tkuehler, 1e-5, 1.0, tc[i], pc[i], omega_af[i],
            alpha_tr, epsilon, sigma, psi, omega, tol=1e-10, y_i_est=1.0)
        p_sat[i] = soln['p']
        z_l[i] = soln['z_l']
        z_v[i] = soln['z_v']
        v_m_l[i] = z_l[i] * r * t_aus_nh3_fl / (
            p_sat[i] * 100000.)  # m^3 / mol
        v_m_v[i] = z_v[i] * r * t_aus_nh3_fl / (
            p_sat[i] * 100000.)  # m^3 / mol
        t_1, t_2 = t_aus_tkuehler, 1.0001 * t_aus_tkuehler
        ps_2 = z_l_v.bubl_p(
            t_2, p_sat[i], 1.0, tc[i], pc[i], omega_af[i],
            alpha_tr, epsilon, sigma, psi, omega, tol=1e-10,
            y_i_est=1.0)['p']
        ps_1 = z_l_v.bubl_p(
            t_1, p_sat[i], 1.0, tc[i], pc[i], omega_af[i],
            alpha_tr, epsilon, sigma, psi, omega, tol=1e-10,
            y_i_est=1.0)['p']
        dps_dt = (ps_2 - ps_1) / (t_2 - t_1)  # bar / °K
        # Clausius-Clapeyron
        delta_h_v[i] = t_aus_nh3_fl * (v_m_v[i] - v_m_l[i]) * dps_dt * \
            100000.  # J / mol
    else:
        p_sat[i] = np.nan
        z_l[i] = np.nan
        z_v[i] = np.nan
        v_m_l[i] = np.nan
        v_m_v[i] = np.nan
        delta_h_v[i] = 0

# Vorkühler-Leistung, inklusive Verflüssigung des entsprechenden
# Anteils bei Trockner-Temperatur (20°C)
q = sum(n_7 * (h_8 - h_7)) + sum(n_8_l * -delta_h_v)  # mol/h * J/mol = J/h

print(
    'Vorkühler (auf T= ' + str(t_aus_nh3_fl) + ' °K), Q: ' +
    '{0:0.20g}'.format(
        q * 1 / 60. ** 2 * 1 / 1000.  # 1h/60^2s * 1kW / 1000W
    ).replace('.', ',') + ' kW'
)

print('')

print('Dampfphase (zum Splitter):')

for i, komb in enumerate(np.array(n_8_v)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum(n_8_v) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(y_i):
    print('y_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('H2/N2 Verhältnis: ' +
      str(y_i[namen.index('H2')] /
          y_i[namen.index('N2')]))
print('Stoechiometrisches Verhältnis: ' + str(3))
print('')
print('T: ' + '{:g}'.format(t_aus_nh3_fl) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum(n_8_v * h_8) / sum(n_8_v) * 1000.) + ' J/kmol')

print('')

print('========================================')

print('Produkt-Flüssigkeit:')

for i, komb in enumerate(np.array(n_8_l)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum(n_8_l) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(x_i):
    print('x_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('T: ' + '{:g}'.format(t_aus_nh3_fl) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum(n_8_l * h_8) / sum(n_8_l) * 1000.) + ' J/kmol')

print('')

print('========================================')
for j in range(2):
    print('')

print('========================================')
print('Abtrennung: Splitter')
print('========================================')

rlv = 1 - 0.2

print('Rücklaufverhältnis = nr/nv = ' + '{:g}'.format(rlv) + '')

print('')

print('Rücklaufstrom:')

n_9 = rlv * n_8_v
h_9 = h_8

for i, komb in enumerate(np.array(n_9)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum(n_9) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(np.array(n_9 / sum(n_9))):
    print('y_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('T: ' + '{:g}'.format(t_aus_nh3_fl) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum(n_9 * h_9) / sum(n_9) * 1000.) + ' J/kmol')

print('========================================')

print('')

print('Abgas:')

for i, komb in enumerate(np.array((1 - rlv) * n_8_v)):
    print('n_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
          ' kmol/h')
print('n_T' + '=' +
      '{0:0.20g}'.format(sum((1 - rlv) * n_8_v) / 1000.).replace('.', ',') +
      ' kmol/h')
print('')
for i, komb in enumerate(np.array((1 - rlv) * n_8_v / sum((1 - rlv) * n_8_v))):
    print('y_{' + namen[i] + '}=' +
          '{0:0.20g}'.format(komb).replace('.', ',')
          )
print('')
print('T: ' + '{:g}'.format(t_aus_nh3_fl) + ' °K')
print('p: ' + '{:g}'.format(p) + ' bar')
print('H: ' + '{0:0.6f}'.format(
    sum((1 - rlv) * n_8_v * h_8) / sum((1 - rlv) * n_8_v) * 1000.) + ' J/kmol')

print('========================================')
for j in range(2):
    print('')

# Rücklaufschleife wiederholen, bis Konvergenz erreicht.
# Bei der 10. Iteration erreicht man  722,65°K=449,5°C
for it_n in range(1, 25):

    print('========================================')
    for j in range(2):
        print('')
    it_str = str(it_n) + '. ' + 'Iteration'
    print(
        '=' * int(np.ceil((40 - len(it_str)) / 2.)) +
        it_str +
        '=' * int(np.floor((40 - len(it_str)) / 2.))
    )
    for j in range(2):
        print('')
    print('========================================')
    for j in range(2):
        print('')

    print('========================================')
    print('Mischung: Rücklauf + Zulauf + Vorwärmer')
    print('========================================')

    # Elim. n<10^-16
    n_8_v_t = sum(n_8_v)
    n_8_v[abs(n_8_v) < eps] = 0
    n_8_v = n_8_v / sum(n_8_v) * n_8_v_t
    n_6 = n_5 + rlv * n_8_v

    t_mischung = optimize.root(
        lambda temp: sum(
            n_5 * h_5 + rlv * n_8_v * h_8 - n_6 * h(temp)
        ), t_aus_vd_kuehler
    ).x.item()
    h_mischung = h(t_mischung)
    # Aufheizer-Leistung
    q = sum(n_6 * (h_6 - h_mischung))  # mol/h * J/mol = J/h

    print('Rücklaufverhältnis = nr/nv = ' + '{:g}'.format(rlv) + '')
    print('Temperatur der Mischung: ' + '{:g}'.format(t_mischung) + ' °K')
    print(
        'Vorwärmer (auf T= ' + str(t_ein_nh3_syn) + ' °K), Q: ' +
        '{0:0.20g}'.format(
            q * 1 / 60. ** 2 * 1 / 1000.  # 1h/60^2s * 1kW / 1000W
        ).replace('.', ',') + ' kW'
    )

    print('')

    for i, komb in enumerate(np.array(n_6)):
        print('n_{' + namen[i] + '}=' +
              '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
              ' kmol/h')
    print('n_T' + '=' +
          '{0:0.20g}'.format(sum(n_6) / 1000.).replace('.', ',') +
          ' kmol/h')
    print('')
    for i, komb in enumerate(np.array(n_6 / sum(n_6))):
        print('y_{' + namen[i] + '}=' +
              '{0:0.20g}'.format(komb).replace('.', ',')
              )
    print('')
    print('T: ' + '{:g}'.format(t_ein_nh3_syn) + ' °K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{0:0.6f}'.format(
        sum(n_6 * h_6) / sum(n_6) * 1000.) + ' J/kmol')

    print('')
    print('========================================')
    for j in range(2):
        print('')

    print('========================================')
    print('Adiabate Ammoniaksynthese: Reaktor')
    print('========================================')

    # Mit Zulauf-Temperatur als erster Ansatz beginnen.
    h_7 = h(t_ein_nh3_syn)
    g_7 = g(t_ein_nh3_syn, h_7)
    k_7 = k(t_ein_nh3_syn, g_7, nuij)
    k_x_7 = np.multiply(k_7, np.power(p / 1., -sum(nuij)))

    print('Berücksichtigte (unabhängige) Reaktionen:')

    for j, row in enumerate(np.array(nuij.T)):
        lhs = '+ '.join([str(abs(row[i])) + ' ' + namen[i]
                         for i in np.where(row < 0)[0]])
        rhs = '+'.join([str(abs(row[i])) + ' ' + namen[i]
                        for i in np.where(row > 0)[0]])
        print(lhs + '<<==>>' + rhs + '    K(T)=' + '{:g}'.format(k_7[j]))

    n_6_norm = n_6 / sum(n_6)  # Normalisieren
    n_7_norm = n_6_norm
    # Entspannung
    n_7_norm, _, xi_7_norm, t_aus_nh3_syn = r_entspannung_adiab(
        k_x_7, nuij, n_6_norm, t_ein_nh3_syn, p, 1)
    # Newton Raphson
    x0 = np.concatenate([xi_7_norm, [t_aus_nh3_syn]])
    soln = nr_ls(
        x0=x0,
        f=lambda xi_t: gg_abstaende(
          k_x_7, nuij, n_6_norm, xi_t, h_6, 'adiabat'),
        j=lambda xi_t: jac_gg_abstaende(
          k_x_7, nuij, n_6_norm, xi_t, h_6, 'adiabat'),
        tol=1e-12,
        max_it=1000,
        inner_loop_condition=lambda xi_t: all(
          [item >= 0 or abs(item) for item in
           n_6_norm + nuij.dot(xi_t[:-1])]),
        notify_status_func=notify_status_func,
        process_func_handle=lambda: logging.debug('no progress'))
    xi_7 = soln['x'][:-1]
    # denormalize
    xi_7 = xi_7 * sum(n_6)
    t_aus_nh3_syn = soln['x'][-1]
    n_7 = (n_6 + nuij.dot(xi_7))
    fehler = gg_abstaende(k_x_7, nuij, n_6,
                          np.concatenate([xi_7, [t_aus_nh3_syn]]),
                          h_6, 'adiabat')

    # Fehler-Verbesserung. Da die Ablauf-Temperatur schon bekannt ist, löst man
    # die Massenbilanzen entkoppelt. Dies vermeidet Skalierung-Probleme bei
    # Gleichungssystemen, deren Jacobi-Matrix zugleich Temperatur und
    # Reaktionslaufzahlen enthalten, da Reaktionslaufzahlen normiert werden können,
    # aber Temperatur nicht. Sie ist nämlich eine intensive Größe.
    h_7 = h(t_aus_nh3_syn)
    g_7 = g(t_aus_nh3_syn, h_7)
    k_7 = k(t_aus_nh3_syn, g_7, nuij)
    k_x_7 = np.multiply(k_7, np.power(p / 1., -sum(nuij)))
    soln = nr_ls(
        x0=xi_7,
        f=lambda xi: gg_abstaende(
          k_x_7, nuij, n_6, xi, h_6, 'isotherm'),
        j=lambda xi: jac_gg_abstaende(
          k_x_7, nuij, n_6, xi, h_6, 'isotherm'),
        tol=1e-6,
        max_it=1000,
        inner_loop_condition=lambda xi: all(
          [item >= 0 or abs(item) < eps for item in
           n_6 + nuij.dot(xi)]),
        notify_status_func=notify_status_func,
        process_func_handle=lambda: logging.debug('no progress'))
    xi_7 = soln['x']
    n_7 = (n_6 + nuij.dot(xi_7))
    fehler = gg_abstaende(k_x_7, nuij, n_6,
                          np.concatenate([xi_7, [t_aus_nh3_syn]]),
                          h_6, 'adiabat')

    q = sum(n_7 * h_7 - n_6 * h_6)  # mol/h * J/mol = J/h

    print(
        'Totale zu tauschende Energie, Q: ' +
        '{0:0.20g}'.format(
            q * 1 / 60.**2 * 1 / 1000.
        ).replace('.', ',') + ' kW (adiabat)'
    )

    print('')

    for i, komb in enumerate(np.array(n_7)):
        print('n_{' + namen[i] + '}=' +
              '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
              ' kmol/h')
    print('n_T' + '=' +
          '{0:0.20g}'.format(sum(n_7) / 1000.).replace('.', ',') +
          ' kmol/h')
    print('')
    for i, komb in enumerate(np.array(n_7 / sum(n_7))):
        print('y_{' + namen[i] + '}=' +
              '{0:0.20g}'.format(komb).replace('.', ',')
              )
    print('')
    print('T: ' + '{:g}'.format(t_aus_nh3_syn) + ' °K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{0:0.6f}'.format(
        sum(n_7 * h_7) / sum(n_7) * 1000.) + ' J/kmol')

    print('Gesamtfehler: ' + str(np.sqrt(fehler.dot(fehler))))

    print('')

    print('========================================')
    for j in range(2):
        print('')

    print('========================================')
    print('Abkühler + Produkt-Abscheider')
    print('========================================')

    t_aus_nh3_fl = 21 + 273.15  # °K
    h_8 = h(t_aus_nh3_fl)
    g_8 = g(t_aus_nh3_fl, h_8)

    # Lösung des isothermischen Verdampfers (des Abscheiders)
    z_i = n_7 / sum(n_7)
    soln = z_l_v.p_i_sat_ceos(t_aus_tkuehler, p, tc, pc, omega_af, alpha_tr, epsilon, sigma, psi, omega, tol=1e-10)
    pisat = np.array([soln['p'][i] for i in range(len(z_i)) if soln['success'][i]])
    zisat = np.array([z_i[i] for i in range(len(z_i)) if soln['success'][i]])
    p_est = sum(pisat * zisat / sum(zisat))
    soln = z_l_v.pt_flash(t_aus_tkuehler, p, z_i, tc, pc, omega_af,
                          alpha_tr, epsilon, sigma, psi, omega, tol=1e-10, p_est_0=p_est)
    v_f = soln['v_f']
    y_i = soln['y_i']
    x_i = soln['x_i']
    k_i_verteilung = soln['k_i']

    n_8_l = (1 - v_f) * sum(n_7) * x_i
    n_8_v = v_f * sum(n_7) * y_i

    # Verdampfungsenthalpie-Berechnung (s. oben).
    p_sat = np.empty_like(n_7)
    v_m_l = np.empty_like(n_7)
    v_m_v = np.empty_like(n_7)
    z_l = np.empty_like(n_7)
    z_v = np.empty_like(n_7)
    delta_h_v = np.empty_like(n_7)

    for i in range(len(p_sat)):
        if not t_aus_tkuehler > tc[i]:
            soln = z_l_v.bubl_p(
                t_aus_tkuehler, 1e-5, 1.0, tc[i], pc[i], omega_af[i],
                alpha_tr, epsilon, sigma, psi, omega, tol=1e-10, y_i_est=1.0)
            p_sat[i] = soln['p']
            z_l[i] = soln['z_l']
            z_v[i] = soln['z_v']
            v_m_l[i] = z_l[i] * r * t_aus_nh3_fl / (
                p_sat[i] * 100000.)  # m^3 / mol
            v_m_v[i] = z_v[i] * r * t_aus_nh3_fl / (
                p_sat[i] * 100000.)  # m^3 / mol
            t_1, t_2 = t_aus_tkuehler, 1.0001 * t_aus_tkuehler
            ps_2 = z_l_v.bubl_p(
                t_2, p_sat[i], 1.0, tc[i], pc[i], omega_af[i],
                alpha_tr, epsilon, sigma, psi, omega, tol=1e-10,
                y_i_est=1.0)['p']
            ps_1 = z_l_v.bubl_p(
                t_1, p_sat[i], 1.0, tc[i], pc[i], omega_af[i],
                alpha_tr, epsilon, sigma, psi, omega, tol=1e-10,
                y_i_est=1.0)['p']
            dps_dt = (ps_2 - ps_1) / (t_2 - t_1)  # bar / °K
            # Clausius-Clapeyron
            delta_h_v[i] = t_aus_nh3_fl * (v_m_v[i] - v_m_l[i]) * dps_dt * \
                100000.  # J / mol
        else:
            p_sat[i] = np.nan
            z_l[i] = np.nan
            z_v[i] = np.nan
            v_m_l[i] = np.nan
            v_m_v[i] = np.nan
            delta_h_v[i] = 0

    # Vorkühler-Leistung, inklusive Verflüssigung des entsprechenden
    # Anteils bei Trockner-Temperatur (20°C)
    q = sum(n_7 * (h_8 - h_7)) + sum(n_8_l * -delta_h_v)  # mol/h * J/mol = J/h

    print(
        'Vorkühler (auf T= ' + str(t_aus_nh3_fl) + ' °K), Q: ' +
        '{0:0.20g}'.format(
            q * 1 / 60. ** 2 * 1 / 1000.  # 1h/60^2s * 1kW / 1000W
        ).replace('.', ',') + ' kW'
    )

    print('')

    print('Dampfphase (zum Splitter):')

    for i, komb in enumerate(np.array(n_8_v)):
        print('n_{' + namen[i] + '}=' +
              '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
              ' kmol/h')
    print('n_T' + '=' +
          '{0:0.20g}'.format(sum(n_8_v) / 1000.).replace('.', ',') +
          ' kmol/h')
    print('')
    for i, komb in enumerate(y_i):
        print('y_{' + namen[i] + '}=' +
              '{0:0.20g}'.format(komb).replace('.', ',')
              )
    print('')
    print('H2/N2 Verhältnis: ' +
          str(y_i[namen.index('H2')] /
              y_i[namen.index('N2')]))
    print('Stoechiometrisches Verhältnis: ' + str(3))
    print('')
    print('T: ' + '{:g}'.format(t_aus_nh3_fl) + ' °K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{0:0.6f}'.format(
        sum(n_8_v * h_8) / sum(n_8_v) * 1000.) + ' J/kmol')

    print('')

    print('========================================')

    print('Produkt-Flüssigkeit:')

    for i, komb in enumerate(np.array(n_8_l)):
        print('n_{' + namen[i] + '}=' +
              '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
              ' kmol/h')
    print('n_T' + '=' +
          '{0:0.20g}'.format(sum(n_8_l) / 1000.).replace('.', ',') +
          ' kmol/h')
    print('')
    for i, komb in enumerate(x_i):
        print('x_{' + namen[i] + '}=' +
              '{0:0.20g}'.format(komb).replace('.', ',')
              )
    print('')
    print('T: ' + '{:g}'.format(t_aus_nh3_fl) + ' °K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{0:0.6f}'.format(
        sum(n_8_l * h_8) / sum(n_8_l) * 1000.) + ' J/kmol')

    print('')

    print('========================================')
    for j in range(2):
        print('')

    print('========================================')
    print('Abtrennung: Splitter')
    print('========================================')

    print('Rücklaufverhältnis = nr/nv = ' + '{:g}'.format(rlv) + '')

    print('')

    print('Rücklaufstrom:')

    n_9 = rlv * n_8_v
    h_9 = h_8

    for i, komb in enumerate(np.array(n_9)):
        print('n_{' + namen[i] + '}=' +
              '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
              ' kmol/h')
    print('n_T' + '=' +
          '{0:0.20g}'.format(sum(n_9) / 1000.).replace('.', ',') +
          ' kmol/h')
    print('')
    for i, komb in enumerate(np.array(n_9 / sum(n_9))):
        print('y_{' + namen[i] + '}=' +
              '{0:0.20g}'.format(komb).replace('.', ',')
              )
    print('')
    print('T: ' + '{:g}'.format(t_aus_nh3_fl) + ' °K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{0:0.6f}'.format(
        sum(n_9 * h_9) / sum(n_9) * 1000.) + ' J/kmol')

    print('========================================')

    print('')

    print('Abgas:')

    for i, komb in enumerate(np.array((1 - rlv) * n_8_v)):
        print('n_{' + namen[i] + '}=' +
              '{0:0.20g}'.format(komb / 1000.).replace('.', ',') +
              ' kmol/h')
    print('n_T' + '=' + '{0:0.20g}'.format(sum((1 - rlv) *
                                               n_8_v) / 1000.).replace('.', ',') + ' kmol/h')
    print('')
    for i, komb in enumerate(
            np.array((1 - rlv) * n_8_v / sum((1 - rlv) * n_8_v))):
        print('y_{' + namen[i] + '}=' +
              '{0:0.20g}'.format(komb).replace('.', ',')
              )
    print('')
    print('T: ' + '{:g}'.format(t_aus_nh3_fl) + ' °K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{0:0.6f}'.format(sum((1 - rlv) * n_8_v *
                                        h_8) / sum((1 - rlv) * n_8_v) * 1000.) + ' J/kmol')

    print('========================================')
    for j in range(2):
        print('')
