import z_l_v
import numpy as np
import logging
import itertools
from numerik import rref, ref, nr_ls
from setup_results_log import notify_status_func, setup_log_file

eps = np.finfo(float).eps
np.set_printoptions(linewidth=200)

# Modell feststellen
z_l_v.use_pr_eos()

p = 35.  # bar
temp = 273.15 + 220.  # K
t_flash = 273.15 + 60  # K
t0_ref = 298.15  # K
r = 8.314  # J/(mol K)

namen = ['CO', 'H2', 'CO2', 'H2O', 'CH4', 'NH3', 'AR', 'O2', 'N2', 'CH3OH']
elemente = ['C', 'O', 'N', 'H', 'AR']

atom_m = np.array([
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 2, 1, 0, 0, 0, 2, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 2, 0],
    [0, 2, 0, 2, 4, 3, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
], dtype=float)

# Molare Masse der Elemente
mm_el = np.array([
    12.011,
    15.999,
    14.007,
    1.008,
    39.948,
]) / 1000.  # kg/gmol

# Molare Masse der Komponente
mm_k = atom_m.T.dot(mm_el)

red_atom_m = rref(ref(atom_m)[0])

rho = int(np.linalg.matrix_rank(red_atom_m))
n_c = len(namen)
n_e = len(elemente)
n_r = n_c - rho

# Thermochemische Daten

# Barin, Ihsan: Thermochemical Data of Pure Substances.
# Weinheim, New York: VCH, 1993.

h_298 = np.array(
    [-110.541, 0., -393.505,
     -241.826, -74.873, -45.940,
     0., 0., 0.,
     -201.167]) * 1000  # J/mol

g_298 = np.array(
    [-169.474, -38.962, -457.240,
     -298.164, -130.393, -103.417,
     -46.167, -61.165, -57.128,
     -272.667
     ]) * 1000  # J/mol

# Kritische Parameter Tc, Pc, omega(azentrischer Faktor)

# e.V., VDI: VDI-Wärmeatlas. Wiesbaden: Springer Berlin Heidelberg, 2013.

tc = np.array([
    132.86, 33.19, 304.13,
    647.10, 190.56, 405.50,
    150.69, 154.60, 126.19,
    513.38,
])  # K

pc = np.array([
    34.98, 13.15, 73.77,
    220.64, 45.99, 113.59,
    48.63, 50.46, 33.96,
    82.16
])  # bar

omega_af = np.array([
    0.050, -0.219, 0.224,
    0.344, 0.011, 0.256,
    -0.002, 0.022, 0.037,
    0.563
])
# umformen (reshape), um direkte Division zu ermöglichen
mm = np.array([
    28.01, 2.02, 44.01,
    18.02, 16.04, 17.03,
    39.95, 32.00, 28.01,
    32.04
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
846,6321  5,7309  –4,8842  –12,8501  78,9997  –127,3725  82,7107
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
            progress_k, stop, outer_it_k, outer_it_j, \
                lambda_ls, accum_step, x, \
                diff, f_val, lambda_ls_y = \
                nr_ls(x0=x0,
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

            soln_xi = x

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
    progress_k, stop, outer_it_k, outer_it_j, \
        lambda_ls, accum_step, x, \
        diff, f_val, lambda_ls_y = \
        nr_ls(x0=temp,
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
    temp = x
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
            progress_k, stop, outer_it_k, outer_it_j, \
                lambda_ls, accum_step, x, \
                diff, f_val, lambda_ls_y = \
                nr_ls(x0=x0,
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

            soln_xi = x[:-1]

            temp = x[-1]

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
