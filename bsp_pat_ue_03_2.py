import numpy as np
from scipy import integrate
from scipy import optimize
import z_l_v
import logging
from numerik import nr_ls

eps = np.finfo(float).eps

# Modell feststellen
z_l_v.use_pr_eos()

p = 50.  # bar
temp = 273.15 + 220.  # K
t_flash = 273.15 + 60  # K
t0_ref = 298.15  # K
r = 8.314  # J/(mol K)

namen = ['CO', 'H2', 'CO2', 'H2O', 'CH4', 'NH3', 'AR', 'O2', 'N2']

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

nuij = np.array([
    [+1, +2, +0, +0, -1, +0, +0, -1 / 2, +0],
    [+1, +3, +0, -1, -1, +0, +0, +0, +0],
    [-1, +1, +1, -1, +0, +0, +0, +0, +0],
    [-1, -3, +0, +1, +1, +0, +0, +0, +0],
    [+0, -4, -1, +2, +1, +0, +0, +0, +0],
    [+0, -3 / 2, 0, 0, +0, +1, +0, +0, -1 / 2]
]).T

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


def k(t, g_t):
    delta_g_t = nuij.T.dot(g_t)
    return np.exp(-delta_g_t / (r * t))


n_ein = (ne_dampf + ne_rohgas + ne_luft) * 1000  # mol/h
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
k_t_ein = k(t_ein, g_t_ein)  # []

t_aus_rdampfr =  995 + 273.15 # °K
h_0 = h_t_ein
cp_0 = cp_t_ein
g_0 = g_t_ein
k_0 = k_t_ein

h_1 = h(t_aus_rdampfr)
cp_1 = r * cp_durch_r(t_aus_rdampfr)
g_1 = g(t_aus_rdampfr, h_1)
k_1 = k(t_aus_rdampfr, g_1)
print(k_1)
print(nuij.T.dot(h_1) )


def r_dampf_reformierung(x_vec):
    n_aus = x_vec[:len(n_ein)]
    xi_aus = x_vec[len(n_ein):]


    n_t = np.sum(n_aus)
    delta_h_1 = nuij.T.dot(h_1)  # J/mol

    f = np.empty_like(x_vec)

    # Stoffbilanzen
    f[:len(n_ein)] = -n_aus + n_ein + np.matrix(nuij.dot(xi_aus))

    # Gleichgewicht-Verhältnisse
    # 0 = - Kj * n_T^sum_i(nuij) *  prod_i(ni^(nuij*deltaij)) + 
    #                               prod_i(ni^(nuij*(1-deltaij)))
    # deltaij = 1 , wenn nuij < 0 ; 0 sonst
    pir = np.ones_like(k_1)
    pip = np.ones_like(k_1)

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
    k_pir = n_t**np.sum(+ nuij) * np.multiply(k_1, pir)
    # Abstand zum Gleichgewicht
    #for i in range(len(f[len(n_ein):])):
    #    if k_pir[i] < pip[i]:
    #        f[len(n_ein)+i] = k_pir[i]/pip[i] - 1
    #    elif k_pir[i] > pip[i]:
    #        f[len(n_ein)+i] = 1 - pip[i]/k_pir[i]
    #    elif k_pir[i] == pip[i]:
    #        f[len(n_ein)+i] = 0
    f_val = n_t**np.sum(+ nuij) * k_1 - np.array(
        [
            np.prod(np.power(n_aus[j], nuij[:,j])) for j in range(nuij.shape[1])
        ])
    if type(f) is np.ndarray:
        f[len(n_ein):] = f_val
    elif type(f) is np.matrixlib.defmatrix.matrix:
        f[len(n_ein):] = np.matrix(f_val).T
    #print(f[len(n_ein):])
    print(f)


    # Energiebilanz
    #q = np.sum(
    #    np.multiply(n_ein, (h_0 - h_298)) -
    #    np.multiply(n_aus, (h_1 - h_298))
    #) + np.dot(xi_aus, -delta_h_1)

    return f

def jac_r_dampf_reformierung(x_vec):
    n_aus = x_vec[:len(n_ein)]
    xi_aus = x_vec[len(n_ein):]

    n_t = np.sum(n_aus)
    prod_n_nuij = np.array([
        np.prod(np.power(n_aus[j],nuij[:,j])) for j in range(nuij.shape[1])
        ])
    sum_nuij = np.sum(nuij)

    len_n_aus = len(n_aus)
    len_xi_aus = len(xi_aus)
    
    jac = np.zeros([len(x_vec), len(x_vec)])
        
    for i in range(len_n_aus):
        jac[i, i] = -1.
        for j in range(len_xi_aus):
            jac[i, len_n_aus + j] = nuij[i, j]
            jac[len_n_aus + j, i] = \
                sum_nuij * n_t**(sum_nuij - 1) * k_1[j] - \
                nuij[i, j] / n_aus[i] * prod_n_nuij[j]

    return jac


def notify_status_func(progress_k, stop_value, k,
                       j_it_backtrack, lambda_ls, accum_step,
                       x, diff, f_val, j_val, lambda_ls_y,
                       method_loops):
    g_min = np.nan
    g1 = np.nan
    y = lambda_ls_y
    pr_str =';k=' + str(k) + \
            ';backtrack=' + str(j_it_backtrack) + \
            ';lambda_ls=' + str(lambda_ls) + \
            ';accum_step=' + str(accum_step) + \
            ';stop=' + str(stop_value) + \
            ';X=' + '[' + ','.join(map(str, x.T)) + ']' + \
            ';||X(k)-X(k-1)||=' + str((diff.T * diff).item()) + \
            ';f(X)=' + '[' + ','.join(map(str, f_val.T.A1)) + ']' + \
            ';||f(X)||=' + str(np.sqrt((f_val.T * f_val).item())) + \
            ';j(X)=' + str(j_val.tolist()) + \
            ';Y=' + '[' + ','.join(map(str, y.T.A1)) + ']' + \
            ';||Y||=' + str(np.sqrt((y.T * y).item())) + \
            ';g=' + str(g_min) + \
            ';|g-g1|=' + str(abs(g_min - g1))
    logging.debug(pr_str)

print(n_ein)
# naus_0 = np.array([
#     13535.13, 57854.92, 5909.098,
#     40496.69, 555.78, 57.88693,
#     150, 0, 11671.06
# ]) * 1000
naus_0 = n_ein
print(-naus_0[:5] + n_ein[:5])
print(nuij)
xi_0 = np.array([
    2*0.21 * 15000, 19444.22, 0, 0, 0, 0
]) * 1000 * 0
print(xi_0)
x0 = np.concatenate([
    naus_0,
    xi_0
])
x0[x0 == 0] = eps
# print(x0)
# print(x0[:len(n_ein)])
# print(nuij.dot(np.zeros(nuij.shape[1])))
soln = optimize.root(r_dampf_reformierung, x0)
print(soln)

n_ein = np.matrix(n_ein).T
progress_k, stop, outer_it_k, outer_it_j, \
        lambda_ls, accum_step, x, \
        diff, f_val, lambda_ls_y, \
        method_loops = \
        nr_ls(x0=np.matrix(x0).T,
              f=lambda x: np.matrix(r_dampf_reformierung(x)),
              j=lambda x: np.matrix(jac_r_dampf_reformierung(x)),
              tol=1e-12,
              max_it=1000,
              inner_loop_condition=lambda x_vec:
              all([item >= 0 for item in
                   x_vec[0:len(n_ein)]]),
              notify_status_func=notify_status_func,
              method_loops=[0, 0],
              process_func_handle=None)

