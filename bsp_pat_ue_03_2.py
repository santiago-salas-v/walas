import numpy as np
from scipy import integrate
from scipy import optimize
import z_l_v

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

te_dampf = 500+273.15 # °K
te_rohgas = 20+273.15 # °K
te_luft = 20+273.15 # °K

nuij = np.array([
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


ne = ne_dampf + ne_rohgas + ne_luft # kmol/h
h_dampf_ein = h(te_dampf)
h_rohgas_ein = h(te_rohgas)
h_luft_ein = h(te_luft)
# Adiabatische Vermischung sum(n_i h_i(T)-n_i h_i(T0))=0
t_ein = optimize.root(lambda temp:
    sum(
        ne_dampf*1000*(h(temp)-h_dampf_ein) + 
        ne_rohgas*1000*(h(temp)-h_rohgas_ein) + 
        ne_luft*1000*(h(temp)-h_luft_ein)
        ), 
    (te_luft+te_dampf+te_rohgas)/3
    ).x

h_ein = h(t_ein)  # J/mol

h_t_ein = h(t_ein)
g_t_ein = g(t_ein, h_t_ein)  # J/mol
k_t_ein = k(493.15, g_t_ein)  # []

print(t_ein)