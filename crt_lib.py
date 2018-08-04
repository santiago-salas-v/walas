import numpy as np
from scipy.integrate import odeint
import locale
import z_l_v

# Theorie und Aufstellung der Gleichungen:
# https://git.io/fdKBI

namen = ['CO', 'CO2', 'H2', 'H2O', 'MeOH',
         'CH4', 'N2', 'EthOH', 'PrOH', 'METHF']
mm = np.array([
    28.01, 44.01, 2.016,
    18.015, 32.042, 16.043,
    28.014, 46.069, 60.096,
    60.053
], dtype=float)  # g/mol
# Parameter der Wärmekapazität-Gleichung aus dem VDI-Wärmeatlas
cp_konstanten = np.zeros([len(namen), 7])
cp_konstanten_text = [
    '407,9796 3,5028 2,8524 –2,3018 32,9055 –100,1815 106,1141',
    '514,5073 3,4923 –0,9306 –6,0861 54,1586 –97,5157 70,9687',
    '392,8422 2,4906 –3,6262 –1,9624 35,6197 –81,3691 62,6668',
    '706,3032 5,1703 –6,0865 –6,6011 36,2723 –63,0965 46,2085',
    '846,6321 5,7309 –4,8842 –12,8501 78,9997 –127,3725 82,7107',
    '1530,8043 4,2038 –16,6150 –3,5668 43,0563 –86,5507 65,5986',
    '432,2027 3,5160 2,8021 –4,1924 42,0153 –114,2500 111,1019',
    '1165,8648 4,7021 9,7786 –1,1769 –135,7676 322,7596 –247,7349',
    '506,0032 12,1539 0,0041 –36,1389 175,9466 –276,1927 171,3886',
    '650,0705 5,0360 –0,4487 –13,5453 126,9502 –219,5695 151,4586'
]
for i in range(len(namen)):
    cp_konstanten[i, :] = np.array(
        cp_konstanten_text[i].replace(
            '–', '-').replace(',', '.').split(' '),
        dtype=float)

# Quelle: The properties of Gases and Liquids Poling Prausnitz
# Lennard-Jones Parameter Anhang B
# epsilon / kB
l_j_epsilon_d_k = np.array([
    91.7, 195.2, 59.7,
    809.1, 481.8, 148.6,
    71.4, 362.6, 576.7,
    469.8
])  # K
# sigma
l_j_sigma = np.array([
    3.69, 3.941, 2.827,
    2.641, 3.626, 3.758,
    3.798, 4.53, 4.549,
    4.936
])  # Angstrom
# T* = k T / epsilon
h_298 = np.array([
    -110.53, -393.51, 0,
    -241.81, -200.94, -74.52,
    0, -234.95, -255.20,
    -352.40
], dtype=float) * 1000.  # J/mol
tc = np.array([
    132.85, 304.12, 32.98,
    647.14, 512.64, 190.56,
    126.2, 513.92, 536.78,
    487.2
])  # K
pc = np.array([
    34.94, 73.74, 12.93,
    220.64, 80.97, 45.99,
    33.98, 61.48, 51.75,
    60.0
])  # bar
omega_af = np.array([
    0.045, 0.225, -0.217,
    0.344, 0.565, 0.011,
    0.037, 0.649, 0.629,
    0.0
])

nuij = np.zeros([len(namen), 3])
# Hydrierung von CO2
nuij[[
    namen.index('CO2'),
    namen.index('H2'),
    namen.index('MeOH'),
    namen.index('H2O'),
    namen.index('CO'),
    namen.index('N2'),
], 0] = np.array([-1, -3, +1, +1, 0, 0], dtype=float)
# Hydrierung von CO
nuij[[
    namen.index('CO2'),
    namen.index('H2'),
    namen.index('MeOH'),
    namen.index('H2O'),
    namen.index('CO'),
    namen.index('N2'),
], 1] = np.array([0, -2, +1, 0, -1, 0], dtype=float)
# RWGS Reverse-Wassergasshiftreaktion (muss gleich sein als die Vorwärtsreaktion,
# wofür die Kinetik verfügbar ist)
nuij[[
    namen.index('CO2'),
    namen.index('H2'),
    namen.index('MeOH'),
    namen.index('H2O'),
    namen.index('CO'),
    namen.index('N2'),
], 2] = - np.array([+1, +1, 0, -1, -1, 0], dtype=float)

delta_h_r_298 = nuij.T.dot(h_298)  # J/mol


def cp_ig_durch_r(t):
    a, b, c, d, e, f, g = cp_konstanten.T
    gamma_var = t / (a + t)
    return b + (c - b) * gamma_var ** 2 * (
        1 + (gamma_var - 1) * (
            d + e * gamma_var + f * gamma_var ** 2 + g * gamma_var ** 3
        ))  # dimensionslos


def int_cp_durch_r_dt_minus_const(t):
    a, b, c, d, e, f, g = cp_konstanten.T
    return b * t + (c - b) * (
        t - (d + e + f + g + 2) * a * np.log(a + t) +
        -(2 * d + 3 * e + 4 * f + 5 * g + 1) * a**2 / (a + t) +
        +(1 * d + 3 * e + 6 * f + 10 * g) * a**3 / 2 / (a + t)**2 +
        -(1 * e + 4 * f + 10 * g) * a**4 / 3 / (a + t)**3 +
        +(1 * f + 5 * g) * a**5 / 4 / (a + t)**4 +
        - g * a**6 / 5 / (a + t)**5
    )


def int_cp_durch_rt_dt_minus_const(t):
    a, b, c, d, e, f, g = cp_konstanten.T
    return b * np.log(t) + (c - b) * (
        np.log(a + t) + (1 + d + e + f + g) * a / (a + t) +
        -(d / 2 + e + 3 * f / 2 + 2 * g) * a**2 / (a + t)**2 +
        +(e / 3 + f + 2 * g) * a**3 / (a + t)**3 +
        -(f / 4 + g) * a**4 / (a + t)**4 +
        +(g / 5) * a**5 / (a + t)**5
    )


def mcph(x, t0_ref, t):
    return sum(x * (
        int_cp_durch_r_dt_minus_const(t) -
        int_cp_durch_r_dt_minus_const(t0_ref))
    ) / sum(x) / (t - t0_ref)


def delta_h_r(t, nuij, delta_h_r_298):
    cp_m = (
        int_cp_durch_r_dt_minus_const(t) -
        int_cp_durch_r_dt_minus_const(298.15)
    ) / (t - 298.15) * 8.3145  # J/mol/K * K = J/mol
    return delta_h_r_298 + nuij.T.dot(cp_m) * (t - 298.15)


def mu(t, y_i):
    # T* = k T / epsilon
    t_st = t / l_j_epsilon_d_k  # J/K
    # Stoßintegral (Bird Tabelle E.2)
    omega_mu = 1.16145 / t_st**0.14874 + \
        0.52487 / np.exp(0.77320 * t_st) + \
        2.16178 / np.exp(2.43787 * t_st)
    konst_1 = 5 / 16. * np.sqrt(
        8.3145 * 1000 * 100 ** 2 / np.pi
    ) * 10**16 / 6.022e23  # g/cm/s
    mu_i = konst_1 * np.sqrt(mm * t) / (
        l_j_sigma**2 * omega_mu
    ) * 100 / 1000
    # g/cm/s * 100cm/m * 1kg/1000g = kg/m/s = Pa s
    phi_ab = 1 / np.sqrt(8) * (
        1 + np.outer(mm, 1 / mm)) ** (-1 / 2.) * (
        1 + np.outer(mu_i, 1 / mu_i) ** (1 / 2.) *
        np.outer(1 / mm, mm) ** (1 / 4.)
    ) ** 2
    mu_mix = np.sum(y_i * mu_i / phi_ab.dot(y_i))
    return mu_mix  # Pa s


# T-abhängige Parameter, dem Artikel nach
def k_t(t):
    r = 8.3145  # Pa m^3/mol/K
    # Geschwindigkeitskonstanten der 3 Reaktionen
    k_1 = 10 ** (3066 / t - 10.592)
    k_2 = 10 ** (5139 / t - 12.621)
    k_3 = 10 ** (2073 / t - 2.029)
    # Angepasste Parameter des kinetischen Modells
    # A(i) exp(B(i)/RT)
    a = np.array([
        0.499, 6.62e-11, 3453.38, 1.07, 1.22e10
    ], dtype=float)
    b = np.array([
        17197, 124119, 0, 36696, -94765
    ], dtype=float)
    k_h2 = a[0] ** 2 * np.exp(2 * b[0] / (r * t))
    k_h2o = a[1] * np.exp(b[1] / (r * t))
    k_h2o_d_k_8_k_9_k_h2 = a[2] * np.exp(b[2] / (r * t))
    k5a_k_2_k_3_k_4_k_h2 = a[3] * np.exp(b[3] / (r * t))
    k1_strich = a[4] * np.exp(b[4] / (r * t))
    return np.array([
        k_1, k_2, k_3,
        k_h2, k_h2o, k_h2o_d_k_8_k_9_k_h2,
        k5a_k_2_k_3_k_4_k_h2, k1_strich
    ])


def r_i(t, p_i):
    p_co2 = p_i[namen.index('CO2')]
    p_co = p_i[namen.index('CO')]
    p_h2 = p_i[namen.index('H2')]
    p_meoh = p_i[namen.index('MeOH')]
    p_h2o = p_i[namen.index('H2O')]
    [k_1, _, k_3,
     k_h2, k_h2o, k_h2o_d_k_8_k_9_k_h2,
     k5a_k_2_k_3_k_4_k_h2, k1_strich] = k_t(t)
    r_meoh = k5a_k_2_k_3_k_4_k_h2 * p_co2 * p_h2 * (
        1 - 1 / k_1 * p_h2o * p_meoh / (
            p_h2 ** 3 * p_co2
        )
    ) / (
        1 + k_h2o_d_k_8_k_9_k_h2 * p_h2o / p_h2 +
        np.sqrt(k_h2 * p_h2) + k_h2o * p_h2o
    ) ** 3
    r_rwgs = k1_strich * p_co2 * (
        1 - k_3 * p_h2o * p_co / (p_co2 * p_h2)
    ) / (
        1 + k_h2o_d_k_8_k_9_k_h2 * p_h2o / p_h2 +
        np.sqrt(k_h2 * p_h2) + k_h2o * p_h2o
    ) ** 1
    return np.array([r_meoh, 0., r_rwgs])


def df_dt(y, _, g, d_t, l_r,
          phi, d_p, rho_b,
          u, t_r):
    # FIXME: Normalize y_i on each iteration.
    # Inerts should not change at all, but there is a 0.00056% increase already
    # on the first time step. Test:
    # g*60**2*n_t*(np.pi / 4 * d_t**2)/sum(y_i*mm/1000.)*y_i[6]*mm[6]/1000.
    y_i = y[:-2]
    p = y[-2]
    t = y[-1]
    mm_m = sum(y_i * mm) * 1 / 1000.  # kg/mol
    cp_m = sum(y_i * cp_ig_durch_r(t) * 8.3145)  # J/mol/K
    cp_g = cp_m / mm_m  # J/kg/K
    z_realgas_f = z_l_v.z_non_sat(t, p, y_i, tc, pc, omega_af)['z']
    c_t = p / (8.3145 * 1e-5 * t) * 1 / z_realgas_f
    # bar * mol/bar/m^3/K*K = mol/m^3
    p_i = y_i * p  # bar
    delta_h_r_t = delta_h_r(t, nuij, delta_h_r_298)
    mu_t_y = mu(t, y_i)
    r_j = r_i(t, p_i)  # mol/kg Kat/s
    dyi_dz = l_r * rho_b * mm_m / g * (
        nuij.dot(r_j) - y_i * sum(nuij.dot(r_j))
    )   # m * kgKat/m^3 * kg/mol * m^2 s/kg * mol/kgKat/s = dimlos
    dp_dz = -l_r * 1 / 10**5 * g / (
        c_t * mm_m * d_p
    ) * (1 - phi) / phi**3 * (
        150 * (1 - phi) * mu_t_y / d_p + 1.75 * g
    )  # Pa * 1bar/(1e5 Pa) = bar
    dt_dz = l_r / (
        g * cp_g
    ) * (
        2 * u / (d_t / 2) * (t_r - t) + rho_b * -delta_h_r_t.dot(r_j)
    )
    result = np.empty_like(y)
    result[:-2] = dyi_dz
    result[-2] = dp_dz
    result[-1] = dt_dz
    return result