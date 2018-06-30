import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib.lines import Line2D
from scipy.integrate import odeint
import z_l_v
import locale

# Lösung der ChP-Übung. Reaktor angepasst.
namen = ['CO', 'CO2', 'H2', 'H2O', 'MeOH',
         'CH4', 'N2', 'EthOH', 'PrOH', 'METHF']
# Katalysator
rho_b = 1190  # kg Kat/m^3 Feststoff
phi = 0.3  # m^3 Gas/m^3 Feststoff
m_kat = 1190 * (1 - 0.3) * np.pi / 4 * (
    0.03)**2 * 7  # kg Kat (pro Rohr)
# Partikeldurchmesser wählen, damit
# Delta P gesamt=-3bar, denn dies ist der Parameter
d_p = 0.0054 / 0.0054 * 0.16232576224693065  # m Feststoff
# Reaktor
n_t = 1620  # Rohre
d_t = 0.04  # m Rohrdurchmesser
sn_param = 3.44  #2.05  # dimensionslos
ntu_param = 1.82  # Parameter
verhaeltnis_co_co2 = 0.74   # Parameter
l_r = 7.  # m Rohrlänge
# Betriebsbedingungen
t0 = 220 + 273.15  # K
p0 = 50  # bar
# Wärmetauschparameter
u = 118.44  # W/m^2/K
# Kühlmitteleigenschaften (VDI-WA)
t_r = (240 - 230) / (33.467 - 27.968) * (
    29 - 33.467) + 240 + 273.15  # K
p_sat = 29  # bar
h_sat_l = (1037.5 - 990.21) / (33.467 - 27.968) * (
    29 - 33.467) + 1037.5  # kJ/kg
h_sat_v = (2803.1 - 2803.0) / (33.467 - 27.968) * (
    29 - 33.467) + 2803.1  # kJ/kg
delta_h_sat = (h_sat_v - h_sat_l)
# Zulaufbedingungen
n_i_0 = np.array([
    1171.15, 1178.85, 3058.84,
    100.65, 0, 0,
    0, 0, 0,
    0
]) / 60**2 * 1000  # mol/s


z_l_v.use_pr_eos()
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


def delta_h_r(t):
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
            1 + np.outer(mu_i, 1 / mu_i) ** (1 / 2.) * np.outer(1 / mm, mm) ** (1 / 4.)
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


def df_dt(y, _):
    # FIXME: Normalize y_i on each iteration.
    # Inerts should not change at all, but there is a 0.00056% increase already
    # on the first time step. Test:
    # u_s*60**2*n_t*(np.pi / 4 * d_t**2)/sum(y_i*mm/1000.)*y_i[6]*mm[6]/1000.
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
    delta_h_r_t = delta_h_r(t)
    mu_t_y = mu(t, y_i)
    r_j = r_i(t, p_i)  # mol/kg Kat/s
    dyi_dz = l_r * rho_b * mm_m / u_s * (
        nuij.dot(r_j) - y_i * sum(nuij.dot(r_j))
    )   # m * kgKat/m^3 * kg/mol * m^2 s/kg * mol/kgKat/s = dimlos
    dp_dz = -l_r * 1 / 10**5 * u_s / (
        c_t * mm_m * d_p
    ) * (1 - phi) / phi**3 * (
        150 * (1 - phi) * mu_t_y / d_p + 1.75 * u_s
    )  # Pa * 1bar/(1e5 Pa) = bar
    dt_dz = l_r / (
        u_s * cp_g
    ) * (
        2 * u / (d_t / 2) * (t_r - t) + rho_b * -delta_h_r_t.dot(r_j)
    )
    result = np.empty_like(y)
    result[:-2] = dyi_dz
    result[-2] = dp_dz
    result[-1] = dt_dz
    return result

# Init.
n_i_1 = n_i_0  # mol/s
p1 = p0
t1 = t0

z_d_l_r = np.linspace(0, 1, 100)
dlr = 1 / (len(z_d_l_r) - 1) * l_r  # m
y_0 = np.empty([len(namen) + 1 + 1])

for i in range(26):
    n_i_1_n_m_1 = n_i_1
    m_dot_i = n_i_1 * mm / 1000.  # kg/s
    m_dot = sum(m_dot_i)
    y_i1 = m_dot_i / mm / sum(m_dot_i / mm)
    # Berechnung der Parameter
    mm_m_1 = sum(y_i1 * mm) * 1 / 1000.  # kg/mol
    cp_m_1 = sum(y_i1 * cp_ig_durch_r(t1) * 8.3145)  # J/mol/K
    cp_g_1 = cp_m_1 / mm_m_1  # J/kg/K
    n_t = ntu_param / (2 * np.pi * d_t / 2 * l_r * u) * (
            m_dot * cp_g_1
    )
    u_s = m_dot / (np.pi / 4 * d_t ** 2) / n_t  # kg/m^2/s
    ntu = l_r * 1 / (u_s * cp_g_1) * 2 * u / (d_t / 2)
    # m * m^2 s/kg * kg K /J * J/s/m^2/K *1/m = [dimensionslose Einheiten]

    y_0[:-2] = y_i1
    y_0[-2] = p1
    y_0[-1] = t1

    # Partikeldurchmesser nach Parameter Delta_P=3bar optimisieren.
    # Es wird direkt Fixpunkt-Iteration angewendet, nach der Form der Ergun Gl.
    # D_p{n+1} = D_p{n} * int(1/D_p{n}*f(D_p{n}) dz) / -3bar
    for j in range(5):
        soln_dp = odeint(df_dt, y_0, z_d_l_r)
        dp_dz = (soln_dp[-1, -2] - soln_dp[0, -2]).item()
        d_p = dp_dz * d_p / -3.0

    soln = odeint(df_dt, y_0, z_d_l_r)
    y_i_soln = soln[:, :len(y_i1)]
    p_soln = soln[:, -2]
    t_soln = soln[:, -1]

    mm_m_soln = np.sum(y_i_soln * mm * 1 / 1000., axis=1)  # kg/mol
    n_soln = u_s * n_t * (np.pi / 4 * d_t ** 2) / mm_m_soln
    # kg/s/m^2 * m^2 / kg*mol = mol/s
    m_soln = u_s * n_t * (np.pi / 4 * d_t ** 2)  # kg/s
    n_i_soln = (y_i_soln.T * n_soln).T  # mol/s
    m_i_soln = n_i_soln * (mm * 1 / 1000.) * 60**2  # kg/h
    # mol/h * g/mol * 1kg/1000g = 1/1000 kg/h
    v_soln = n_soln * 8.3145 * 1e-5 * t_soln / p_soln
    # mol/s * 8,3145Pa m^3/mol/K * 1e-5bar/Pa * K/bar = m^3/s

    m_km_soln = np.zeros_like(z_d_l_r)
    ums_soln = (n_i_soln[0, namen.index('CO')] -
                n_i_soln[:, namen.index('CO')]
                ) / n_i_soln[0, namen.index('CO')]
    m_km_soln = u * (2 / (d_t / 2)) * (t_soln - t_r) * (
            np.pi / 4 * d_t ** 2) / delta_h_sat * n_t * 60 ** 2 / 1000.
    # J/s/K/m^2 * 1/m * K * m^2 * kg/kJ * 60^2s/h * 1kJ/(1000J) = kg/h/m
    n_i_2 = n_i_soln[-1]  # mol/s

    n_0 = sum(n_i_0)  # mol/h
    n_2 = sum(n_i_2)  # mol/h
    t_2 = 60 + 273.15  # K
    p_2 = p_soln[-1]  # bar
    z_i = y_i_soln[-1]  # dimensionslos
    verfluessigung = z_l_v.isot_flash_solve(
        t_2, p_2, z_i, tc, pc, omega_af)
    v_f = verfluessigung['v_f']
    x_i = verfluessigung['x_i']
    y_i = verfluessigung['y_i']

    y_co2_r = y_i[namen.index('CO2')]
    y_co_r = y_i[namen.index('CO')]
    y_h2_r = y_i[namen.index('H2')]

    y_h2_0 = n_i_0[namen.index('H2')] / n_0
    y_co_0 = n_i_0[namen.index('CO')] / n_0
    y_co2_0 = n_i_0[namen.index('CO2')] / n_0

    n_i_1 = n_i_0 + v_f * n_2 * y_i
    n_1 = sum(n_i_1)
    y_i_1 = n_i_1 / n_1

    y_co_1 = y_i_1[namen.index('CO')]
    y_co2_1 = y_i_1[namen.index('CO2')]

    n_co_zus = n_1 * (
            verhaeltnis_co_co2 * y_co2_1 - y_co_1
    )

    n_h2_zus = v_f * n_2 * (
            sn_param * (y_co2_r + y_co_r) +
            y_co2_r - y_h2_r
    ) + n_0 * (
                       sn_param * (y_co2_0 + y_co_0) +
                       y_co2_0 - y_h2_0
               )

    n_i_1[namen.index('H2')] = \
        n_i_1[namen.index('H2')] + n_h2_zus
    n_i_1[namen.index('CO')] = \
        n_i_1[namen.index('CO')] + n_co_zus

    n_i_r = v_f * sum(n_i_2) * y_i
    mm_0 = sum(n_i_0 / sum(n_i_0) * mm) / 1000.  # kg/mol
    mm_1 = sum(n_i_1 / sum(n_i_1) * mm) / 1000.  # kg/mol
    mm_2 = sum(n_i_2 / sum(n_i_2) * mm) / 1000.  # kg/mol
    mm_r = sum(n_i_r / sum(n_i_r) * mm) / 1000.  # kg/mol
    bilanz = sum(n_i_1 * mm_1) - sum(n_i_r * mm_r) \
          - sum(n_i_0 * mm_0) \
          - n_h2_zus * mm[namen.index('H2')]/1000. \
          - n_co_zus * mm[namen.index('CO')]/1000.
    aend = sum(n_i_1 - n_i_1_n_m_1) / sum(n_i_1) * 100
    umsatz = 1 - n_i_2[namen.index('CO')] - n_i_1[namen.index('CO')]
    print('It.  ' + '{:2d}'.format(i) + ', Bilanz: ' +
          '{:g}'.format(np.sqrt(bilanz.dot(bilanz))) + '\t' +
          ' Änderung: '+
          '{:5.4g}'.format(np.sqrt(aend**2)) + '%\t')

v_soln_real = np.empty_like(v_soln)
for i in range(len(z_d_l_r)):
    z_realgas_f = z_l_v.z_non_sat(
        t_soln[i], p_soln[i], y_i_soln[i],
        tc, pc, omega_af)['z']
    v_soln_real[i] = v_soln[i] * z_realgas_f

# Tatsächliche stöchiometrische Zahl
sn = (y_i1[namen.index('H2')] - y_i1[namen.index('CO2')]) / (
        y_i1[namen.index('CO2')] + y_i1[namen.index('CO')]
)
verhaeltnis_h2_co2 = y_i1[namen.index('H2')] / y_i1[namen.index('CO2')]
verhaeltnis_co_co2 = y_i1[namen.index('CO')] / y_i1[namen.index('CO2')]

# Energie-Analyse
t_m_1 = 1 / 2 * (36696 / 8.3145 - np.sqrt(36696 /
                                          8.3145 * (36696 / 8.3145 - 4 * t_r)))
t_m_2 = 1 / 2 * (-94765 / 8.3145 - np.sqrt(-94765 /
                                           8.3145 * (-94765 / 8.3145 - 4 * t_r)))

vars_1 = [
    [r'\rho_b', rho_b, r'\frac{kg_{Kat}}{m^3_{Fest}}'],
    ['\phi', phi, r'\frac{m^3_{Gas}}{m^3_{Fest}}'],
    ['D_p', d_p, 'm_{Fest}'],
    ['D_t', d_t, 'm'],
    ['L_R', l_r, 'm'],
]
vars_2 = [
    ['n_T', n_t, ''],
    ['U', u, r'\frac{W}{m^2\cdot K}'],
    ['\dot m', m_dot, 'kg/s'],
    ['C_{p_g}', cp_g_1 / 1000., r'\frac{kJ}{kg\cdot K}'],
    ['NTU', ntu, ''],
]
vars_3 = [
    ['T_0', t0 - 273.15, '°C'],
    ['P_0', p0, 'bar'],
    ['T_r', t_r - 273.15, '°C_{Kühlmittel}'],
    ['P_{Sät}', p_sat, 'bar_{Kühlmittel}'],
    ['\Delta H_{Sät}', delta_h_sat, r'\frac{kJ}{kg_{Kühlmittel}}'],
    ['SN', sn, '']
]
text_1 = '\n'.join(['$' + ' = '.join([line[0], '{:g}'.format(line[1]) +
                                      ' ' + line[2]]) + '$'
                    for line in vars_1])
text_2 = '\n'.join(['$' + ' = '.join([line[0], '{:g}'.format(line[1]) +
                                      ' ' + line[2]]) + '$'
                    for line in vars_2])
text_3 = '\n'.join(['$' + ' = '.join([line[0], '{:g}'.format(line[1]) +
                                      ' ' + line[2]]) + '$'
                    for line in vars_3])


fig = plt.figure(1)
fig.suptitle('Lösung der Zusammensetzung ' +
             '{:g}'.format(round(verhaeltnis_h2_co2.item(), 2)) +
             ':1:' +
             '{:g}'.format(round(verhaeltnis_co_co2.item(), 2)) +
             '(H2:CO2:CO)')
fig.text(0.05, 0.945, text_1, va='top', fontsize=8)
fig.text(0.33, 0.935, text_2, va='top', fontsize=8)
fig.text(0.66, 0.930, text_3, va='top', fontsize=8)
ax = plt.subplot2grid([2, 3], [0, 0])
ax.plot(z_d_l_r, v_soln, label='$Idealgas$')
ax.plot(z_d_l_r, v_soln_real, label='$Realgas$')
ax.set_ylabel(r'$\frac{\dot V}{m^3/h}$')
ax.set_xlabel('Reduzierte Position, $z/L_R$')
ax.legend(fontsize='xx-small')
ax2 = plt.subplot2grid([2, 3], [1, 0])
ax2.plot(z_d_l_r, m_km_soln)
ax2.fill(z_d_l_r, m_km_soln, color='orange')
ax2.fill([0, 1, 1, 0],
         [m_km_soln[0], m_km_soln[-1],
          m_km_soln[0], m_km_soln[0]], color='orange')
ax2.text(0.3, 1 / 2. * (m_km_soln[0] + m_km_soln[-1]),
         '{:g}'.format(sum(m_km_soln * dlr)) + 'kg/h')
ax2.set_ylabel(r'$\frac{\dot m_{Kuehlmittel}}{kg/h}$')
ax2.set_xlabel('Reduzierte Position, $z/L_R$')

ax3 = plt.subplot2grid([2, 3], [1, 1], colspan=2)
ax3.set_ylabel('Massenstrom / (kg/h)')
ax3.set_xlabel('Reduzierte Position, $z/L_R$')
for item in ['CO', 'H2O', 'MeOH', 'CO2']:
    marker = np.random.choice(list(lines.lineMarkers.keys()))
    index = namen.index(item)
    ax3.plot(z_d_l_r, m_i_soln[:, index], label=item,
             marker=marker)
ax3.legend(loc=1)
ax4 = plt.subplot2grid([2, 3], [0, 1])
ax4_1 = ax4.twinx()
ax4_1.set_ylabel('Umsatz (CO)')
ax4.set_ylabel('Temperatur / °C')
ax4.set_xlabel('Reduzierte Position, $z/L_R$')
ax4.plot(z_d_l_r, t_soln - 273.15, label='T / °C')
ax4_1.plot(z_d_l_r, ums_soln, label='Umsatz (CO)', ls='--', color='gray')
ax5 = plt.subplot2grid([2, 3], [0, 2], colspan=2)
ax5.set_ylabel('Druck / bar')
ax5.set_xlabel('Reduzierte Position, $z/L_R$')
ax5.plot(z_d_l_r, p_soln, label='p / bar')
plt.tight_layout(rect=[0, 0, 0.95, 0.75])

locale.setlocale(locale.LC_ALL, '')

print('')
print('=== GAS AUS REAKTOR ===')
print('SN= ' + str(sn))
print('NTU= ' + str(ntu))
print('y(CO)/y(CO2)= ' + str(verhaeltnis_co_co2))
print('\n'.join([
    namen[i] + ': ' + locale.format('%.8g', x) + ' kg/h'
    for i, x in enumerate(m_i_soln[-1])
]))
print('T= ' + str(t_soln[-1] - 273.15) + '°C')
print('P= ' + str(p_soln[-1]) + 'bar')
print('V0= ' + str(v_soln[0]) + 'm^3/h')
print('V= ' + str(v_soln[-1]) + 'm^3/h')
print('Cpg= ' + str(cp_g_1) + 'J/kg/K')
print('Partikeldurchmesser für (DeltaP= ' +
      '{:g}'.format(p_soln[0] - p_soln[-1]) + ' bar): ' +
      '{:g}'.format(d_p) + ' m'
      )
print('Kühlmittel: Gesättigtes H_2O(l) ' +
      ' bei ' + '{:g}'.format(p_sat) + ' bar' + '\n' +
      'Verdampfungsenthalpie: ' + '{:g}'.format(delta_h_sat) +
      'kJ/kg' + '\n' + 'Kühlmittelmassenstrom: ' +
      '{:g}'.format(sum(m_km_soln * dlr)) + 'kg/h')

print('')
print('======'*3)
print('=== MAKEUP-STROM ===')
print('\n'.join([
    namen[i] + ': ' + locale.format('%.8g', x) + ' kg/h'
    for i, x in enumerate(n_i_0 * mm / 1000. * 60**2)
]))
print('')
print('======'*3)
print('=== RÜCKLAUFSTROM ===')
print('\n'.join([
    namen[i] + ': ' + locale.format('%.8g', x) + ' kg/h'
    for i, x in enumerate(n_i_r * mm / 1000. * 60**2)
]))
print('')
print('======'*3)
print('=== ERFORDERLICHE CO UND H2 STRÖME, UM SN UND CO/CO2 ANZUPASSEN ===')
print('H2: ' +
      locale.format('%.8g', n_h2_zus.item() * mm[namen.index('H2')] /
          1000. * 60 ** 2) + ' kg/h')
print('CO: ' +
      locale.format('%.8g', n_co_zus.item() * mm[namen.index('CO')] /
          1000. * 60**2) + ' kg/h')
print('auf kmol/h')
print('H2: ' +
      locale.format('%.8g', n_h2_zus.item() / 1000. * 60 ** 2) + ' kmol/h')
print('CO: ' +
      locale.format('%.8g', n_co_zus.item() / 1000. * 60 ** 2) + ' kmol/h')
print('')
print('======'*3)
print('=== REAKTOR ZULAUFSTROM ===')
print('\n'.join([
    namen[i] + ': ' + locale.format('%.8g', x) + ' kg/h'
    for i, x in enumerate(n_i_1 * mm / 1000. * 60**2)
]))

plt.show()
