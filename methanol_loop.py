import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from scipy.integrate import odeint
import z_l_v
import locale
import ctypes, os

# Theorie und Aufstellung der Gleichungen:
# https://git.io/fdKBI

sn_param = (22222.0112180475-5488.76285475286)/(
        5488.76285475286+2206.10385252181) # FIXME: 2,19 SN crash
sn_param = 2.55 # 2.05  # dimensionslos
ntu_param = 3.09  # Parameter
optimisieren_nach_param = ['n_t', 'l_r_n_t', 'keine_opt'][2]

namen = ['CO', 'CO2', 'H2', 'H2O', 'MeOH',
         'CH4', 'N2', 'EthOH', 'PrOH', 'METHF']
# Katalysator
rho_b = 1190  # kg Kat/m^3 Kat-Schüttung
phi = 0.3  # m^3 Hohl/m^3 Kat-Schüttung
m_kat = 1190 / (1 - phi) * np.pi / 4 * (
    0.04)**2  # kg Kat (pro Rohr)
# Partikeldurchmesser wählen, damit
# Delta P gesamt=-3bar, denn dies ist der Parameter
d_p = 0.0037  # m Feststoff
# Reaktor
n_t = 10921  # Rohre
d_t = 0.04  # m Rohrdurchmesser
l_r = 6.  # m Rohrlänge
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
    816.040561587593, 742.463438412464, 1428.19943841246,
    130.296561587538+76.54944195313007, 0, 0,
    0, 0, 0,
    0
]) / 60**2 * 1000  # mol/s
mm = np.array([
    28.01, 44.01, 2.016,
    18.015, 32.042, 16.043,
    28.014, 46.069, 60.096,
    60.053
], dtype=float)  # g/mol

z_l_v.use_pr_eos()
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


def df_dt(y, _, g, d_p, l_r):
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
    delta_h_r_t = delta_h_r(t)
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


def profile(n_i_1_ein, d_p_ein, optimisieren_nach='n_t'):
    m_dot_i_ein = n_i_1_ein * mm / 1000.  # kg/s
    m_dot_ein = sum(m_dot_i_ein)
    y_i1_ein = m_dot_i_ein / mm / sum(m_dot_i_ein / mm)
    # Berechnung der Parameter
    mm_m_1_ein = sum(y_i1_ein * mm) * 1 / 1000.  # kg/mol
    cp_m_1_ein = sum(y_i1_ein * cp_ig_durch_r(t1) * 8.3145)  # J/mol/K
    cp_g_1_ein = cp_m_1_ein / mm_m_1_ein  # J/kg/K
    if optimisieren_nach == 'n_t':
        # Anzahl an Rohre anpassen
        n_t_aus = ntu_param / (2 * np.pi * d_t / 2 * l_r * u) * (
            m_dot_ein * cp_g_1_ein
        )
        l_r_aus = l_r
    elif optimisieren_nach == 'l_r_n_t':
        # Anzahl an Rohre und Länge anpassen
        verhaeltnis = 1 / (l_r * n_t) * (
            ntu_param * m_dot_ein * cp_g_1_ein / (
                2 * np.pi * d_t / 2 * u)
        )
        l_r_aus = np.sqrt(verhaeltnis) * l_r
        n_t_aus = np.sqrt(verhaeltnis) * n_t
    elif optimisieren_nach == 'keine_opt':
        # Keine Optimsierung von weder Rohrenzahl noch Länge
        l_r_aus = l_r
        n_t_aus = n_t
    g_neu = m_dot_ein / (np.pi / 4 * d_t ** 2) / n_t_aus  # kg/m^2/s
    y_0 = np.empty([len(namen) + 1 + 1])
    y_0[:-2] = y_i1_ein
    y_0[-2] = p1
    y_0[-1] = t1

    soln = odeint(lambda y, z0: df_dt(y, z0, g_neu, d_p_ein, l_r_aus),
                  y_0, z_d_l_r)
    y_i_soln = soln[:, :len(y_i1_ein)]
    p_soln = soln[:, -2]
    t_soln = soln[:, -1]

    mm_m_soln = np.sum(y_i_soln * mm * 1 / 1000., axis=1)  # kg/mol
    n_soln = g_neu * n_t_aus * (np.pi / 4 * d_t ** 2) / mm_m_soln
    # kg/s/m^2 * m^2 / kg*mol = mol/s
    n_i_soln = (y_i_soln.T * n_soln).T  # mol/s

    n_i_2 = n_i_soln[-1]  # mol/s

    n_0 = sum(n_i_0)  # mol/s
    n_2 = sum(n_i_2)  # mol/s
    t_2 = 60 + 273.15  # K
    p_2 = p_soln[-1]  # bar
    z_i = y_i_soln[-1]  # dimensionslos
    verfluessigung = z_l_v.isot_flash_solve(
        t_2, p_2, z_i, tc, pc, omega_af)
    v_f_flash = verfluessigung['v_f'].item()
    x_i_flash = verfluessigung['x_i']
    y_i_flash = verfluessigung['y_i']

    y_co2_r = y_i_flash[namen.index('CO2')]
    y_co_r = y_i_flash[namen.index('CO')]
    y_h2_r = y_i_flash[namen.index('H2')]

    y_h2_0 = n_i_0[namen.index('H2')] / n_0
    y_co_0 = n_i_0[namen.index('CO')] / n_0
    y_co2_0 = n_i_0[namen.index('CO2')] / n_0

    n_i_1_aus = n_i_0 + v_f_flash * n_2 * y_i_flash
    n_1 = sum(n_i_1_aus)
    y_i_1 = n_i_1_aus / n_1

    y_co_1 = y_i_1[namen.index('CO')]
    y_co2_1 = y_i_1[namen.index('CO2')]

    n_h2_zus_aus = v_f_flash * n_2 * (
        sn_param * (y_co2_r + y_co_r) +
        y_co2_r - y_h2_r
    ) + n_0 * (
        sn_param * (y_co2_0 + y_co_0) +
        y_co2_0 - y_h2_0
    )

    n_i_1_aus[namen.index('H2')] = \
        n_i_1_aus[namen.index('H2')] + n_h2_zus_aus

    # Partikeldurchmesser nach Parameter Delta_P=3bar optimisieren.
    # Es wird direkt Fixpunkt-Iteration angewendet, nach der Form der Ergun Gl.
    # D_p{n+1} = D_p{n} * int(1/D_p{n}*f(D_p{n}) dz) / -3bar
    deltap_deltaz = (soln[-1][-2] - soln[0][-2]).item()
    d_p_aus = deltap_deltaz * d_p_ein / -3.0
    d_p_aus = d_p_ein

    n_i_r_rueckf = v_f_flash * sum(n_i_2) * y_i_flash

    n_i_0_vollst = n_i_0.copy()
    n_i_0_vollst[namen.index('H2')] += n_h2_zus_aus

    mm_0 = sum(n_i_0 / sum(n_i_0) * mm) / 1000.  # kg/mol
    mm_0_vollst = sum(n_i_0_vollst / sum(n_i_0_vollst) * mm) / 1000.  # kg/mol
    mm_2 = sum(n_i_2 / sum(n_i_2) * mm) / 1000.  # kg/mol
    mm_v = sum(n_i_r_rueckf / sum(n_i_r_rueckf) * mm) / 1000.  # kg/mol
    mm_1 = (v_f_flash * sum(n_i_2) /
            sum(n_i_0_vollst) * mm_v + mm_0_vollst) / (
        v_f_flash * sum(n_i_2) / sum(n_i_0_vollst) + 1
    )
    massenbilanz = mm_1 * sum(n_i_1_aus) - mm_v * sum(n_i_r_rueckf) \
        - mm_0_vollst * sum(n_i_0_vollst)
    mengenbilanz = sum(n_i_1_aus) - sum(n_i_r_rueckf) \
        - sum(n_i_0_vollst)
    aend = sum(n_i_1_aus - n_i_1_ein) / sum(n_i_1_ein) * 100
    umsatz = 1 - n_i_2[namen.index('CO')] / n_i_1_aus[namen.index('CO')]
    print('It.  ' + '{:2d}'.format(i) + ' Änderung(n_1):\t' +
          '{:5.4f}'.format(np.sqrt(aend**2)) + '%\t' + ', Massenbilanz: ' +
          '{:3.3e}'.format(np.sqrt(massenbilanz**2)) + '\t' + ', Mengenbilanz: ' +
          '{:3.3e}'.format(np.sqrt(mengenbilanz**2)) + '\t' +
          'Umsatz: ' +
          '{:3.4f}'.format(umsatz) + '\t' + 'd_p= ' +
          '{:3.4f}'.format(d_p_aus) + 'm')
    return n_i_1_aus, y_i1_ein, n_t_aus, l_r_aus, d_p_aus, g_neu, \
        cp_g_1_ein, m_dot_ein, t_soln, p_soln, \
        n_soln, n_i_soln, y_i_soln, \
        n_i_r_rueckf, n_h2_zus_aus, \
        n_i_0_vollst


# Init.
m_dot_i_0 = n_i_0 * mm / 1000.  # kg/s
m_dot_0 = sum(m_dot_i_0)
y_i0 = m_dot_i_0 / mm / sum(m_dot_i_0 / mm)
# Berechnung der Parameter
mm_m_0 = sum(y_i0 * mm) * 1 / 1000.  # kg/mol
cp_m_0 = sum(y_i0 * cp_ig_durch_r(t0) * 8.3145)  # J/mol/K
cp_g_0 = cp_m_0 / mm_m_0  # J/kg/K

n_i_1 = n_i_0  # mol/s
p1 = p0
t1 = t0

z_d_l_r = np.linspace(0, 1, 100)
dlr = 1 / (len(z_d_l_r) - 1) * l_r  # m

for i in range(50):
    n_i_1, y_i1, n_t, l_r, d_p, g, cp_g_1, \
        m_dot, t_z, p_z, n_z, n_i_z, \
        y_i_z, n_i_r, n_h2_zus, \
        n_i_0_vollst = profile(
            n_i_1, d_p,
            optimisieren_nach=optimisieren_nach_param
        )

ntu = l_r * 1 / (g * cp_g_1) * 2 * u / (d_t / 2)
# m * m^2 s/kg * kg K /J * J/s/m^2/K *1/m = [dimensionslose Einheiten]
m_i_z = n_i_z * (mm * 1 / 1000.) * 60**2  # kg/h
# mol/h * g/mol * 1kg/1000g = 1/1000 kg/h
m_z = g * n_t * (np.pi / 4 * d_t ** 2)  # kg/s
v_z = n_z * 8.3145 * 1e-5 * t_z / p_z
# mol/s * 8,3145Pa m^3/mol/K * 1e-5bar/Pa * K/bar = m^3/s
ums_z = 1 - n_i_z[:, namen.index('CO')] / \
    n_i_z[0, namen.index('CO')]
m_km_z = u * (2 / (d_t / 2)) * (t_z - t_r) * (
    np.pi / 4 * d_t ** 2) / delta_h_sat * n_t * 60 ** 2 / 1000.
# J/s/K/m^2 * 1/m * K * m^2 * kg/kJ * 60^2s/h * 1kJ/(1000J) = kg/h/m
v_z_real = np.empty_like(v_z)
for i in range(len(z_d_l_r)):
    z = z_l_v.z_non_sat(
        t_z[i], p_z[i], y_i_z[i],
        tc, pc, omega_af)['z']
    v_z_real[i] = v_z[i] * z

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
    [r'\rho_b', rho_b, r'\frac{kg_{Kat}}{m^3_{Schüttung}}'],
    ['\phi', phi, r'\frac{m^3_{Hohlraum}}{m^3_{Schüttung}}'],
    ['D_p', d_p, 'm'],
]
vars_2 = [
    ['D_t', d_t, 'm'],
    ['L_R', l_r, 'm'],
    ['n_T', n_t, ''],
    ['T_0', t0 - 273.15, '°C'],
    ['P_0', p0, 'bar'],
]
vars_3 = [
    ['U', u, r'\frac{W}{m^2\cdot K}'],
    ['\dot m', m_dot, 'kg/s'],
    ['C_{p_g}', cp_g_1 / 1000., r'\frac{kJ}{kg\cdot K}'],
    ['NTU', ntu, ''],
    ['SN', sn, ''],
]
vars_4 = [
    ['T_r', t_r - 273.15, '°C_{Kühlmittel}'],
    ['P_{Sät}', p_sat, 'bar_{Kühlmittel}'],
    ['\Delta h_{v}', delta_h_sat, r'\frac{kJ}{kg_{Kühlmittel}}'],
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
text_4 = '\n'.join(['$' + ' = '.join([line[0], '{:g}'.format(line[1]) +
                                      ' ' + line[2]]) + '$'
                    for line in vars_4])

fig = plt.figure(1)
fig.suptitle('Lösung der Zusammensetzung ' +
             '{:g}'.format(round(verhaeltnis_h2_co2.item(), 2)) +
             ':1:' +
             '{:g}'.format(round(verhaeltnis_co_co2.item(), 2)) +
             '(H2:CO2:CO)')
fig.text(0.05, 0.935, text_1, va='top', fontsize=8)
fig.text(0.25, 0.935, text_2, va='top', fontsize=8)
fig.text(0.50, 0.935, text_3, va='top', fontsize=8)
fig.text(0.75, 0.935, text_4, va='top', fontsize=8)
ax = plt.subplot2grid([2, 3], [0, 0])
ax.plot(z_d_l_r, v_z, label='$Idealgas$')
ax.plot(z_d_l_r, v_z_real, label='$Realgas$')
ax.set_ylabel(r'$\frac{\dot V}{m^3/s}$')
ax.set_xlabel('Reduzierte Position, $z/L_R$')
ax.legend(fontsize='xx-small')
ax2 = plt.subplot2grid([2, 3], [1, 0])
ax2.plot(z_d_l_r, m_km_z)
ax2.fill_between(z_d_l_r, 0, m_km_z, color='orange')
ax2.text(0.3, 1 / 2. * (m_km_z[0] + m_km_z[-1]),
         '{:g}'.format(sum(m_km_z * dlr)) + 'kg/h')
ax2.set_ylabel(r'$\frac{\dot m_{Kuehlmittel}}{kg/h}$')
ax2.set_xlabel('Reduzierte Position, $z/L_R$')

ax3 = plt.subplot2grid([2, 3], [1, 1], colspan=2)
ax3.set_ylabel('Massenstrom / (kg/h)')
ax3.set_xlabel('Reduzierte Position, $z/L_R$')
for item in ['CO', 'H2O', 'MeOH', 'CO2']:
    marker = np.random.choice(list(lines.lineMarkers.keys()))
    index = namen.index(item)
    ax3.plot(z_d_l_r, m_i_z[:, index], label=item,
             marker=marker)
ax3.legend(loc=1)
ax4 = plt.subplot2grid([2, 3], [0, 1])
ax4_1 = ax4.twinx()
ax4_1.set_ylabel('Umsatz (CO)')
ax4.set_ylabel('Temperatur / °C')
ax4.set_xlabel('Reduzierte Position, $z/L_R$')
ax4.plot(z_d_l_r, t_z - 273.15, label='T / °C')
ax4_1.plot(z_d_l_r, ums_z, label='Umsatz (CO)', ls='--', color='gray')
ax5 = plt.subplot2grid([2, 3], [0, 2], colspan=2)
ax5.set_ylabel('Druck / bar')
ax5.set_xlabel('Reduzierte Position, $z/L_R$')
ax5.plot(z_d_l_r, p_z, label='p / bar')
plt.tight_layout(rect=[0, 0, 0.95, 0.75])

locale.setlocale(locale.LC_ALL, '')

print('')
print('=== GAS AUS REAKTOR ===')
print('SN= ' + str(sn))
print('NTU= ' + str(ntu))
print('y(CO)/y(CO2)= ' + str(verhaeltnis_co_co2))
print('\n'.join([
    namen[i] + ': ' + locale.format_string('%.8g', x) + ' kg/h'
    for i, x in enumerate(m_i_z[-1])
]))
print('T= ' + str(t_z[-1] - 273.15) + '°C')
print('P= ' + str(p_z[-1]) + 'bar')
print('V0= ' + str(v_z[0]) + 'm^3/s')
print('V= ' + str(v_z[-1]) + 'm^3/s')
print('Cpg= ' + str(cp_g_1) + 'J/kg/K')
print('Partikeldurchmesser für (DeltaP= ' +
      '{:g}'.format(p_z[0] - p_z[-1]) + ' bar): ' +
      '{:g}'.format(d_p) + ' m'
      )
print('Kühlmittel: Gesättigtes H_2O(l) ' +
      ' bei ' + '{:g}'.format(p_sat) + ' bar' + '\n' +
      'Verdampfungsenthalpie: ' + '{:g}'.format(delta_h_sat) +
      'kJ/kg' + '\n' + 'Kühlmittelmassenstrom: ' +
      '{:g}'.format(sum(m_km_z * dlr)) + 'kg/h')

print('')
print('======' * 3)
print('=== MAKEUP-STROM ===')
print('\n'.join([
    namen[i] + ': ' + locale.format_string('%.8g', x) + ' kg/h'
    for i, x in enumerate(n_i_0 * mm / 1000. * 60**2)
]))
print('')
print('\n'.join([
    namen[i] + ': ' + locale.format_string('%.8g', x) + 'kmol/h'
    for i, x in enumerate(n_i_0 * 60**2 / 1000.)
]))
print('')
print('======' * 3)
print('=== RÜCKLAUFSTROM ===')
print('\n'.join([
    namen[i] + ': ' + locale.format_string('%.8g', x) + ' kg/h'
    for i, x in enumerate(n_i_r * mm / 1000. * 60**2)
]))
print('')
print('======' * 3)
print('=== ERFORDERLICHE CO UND H2 STRÖME, UM SN UND CO/CO2 ANZUPASSEN ===')
print('H2: ' +
      locale.format_string('%.8g', n_h2_zus.item() * mm[namen.index('H2')] /
                    1000. * 60 ** 2) + ' kg/h')
print('auf kmol/h')
print('H2: ' +
      locale.format_string('%.8g', n_h2_zus.item() / 1000. * 60 ** 2) + ' kmol/h')
print('')
print('MAKE-UP MIT ERFORDERLICHEN H2 UND CO-STRÖMEN')
print('\n'.join([
    namen[i] + ': ' + locale.format_string('%.8g', x) + 'kmol/h'
    for i, x in enumerate(n_i_0_vollst / 1000. * 60**2)
]))
print('')
print('======' * 3)
print('=== REAKTOR ZULAUFSTROM ===')
print('\n'.join([
    namen[i] + ': ' + locale.format_string('%.8g', x) + ' kg/h'
    for i, x in enumerate(n_i_1 * mm / 1000. * 60**2)
]))
print('\nauf kmol/h\n')
print('\n'.join([
    namen[i] + ': ' + locale.format_string('%.8g', x) + ' kmol/h'
    for i, x in enumerate(n_i_1 / 1000. * 60**2)
]))
print('SN0: ' + str((n_i_0_vollst[namen.index('H2')] -
                     n_i_0_vollst[namen.index('CO2')]) / (
            n_i_0_vollst[namen.index('CO2')]+
            n_i_0_vollst[namen.index('CO')])))

if os.name == 'nt':
    thisappid = plt.matplotlib.__package__+plt.matplotlib.__version__
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(thisappid)
plt.show()