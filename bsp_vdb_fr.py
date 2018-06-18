import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Katalysator
rho_b = 1775  # kg Kat/m^3 Feststoff
phi = 0.5  # m^3 Gas/m^3 Feststoff
m_kat = 34.8 / 1000.  # kg Kat
d_p = 0.0005  # m Feststoff
# Reaktor
d_t = 0.016  # m Rohrdurchmesser
l_r = 0.15  # m Rohrlänge
# Betriebsbedingungen
t0 = 493.2  # K
p0 = 50.  # bar
m_dot = 1e-5  # kg/s
# Zulaufbedingungen
namen = ['CO', 'H2O', 'MeOH', 'H2', 'CO2', 'N2']
y_i0 = np.array([
    4, 0, 0, 82, 3, 11
], dtype=float) / 100.
mm = np.array([
    28, 18, 32, 2, 44, 28
], dtype=float)  # g/mol
# Wärmetausch
t_r = 25 + 273.15  # K
u = 0  # Adiabat

# Berechnung der Parameter
u_s = m_dot / (np.pi / 4 * d_t**2)  # kg/m^2/s
# Stoechiometrische Koeffizienten
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
# RWGS Rückwassergasshiftreaktion
nuij[[
    namen.index('CO2'),
    namen.index('H2'),
    namen.index('MeOH'),
    namen.index('H2O'),
    namen.index('CO'),
    namen.index('N2'),
], 2] = np.array([+1, +1, 0, -1, -1, 0], dtype=float)

# Quelle: Chemical thermodynamics for process simulation
h_298 = np.array([
    -110540, -241820, -201160, 0, -393500, 0
], dtype=float)  # J/mol

# Parameter der Gleichung aus dem VDI-Wärmeatlas
cp_konstanten = np.zeros([len(namen), 7])
cp_konstanten_text = [
    '407,9796 3,5028 2,8524 –2,3018 32,9055 –100,1815 106,1141',
    '706,3032 5,1703 –6,0865 –6,6011 36,2723 –63,0965 46,2085',
    '846,6321 5,7309 –4,8842 –12,8501 78,9997 –127,3725 82,7107',
    '392,8422 2,4906 –3,6262 –1,9624 35,6197 –81,3691 62,6668',
    '514,5073 3,4923 –0,9306 –6,0861 54,1586 –97,5157 70,9687',
    '432,2027 3,5160 2,8021 –4,1924 42,0153 –114,2500 111,1019',
]
for i in range(len(namen)):
    cp_konstanten[i, :] = np.array(
        cp_konstanten_text[i].replace(
            '–', '-').replace(',', '.').split(' '),
        dtype=float)


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


def mcph(x, t0, t):
    return sum(x * (
        int_cp_durch_r_dt_minus_const(t) -
        int_cp_durch_r_dt_minus_const(t0))
    ) / sum(x) / (t - t0)


def mu(t, y_i):
    mm_int = np.array([
        28, 18, 32, 2, 44, 28
    ], dtype=float)
    # Lennard-Jones Parameter Prausnitz Anhang B
    # epsilon / kB
    l_j_epsilon_d_k = np.array([
        91.7, 809.1, 481.8, 59.7, 195.2, 71.4
    ])  # K
    # sigma
    l_j_sigma = np.array([
        3.690, 2.641, 3.626, 2.827, 3.941, 3.798
    ])  # Angstrom
    # T* = k T / epsilon
    t_st = t / l_j_epsilon_d_k  # J/K
    # Stoßintegral (Bird Tabelle E.2)
    omega_mu = 1.16145 / t_st**0.14874 + \
        0.52487 / np.exp(0.77320 * t_st) + \
        2.16178 / np.exp(2.43787 * t_st)
    konst_1 = 5 / 16. * np.sqrt(
        8.3145 * 1000 * 100 ** 2 / np.pi
    ) * 10**16 / 6.022e23  # g/cm/s
    mu_i = konst_1 * np.sqrt(mm_int * t) / (
        l_j_sigma**2 * omega_mu
    ) * 100 / 1000
    # g/cm/s * 100cm/m * 1kg/1000g = kg/m/s = Pa s
    phi_ab = np.zeros([mm_int.size, mu_i.size])
    for alpha in range(phi_ab.shape[0]):
        for beta in range(phi_ab.shape[1]):
            phi_ab[alpha, beta] = 1 / np.sqrt(8) * (
                1 + mm_int[alpha] / mm_int[beta]) ** (-1 / 2.) * (
                1 + (mu_i[alpha] / mu_i[beta]) ** (1 / 2.) * (
                    mm_int[beta] / mm_int[alpha]) ** (1 / 4.)
            ) ** 2
    mu_mix = sum(y_i * mu_i / phi_ab.dot(y_i))
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
        0.499, 6.62e-11, 3453.38, 1.07, 1.22e-10
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
    p_ch3oh = p_i[namen.index('MeOH')]
    p_h2o = p_i[namen.index('H2O')]
    [k_1, k_2, k_3,
     k_h2, k_h2o, k_h2o_d_k_8_k_9_k_h2,
     k5a_k_2_k_3_k_4_k_h2, k1_strich] = k_t(t)
    r_meoh = k5a_k_2_k_3_k_4_k_h2 * p_co2 * p_h2 * (
        1 - 1 / k_1 * p_h2o * p_ch3oh / (
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
    return np.array([r_meoh, 0, r_rwgs])


def df_dt(y, t0_zeit):
    y_i = y[:-2]
    p = y[-2]
    t = y[-1]
    mm_m = sum(y_i * mm) * 1 / 1000.  # kg/mol
    cp_m = sum(y_i * cp_ig_durch_r(t) * 8.3145) / mm  # J/kg/K
    cp_g = cp_m / mm_m
    c_t = p / (8.3145 * 1e-5 * t)  # bar * mol/bar/m^3/K*K = mol/m^3
    p_i = y_i * p  # bar
    r_j = r_i(t, p_i)  # mol/kg Kat/s
    dyi_dz = l_r * rho_b * mm_m / u_s * (
        nuij.dot(r_j) - y_i * sum(nuij.dot(r_j))
    )   # m * kgKat/m^3 * kg/mol * m^2 s/kg * mol/kgKat/s = dimlos
    dp_dz = -l_r * u_s / (
        c_t * mm_m / 1000. * d_p
    ) * (1 - phi) / phi**3 * (
        150 * (1 - phi) * mu(t, y_i) / d_p + 1.75 * u_s
    )
    dt_dz = l_r * 1 / (
        u_s * cp_g
    ) * (
        2 * u / (d_t / 2) * (t_r - t) + rho_b * 2222222222222222222222222222222222
    )


t_r = np.linspace(100, 1000, 200)
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(t_r, np.array(
    [mu(t, y_i0) for t in t_r]) / 1e-5, label='Berechnet')
ax.plot(t_r, np.array(
    [67.2e-7 + 0.21875e-7 * t for t in t_r]) / 1e-5, label='Bezugsdaten')
ax.set_xlabel('T/K')
ax.set_ylabel(r'$\frac{\mu}{(Pa s) \cdot 1e-5}$')
ax.legend()

ax2 = fig.add_subplot(212)
ax2.plot(t_r, np.array(
    [sum(cp_ig_durch_r(t)*y_i0) * 8.3145/sum(mm/1000. * y_i0)
     for t in t_r]) / 1e-5, label='Berechnet')
ax2.set_xlabel('T/K')
ax2.set_ylabel(r'$C_{p_g} /(\frac{J}{kg K}})$')


plt.show()
