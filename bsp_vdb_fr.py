import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from scipy.integrate import odeint

# Journal of Catalysis, 1996, 161. Jg., Nr. 1, S. 1-10.

# Bezugdaten: 'Løvik, Ingvild. "Modelling, estimation and optimization
# of the methanol synthesis with catalyst deactivation." (2001).

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
m_dot = 2.8 * 1e-5  # kg/s
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

# Quelle: Chemical thermodynamics for process simulation
h_298 = np.array([
    -110540, -241820, -201160, 0, -393500, 0
], dtype=float)  # J/mol

delta_h_r_298 = nuij.T.dot(h_298)  # J/mol

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

# Lennard-Jones Parameter Prausnitz Anhang B
    # epsilon / kB
    l_j_epsilon_d_k = np.array([
        91.7, 809.1, 481.8, 59.7, 195.2, 71.4
    ])  # K
    # sigma
    l_j_sigma = np.array([
        3.690, 2.641, 3.626, 2.827, 3.941, 3.798
    ])  # Angstrom


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
    ) / (t - 298.15) * 8.3145  # J/mol/K
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
    phi_ab = np.zeros([mm.size, mu_i.size])
    for alpha in range(phi_ab.shape[0]):
        for beta in range(phi_ab.shape[1]):
            phi_ab[alpha, beta] = 1 / np.sqrt(8) * (
                1 + mm[alpha] / mm[beta]) ** (-1 / 2.) * (
                1 + (mu_i[alpha] / mu_i[beta]) ** (1 / 2.) * (
                    mm[beta] / mm[alpha]) ** (1 / 4.)
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
    c_t = p / (8.3145 * 1e-5 * t)  # bar * mol/bar/m^3/K*K = mol/m^3
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


y_0 = np.empty([len(y_i0)+1+1])
y_0[:-2] = y_i0
y_0[-2] = p0
y_0[-1] = t0
z_d_l_r = np.linspace(0, 1, 200)
soln = odeint(df_dt, y_0, z_d_l_r)
y_i_soln = soln[:, :len(y_i0)]
p_soln = soln[:, -2]
t_soln = soln[:, -1]

t_r = np.linspace(100, 1000, 200)
fig = plt.figure(1)
fig.suptitle('Adiabates System (Journal of Catalysis, ' +
             '1996, 161. Jg., Nr. 1, S. 1-10. )')
ax = plt.subplot2grid([2, 3], [0, 0])
ax.plot(t_r, np.array(
    [mu(t, y_i0) for t in t_r]) / 1e-5, label='Berechnet')
ax.plot(t_r, np.array(
    [67.2e-7 + 0.21875e-7 * t for t in t_r]) / 1e-5, label='Bezugdaten')
ax.set_ylabel(r'$\frac{\mu}{(Pa s) \cdot 1e-5}$')
plt.setp(ax.get_xticklabels(), visible=False)
ax.legend()

ax2 = plt.subplot2grid([2, 3], [1, 0], sharex=ax)
delta_h_r_t_0 = np.array([delta_h_r(t) for t in t_r])
ax2.plot(t_r, delta_h_r_t_0[:, 0] / 1000.,
         label='$\Delta_R H_1 Hyd. CO_2 - berechnet$')
ax2.plot(t_r, delta_h_r_t_0[:, 2] / 1000.,
         label='$\Delta_R H_3 RWGS - berechnet$')
ax2.plot(t_r, -(57980 + 35 * (t_r - 498.15)) / 1000.,
         label='$\Delta_R H_{rx}^1 Hyd. CO_2 - Bezugsdaten$')
ax2.plot(t_r, -(-39892 + 8 * (t_r - 498.15)) / 1000.,
         label='$\Delta H_{rx}^2 RWGS - Bezugsdaten$')
ax2.set_xlabel('T/K')
ax2.set_ylabel(r'$\frac{\Delta_R H(T)}{(J/mol) \cdot 10^3 }$')
ax2.legend(fontsize='xx-small', loc='best')
plt.tight_layout()


ax3 = plt.subplot2grid([2, 3], [1, 1], colspan=2)
ax3.set_ylabel('Konzentration / Mol%')
ax3.set_xlabel('Reduzierte Position, $z/L_R$')
for item in ['CO', 'H2O', 'MeOH', 'CO2']:
    marker = np.random.choice(list(lines.lineMarkers.keys()))
    index = namen.index(item)
    ax3.plot(z_d_l_r, y_i_soln[:, index] * 100., label=item,
             marker=marker)
ax3.legend(loc=1)
plt.tight_layout()
ax4 = plt.subplot2grid([2, 3], [0, 1])
ax4.set_ylabel('Temperatur / K')
ax4.set_xlabel('Reduzierte Position, $z/L_R$')
ax4.plot(z_d_l_r, t_soln, label='T / K')
plt.tight_layout(rect=[0, 0, 1, 0.9])
ax5 = plt.subplot2grid([3, 3], [0, 2], colspan=2)
ax5.set_ylabel('Druck / bar')
ax5.set_xlabel('Reduzierte Position, $z/L_R$')
ax5.plot(z_d_l_r, p_soln, label='p / bar')

# Chem. Eng. Technol. 2011, 34, No. 5, 817–822

# Katalysator
rho_b = 1190  # kg Kat/m^3 Feststoff
phi = 0.285  # m^3 Gas/m^3 Feststoff
m_kat = 1190*(1-0.285)*np.pi/4*(
    0.04)**2 * 7  # kg Kat (pro Rohr)
d_p = 0.0054  # m Feststoff
# Reaktor
n_t = 1620  # Rohre
d_t = 0.04  # m Rohrdurchmesser
l_r = 7.  # m Rohrlänge
# Betriebsbedingungen
t0 = 225+273.15  # K
p0 = 69.7  # bar
m_dot = 57282.8 / 60**2 / n_t  # kg/s
# Zulaufbedingungen
namen = ['CO', 'CO2', 'H2', 'H2O', 'MeOH',
         'CH4', 'N2', 'EthOH', 'PrOH', 'METHF']
m_dot_i = np.array([
    10727.9, 23684.2, 9586.5,
    108.8, 756.7, 4333.1,
    8072.0, 0.6, 0.0,
    13.0
], dtype=float) / 60**2 /n_t   # kg/s
m_dot = sum(m_dot_i)
mm = np.array([
    28.01, 44.01, 2.02,
    18.02, 32.04, 16.04,
    28.01, 46.07, 60.10,
    60.05
], dtype=float)  # g/mol
y_i0 = m_dot_i/mm / sum(m_dot_i/mm)
# Wärmetauschparameter
t_r = 232+273.15  # K
u = 118.44  # W/m^2/K

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

# Parameter der Gleichung aus dem VDI-Wärmeatlas
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

# Lennard-Jones Parameter Prausnitz Anhang B
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

# Quelle: The properties of Gases and Liquids
h_298 = np.array([
    -110.53, -393.51, 0,
    -241.81, -200.94, -74.52,
    0, -234.95, -255.20,
    -352.40
], dtype=float)*1000.  # J/mol

delta_h_r_298 = nuij.T.dot(h_298)  # J/mol

t_reihe = np.linspace(100, 1000, 200)
fig = plt.figure(2)
fig.suptitle('Nicht adiabates System')
ax = plt.subplot2grid([2, 3], [0, 0])
ax.plot(t_reihe, np.array(
    [mu(t, y_i0) for t in t_reihe]) / 1e-5, label='Berechnet')
ax.plot(t_reihe, np.array(
    [67.2e-7 + 0.21875e-7 * t for t in t_reihe]) / 1e-5, label='Bezugdaten')
ax.set_ylabel(r'$\frac{\mu}{(Pa s) \cdot 1e-5}$')
plt.setp(ax.get_xticklabels(), visible=False)
ax.legend()

y_0 = np.empty([len(y_i0)+1+1])
y_0[:-2] = y_i0
y_0[-2] = p0
y_0[-1] = t0
z_d_l_r = np.linspace(0, 1, 200)
soln = odeint(df_dt, y_0, z_d_l_r)
y_i_soln = soln[:, :len(y_i0)]
p_soln = soln[:, -2]
t_soln = soln[:, -1]

n_i_soln = np.zeros_like(y_i_soln)
m_i_soln = np.zeros_like(y_i_soln)
n_soln = np.zeros_like(z_d_l_r)
mm_m_soln = np.zeros_like(z_d_l_r)
m_soln = np.zeros_like(z_d_l_r)
for i in range(len(z_d_l_r)):
    mm_m_soln[i] = sum(y_i_soln[i] * mm * 1/1000.)  # kg/mol
    n_soln[i] = u_s * n_t * 60**2 * (np.pi/4 * d_t**2) / mm_m_soln[i]
    # kg/s/m^2 * 60^2s/h * m^2 / kg*mol = mol/h
    m_soln[i] = u_s * n_t * 60**2 * (np.pi/4 * d_t**2)  # kg/h
    n_i_soln[i] = n_soln[i] * y_i_soln[i]  # mol/h
    m_i_soln[i] = n_soln[i] * y_i_soln[i] * (mm * 1 / 1000.)
    # mol/h * g/mol * 1kg/1000g

ax3 = plt.subplot2grid([2, 3], [1, 1], colspan=2)
ax3.set_ylabel('Massenstrom / (kg/h)')
ax3.set_xlabel('Reduzierte Position, $z/L_R$')
for item in ['CO', 'H2O', 'MeOH', 'CO2']:
    marker = np.random.choice(list(lines.lineMarkers.keys()))
    index = namen.index(item)
    ax3.plot(z_d_l_r, m_i_soln[:, index], label=item,
             marker=marker)
ax3.legend(loc=1)
plt.tight_layout()
ax4 = plt.subplot2grid([2, 3], [0, 1])
ax4.set_ylabel('Temperatur / °C')
ax4.set_xlabel('Reduzierte Position, $z/L_R$')
ax4.plot(z_d_l_r, t_soln-273.15, label='T / °C')
plt.tight_layout(rect=[0, 0, 1, 0.9])
ax5 = plt.subplot2grid([3, 3], [0, 2], colspan=2)
ax5.set_ylabel('Druck / bar')
ax5.set_xlabel('Reduzierte Position, $z/L_R$')
ax5.plot(z_d_l_r, p_soln, label='p / bar')

print('\n'.join([str(x) for x in m_i_soln[-1]]))

plt.show()
