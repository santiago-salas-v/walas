from crt_lib import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import lines

# Journal of Power Sources 173 (2007) 467–477

# Reaktor
n_t = 1  # Rohre
d_t = 0.06  # m Rohrdurchmesser
l_r = 0.3  # m Rohrlänge
# Katalysator
d_p = 0.0005  # m Feststoff
rho_b = 1190  # kg Kat/m^3 Feststoff
phi = 0.38+0.073*(1-(d_t/d_p-2)**2/(d_t/d_p)**2)  # m^3 Gas/m^3 Feststoff
m_kat = 1190 * (1 - 0.3) * np.pi / 4 * (
    0.03)**2 * 7  # kg Kat (pro Rohr)
# Betriebsbedingungen
t0 = 187.5 + 273.15  # K
p0 = 1.01325  # bar
# Wärmetauschparameter
u = 0  # W/m^2/K
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
    6.6, 9.1, 36.0,
    26.4, 0, 4.7,
    0, 0, 0,
    0
]) / 60**2  # mol/s
y_i0 = n_i_0 / sum(n_i_0)
m_dot = sum(n_i_0 * mm / 1000.) # kg/s

z_l_v.use_pr_eos()

# Stoechiometrische Koeffizienten
nuij = np.zeros([len(namen), 1])
# RWGS Reverse-Wassergasshiftreaktion (muss gleich sein als die Vorwärtsreaktion,
# wofür die Kinetik verfügbar ist)
nuij[[
    namen.index('CO2'),
    namen.index('H2'),
    namen.index('MeOH'),
    namen.index('H2O'),
    namen.index('CO'),
    namen.index('N2'),
], 0] = - np.array([+1, +1, 0, -1, -1, 0], dtype=float)


delta_h_r_298 = nuij.T.dot(h_298)  # J/mol

# Berechnung der Parameter
g = m_dot / (np.pi / 4 * d_t**2) / n_t  # kg/m^2/s
mm_m_0 = sum(y_i0 * mm) * 1 / 1000.  # kg/mol
cp_m_0 = sum(y_i0 * cp_ig_durch_r(t0) * 8.3145)  # J/mol/K
cp_g_0 = cp_m_0 / mm_m_0  # J/kg/K
# Anzahl an Übertragungseinheiten (NTU)
ntu = l_r * 1 / (g * cp_g_0) * 2 * u / (d_t / 2)
# m * m^2 s/kg * kg K /J * J/s/m^2/K *1/m = [dimensionslose Einheiten]
# Stöchiometrische Zahl
sn = (y_i0[namen.index('H2')] - y_i0[namen.index('CO2')]) / (
    y_i0[namen.index('CO2')] + y_i0[namen.index('CO')]
)

y_0 = np.empty([len(y_i0) + 1 + 1])
y_0[:-2] = y_i0
y_0[-2] = p0
y_0[-1] = t0
z_d_l_r = np.linspace(0, 1, 100)
d_z_dimlos = 1 / (len(z_d_l_r) - 1)  # dimlos
dlr = d_z_dimlos * l_r  # m
soln = odeint(lambda y, z0:
              df_dt(y, z0, g, d_t, l_r,
                    phi, d_p, rho_b,
                    u, t_r),
              y_0, z_d_l_r)
y_i_soln = soln[:, :len(y_i0)]
p_soln = soln[:, -2]
t_soln = soln[:, -1]

n_i_soln = np.zeros_like(y_i_soln)
m_i_soln = np.zeros_like(y_i_soln)
n_soln = np.zeros_like(z_d_l_r)
mm_m_soln = np.zeros_like(z_d_l_r)
m_soln = np.zeros_like(z_d_l_r)
v_soln = np.zeros_like(z_d_l_r)
m_km_soln = np.zeros_like(z_d_l_r)
ums_soln = np.zeros_like(z_d_l_r)
for i in range(len(z_d_l_r)):
    mm_m_soln[i] = sum(y_i_soln[i] * mm * 1 / 1000.)  # kg/mol
    n_soln[i] = g * n_t * 60**2 * (np.pi / 4 * d_t**2) / mm_m_soln[i]
    # kg/s/m^2 * 60^2s/h * m^2 / kg*mol = mol/h
    m_soln[i] = g * n_t * 60**2 * (np.pi / 4 * d_t**2)  # kg/h
    n_i_soln[i] = n_soln[i] * y_i_soln[i]  # mol/h
    m_i_soln[i] = n_soln[i] * y_i_soln[i] * (mm * 1 / 1000.)
    # mol/h * g/mol * 1kg/1000g
    v_soln[i] = n_soln[i] * 8.3145 * 1e-5 * t_soln[i] / p_soln[i]
    ums_soln[i] = (n_i_soln[0][namen.index('CO')] -
                   n_i_soln[i][namen.index('CO')]
                   ) / n_i_soln[0][namen.index('CO')]
    m_km_soln[i] = u * (2 / (d_t / 2)) * (t_soln[i] - t_r) * (
        np.pi / 4 * d_t**2) / delta_h_sat * n_t * 60**2 / 1000.
    # J/s/K/m^2 * 1/m * K * m^2 * kg/kJ * 60^2s/h * 1kJ/(1000J) = kg/h/m

# Energie-Analyse
t_m_1 = 1 / 2 * (36696 / 8.3145 - np.sqrt(36696 /
                                          8.3145 * (36696 / 8.3145 - 4 * t_r)))
t_m_2 = 1 / 2 * (-94765 / 8.3145 - np.sqrt(-94765 /
                                           8.3145 * (-94765 / 8.3145 - 4 * t_r)))

vars_1 = [
    [r'\rho_b', rho_b, r'\frac{kg_{Kat}}{m^3_{Fest}}'],
    ['\phi', phi, r'\frac{m^3_{Gas}}{m^3_{Fest}}'],
    ['D_p', d_p, 'm_{Fest}'],
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
    ['C_{p_g}', cp_g_0 / 1000., r'\frac{kJ}{kg\cdot K}'],
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
fig.suptitle('Adiabates System' +
             '(Journal of Power Sources 173 (2007) 467–477)')
fig.text(0.05, 0.935, text_1, va='top', fontsize=8)
fig.text(0.25, 0.935, text_2, va='top', fontsize=8)
fig.text(0.50, 0.935, text_3, va='top', fontsize=8)
fig.text(0.75, 0.935, text_4, va='top', fontsize=8)
ax = plt.subplot2grid([2, 3], [0, 0])
ax.plot(z_d_l_r, v_soln, label='$\dot V$')
ax.set_ylabel(r'$\frac{\dot V}{m^3/h}$')
ax.set_xlabel('Reduzierte Position, $z/L_R$')
ax2 = plt.subplot2grid([2, 3], [1, 0])
ax2.plot(z_d_l_r, m_km_soln)
ax2.fill_between(z_d_l_r, 0, m_km_soln, color='orange')
ax2.text(0.3, 1 / 2. * (m_km_soln[0] + m_km_soln[-1]),
         '{:g}'.format(sum(m_km_soln * dlr)) + 'kg/h \n')
ax2.set_ylabel(r'$\frac{\dot m_{Kuehlmittel}}{\frac{kg}{h\cdot m}}$')
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
ax4_1.set_ylabel('CO - Molanteil')
ax4.set_ylabel('Temperatur / °C')
ax4.set_xlabel('Reduzierte Position, $z/L_R$')
ax4.plot(z_d_l_r, t_soln - 273.15, label='T / °C')
ax4_1.plot(z_d_l_r, y_i_soln[:, namen.index('CO')],
           ls='--', color='gray')
ax5 = plt.subplot2grid([2, 3], [0, 2], colspan=2)
ax5.set_ylabel('Druck / bar')
ax5.set_xlabel('Reduzierte Position, $z/L_R$')
ax5.plot(z_d_l_r, p_soln, label='p / bar')
plt.tight_layout(rect=[0, 0, 0.95, 0.75])

print('')
print('NTU= ' + '{:g}'.format(ntu))
print('\n'.join([
    namen[i] + ': ' + '{:g}'.format(x) + ' kg/h'
    for i, x in enumerate(m_i_soln[-1])
]))
print('T=' + str(t_soln[-1] - 273.15) + '°C')
print('P=' + str(p_soln[-1]) + 'bar')
print('V0=' + str(v_soln[0]) + 'm^3/h')
print('V=' + str(v_soln[-1]) + 'm^3/h')
print('Kühlmittel: Gesättigtes $H_2O(l)$' +
      ' bei ' + '{:g}'.format(p_sat) + ' bar' +
      '\n' + 'Verdampfungsenthalpie: ' +
      '{:g}'.format(delta_h_sat) +
      'kJ/kg' + '\n' + 'Kühlmittelmassenstrom: ' +
      '{:g}'.format(sum(m_km_soln * dlr)) + 'kg/h')
print('Partikeldurchmesser für DeltaP=' +
      '{:g}'.format(p_soln[0] - p_soln[-1]) + ' bar: ' +
      '{:g}'.format(d_p) + ' m'
      )

plt.show()