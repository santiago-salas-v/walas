import numpy as np
import z_l_v

# Modell feststellen
z_l_v.use_pr_eos()

p = 50.  # bar
temp = 273.15 + 220.  # K
t_flash = 273.15 + 60  # K
t0_ref = 298.15  # K
r = 8.314  # J/(mol K)

namen = ['CO', 'H2', 'CO2', 'H2O', 'CH4', 'NH3', 'AR', 'O2', 'N2']
ne = np.array([
0, 0, 0, 60000, 20000, 0, 
0.01*15000,
0.21*15000,
0.78*15000
], dtype=float) # kmol/h

nuij = np.array([
    [-1, +1, +1, -1, +0, +0, +0, +0, +0],
    [-1, -3, +0, +1, +1, +0, +0, +0, +0],
    [+0, -4, -1, +2, +1, +0, +0, +0, +0],
    [+0, -3/2, 0, 0, +0, +1, +0, +0, -1/2]
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

mm = np.array([
    28.01, 2.02, 44.01, 
    18.02, 16.04, 17.03, 
    39.95, 32.00, 28.01
    ]).reshape([len(namen),1])

# Koeffizienten für Cp(T)/R = A + B*T + C*T^2 + D*T^-2, T[=]K
# Nach rechts hin: A, B, C, D

# Smith, J.M.; Van Ness, Hendrick ; Abbott, Michael 
# Introduction to Chemical Engineering Thermodynamics. 
# New York: McGraw-Hill Education, 2017.
cp_coefs = np.array([z for z in [
    [
        y.replace(',', '.') for y in x.split('  ')
    ] for x in """
3,3760E+00  5,5700E-04  0,0000E+00  -3,1000E+03
3,2490E+00  4,2200E-04  0,0000E+00  8,3000E+03
5,4570E+00  1,0450E-03  0,0000E+00  -1,1570E+05
3,4700E+00  1,4500E-03  0,0000E+00  1,2100E+04
1,7020E+00  9,0810E-03  -2,1640E-06  0,0000E+00
3,5780E+00  3,0200E-03  0,0000E+00  -1,8600E+04
""".split('\n') if len(x) > 0] if len(z) > 1], dtype=float)

# Koeffizienten für Cp(T)/R = B+(C-B)(T/(A+T))^2*(
# 1-A/(A+T)*(D+E*T/(A+T)+F*(T/(A+T))^2+G*(T/(A+T))^3))
# Nach rechts hin: A, B, C, D

# e.V., VDI: VDI-Wärmeatlas. Wiesbaden: Springer Berlin Heidelberg, 2013.

cp_coefs = np.array([z for z in [
    [
        y.replace(',', '.').replace('–','-') for y in x.split('  ')
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

def cp_durch_r(t):
    a,b,c,d,e,f,g = np.split(cp_coefs,cp_coefs.shape[1], axis=1)
    return b+(c-b)*(t/(a+t))**2*(
    1-a/(a+t)*(
    d+e*t/(a+t)+f*(t/(a+t))**2+g*(t/(a+t))**3
    )) # dimensionslos

print('298,15°K')
print(cp_durch_r(298.15)*8.314/mm)
print('773,15°K')
print(cp_durch_r(500+273.15)*8.314/mm)

# Berechne H(T), G(T) und K(T) mit Cp(T)

def h(t):
    return (
        h_298 +
        r * cp_coefs[:, 0] * (t - t0_ref) +
        r * cp_coefs[:, 1] / 2. * (t ** 2 - t0_ref ** 2) +
        r * cp_coefs[:, 2] / 3. * (t ** 3 - t0_ref ** 3) -
        r * cp_coefs[:, 3] * (1 / t - 1 / t0_ref)
    )  # J/mol

def g(t, h_t):
    return (
        h_t - t / t0_ref * (h_298 - g_298) -
        r * cp_coefs[:, 0] * t * np.log(t / t0_ref) -
        r * cp_coefs[:, 1] * t ** 2 * (1 - t0_ref / t) -
        r * cp_coefs[:, 2] / 2. * t ** 3 * (1 - (t0_ref / t) ** 2) +
        r * cp_coefs[:, 3] / 2. * 1 / t * (1 - (t / t0_ref) ** 2)
    )  # J/mol

def k(t, g_t):
    delta_g_t = nuij.T.dot(g_t)
    return np.exp(-delta_g_t / (r * t))

delta_gr_298 = nuij.T.dot(g_298)

delta_hr_298 = nuij.T.dot(h_298)

cp_493 = cp(493.15)  # J/(mol K)
h_493 = h(493.15)  # J/mol
g_493 = g(493.15, h_493)  # J/mol
k_493 = k(493.15, g_493)  # []
