import numpy as np
import numbers

# Basic Constants I: 468 Components in table
# ======================
# Values in Table:
no = np.array([455, 460, 461, 463, 95])
formula = np.array(['N2', 'O2', 'O2S', 'O3S', 'C3H8'])
name = np.array(['nitrogen', 'oxygen',
                 'sulfur dioxide', 'sulfur trioxide',
                 'propane'])
cas_no = np.array(['7727-37-9', '7782-44-7',
                   '7446-09-5', '7446-11-9', '74-98-6'])
mol_wt = np.array([28.014, 31.999, 64.065, 80.064, 44.097])
tfp = np.array([63.15, 54.36, 197.67, 289.95, 91.45])
tb = np.array([77.35, 90.17, 263.13, 317.9, 231.02])
tc = np.array([126.2, 154.58, 430.8, 490.9, 369.83])
pc = np.array([33.98, 50.43, 78.84, 82.1, 42.48])
zc = np.array([0.289, 0.288, 0.269, 0.254, 0.276])
omega = np.array([0.037, 0.0, 0.0, 0.0, 0.152])
# Basic Constants II: 458 Components in table
# ======================
# Values in Table:
no = np.array([455, 460, 461, 463, 95])
formula = np.array(['N2', 'O2', 'O2S', 'O3S', 'C3H8'])
name = np.array(['nitrogen', 'oxygen',
                 'sulfur dioxide', 'sulfur trioxide',
                 'propane'])
cas_no = np.array(['7727-37-9', '7782-44-7',
                   '7446-09-5', '7446-11-9', '74-98-6'])
delHf0 = np.array([0.0, 0.0, -296.81, -395.72, -104.68])
delGf0 = np.array([0.0, 0.0, -300.14, -370.93, -24.29])
delHb = np.array([5.58, 6.82, 24.94, 40.69, 19.04])
delHm = np.array([0.72, 0.44, 7.4, 7.53, 3.53])
v_liq = np.array([34.84, 27.85, 44.03, 42.1, 74.87])
t_liq = np.array([78.0, 90.0, 263.15, 298.15, 233.15])
dipole = np.array([0.0, 0.0, 1.6, 0.0, 0.0])
# Ideal Gas and Liquid Heat Capacities: 368 Components in table
# ======================
# Values in Table:
no = np.array([455, 460, 461, 463, 95])
formula = np.array(['N2', 'O2', 'O2S', 'O3S', 'C3H8'])
name = np.array(['nitrogen', 'oxygen',
                 'sulfur dioxide', 'sulfur trioxide',
                 'propane'])
cas_no = np.array(['7727-37-9', '7782-44-7',
                   '7446-09-5', '7446-11-9', '74-98-6'])
trange = np.array(['50-1000', '50-1000',
                   '50-1000', '50-1000', '50-1000'])
a0 = np.array([3.539, 3.63, 4.417, 3.426, 3.847])
a1 = np.array([-0.000261, -0.001794, -0.002234,
               0.006479, 0.005131])
a2 = np.array([7e-08, 6.5800000000000005e-06,
               2.344e-05, 1.6910000000000002e-05,
               6.0110000000000006e-05])
a3 = np.array([1.57e-09, -6.01e-09, -3.271e-08,
               -3.356e-08, -7.893e-08])
a4 = np.array([-9.9e-13, 1.7899999999999998e-12,
               1.3929999999999999e-11, 1.59e-11, 3.079e-11])
cpig = np.array([29.12, 29.38, 40.05, 50.86, 73.6])
cpliq = np.array([0.0, 0.0, 0.0, 226.8, 120.0])
cpig_test = np.array([29.110669036107197,
                      29.39089107430498, 42.21885464290135,
                      50.69420598839356, 73.76265226949711])

nuij = np.zeros([len(formula), 1])
nuij[[
    np.where(formula == 'C3H8')[0][0],
    np.where(formula == 'N2')[0][0],
    np.where(formula == 'O2')[0][0],
    np.where(formula == 'O2S')[0][0],
    np.where(formula == 'O3S')[0][0],
], 0] = np.array([0, 0, -1 / 2., -1, +1], dtype=float)

delta_h_r_298 = nuij.T.dot(delHf0)  # kJ / mol


def cp_ig_durch_r(t):
    cp = a0 + a1 * t + a2 * t**2 + \
        a3 * t**3 + a4 * t**4
    return cp  # dimensionslos


def int_cp_durch_r_dt_minus_const(t):
    integ = a0 * t + a1 / 2 * t**2 + \
        a2 / 3 * t**3 + a3 / 4 * t**4 + \
        a4 / 5 * t**5
    return integ


def int_cp_durch_rt_dt_minus_const(t):
    integ = a0 * np.log(t) + a1 * t + \
        a2 / 2 * t**2 + a3 / 3 * t**3 + \
        a4 / 4 * t**4
    return integ


def mcph(t0_ref, t, x=None):
    icp_r_dt_t = int_cp_durch_r_dt_minus_const(t)
    icp_r_dt_t0 = int_cp_durch_r_dt_minus_const(t0_ref)
    if not x:
        return (
            icp_r_dt_t - icp_r_dt_t0
        ) / (t - t0_ref)
    else:
        return sum(x * (
            icp_r_dt_t - icp_r_dt_t0)
        ) / sum(x) / (t - t0_ref)


def delta_h_r(t, nuij, delta_h_r_298):
    cp_m = mcph(298.15, t) * 8.3145 / 1.0e3  # kJ / mol K
    print(nuij)
    print(delta_h_r_298)
    print(cp_m)
    return delta_h_r_298 + \
        nuij.T.dot(cp_m) * (t - 298.15)  # kJ / mol K

t = 480 + 273.15 # °C
delta_h_r_t = delta_h_r(t, nuij, delta_h_r_298)
output = [
    ['formula', formula, ''],
    ['cp_298', cp_ig_durch_r(298.15) * 8.3145,
     'kJ/mol/°C'],
    ['cp_t', cp_ig_durch_r(t) * 8.3145,
     'kJ/mol/°C'],
    ['delta_h_298', delta_h_r_298, 'kJ/mol'],
    ['delta_h_t', delta_h_r_t, 'kJ/mol'],
    ['h_t', mcph(
        298.15, t
    ) * 8.3145 / 1.0e3 * (t - 298.15),
        'kJ/mol']
]

for item in output:
    col_2 = ''
    for x in item[1]:
        if isinstance(x, numbers.Number):
            col_2 = col_2 + '{:5.6g}'.format(x) + '\t'
        else:
            col_2 = col_2 + x + '\t'
    print(item[0] + '\t' + col_2 + '\t' + item[2])
