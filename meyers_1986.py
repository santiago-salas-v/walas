import numpy as np
from scipy import optimize

p = 0.101325  # MPa
temp = 1478.  # °K
t0_ref = 298.15  # K
r = 8.314  # J/(mol K)

namen = ['CO2', 'SO2', 'CO', 'H2O', 'COS', 'CS2', 'H2S', 'H2', 'S2']



ne = np.array([
    113.15,
    0.0,
    -81.25,
    0,
    0,
    0,
    0,
    89.1,
    18.73]) # mol

g_t =r * t * np.array([
    -32.2528,
    -20.4331,
    -19.7219,
    -13.4872,
    -18.0753,
    -1.9250,
    -1.4522,
    0.,
    0.]) # J/mol
k_t = nuij.T.dot(g_t)