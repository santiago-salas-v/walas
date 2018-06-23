import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
import matplotlib.pyplot as plt

t0 = 352 + 273.15 # °K (335-415°C Betriebstemperatur)
t = t0
p = 1 # atm
# Ordnung für die Eigenschaften: N2, O2, Ar, Benzol
komponente = np.array(['N2', 'O2', 'Ar', 'Benzol'])
y_i = np.array([78,21,1,1.82])/sum(
    np.array([78,21,1,1.82], dtype=float))
mm_g = np.array([28, 32, 40, 78.11]) # g/mol
# IG-Eigenschaften
rho_g = 101325./(8.314*t)*mm_g/1000. # kg/m^3
# VDI Wärmeatlas - Cv bei 352°C
# Gasen bei 352°C
cv_g = np.array([
    (0.7640-0.7500)/(400-350)*(352-400)+0.7640 ,
    (0.795-0.783)/(400-350)*(352-400)+0.795 ,
    3/2*8.3145/40,
    (2.212-1.991)/(400-300)*(352-400)+2.212 ,
])
# kJ/(kg K) = J/g/K
cp_g = (8.3145+cv_g*mm_g)/mm_g # Nach Idealgasmodell
# Lennard-Jones Parameter (Bird Tabelle E.1)
l_j_epsilon_d_k = np.array([99.8,113,122.4,387.]) # K
l_j_sigma = np.array([3.667,3.433,3.432,5.443]) # Angstrom
k_t_d_e = t / l_j_epsilon_d_k
# Stoßintegral (Bird Tabelle E.2)
stossintegral_k_mu = interp1d(
    [1.60,1.65,5.0, 6.0, 7.0],
    [1.280,1.264,0.9268,0.8962,0.8727]
)(k_t_d_e)
konst_1 = 5 / 16 * np.sqrt(
    1.3806e-23 * 1000 * 100**2 / 6.022e23 / np.pi
) * (10**10 / 100)**2  # 1/cm/s
konst_2 = (9 / 4 * 8.3145 + cv_g * mm_g
          ) * 1 / 4.184 * konst_1 # cal/cm/s/K
mu = konst_1 * np.sqrt(mm_g * t) / (
    l_j_sigma**2 * stossintegral_k_mu)*100/1000.
# g/cm/s * 100cm/1000g * 1kg/m = kg/m/s = Pa s
k = konst_2 * np.sqrt(t / mm_g) / (
    l_j_sigma**2 * stossintegral_k_mu
) * 4.184 * 100 # W/m/K

def phi_alpha_beta(mm_i, mu):
    phi_ab = np.zeros([mm_i.size, mu.size])
    for alpha in range(phi_ab.shape[0]):
        for beta in range(phi_ab.shape[1]):
            phi_ab[alpha, beta] = 1/np.sqrt(8)*(
                1+mm_i[alpha]/mm_i[beta])**(-1/2.)*(
                1+(mu[alpha]/mu[beta])**(1/2.)*(
                    mm_i[beta]/mm_i[alpha]
                )**(1/4.)
            )**2
    return phi_ab

mu_mix = sum(y_i * mu / phi_alpha_beta(
    mm_g,mu).dot(y_i))
k_mix = sum(y_i * k / phi_alpha_beta(
    mm_g,k).dot(y_i))

# Eigenschaften als konstant für die Mischung angenommen
rho_g = (sum(y_i * rho_g/mm_g)*sum(y_i * mm_g)).item()
cp_g = (sum(y_i * cp_g/mm_g)*sum(y_i * mm_g)).item()
cv_g = (sum(y_i * cv_g/mm_g)*sum(y_i * mm_g)).item()
mm_g = sum(y_i * mm_g).item()
k = k_mix
mu = mu_mix
lambda_g = k_mix

output = [
    'Prozessstrom, Luft mit verdünntem o-Xylen-Anteil',
    'mm = ' + '{:g}'.format(mm_g) + ' ' + 'g/mol',
    'cv_g = ' + '{:g}'.format(cv_g) + ' ' + 'kJ/kg/K' +
    ' (VDI-Wärmeatlas)',
    'cp_g = ' + '{:g}'.format(cp_g) + ' ' + 'kJ/kg/K' +
    ' ... = (cv_g*M+R)/M Idealgas',
    'rho_g = ' + '{:g}'.format(rho_g) + ' ' + 'kg/m^3' +
    ' ... Idealgas',
    'Bird Tabelle E.1: ',
    'epsilon/k = ' + str(l_j_epsilon_d_k) + ' ' + 'K',
    'sigma = ' + str(l_j_sigma) + ' ' +  'Angstrom',
    'Bird Tabelle E.2: ',
    'Omega_mu=Omega_k = ' + str(
        stossintegral_k_mu) + ' ',
    'Bird Gl. 1.4-14, 1.4-15, 1.4-16, 9.3-13: ',
    'mu = ' + '{:g}'.format(mu) + ' ' + 'Pa s',
    'k = ' + '{:g}'.format(k) + ' ' + 'W/m/K',
    'k = lambda_g = ' + '{:g}'.format(
        k*1/4.184*60**2) + ' ' + 'kcal/m/h/°C'
]
print('\n'.join(output))

# Wasser als Kühlmittel: Gesättigte Flüssigkeit bei
# 230°C, 27,968 bar
rho_l = 827.12 # kg/m^3
cp_l = 4.68318 # kJ/kg/K
lambda_l = 636.6*1e-3 # W/m/K
eta_l = 116.2*1e-6 # Pa s
pr_l = eta_l/(lambda_l/(cp_l*1000)) # [dimlos]
d_i = 2.54*np.sqrt(2500)/2.6/np.sqrt(33*2)*31.0 / 100 #m
# Wanddicke und Wärmeleitfähigkeit: St. 35.8. (1.0305)
w_d = 0.133 * 1 / 12 * 30.48 / 100.  # m
lambda_m = (
                   (45 - 50) / (400 - 300) * (352 - 400) + 45
           ) * 1 / 4.184 / 1000. * 60 ** 2  # kcal/h/m^2/K
re_l = (
    1/(1/82.7-1/88.9339 - w_d/lambda_m
      )*1000*4.184/60**2 * \
    d_i/lambda_l/(pr_l**0.333)/0.026)**(1/0.8)
xi_l = (1.8*np.log10(re_l)-1.5)**(-2.)
nu_l = xi_l/8.*re_l*pr_l/(
    1+12.7*np.sqrt(xi_l/8.)*(pr_l**(2/3.)-1)
)*(1+(0.)**(2/3)) # d_i/l<<1
# (wesentlich höhere Länge als Durchmesser)
nu_l = 0.026*re_l**0.8*pr_l**0.333*1**0.14 # Bird
alpha_o = nu_l * lambda_l/d_i * \
    60**2 * 1/4.184 * 1/1000 # W/m/K * 1000cal/4184J *
# 60^2s/h
output = [
    'Kühlmittel: Wasser bei Sättigung bei 230°C '+
    '(28bar) (VDI-Wärmeatlas)',
    'rho_l = ' + '{:g}'.format(rho_l) + ' kg/m^3',
    'cp_l = ' + '{:g}'.format(cp_l) + ' kJ/kg/K',
    'eta_l = ' + '{:g}'.format(eta_l) + ' Pa s',
    'Pr_l = ' + '{:g}'.format(pr_l) + ' ',
    'Voll-ausgebildete turbulente Strömung:',
    'Re_l = ' + '{:g}'.format(re_l) + ' ',
    'Nusselt-Zahl bei voll-ausgebildeter turbulenter' +
    'Strömung (Gl. 26 Kap. G1 VDI-Wärmeatlas)',
    'xi_l = ' + '{:g}'.format(xi_l) + ' ',
    'Nu_l = ' + '{:g}'.format(nu_l) + ' ',
    'Bezugslänge: Innendurchmesser des Rohrbündels ' +
    'mit 2500 Rohren, je 2,54cm',
    'd_i = ' + '{:g}'.format(d_i) + ' m',
    'Wärmeübergangskoeffizient im Mantel',
    'alpha_o = '  + '{:g}'.format(alpha_o) +
    ' kcal/h/m^2/°C',
]
print('\n'.join(output))

l_r = 3  # m
d = 2.54 * 1 / 100.  # m
n = 2500  # Rohre
t = t0
dp = 3 / 1000.  # m
rho_b = 1300  # Bulk density = rhoc*(1-phi) # kgKat/m^3
ya0 = 1 / 100.  # < 1 mol%
p = 1  # atm
n_p = 1650  # t/a
g = 1650*1000./365./24./2500./(3.14/4*0.025**2)
# kg / m^2/h * 1h/(60^2 s) = kg/m^2/s
g = 4684 # kg / m^2/h * 1h/(60^2 s) = kg/m^2/s
rho_g = 1.293  # kg/m^3
u_s = g / rho_g # kg/m^2/h / kg*m^3 = m/h
delta_h_r = -307000.  # kcal/kmol
cp = 0.237  # kcal/(kg °C)
pb0 = y_i[1] * 1  # atm

re = dp * g / mu * 1/60.**2   # [=] m * kg/m^2/h /kg *m*s
# = [dimlos]
pr = mu / (lambda_g / (cp_g*4.184*1000))  # [dimlos]
# Levas Korrelation
nu = 3.50 * (re) ** 0.7 * np.exp(-4.6 * dp / d)
alpha_i = nu * lambda_g / d / 4.184 / 1000 * 60 ** 2  # W/m^2/K
# * 1cal/4.184J * 1kcal/1000cal * 60^2s/h = kcal/h/m^2/K
u = 1 / (1 / alpha_i + w_d / lambda_m + 1 / alpha_o)


def df_dy(y, z0):
    p = y[0]
    t = y[1]
    k_t = np.exp(19.837 - 13636 / t)
    # k_t[=] kmol/kgKat/h * atm^-2
    r_a = k_t * pb0 * p  # kmol/kgKat/h
    dp_dz = -mm_g * 1 * rho_b / rho_g * r_a / u_s
    dt_dz = -delta_h_r / (
            rho_g * cp
    ) * rho_b * r_a / (u_s) - 4 / d * u / (
                    rho_g * cp) / (u_s) * (t - t0)
    return np.array([dp_dz, dt_dz])


z = np.linspace(0, 3.0, 100)

pb0 = y_i[1] * 1  # atm
mm_g = np.array([28, 32, 40, 78.11])  # g/mol
mm_g = sum(y_i * mm_g).item()
p0_t0 = np.array([y_i[-1] * 1, t0])
y, info = integrate.odeint(
    df_dy, p0_t0, z, full_output=True
)

output = [
    'Prozessstrom Kennzahlen',
    'Pr = ' + '{:g}'.format(pr) + ' ',
    'Re = ' + '{:g}'.format(re) + ' ',
    'Nusselt-Zahl mit ruhenden Feststoffpartikeln\n' +
    '(Schüttschicht), nach Levas Korrelation in \n' +
    'Behr Gmehling Techn. Chemie',
    'Nu = ' + '{:g}'.format(nu) + ' ',
    'Bezugslänge: Innendurchmesser des Rohrbündels ',
    'd = ' + '{:g}'.format(d) + ' m',
    'Wärmeübergangskoeffizient im Rohr',
    'alpha_i = ' + '{:g}'.format(alpha_i) +
    ' kcal/h/m^2/°C',
    'Mittlerer Wärmeübergangskoeffizient',
    'U = ' + '{:g}'.format(u) +
    ' kcal/h/m^2/°C',
]
print('\n'.join(output))

fig = plt.figure(figsize=(20 * 12 / 30.48, 30 * 12 / 30.48))
ax1 = plt.subplot(211)
plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = plt.subplot(212, sharex=ax1)
ax1.set_ylim([0, 0.02])
ax2.set_ylim([625, 725])
ax1.set_xlim([0, 1.25])
ax1.set_ylabel('$p_0 / atm$')
ax2.set_ylabel('T / K')
ax2.set_xlabel('z / m')

for p0 in [0.011, 0.012, 0.013, 0.015,
           0.016, 0.017, 0.018, 0.0181,
           0.0182, 0.019]:
    y_i = np.array([78, 21, 1, p0 * 100]) / sum(
        np.array([78, 21, 1, p0 * 100], dtype=float))
    pb0 = y_i[1] * 1  # atm
    # mm_g = np.array([28, 32, 40, 78.11]) # g/mol
    # mm_g = sum(y_i * mm_g).item()
    p0_t0 = np.array([y_i[-1], t0])
    y, info = integrate.odeint(
        df_dy, p0_t0, z, full_output=True
    )

    ax1.plot(z, y[:, 0], label=str(p0))
    ax2.plot(z, y[:, 1], label=str(p0))
    index_max = np.argmax(y[:, 1])
    x_max = z[index_max]
    y_max = y[index_max, 1]
    ax2.annotate('$p_0=' + str(p0) + '$',
                 xy=(x_max, y_max))

ax1.legend()
ax2.legend();
plt.show()