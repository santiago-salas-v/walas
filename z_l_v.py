import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

r = 8.314 * 10.**6 / 10.**5  # bar cm^3/(mol K)
rvg = 0.8  # Rückvermischungsgrad
t_flash = 273.16 + 60  # K

# Nach unten hin: CO, H2, CO2, H2O, CH3OH, N2, CH4

# Kritische Parameter Tc, Pc, omega(azentrischer Faktor)
tc = np.array([
    132.86, 33.19, 304.13, 647.10, 513.38, 126.19
])  # K

pc = np.array([
    34.98, 13.15, 73.77, 220.64, 82.16, 33.96
])  # bar

omega_af = np.array([
    0.050, -0.219, 0.224, 0.344, 0.563, 0.037
])


def use_pr_eos():
    # PR (1976) params
    global epsilon
    global sigma
    global omega
    global psi
    global alpha

    epsilon = 1 - np.sqrt(2)
    sigma = 1 + np.sqrt(2)
    omega = 0.07780
    psi = 0.45724

    def alpha(tr, af_omega): return \
        (1 + (
            0.37464 + 1.54226 * af_omega - 0.26992 * af_omega ** 2
        ) * (1 - tr ** (1 / 2.))) ** 2


def use_srk_eos():
    # SRK (1972) params
    global epsilon
    global sigma
    global omega
    global psi
    global alpha
    epsilon = 0.
    sigma = 1.
    omega = 0.08664
    psi = 0.42748

    def alpha(tr, af_omega): return \
        (1 + (
            0.480 + 1.574 * af_omega - 0.176 * af_omega ** 2
        ) * (1 - tr ** (1 / 2.))) ** 2


def use_srk_eos_simple_alpha():
    # SRK (1972) params
    # SRK without acentric factor alpha(Tr) = Tr^(-1/2)
    global epsilon
    global sigma
    global omega
    global psi
    global alpha
    epsilon = 0.
    sigma = 1.
    omega = 0.08664
    psi = 0.42748

    def alpha(tr, af_omega): return \
        (tr) ** (-1 / 2.)


def beta(tr, pr):
    return omega * pr / tr


def q(tr, pr, af_omega):
    return psi * alpha(tr, af_omega) / (omega * tr)


def z_v_func(z, beta, q):
    return -z + 1 + beta - \
        q * beta * (z - beta) / (
            (z + epsilon * beta) * (z + sigma * beta))


def z_l_func(z, beta, q):
    return -z + beta + \
        (z + epsilon * beta) * (z + sigma * beta) * (
            (1 + beta - z) / (q * beta)
        )

# Dampfdruck nach VDI-Wärmeatlas


def p_sat_func(psat, t, af_omega, tc, pc):
    tr = t / tc
    pr = psat / pc
    beta_i = beta(tr, pr)
    q_i = q(tr, pr, af_omega)

    z_l = optimize.root(
        lambda z_l: z_l_func(
            z_l, beta_i, q_i),
        beta_i
    ).x.item()
    z_v = optimize.root(
        lambda z_v: z_v_func(z_v, beta_i, q_i),
        1.0
    ).x.item()

    i_i_l = +1 / (sigma - epsilon) * np.log(
        (z_l + sigma * beta_i) / (z_l + epsilon * beta_i)
    )
    i_i_v = +1 / (sigma - epsilon) * np.log(
        (z_v + sigma * beta_i) / (z_v + epsilon * beta_i)
    )
    ln_phi_l = + z_l - 1 - \
        np.log(z_l - beta_i) - q_i * i_i_l
    ln_phi_v = + z_v - 1 - \
        np.log(z_v - beta_i) - q_i * i_i_v
    f1 = -ln_phi_v + ln_phi_l
    return f1


def p_sat(t, af_omega, tc, pc):
    def fs(psat): return p_sat_func(psat, t, af_omega, tc, pc)
    # Das Levenberg-Marquardt Algorithmus weist wenigere Sprunge auf.
    return optimize.root(fs, 1., method='lm')


def z_non_sat(t, p, x_i, tc_i, pc_i, af_omega_i):
    tr_i = t / tc_i
    # vorsicht: Alpha oder Tr^(-1/2)
    a_i = psi * alpha(tr_i, af_omega_i) * r**2 * tc_i**2 / pc_i
    b_i = omega * r * tc_i / pc_i
    beta_i = b_i * p / (r * t)
    q_i = a_i / (b_i * r * t)
    a_ij = np.empty([len(x_i), len(x_i)])
    for i in range(len(x_i)):
        for j in range(len(x_i)):
            a_ij[i, j] = np.sqrt(a_i[i] * a_i[j])
    b = sum(x_i * b_i)
    a = 0
    for i in range(len(x_i)):
        for j in range(len(x_i)):
            a = a + x_i[i] * x_i[j] * a_ij[i, j]
    beta = b * p / (r * t)
    q = a / (b * r * t)
    s_x_j_a_ij = np.empty([len(x_i)])
    for i in range(len(x_i)):
        s_x_j_a_ij[i] = 0
        for j in range(len(x_i)):
            if i != j:
                s_x_j_a_ij[i] = s_x_j_a_ij[i] + x_i[j] * a_ij[i, j]
            elif i == j:
                # 0 summieren
                pass
    a_mp_i = -a + 2 * s_x_j_a_ij + 2 * x_i * a_i  # partielles molares a_i
    b_mp_i = b_i  # partielles molares b_i
    q_mp_i = q * (1 + a_mp_i / a - b_i / b)  # partielles molares q_i
    z = optimize.root(
        lambda z_var: z_v_func(z_var, beta, q),
        1.0).x
    i_int = 1 / (sigma - epsilon) * \
        np.log((z + sigma * beta) / (z + epsilon * beta))
    ln_phi = b_i / b * (z - 1) - np.log(z - beta) - q_mp_i * i_int
    phi = np.exp(ln_phi)

    soln = dict()
    for item in ['a_i', 'b_i', 'b', 'a', 'q',
                 'a_mp_i', 'b_mp_i', 'q_mp_i',
                 'beta', 'z', 'i_int', 'ln_phi', 'phi']:
        soln[item] = locals().get(item)
    return soln


def phi_l(t, p, x_i, tc_i, pc_i, af_omega_i):
    tr_i = t / tc_i
    a_i = psi * alpha(tr_i, af_omega_i) * r ** 2 * tc_i ** 2 / pc_i
    b_i = omega * r * tc_i / pc_i
    beta_i = b_i * p / (r * t)
    q_i = a_i / (b_i * r * t)
    a_ij = np.empty([len(x_i), len(x_i)])
    for i in range(len(x_i)):
        for j in range(len(x_i)):
            a_ij[i, j] = np.sqrt(a_i[i] * a_i[j])

    # Variablen, die von der Flüssigkeit-Zusammensetzung abhängig sind
    b_l = sum(x_i * b_i)
    a_l = 0
    for i in range(len(x_i)):
        for j in range(len(x_i)):
            a_l = a_l + x_i[i] * x_i[j] * a_ij[i, j]
    beta_l = b_l * p / (r * t)
    q_l = a_l / (b_l * r * t)
    s_x_j_a_ij = np.empty([len(x_i)])
    for i in range(len(x_i)):
        s_x_j_a_ij[i] = 0
        for j in range(len(x_i)):
            if i != j:
                s_x_j_a_ij[i] = s_x_j_a_ij[i] + x_i[j] * a_ij[i, j]
            elif i == j:
                # 0 summieren
                pass
    a_mp_i_l = -a_l + 2 * s_x_j_a_ij + 2 * x_i * a_i  # partielles molares a_i
    b_mp_i_l = b_i  # partielles molares b_i
    q_mp_i_l = q_l * (1 + a_mp_i_l / a_l - b_i / b_l)  # partielles molares q_i
    z_soln = optimize.root(
        lambda z_var: z_l_func(z_var, beta_l, q_l),
        beta_l)
    z_l = z_soln.x
    success = z_soln.success
    opt_func = z_soln.fun
    nfev = z_soln.nfev
    i_int_l = 1 / (sigma - epsilon) * \
        np.log((z_l + sigma * beta_l) / (z_l + epsilon * beta_l))
    ln_phi_l = b_i / b_l * (z_l - 1) - np.log(z_l -
                                              beta_l) - q_mp_i_l * i_int_l
    phi_l = np.exp(ln_phi_l)

    soln = dict()
    for item in ['a_i', 'b_i',
                 'b_l', 'a_l', 'q_l',
                 'a_mp_i_l', 'b_mp_i_l', 'q_mp_i_l',
                 'beta_l', 'z_l', 'i_int_l', 'ln_phi_l', 'phi_l',
                 'opt_func', 'nfev', 'success']:
        soln[item] = locals().get(item)
    return soln


def phi_v(t, p, y_i, tc_i, pc_i, af_omega_i):
    tr_i = t / tc_i
    a_i = psi * alpha(tr_i, af_omega_i) * r ** 2 * tc_i ** 2 / pc_i
    b_i = omega * r * tc_i / pc_i
    beta_i = b_i * p / (r * t)
    q_i = a_i / (b_i * r * t)
    a_ij = np.empty([len(y_i), len(y_i)])
    for i in range(len(y_i)):
        for j in range(len(y_i)):
            a_ij[i, j] = np.sqrt(a_i[i] * a_i[j])
    # Variablen, die von der Gasphase-Zusammensetzung abhängig sind
    b_v = sum(y_i * b_i)
    a_v = 0
    for i in range(len(y_i)):
        for j in range(len(y_i)):
            a_v = a_v + y_i[i] * y_i[j] * a_ij[i, j]
    beta_v = b_v * p / (r * t)
    q_v = a_v / (b_v * r * t)
    s_x_j_a_ij = np.empty([len(y_i)])
    for i in range(len(y_i)):
        s_x_j_a_ij[i] = 0
        for j in range(len(y_i)):
            if i != j:
                s_x_j_a_ij[i] = s_x_j_a_ij[i] + y_i[j] * a_ij[i, j]
            elif i == j:
                # 0 summieren
                pass
    a_mp_i_v = -a_v + 2 * s_x_j_a_ij + 2 * y_i * a_i  # partielles molares a_i
    b_mp_i_v = b_i  # partielles molares b_i
    q_mp_i_v = q_v * (1 + a_mp_i_v / a_v - b_i / b_v)  # partielles molares q_i
    z_soln = optimize.root(
        lambda z_var: z_v_func(z_var, beta_v, q_v),
        1.0)
    z_v = z_soln.x
    success = z_soln.success
    opt_func = z_soln.fun
    nfev = z_soln.nfev
    i_int_v = 1 / (sigma - epsilon) * \
        np.log((z_v + sigma * beta_v) / (z_v + epsilon * beta_v))
    ln_phi_v = b_i / b_v * (z_v - 1) - np.log(z_v -
                                              beta_v) - q_mp_i_v * i_int_v
    phi_v = np.exp(ln_phi_v)

    soln = dict()
    for item in ['a_i', 'b_i',
                 'b_v', 'a_v', 'q_v',
                 'a_mp_i_v', 'b_mp_i_v', 'q_mp_i_v',
                 'beta_v', 'z_v', 'i_int_v', 'ln_phi_v', 'phi_v',
                 'opt_func', 'nfev', 'success']:
        soln[item] = locals().get(item)
    return soln


def siedepunkt(t, p, x_i, y_i, tc_i, pc_i, af_omega_i, max_it):
    soln_l = phi_l(t, p, x_i, tc_i, pc_i, af_omega_i)
    soln_v = phi_v(t, p, y_i, tc_i, pc_i, af_omega_i)
    k_i = soln_l['phi_l'] / soln_v['phi_v']
    y_i = k_i * x_i / sum(k_i * x_i)
    i = 0
    while i < max_it and np.abs(1 - sum(y_i)) > 0:
        i = i + 1
        soln_v = phi_v(t, p, y_i, tc_i, pc_i, af_omega_i)
        k_i = soln_l['phi_l'] / soln_v['phi_v']
        y_i = k_i * x_i / sum(k_i * x_i)
        # print('i='+str(i))

    opt_func = 1 - sum(k_i * x_i)

    soln = dict()
    for item in ['soln_l', 'soln_v', 'k_i', 'y_i', 'opt_func']:
        soln[item] = locals().get(item)
    return soln


def isot_flash(t, p, x_i, y_i, z_i, tc_i, pc_i, af_omega_i):
    soln_l = phi_l(t, p, x_i, tc_i, pc_i, af_omega_i)
    soln_v = phi_v(t, p, y_i, tc_i, pc_i, af_omega_i)
    k_i = soln_l['phi_l'] / soln_v['phi_v']
    soln_v_f = optimize.root(
        lambda v_f: sum(
            z_i * (1 - k_i) / (1 + v_f * (k_i - 1))),
        0.5
    )
    v_f = soln_v_f.x
    x_i = z_i / (1 + v_f * (k_i - 1))
    y_i = x_i * k_i / sum(x_i * k_i)

    # Normalize

    x_i = x_i / sum(x_i)
    y_i = y_i / sum(y_i)
    soln = dict()
    for item in ['soln_l', 'soln_v', 'k_i', 'v_f', 'x_i', 'y_i']:
        soln[item] = locals().get(item)
    return soln


def beispiel_wdi_atlas():
    use_pr_eos()
    print(
        'ref. VDI Wärmeatlas H2 Dampfdruck um -256.6K: ' +
        '{:.4g}'.format(
            p_sat(-256.6 + 273.15, -0.216, 33.19, 13.13).x.item() * 1000
        ) + ' mbar. (Literaturwert 250mbar)'
    )
    t = 273.15 + np.linspace(-200, 500, 30)
    res = np.empty_like(t)
    success = True
    i = 0
    res_0 = 1.
    while i < len(t) and success:
        temp = t[i]

        def fs(psat): return p_sat_func(
            psat, temp, omega_af[0], tc[0], pc[0]
        )
        root = optimize.root(fs, 1.0, method='lm')
        if root.success:
            res[i] = root.x
            res_0 = res[i]
        else:
            res[i:] = np.nan
            success = False
        i += 1

    print(res)


def beispiel_svn_14_1():
    use_srk_eos_simple_alpha()
    x_i = np.array([0.4, 0.6])
    tc_i = np.array([126.2, 190.6])
    pc_i = np.array([34., 45.99])
    af_omega_i = np.array([0.038, 0.012])
    print(z_non_sat(200, 30, x_i, tc_i, pc_i, af_omega_i))


def beispiel_svn_14_2():
    use_srk_eos()
    x_i = np.array([0.2, 0.8])
    y_i = np.array([0.5, 0.5])  # Est
    tc_i = np.array([190.6, 425.1])
    pc_i = np.array([45.99, 37.96])
    af_omega_i = np.array([0.012, 0.200])
    max_it = 100
    print(siedepunkt(310.92, 30, x_i, y_i, tc_i, pc_i, af_omega_i, max_it))
    y_i = siedepunkt(
        310.92,
        30,
        x_i,
        y_i,
        tc_i,
        pc_i,
        af_omega_i,
        max_it)['y_i']
    for i in range(5):
        soln = siedepunkt(310.92, 30, x_i, y_i, tc_i, pc_i, af_omega_i, max_it)
        y_i = soln['y_i']
        k_i = soln['k_i']
        print(y_i)
        print(1 - sum(y_i))
        print(sum(k_i * x_i))
    soln = optimize.root(
        lambda p: siedepunkt(
            310.92, p, x_i, y_i, tc_i, pc_i, af_omega_i, max_it
        )['opt_func'], 1.
    )
    print(soln)
    print(siedepunkt(310.92, soln.x, x_i, y_i, tc_i, pc_i, af_omega_i, max_it))

    x = np.linspace(0.0, 0.8, 30)
    y = np.empty_like(x)
    p_v = np.empty_like(x)
    p_v0 = 1.0
    for i in range(len(x)):
        x_i = np.array([x[i], 1 - x[i]])
        soln = optimize.root(
            lambda p: siedepunkt(
                310.92, p, x_i, y_i, tc_i, pc_i, af_omega_i, max_it
            )['opt_func'], p_v0
        )
        p_v[i] = soln.x
        output = siedepunkt(
            310.92, p_v[i], x_i, y_i, tc_i, pc_i, af_omega_i, max_it
        )
        y[i] = output['y_i'][0]
        if soln.success:
            p_v0 = p_v[i]
    plt.plot(x, p_v, y, p_v)
    plt.show()


def beispiel_pat_ue_03_flash():
    use_pr_eos()
    n = np.array([
        205.66,
        14377.78,
        1489.88,
        854.75,
        1348.86,
        2496.13
    ])

    z_i = n / sum(n)
    x_i = 1 / len(n) * np.ones(len(n))
    y_i = 1 / len(n) * np.ones(len(n))
    tc_i = tc
    pc_i = pc
    af_omega_i = omega_af
    t = 60 + 273.15
    p = 50.

    for i in range(20):
        soln = isot_flash(t, p, x_i, y_i, z_i, tc_i, pc_i, af_omega_i)
        y_i = soln['y_i']
        x_i = soln['x_i']
        v_f = soln['v_f']
        k_i = soln['k_i']
        print(z_i * sum(n))
        print(y_i * sum(n) * v_f)
        print(x_i * sum(n) * (1 - v_f))
        print(v_f)


def beispiel_isot_flash_seader_4_1():
    use_pr_eos()
    n = np.array([
        10,
        20,
        30,
        40
    ])
    z_i = n / sum(n)
    x_i = 1 / len(n) * np.ones(len(n))
    y_i = 1 / len(n) * np.ones(len(n))
    tc_i = np.array([
        369.82,
        425.13,
        469.66,
        507.79
    ])
    pc_i = np.array([
        42.48,
        37.96,
        33.69,
        30.42
    ])
    af_omega_i = np.array([
        0.152,
        0.201,
        0.252,
        0.300
    ])
    t = (200 - 32) * 5 / 9 + 273.15
    p = 6.895
    for i in range(10):
        soln = isot_flash(t, p, x_i, y_i, z_i, tc_i, pc_i, af_omega_i)
        x_i = soln['x_i']
        y_i = soln['y_i']
        v_f = soln['v_f']
        print(x_i)
        print(y_i)
        print(v_f)


def beispiel_pat_ue_03_komplett():
    use_pr_eos()

    p = 50.  # bar
    temp = 273.15 + 220.  # K
    t_flash = 273.16 + 60  # K
    t0_ref = 298.15  # K
    r = 8.314  # J/(mol K)
    rvg = 0.8  # Rückvermischungsgrad
    t_flash = 273.16 + 60  # K
    p = 50.  # bar

    namen = ['CO', 'H2', 'CO2', 'H2O', 'CH3OH', 'N2']

    n0co = 750.  # kmol/h
    n0h2 = 5625.  # kmol/h
    n0co2 = 750.  # kmol/h
    n0h2o = 375.  # kmol/h
    n0ch3oh = 0.  # kmol/h
    n0n2 = 500.  # kmol/h

    ne = np.array([n0co, n0h2, n0co2, n0h2o, n0ch3oh, n0n2])

    nuij = np.array([[-1, -2, 0, 0, +1, 0],
                     [0, -3, -1, +1, +1, 0],
                     [-1, +1, +1, -1, 0, 0]]).T

    h_298 = np.array(
        [-110.541, 0., -393.505, -241.826, -201.167, 0.]) * 1000  # J/mol

    g_298 = np.array([-169.474, -38.962, -457.240, -
                      298.164, -272.667, -57.128]) * 1000  # J/mol

    # Kritische Parameter Tc, Pc, omega(azentrischer Faktor)
    tc = np.array([
        132.86, 33.19, 304.13, 647.10, 513.38, 126.19
    ])  # K

    pc = np.array([
        34.98, 13.15, 73.77, 220.64, 82.16, 33.96
    ])  # bar

    omega_af = np.array(
        [0.050, -0.219, 0.224, 0.344, 0.563, 0.037]
    )

    # Berechne delta Cp(T) mit Temperaturfunktionen für ideale Gase (SVN).

    # Koeffizienten für Cp(T)/R = A + B*T + C*T^2 + D*T^-2, T[=]K
    # Nach rechts hin: A, B, C, D
    # Nach unten hin: CO, H2, CO2, H2O, CH3OH, N2
    cp_coefs = np.array([z for z in [
        [
            y.replace(',', '.') for y in x.split('\t')
        ] for x in """
    3,3760E+00	5,5700E-04	0,0000E+00	-3,1000E+03
    3,2490E+00	4,2200E-04	0,0000E+00	8,3000E+03
    5,4570E+00	1,0450E-03	0,0000E+00	-1,1570E+05
    3,4700E+00	1,4500E-03	0,0000E+00	1,2100E+04
    2,2110E+00	1,2216E-02	-3,4500E-06	0,0000E+00
    3,2800E+00	5,9300e-04	0,0000E+00	4,0000e+03
    """.split('\n') if len(x) > 0] if len(z) > 1], dtype=float)

    def cp(t):
        return r * (
            cp_coefs[:, 0] +
            cp_coefs[:, 1] * t +
            cp_coefs[:, 2] * t ** 2 +
            cp_coefs[:, 3] * t ** -2
        )  # J/(mol K)

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

    sol_x_2 = 862.886897607255377806723117828
    sol_x_4 = 149.394609392209389397976337932
    sol_x_m_1 = 579.830273084205259692680556327
    n0 = np.array([n0co, n0h2, n0co2, n0h2o, sol_x_2, n0n2])

    # n0 = ne

    # Lösung des einfacheren Falls in schwierigerem Fall einwenden.
    def fun(x_vec):
        n2co = x_vec[0]
        n2h2 = x_vec[1]
        n2co2 = x_vec[2]
        n2h2o = x_vec[3]
        n2ch3oh = x_vec[4]
        n2n2 = x_vec[5]
        xi1 = x_vec[6]
        xi2 = x_vec[7]
        xi3 = x_vec[8]
        t2 = x_vec[9]

        # Stoffströme am Ausgang des Reaktors
        n2 = np.array([n2co, n2h2, n2co2, n2h2o, n2ch3oh, n2n2])
        # Stoffströme am Austritt des Systems (Gas)
        # n = np.array([nco, nh2, nco2, nh2o, nch3oh, nn2])
        # Stoffströme am Austritt des Systems (Flüssigkeit)
        # nl = n0 - n
        # Stoffströme im Rücklaufstrom
        # nr = rvg * n
        # Reaktionslaufzahlen
        xi = np.array([xi1, xi2, xi3])

        h_0 = h_493
        cp_0 = cp_493
        cp_t2 = cp(t2)
        h_t2 = h(t2)
        g_t2 = g(t2, h_t2)
        k_t2 = k(t2, g_t2)

        # phi_l, phi_v, k_i. Lösung des isothermischen Verdampfers
        z_i = n2 / sum(n2)
        x_i = np.array([
            2.20682E-05,
            0.000827379,
            0.005554105,
            0.420278293,
            0.572815988,
            0.000502009
        ])
        y_i = np.array([
            0.010909844,
            0.762776308,
            0.078483105,
            0.002406051,
            0.013035567,
            0.132389231,
        ])
        x_i = x_i / sum(x_i)
        y_i = y_i / sum(y_i)
        for i in range(10):
            soln = isot_flash(
                t2, p, x_i, y_i, z_i, tc, pc, omega_af
            )
            y_i = soln['y_i']
            x_i = soln['x_i']
            v_f = soln['v_f']
            k_i_verteilung = soln['k_i']

        n = n2 * v_f / (1 + rvg)
        nco = n[0]
        nh2 = n[1]
        nco2 = n[2]
        nh2o = n[3]
        nch3oh = n[4]
        nn2 = n[5]

        delta_h_t2 = nuij.T.dot(h_t2)  # J/mol

        f1 = -n2co + rvg * nco + n0co - xi1 + 0 - xi3
        f2 = -n2h2 + rvg * nh2 + n0h2 - 2 * xi1 - 3 * xi2 + xi3
        f3 = -n2co2 + rvg * nco2 + n0co2 + 0 - xi2 + xi3
        f4 = -n2h2o + rvg * nh2o + n0h2o + 0 + xi2 - xi3
        f5 = -n2ch3oh + rvg * nch3oh + n0ch3oh + xi1 + xi2 - 0
        f6 = -n2n2 + rvg * nn2 + n0n2 + 0
        f7 = -k_t2[0] * (n2co * n2h2 ** 2) + \
            n2ch3oh * (p / 1.) ** -2 * (n2) ** -(-2)
        f8 = -k_t2[1] * (n2co2 * n2h2 ** 3) + \
            n2ch3oh * n2h2o * (p / 1.) ** -2 * (n2) ** -(-2)
        f9 = -k_t2[2] * (n2co * n2h2o) + \
            n2co2 * n2h2 * (p / 1.) ** 0 * (n2) ** -0
        f10 = np.sum(
            np.multiply(n0, (h_0 - h_298)) -
            np.multiply(n, (h_t2 - h_298))) + np.dot(xi, -delta_h_t2)

        return [
            f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

    x0 = np.append(n0, [0., 0., 0., sol_x_m_1])

    sol = optimize.root(fun, x0)

    print(sol)


# beispiel_wdi_atlas()
# beispiel_svn_14_1()
# beispiel_svn_14_2()
# beispiel_pat_ue_03_flash()
beispiel_isot_flash_seader_4_1()
# beispiel_pat_ue_03_komplett()
