from poly_3_4 import solve_cubic, solve_quartic
from numerik import secant_ls_3p, nr_ls
from poly_n import zroots
from numpy import array, append, zeros, abs, ones, empty_like, empty, argwhere, unique, asarray, concatenate, setup
from numpy import sqrt, outer, sum, log, exp, multiply, diag, sign
from numpy import linspace, dot, nan, finfo, isnan, isinf
from numpy.random import randint
from scipy import optimize
from matplotlib import pyplot as plt
import sys, re
# from setup_results_log import notify_status_func, setup_log_file

r = 8.314 * 10. ** 6 / 10. ** 5  # bar cm^3/(mol K)
rlv = 0.8  # Rücklaufverhältnis
t_flash = 273.16 + 60  # K
tol = finfo(float).eps
# setup_log_file('log_z_l_v.log', with_console=True)

# Nach unten hin: CO, H2, CO2, H2O, CH3OH, N2, CH4

# Kritische Parameter Tc, Pc, omega(azentrischer Faktor)
tc = array([132.85, 32.98, 304.12, 647.14, 512.64, 126.2, 190.56])  # K
pc = array([34.94, 12.93, 73.74, 220.64, 80.92, 34.0, 45.992])  # bar
omega_af = array([0.045, -0.217, 0.225, 0.344, 0.565, 0.037, 0.011])
ant_a = array([6.72828, 6.14858, 9.81367, 8.05573, 8.08404, 6.72531, 6.84377])
ant_b = array([295.2279, 80.948, 1340.9768, 1723.6425, 1580.4585, 285.5727, 435.4534])
ant_c = array([268.243, 277.53, 271.883, 233.08, 239.096, 270.09, 271.361])
labels = ['CO', 'H2', 'CO2', 'H2O', 'CH3OH', 'N2', 'CH4']

class State:
    def __init__(self, t, p, z_i, mm_i, tc_i, pc_i, af_omega_i, eos_name='pr',
                 max_it=100, tol=1e-10):
        self.eos = Eos(t, p, z_i, mm_i, tc_i, pc_i, af_omega_i, eos_name)
        self.v_f = 1.0
        self.z_i = z_i
        self.mm_i = mm_i
        self.p = p  # bar
        self.t = t  # K
        self.tc_i = tc_i
        self.pc_i = pc_i
        self.af_omega_i = af_omega_i
        self.max_it = 100
        self.tol = tol

    def set_t(self, t):
        self.eos.t = t
        self.solve()

    def set_p(self, p):
        self.eos.p = p
        self.solve()

    def get_z(self):
        return self.z

    def solve(self):
        self.soln = self.eos.solve()
        self.z = self.soln['z']

        # dew_point = dew_p()

        return self.soln


class Eos:
    def __init__(self, t, p, z_i, mm_i, tc_i, pc_i, af_omega_i, eos_name):
        self.eos = eos_name
        if eos_name == 'pr':
            epsilon = 1 - sqrt(2)
            sigma = 1 + sqrt(2)
            omega = 0.07780
            psi = 0.45724
            m = 0.37464 + 1.54226 * af_omega_i - 0.26992 * af_omega_i ** 2
        elif eos_name == 'srk':
            epsilon = 0.
            sigma = 1.
            omega = 0.08664
            psi = 0.42748
            m = 0.480 + 1.574 * af_omega_i - 0.176 * af_omega_i ** 2
        elif eos_name == 'srk_simple_alpha':
            epsilon = 0.
            sigma = 1.
            omega = 0.08664
            psi = 0.42748
            m = 1.0
        self.epsilon = epsilon
        self.sigma = sigma
        self.omega = omega
        self.psi = psi
        self.m = m

        self.z_i = z_i
        self.tr_i = t / tc_i
        self.pr_i = p / pc_i
        self.tc_i = tc_i
        self.pc_i = pc_i
        self.af_omega_i = af_omega_i
        self.t = t
        self.p = p

    def alpha_tr(self, tr, af_omega):
        eos = self.eos
        m = self.m
        if eos == 'pr':
            return \
                (1 + m * (1 - tr ** (1 / 2.))) ** 2
        elif eos == 'srk':
            return \
                (1 + m * (1 - tr ** (1 / 2.))) ** 2
        elif eos == 'srk_simple_alpha':
            return \
                tr ** (-1 / 2.)

    def dalphadt(self, t, tc, alpha_i, m):
        eos = self.eos
        if eos == 'pr':
            return - alpha_i ** (1/2.) * m / sqrt(t) / sqrt(tc)
        elif eos == 'srk':
            return - alpha_i ** (1/2.) * m / sqrt(t) / sqrt(tc)
        if eos == 'srk_simple_alpha':
            pass

    def q(self, tr, pr, af_omega):
        psi = self.psi
        omega = self.omega
        return psi * alpha_tr(tr, af_omega) / (omega * tr)


    def phi(self, phase):
        return phi(
            self.t, self.p, self.z_i,
            self.tc_i, self.pc_i, self.af_omega_i,
            phase, self.alpha_tr,
            self.epsilon, self.sigma, self.psi, self.omega
        )

    def solve(self):
        bubl_p_val = bubl_p(
            self.t, self.p, self.z_i, self.tc_i, self.pc_i, self.af_omega_i,
            self.alpha_tr, self.epsilon, self.sigma, self.psi, self.omega,
            self.max_it)





    def solve_old(self):
        epsilon = self.epsilon
        sigma = self.sigma
        omega = self.omega
        psi = self.psi

        omega_i = self.af_omega_i
        tc_i = self.tc_i
        pc_i = self.pc_i
        tr_i = self.tr_i
        ptr_i = self.pr_i
        t = self.t
        p = self.p
        z_i = self.z_i
        n = len(z_i)

        m_i = self.m
        alpha_i = self.alpha_tr(tr_i, m_i)
        dalphadt_i = self.dalphadt(t, tc_i, alpha_i, m_i)

        a_i = psi * alpha_i * r ** 2 * tc_i ** 2 / pc_i
        a_i_t = a_i.reshape([n, 1])
        da_idt = a_i / alpha_i * dalphadt_i
        b_i = omega * r * tc_i / pc_i
        beta_i = b_i * p / (r * t)
        q_i = a_i / (b_i * r * t)
        a_ij = sqrt(a_i_t.dot(a_i_t.T))

        # Variablen, die von der Phasen-Zusammensetzung abhängig sind
        b = sum(z_i * b_i)
        a = z_i.dot(a_ij.dot(z_i))

        beta = b * p / (r * t)
        q = a / (b * r * t)
        s_x_j_a_ij = a_ij.dot(z_i)
        mat_1 = 1/2 * diag(a_i).dot(1/sqrt(a_i_t)).dot(
            diag(da_idt).dot(1/sqrt(a_i_t)).T
        )
        da_ijdt = mat_1 + mat_1.T
        da_dt = z_i.dot(da_ijdt.dot(z_i))

        a_mp_i = -a + 2 * s_x_j_a_ij + 2 * z_i * a_i  # partielles molares a_i
        b_mp_i = b_i  # partielles molares b_i
        q_mp_i = q * (1 + a_mp_i / a - b_i / b)  # partielles molares q_i

        a1 = 1.0
        a2 = beta * (epsilon + sigma) - beta - 1
        a3 = q * beta + epsilon * sigma * beta ** 2 \
               - beta * (epsilon + sigma) * (1 + beta)
        a4 = -(epsilon * sigma * beta ** 2 * (1 + beta) +
                 q * beta ** 2)

        soln = solve_cubic([a1, a2, a3, a4])
        roots_z = array(soln['roots'])
        disc = soln['disc']

        if disc <= 0:
            # 2 phases Region
            # 3 real roots. smallest ist liq. largest is gas.
            z_l = roots_z[-1][0]
            z_v = roots_z[0][0]
            phasen = 'L,V'


        elif disc > 0:
            # 1 phase region
            # one real root, 2 complex. First root is the real one.
            z = roots_z[0][0]
            v = z * r * t / p
            dp_dt = r / (v - b) - da_dt * 1 / (
                    (v + epsilon * b)*(v + sigma * b)
            )
            dp_dv = - r * t / (v - b) ** 2 + a * 1 / (
                    (v + epsilon * b) * (v + sigma * b)
            ) * (1 / (v + epsilon * b) + 1 / (v + sigma * b))
            d2p_dvdt = - r/(v - b)**2 + da_dt * 1 / (
                (v + epsilon * b) * (v + sigma * b)
            ) * (1 / (v + epsilon * b) + 1 / (v + sigma * b))
            d2p_dv2 = - 2 * r * t / (v - b)**3 - 2 * a * 1 / (
                (v + epsilon * b) * (v + sigma * b)
            ) * (1 / (v + epsilon * b)**2 + 1 / (
                    (v + epsilon * b) * (v + sigma * b)
            ) + 1 / (v + sigma * b)**2
            )

            # Phase parameter :
            # Ref. Fluid Phase Equilibria 301 (2011) 225–233
            pi_ph = v * (
                    d2p_dvdt / dp_dt - d2p_dv2 / dp_dv
            )

            if pi_ph > 1:
                # liquid or liquid-like vapor
                v_f = 0.0
                phasen = 'L'
            elif pi_ph <= 1:
                # vapor
                v_f = 1.0
                phasen = 'V'

        result = dict()
        for item in [
            'v_f', 'phasen', 'z', 'a', 'b',
            'phi', 'ln_phi', 'i_int'
        ]:
            result[item] = locals().get(item)

        return result





def use_pr_eos():
    # PR (1976) params
    global epsilon
    global sigma
    global omega
    global psi
    global alpha_tr

    epsilon = 1 - sqrt(2)
    sigma = 1 + sqrt(2)
    omega = 0.07780
    psi = 0.45724

    def alpha_tr(tr, af_omega): return \
        (1 + (
            0.37464 + 1.54226 * af_omega - 0.26992 * af_omega ** 2
        ) * (1 - tr ** (1 / 2.))) ** 2


    return alpha_tr, epsilon, sigma, psi, omega


def use_srk_eos():
    # SRK (1972) params
    global epsilon
    global sigma
    global omega
    global psi
    global alpha_tr
    epsilon = 0.
    sigma = 1.
    omega = 0.08664
    psi = 0.42748

    def alpha_tr(tr, af_omega): return \
        (1 + (
            0.480 + 1.574 * af_omega - 0.176 * af_omega ** 2
        ) * (1 - tr ** (1 / 2.))) ** 2


    return alpha_tr, epsilon, sigma, psi, omega


def use_srk_eos_simple_alpha():
    # SRK (1972) params
    # SRK without acentric factor alpha(Tr) = Tr^(-1/2)
    global epsilon
    global sigma
    global omega
    global psi
    global alpha_tr
    epsilon = 0.
    sigma = 1.
    omega = 0.08664
    psi = 0.42748

    def alpha_tr(tr, af_omega): return \
        (tr) ** (-1 / 2.)


    return alpha_tr, epsilon, sigma, psi, omega


def beta(tr, pr):
    return omega * pr / tr


def q(tr, pr, af_omega):
    return psi * alpha_tr(tr, af_omega) / (omega * tr)


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


def p_sat_func(psat, t, af_omega, tc, pc, full_output=False):
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

    i_i_l = +1 / (sigma - epsilon) * log(
        (z_l + sigma * beta_i) / (z_l + epsilon * beta_i)
    )
    i_i_v = +1 / (sigma - epsilon) * log(
        (z_v + sigma * beta_i) / (z_v + epsilon * beta_i)
    )
    ln_phi_l = + z_l - 1 - \
        log(z_l - beta_i) - q_i * i_i_l
    ln_phi_v = + z_v - 1 - \
        log(z_v - beta_i) - q_i * i_i_v
    f1 = -ln_phi_v + ln_phi_l
    if full_output:
        opt_func = f1
        phi_l = exp(ln_phi_l)
        phi_v = exp(ln_phi_v)
        soln = dict()
        for item in ['z_l', 'z_v', 'phi_l', 'phi_v', 'opt_func']:
            soln[item] = locals().get(item)
        return soln
    else:
        return f1


def p_sat(t, af_omega, tc, pc):
    def fs(psat): return p_sat_func(psat, t, af_omega, tc, pc)

    # Das Levenberg-Marquardt Algorithmus weist wenigere Sprunge auf.
    return optimize.root(fs, 1., method='lm')


def p_i_sat_ceos(t, p, tc_i, pc_i, af_omega_i,
                 alpha_tr, epsilon, sigma, psi, omega,
                 max_it=100, tol=tol):
    n_comps = asarray(tc_i).size
    p_sat_list = empty(n_comps)
    success = empty(n_comps, dtype=bool)
    n_fev = empty(n_comps, dtype=int)
    zero_fun = empty(n_comps)
    z_l = empty(n_comps)
    z_v = empty(n_comps)
    phi_l = empty(n_comps)
    phi_v = empty(n_comps)
    for i in range(n_comps):
        if n_comps > 1:
            tc, pc, af_omega = tc_i[i], pc_i[i], af_omega_i[i]
        else:
            tc, pc, af_omega = tc_i, pc_i, af_omega_i
        if tc < t:
            # no p_sat if supercritical
            success[i] = False
        else:
            # approach saturation if possible
            first_step_soln = approach_pt_i_sat_ceos(
                t, p, tc, pc, af_omega, alpha_tr, epsilon, sigma, psi, omega,
                p_or_t='p', max_it=max_it, tol=tol)
            success[i] = first_step_soln['success']
        if not success[i]:
            p_sat_list[i] = nan
        else:
            p0_it = first_step_soln['p']
            inv_slope = 1 / first_step_soln['ddisc_dp']
            p1_it = p0_it * (1 + sign(-inv_slope) * 0.01)
            if first_step_soln['n_fev'] == 1:
                # initial slope not necessarily convergent, direct it toward descending disc
                disc_0 = first_step_soln['disc']
                disc_1 = approach_pt_i_sat_ceos(
                    t, p1_it, tc, pc, af_omega, alpha_tr, epsilon, sigma, psi, omega,
                    p_or_t='p', max_it=max_it, tol=tol)['disc']
                inv_slope = -(p1_it - p0_it) / (disc_1 - disc_0)
                p1_it = p0_it * (1 + sign(-inv_slope) * 0.001)
            soln_temp = secant_ls_3p(
                lambda p_var: phi_sat_ceos(
                    t, p_var, tc, pc, af_omega,
                    alpha_tr, epsilon, sigma, psi, omega)['zero_fun'],
                p0_it, tol, x_1=p1_it,
                restriction=lambda p_var: p_var > 0 and phi_sat_ceos(
                    t, p_var, tc, pc, af_omega, alpha_tr, epsilon,
                    sigma, psi, omega)['success'],
                print_iterations=False
            )
            n_fev[i] = soln_temp['iterations'] + soln_temp['total_backtracks']
            p_sat_list[i] = soln_temp['x']
            soln_temp = phi_sat_ceos(t, p_sat_list[i], tc, pc, af_omega,
                                     alpha_tr, epsilon, sigma, psi, omega)
            success[i] = soln_temp['success']
            zero_fun[i] = soln_temp['zero_fun']
            z_l[i] = soln_temp['z_l']
            z_v[i] = soln_temp['z_v']
            phi_l[i] = soln_temp['phi_l']
            phi_v[i] = soln_temp['phi_v']

    p = p_sat_list
    soln = dict()
    for item in ['p', 't', 'success', 'n_fev', 'zero_fun', 'z_l', 'z_v', 'phi_l', 'phi_v']:
        soln[item] = locals().get(item)
    return soln


def t_i_sat_ceos(t, p, tc_i, pc_i, af_omega_i,
                 alpha_tr, epsilon, sigma, psi, omega,
                 max_it=100, tol=tol):
    n_comps = asarray(tc_i).size
    t_sat_list = empty(n_comps)
    success = empty(n_comps, dtype=bool)
    n_fev = empty(n_comps, dtype=int)
    zero_fun = empty(n_comps)
    z_l = empty(n_comps)
    z_v = empty(n_comps)
    phi_l = empty(n_comps)
    phi_v = empty(n_comps)
    for i in range(n_comps):
        if n_comps > 1:
            tc, pc, af_omega = tc_i[i], pc_i[i], af_omega_i[i]
        else:
            tc, pc, af_omega = tc_i, pc_i, af_omega_i
        # approach saturation if possible
        first_step_soln = approach_pt_i_sat_ceos(
            t, p, tc, pc, af_omega, alpha_tr, epsilon, sigma, psi, omega,
            p_or_t='t', max_it=max_it, tol=tol)
        if pc < p or not first_step_soln['success']:
            # no p_sat if supercritical
            t_sat_list[i] = nan
        else:
            t0_it = first_step_soln['t']
            inv_slope = 1 / first_step_soln['ddisc_dt']
            t1_it = t0_it * (1 + sign(-inv_slope) * 0.001)
            if first_step_soln['n_fev'] == 1:
                # initial slope not necessarily convergent, direct it toward descending disc
                disc_0 = first_step_soln['disc']
                disc_1 = approach_pt_i_sat_ceos(
                    t1_it, p, tc, pc, af_omega, alpha_tr, epsilon, sigma, psi, omega,
                    p_or_t='t', max_it=max_it, tol=tol)['disc']
                inv_slope = -(t1_it - t0_it) / (disc_1 - disc_0)
                t1_it = t0_it * (1 + sign(-inv_slope) * 0.001)
            soln_temp = secant_ls_3p(
                lambda t_var: phi_sat_ceos(
                    t_var, p, tc, pc, af_omega,
                    alpha_tr, epsilon, sigma, psi, omega)['zero_fun'],
                t0_it, tol, x_1=t1_it,
                restriction=lambda t_var: t_var > 0 and phi_sat_ceos(
                    t_var, p, tc, pc, af_omega, alpha_tr, epsilon,
                    sigma, psi, omega)['success'],
                print_iterations=False
            )
            n_fev[i] = soln_temp['iterations'] + soln_temp['total_backtracks']
            t_sat_list[i] = soln_temp['x']
            soln_temp = phi_sat_ceos(t_sat_list[i], p, tc, pc, af_omega,
                                     alpha_tr, epsilon, sigma, psi, omega)
            success[i] = soln_temp['success']
            zero_fun[i] = soln_temp['zero_fun']
            z_l[i] = soln_temp['z_l']
            z_v[i] = soln_temp['z_v']
            phi_l[i] = soln_temp['phi_l']
            phi_v[i] = soln_temp['phi_v']

    t = t_sat_list
    soln = dict()
    for item in ['p', 't', 'success', 'n_fev', 'zero_fun', 'z_l', 'z_v', 'phi_l', 'phi_v']:
        soln[item] = locals().get(item)
    return soln


def approach_pt_i_sat_ceos(t, p, tc_i, pc_i, af_omega_i,
                           alpha_tr, epsilon, sigma, psi, omega,
                           p_or_t='p', max_it=100, tol=tol):
    b_i = omega * r * tc_i / pc_i
    n_fev = 0
    it_outer = 0
    while n_fev in range(max_it):
        n_fev += 1
        tr_i = t / tc_i
        a_i = psi * alpha_tr(tr_i, af_omega_i) * r ** 2 * tc_i ** 2 / pc_i
        beta = b_i * p / (r * t)
        q = a_i / (b_i * r * t)
        a1 = beta * (epsilon + sigma) - beta - 1
        a2 = q * beta + epsilon * sigma * beta ** 2 \
             - beta * (epsilon + sigma) * (1 + beta)
        a3 = -(epsilon * sigma * beta ** 2 * (1 + beta) +
               q * beta ** 2)
        # partial derivative of discriminant with respect to p
        da1_dp = 1 / p * beta * (epsilon + sigma - 1)
        da2_dp = 1 / p * (q * beta + 2 * epsilon * sigma * beta ** 2
                          - beta * (epsilon + sigma) * (1 + 2 * beta))
        da3_dp = 1 / p * (-(epsilon * sigma * beta ** 2 * (2 + 3 * beta) +
                            2 * q * beta ** 2))
        disc = 1 / 27 * (- 1 / 3 * a1 ** 2 + a2) ** 3 + \
               1 / 4 * (2 / 27 * a1 ** 3 - 1 / 3 * a1 * a2 + a3) ** 2
        ddisc_dp = 1 / 9 * (- 1 / 3 * a1 ** 2 + a2) ** 2 * (
                -2 / 3 * a1 * da1_dp + da2_dp) + \
                   1 / 2 * (2 / 27 * a1 ** 3 - 1 / 3 * a1 * a2 + a3) * (
                           2 / 9 * a1 ** 2 * da1_dp
                           - 1 / 3 * (a1 * da2_dp + a2 * da1_dp) + da3_dp)
        # partial derivative of discriminant with respect to t
        da1_dt = -1 / t * beta * (epsilon + sigma - 1)
        da2_dt = -2 / t * (q * beta + epsilon * sigma * beta ** 2
                           - beta * (epsilon + sigma) * (1 / 2 + beta))
        da3_dt = -1 / t * (-(epsilon * sigma * beta ** 2 * (2 + 3 * beta) +
                             3 * q * beta ** 2))
        ddisc_dt = 1 / 9 * (- 1 / 3 * a1 ** 2 + a2) ** 2 * (
                -2 / 3 * a1 * da1_dt + da2_dt) + \
                   1 / 2 * (2 / 27 * a1 ** 3 - 1 / 3 * a1 * a2 + a3) * (
                           2 / 9 * a1 ** 2 * da1_dt
                           - 1 / 3 * (a1 * da2_dt + a2 * da1_dt) + da3_dt)
        success = disc < -tol
        if success:
            break
        f = disc + tol
        if p_or_t == 'p':
            df_dp = ddisc_dp
            inv_slope = 1 / df_dp
            p_old = p
            p = p_old - inv_slope * f
        elif p_or_t == 't':
            df_dt = ddisc_dt
            inv_slope = 1 / df_dt
            t_old = t
            t = t_old - inv_slope * f
    soln = dict()
    for item in ['p', 't', 'success', 'n_fev', 'ddisc_dt', 'ddisc_dp', 'f', 'disc']:
        soln[item] = locals().get(item)
    return soln


def phi_sat_ceos(t, p, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega):
    tr_i = t / tc_i
    a_i = psi * alpha_tr(tr_i, af_omega_i) * r ** 2 * tc_i ** 2 / pc_i
    b_i = omega * r * tc_i / pc_i
    beta = b_i * p / (r * t)
    q = a_i / (b_i * r * t)
    a1 = beta * (epsilon + sigma) - beta - 1
    a2 = q * beta + epsilon * sigma * beta ** 2 \
         - beta * (epsilon + sigma) * (1 + beta)
    a3 = -(epsilon * sigma * beta ** 2 * (1 + beta) +
           q * beta ** 2)

    soln = solve_cubic([1, a1, a2, a3])
    success = soln['disc'] <= 0
    z_l = soln['roots'][-1][0]
    z_v = soln['roots'][0][0]

    i_i_l = +1 / (sigma - epsilon) * log(
        (z_l + sigma * beta) / (z_l + epsilon * beta)
    )
    i_i_v = +1 / (sigma - epsilon) * log(
        (z_v + sigma * beta) / (z_v + epsilon * beta)
    )
    ln_phi_l = + z_l - 1 - \
               log(z_l - beta) - q * i_i_l
    ln_phi_v = + z_v - 1 - \
               log(z_v - beta) - q * i_i_v
    phi_l = exp(ln_phi_l)
    phi_v = exp(ln_phi_v)
    zero_fun = -ln_phi_v + ln_phi_l

    soln = dict()
    for item in ['z_l', 'z_v', 'phi_l', 'phi_v', 'success', 'zero_fun']:
        soln[item] = locals().get(item)
    return soln


def z_non_sat(t, p, x_i, tc_i, pc_i, af_omega_i):
    tr_i = t / tc_i
    # vorsicht: Alpha oder Tr^(-1/2)
    a_i = psi * alpha_tr(tr_i, af_omega_i) * r ** 2 * tc_i ** 2 / pc_i
    b_i = omega * r * tc_i / pc_i
    beta_i = b_i * p / (r * t)
    q_i = a_i / (b_i * r * t)
    a_ij = sqrt(outer(a_i, a_i))
    b = sum(x_i * b_i)
    a = sum(outer(x_i, x_i) * a_ij)
    beta = b * p / (r * t)
    q = a / (b * r * t)
    s_x_j_a_ij = a_ij.dot(x_i) - diag(diag(a_ij)).dot(x_i)
    a_mp_i = -a + 2 * s_x_j_a_ij + 2 * x_i * a_i  # partielles molares a_i
    b_mp_i = b_i  # partielles molares b_i
    q_mp_i = q * (1 + a_mp_i / a - b_i / b)  # partielles molares q_i
    z = optimize.root(
        lambda z_var: z_v_func(z_var, beta, q),
        1.0).x
    i_int = 1 / (sigma - epsilon) * \
        log((z + sigma * beta) / (z + epsilon * beta))
    ln_phi = b_i / b * (z - 1) - log(z - beta) - q_mp_i * i_int
    phi = exp(ln_phi)

    soln = dict()
    for item in ['a_i', 'b_i', 'b', 'a', 'q',
                 'a_mp_i', 'b_mp_i', 'q_mp_i',
                 'beta', 'z', 'i_int', 'ln_phi', 'phi']:
        soln[item] = locals().get(item)
    return soln

def phi(t, p, z_i, tc_i, pc_i, af_omega_i, phase, alpha_tr, epsilon, sigma, psi, omega):
    z_i = asarray(z_i)
    tr_i = t / tc_i
    a_i = psi * alpha_tr(tr_i, af_omega_i) * r ** 2 * tc_i ** 2 / pc_i
    b_i = omega * r * tc_i / pc_i
    beta_i = b_i * p / (r * t)
    q_i = a_i / (b_i * r * t)
    a_ij = sqrt(outer(a_i, a_i))

    # composition-dependent variables
    b = sum(z_i * b_i)
    a = z_i.dot(a_ij).dot(z_i).item()
    beta = b * p / (r * t)
    q = a / (b * r * t)
    a_mp_i = -a + 2 * a_ij.dot(z_i)  # partielles molares a_i
    b_mp_i = b_i  # partielles molares b_i
    q_mp_i = q * (1 + a_mp_i / a - b_i / b)  # partielles molares q_i

    phase_soln = z_phase(t, p, z_i, tc_i, pc_i, af_omega_i, phase,
                         alpha_tr, epsilon, sigma, psi, omega, tol)
    z = phase_soln['z']
    if epsilon != sigma:
        # vdw: epsilon = sigma
        i_int = 1 / (sigma - epsilon) * \
            log((z + sigma * beta) / (z + epsilon * beta))
    elif epsilon == sigma:
        # only vdw
        i_int = beta / (z + epsilon * beta)
    if z >= 0:
        # $G^R/(RT) = Z - 1 - ln(1-\rho b) - ln(Z) - q I$
        # and $\beta = \rho b Z$
        ln_phi = b_i / b * (z - 1) - log(z - beta) - q_mp_i * i_int
    else:
        # $G^R/(RT) = Z - 1 - ln(1-\rho b) + ln(-Z) - q I$
        # and $\beta = \rho b Z$
        ln_phi = b_i / b * (z - 1) - log(beta / z**2 - 1 / z) - q_mp_i * i_int
    phi_calc = exp(ln_phi)
    if phase == 'l':
        # correction for pseudoproperties in phi (Matthias et al. 1984)
        v = phase_soln['v']
        p_calc = r * t / (v - b) - a / ((v + epsilon * b)*(v + sigma * b))
        phi = phi_calc * p_calc / p
        ln_phi = log(phi)
    else:
        phi = phi_calc

    soln = dict()
    for item in ['a_i', 'b_i',
                 'b', 'a', 'q',
                 'a_mp_i', 'b_mp_i', 'q_mp_i',
                 'beta', 'z', 'i_int', 'ln_phi', 'phi']:
        soln[item] = locals().get(item)
    return soln


def bubl_p(t, p, x_i, tc_i, pc_i, af_omega_i,
           alpha_tr, epsilon, sigma, psi, omega,
           sec_j=None, nu_ij=None, unifac_data_dict=None,
           max_it=100, tol=tol, y_i_est=None, print_iterations=False):
    """Bubble pressure determination

        Determines bubble pressure at given temperature and liquid composition. Lee-Kessler method.
        If sec_j and nu_ij are provided, gamma-phi approach is used, otherwise phi-phi.

        ref. Smith, Joseph Mauk ; Ness, Hendrick C. Van ; Abbott, Michael M.: \
        Introduction to chemical engineering thermodynamics. New York: McGraw-Hill, 2005.\
        (Fig. 14.9)

        ref. e.V., VDI: VDI-Wärmeatlas. Wiesbaden: Springer Berlin Heidelberg, 2013. \
        (D5.1. Abb.6.)

        :param t: temperature (known) / K
        :param p: pressure (estimate) / bar
        :param x_i: liquid composition (known)
        :param tc_i: critical temperatures / K
        :param pc_i: critical pressures / bar
        :param af_omega_i: Pitzer-factors
        :param alpha_tr: CEOS alpha(Tr_i, af_omega_i) function
        :param epsilon: CEOS epsilon
        :param sigma: CEOS sigma
        :param psi: CEOS psi
        :param omega: CEOS omega
        :param sec_j: unifac secondary group / subgroup
        :param nu_ij: coefficients in unifac subgroups per component
        :param max_it: maximum iterations
        :param tol: tolerance of method
        :param y_i_est: optional vapor composition (estimate)
        :param print_iterations: optionally print Lee-Kessler iterations
        :return: dictionary with 'x_i', 'y_i', 'phi_l', 'phi_v', 'k_i', 'sum_ki_xi', 'p', 'success'
        """
    if sec_j is None:
        def obj_fun(p_var): return bubl_point_step_l_k(
        t, p_var, x_i, tc_i, pc_i, af_omega_i,
        alpha_tr, epsilon, sigma, psi, omega,
        max_it, tol=tol, y_i_est=y_i_est)
        soln = secant_ls_3p(obj_fun, p, tol=tol, x_1=1.001 * p,
                            restriction=lambda p_val: p_val > 0,
                            print_iterations=print_iterations)
        p = soln['x']
        soln = bubl_point_step_l_k(t, p, x_i, tc_i, pc_i, af_omega_i,
                                   alpha_tr, epsilon, sigma, psi, omega, max_it,
                                   full_output=True, y_i_est=y_i_est)
    else:
        soln = bubl_p_gamma_phi(
            t, p, x_i, tc_i, pc_i, af_omega_i,
            alpha_tr, epsilon, sigma, psi, omega,
            sec_j, nu_ij, unifac_data_dict,
            max_it, tol=tol, y_i_est=y_i_est)
    return soln


def bubl_t(t, p, x_i, tc_i, pc_i, af_omega_i,
           alpha_tr, epsilon, sigma, psi, omega,
           sec_j=None, nu_ij=None, unifac_data_dict=None,
           max_it=100, tol=tol, y_i_est=None, print_iterations=False):
    """Bubble temperature determination

        Determines bubble temperature at given pressure and liquid composition. Lee-Kessler method.
        If sec_j and nu_ij are provided, gamma-phi approach is used, otherwise phi-phi.

        ref. Smith, Joseph Mauk ; Ness, Hendrick C. Van ; Abbott, Michael M.: \
        Introduction to chemical engineering thermodynamics. New York: McGraw-Hill, 2005.\
        (Fig. 14.5 gama-phi; 14.9 phi-phi)

        ref. e.V., VDI: VDI-Wärmeatlas. Wiesbaden: Springer Berlin Heidelberg, 2013. \
        (D5.1. Abb.6.)

        :param t: temperature (estimate) / K
        :param p: pressure (known) / bar
        :param x_i: liquid composition (known)
        :param tc_i: critical temperatures / K
        :param pc_i: critical pressures / bar
        :param af_omega_i: Pitzer-factors
        :param alpha_tr: CEOS alpha(Tr_i, af_omega_i) function
        :param epsilon: CEOS epsilon
        :param sigma: CEOS sigma
        :param psi: CEOS psi
        :param omega: CEOS omega
        :param sec_j: unifac secondary group / subgroup
        :param nu_ij: coefficients in unifac subgroups per component
        :param max_it: maximum iterations
        :param tol: tolerance of method
        :param y_i_est: optional vapor composition (estimate)
        :param print_iterations: optionally print Lee-Kessler iterations
        :return: dictionary with 'x_i', 'y_i', 'phi_l', 'phi_v', 'k_i', 'sum_ki_xi', 'p', 'success'
        """
    if sec_j is None:
        def obj_fun(t_var): return bubl_point_step_l_k(
            t_var, p, x_i, tc_i, pc_i, af_omega_i,
            alpha_tr, epsilon, sigma, psi, omega,
            max_it, tol=tol, y_i_est=y_i_est)
        soln = secant_ls_3p(obj_fun, t, tol=tol, x_1=1.001 * t,
                            restriction=lambda t_val: t_val > 0,
                            print_iterations=print_iterations)
        t = soln['x']
        soln = bubl_point_step_l_k(t, p, x_i, tc_i, pc_i, af_omega_i,
                                   alpha_tr, epsilon, sigma, psi, omega, max_it,
                                   full_output=True, y_i_est=y_i_est)
    else:
        soln = bubl_t_gamma_phi(
            t, p, x_i, tc_i, pc_i, af_omega_i,
            alpha_tr, epsilon, sigma, psi, omega,
            sec_j, nu_ij, unifac_data_dict,
            max_it, tol=tol, y_i_est=y_i_est)
    return soln


def bubl_point_step_l_k(t, p, x_i, tc_i, pc_i, af_omega_i,
                        alpha_tr, epsilon, sigma, psi, omega,
                        max_it, tol=tol, full_output=False, y_i_est=None):
    soln_l = phi(t, p, x_i, tc_i, pc_i, af_omega_i, 'l',
                alpha_tr, epsilon, sigma, psi, omega)
    phi_l = soln_l['phi']
    if y_i_est is not None:
        y_i_est = asarray(y_i_est)
        soln_v = phi(t, p, y_i_est, tc_i, pc_i, af_omega_i, 'v',
                    alpha_tr, epsilon, sigma, psi, omega)
        phi_v = soln_v['phi']
    else:
        soln_v = phi(t, p, x_i, tc_i, pc_i, af_omega_i, 'v',
                    alpha_tr, epsilon, sigma, psi, omega)
        phi_v = soln_v['phi']
    k_i = phi_l / phi_v
    sum_ki_xi = sum(k_i * x_i)
    y_i = k_i * x_i / sum_ki_xi
    stop = False
    i = 0
    success = True
    while not stop:
        sum_ki_xi_k_minus_1 = sum_ki_xi
        soln_v = phi(t, p, y_i, tc_i, pc_i, af_omega_i, 'v',
                    alpha_tr, epsilon, sigma, psi, omega)
        phi_v = soln_v['phi']
        k_i = phi_l / phi_v
        sum_ki_xi = sum(k_i * x_i)
        y_i = k_i * x_i / sum_ki_xi
        if abs((sum_ki_xi - sum_ki_xi_k_minus_1) /
               sum_ki_xi_k_minus_1) <= tol:
            stop = True
        i += 1
        if i >= max_it:
            stop = True
            success = False
    if full_output:
        z_l = soln_l['z']
        z_v = soln_v['z']
        soln = dict()
        for item in ['x_i', 'y_i', 'phi_l', 'phi_v', 'z_l', 'z_v', 'k_i', 'sum_ki_xi', 'p', 't', 'success']:
            soln[item] = locals().get(item)
        return soln
    else:
        return 1 - sum_ki_xi


def bubl_t_gamma_phi(t, p, x_i, tc_i, pc_i, af_omega_i,
                        alpha_tr, epsilon, sigma, psi, omega,
                        sec_j, nu_ij, unifac_data_dict,
                        max_it, tol=tol, full_output=False, y_i_est=None):
    x_i = asarray(x_i)
    soln = t_i_sat_ceos(t, p, tc_i, pc_i, af_omega_i,
                        alpha_tr, epsilon, sigma, psi, omega, max_it=max_it, tol=tol)
    t_i_sat = soln['t']
    t = sum(x_i * t_i_sat)
    gamma_i = gamma_u(t, x_i, sec_j, nu_ij, unifac_data_dict)
    phi_coef_fun_i = ones(x_i.size)
    soln = p_i_sat_ceos(t, p, tc_i, pc_i, af_omega_i,
                        alpha_tr, epsilon, sigma, psi, omega, max_it=max_it, tol=tol)
    p_i_sat = soln['p']

    for n_it in range(max_it):
        # find closest to mean value, use as key component j for vapor pressure ratios
        j = abs(p_i_sat - p_i_sat.mean()).argmin()
        p_j_sat = p_i_sat[j]
        p_j_sat = p / sum(x_i * gamma_i / phi_coef_fun_i * p_i_sat / p_j_sat)
        # with sat. pressure of key component determine sat. temperature
        t_old = t
        t = t_i_sat_ceos(
            t, p_j_sat, tc_i[j], pc_i[j], af_omega_i[j],
            alpha_tr, epsilon, sigma, psi, omega, max_it=max_it, tol=tol
        )['t'].item()

        delta_t = t - t_old
        success = abs(delta_t) <= tol
        if success:
            break

        soln = p_i_sat_ceos(t, p, tc_i, pc_i, af_omega_i,
                            alpha_tr, epsilon, sigma, psi, omega, max_it=max_it, tol=tol)
        p_i_sat = soln['p']
        phi_i_sat = soln['phi_v']
        z_l_i = soln['z_l']
        v_l_i = z_l_i * r * t / p
        y_i = x_i * gamma_i * p_i_sat / phi_coef_fun_i / p
        poynting_i = exp(-v_l_i * (p - p_i_sat) / (r * t))
        phi_i_v = phi(t, p, y_i, tc_i, pc_i, af_omega_i, 'v',
                      alpha_tr, epsilon, sigma, psi, omega)['phi']

        gamma_i = gamma_u(t, x_i, sec_j, nu_ij, unifac_data_dict)
        phi_coef_fun_i = phi_i_v / phi_i_sat * poynting_i

    k_i = gamma_i * p_i_sat / phi_coef_fun_i / p
    phi_v = phi_i_v
    soln = dict()
    for item in ['t', 'p', 'success', 'n_it', 'gamma_i', 'phi_coef_fun_i',
                 'phi_v', 'phi_i_sat', 'p_i_sat', 'z_l_i', 'v_l_i', 'poynting_i',
                 'y_i', 'x_i', 'k_i']:
        soln[item] = locals().get(item)
    return soln


def bubl_p_gamma_phi(t, p, x_i, tc_i, pc_i, af_omega_i,
                        alpha_tr, epsilon, sigma, psi, omega,
                        sec_j, nu_ij, unifac_data_dict,
                        max_it, tol=tol, full_output=False, y_i_est=None):
    x_i = asarray(x_i)
    gamma_i = gamma_u(t, x_i, sec_j, nu_ij, unifac_data_dict)
    soln = p_i_sat_ceos(t, p, tc_i, pc_i, af_omega_i,
                        alpha_tr, epsilon, sigma, psi, omega, max_it=max_it, tol=tol)
    p_i_sat = soln['p']
    phi_i_sat = soln['phi_v']
    z_l_i = soln['z_l']

    phi_coef_fun_i = ones(x_i.size)

    for n_it in range(max_it):
        p_old = p
        p = sum(x_i * gamma_i * p_i_sat / phi_coef_fun_i)
        delta_p = p - p_old
        y_i = x_i * gamma_i * p_i_sat / phi_coef_fun_i / p
        y_i = y_i / sum(y_i)
        phi_i_v = phi(t, p, y_i, tc_i, pc_i, af_omega_i, 'v',
                  alpha_tr, epsilon, sigma, psi, omega)['phi']
        v_l_i = z_l_i * r * t / p
        poynting_i = exp(-v_l_i * (p - p_i_sat) / (r * t))
        phi_coef_fun_i = phi_i_v / phi_i_sat * poynting_i
        success = abs(delta_p) <= tol
        if success:
            break

    k_i = gamma_i * p_i_sat / phi_coef_fun_i / p
    phi_v = phi_i_v
    soln = dict()
    for item in ['t', 'p', 'success', 'n_it', 'gamma_i', 'phi_coef_fun_i',
                 'phi_v', 'phi_i_sat', 'p_i_sat', 'z_l_i', 'v_l_i', 'poynting_i',
                 'y_i', 'x_i', 'k_i']:
        soln[item] = locals().get(item)
    return soln


def dew_p(t, p, y_i, tc_i, pc_i, af_omega_i,
          alpha_tr, epsilon, sigma, psi, omega,
          sec_j=None, nu_ij=None, unifac_data_dict=None,
          max_it=100, tol=tol, x_i_est=None, print_iterations=False):
    """Dew pressure determination

        Determines dew pressure at given temperature and vapor composition. Lee Kessler method.
        If sec_j and nu_ij are provided, gamma-phi approach is used, otherwise phi-phi.

        ref. Smith, Joseph Mauk ; Ness, Hendrick C. Van ; Abbott, Michael M.: \
        Introduction to chemical engineering thermodynamics. New York: McGraw-Hill, 2005.\
        (Fig. 14.9)

        ref. e.V., VDI: VDI-Wärmeatlas. Wiesbaden: Springer Berlin Heidelberg, 2013. \
        (D5.1. Abb.6.)

        :param t: temperature / K (known)
        :param p: pressure / bar (estimated)
        :param y_i: vapor composition (known)
        :param tc_i: critical temperatures / K
        :param pc_i: critical pressures / bar
        :param af_omega_i: Pitzer-factors
        :param alpha_tr: CEOS alpha(Tr_i, af_omega_i) function
        :param epsilon: CEOS epsilon
        :param sigma: CEOS sigma
        :param psi: CEOS psi
        :param omega: CEOS omega
        :param sec_j: unifac secondary group / subgroup
        :param nu_ij: coefficients in unifac subgroups per component
        :param max_it: maximum iterations
        :param tol: tolerance of method
        :param x_i_est: optional liquid composition (estimate)
        :param print_iterations: optionally print Lee-Kessler iterations
        :return: dictionary with 'x_i', 'y_i', 'phi_l', 'phi_v', 'k_i', 'sum_ki_xi', 'p', 'success'
        """
    if sec_j is None:
        def obj_fun(p_var): return dew_point_step_l_k(
            t, p_var, y_i, tc_i, pc_i, af_omega_i,
            alpha_tr, epsilon, sigma, psi, omega,
            max_it, tol=tol, x_i_est=x_i_est)
        soln = secant_ls_3p(obj_fun, p, tol=tol, x_1=1.001 * p,
                            restriction=lambda p_val: p_val > 0,
                            print_iterations=print_iterations)
        p = soln['x']
        soln = dew_point_step_l_k(t, p, y_i, tc_i, pc_i, af_omega_i,
                                  alpha_tr, epsilon, sigma, psi, omega, max_it,
                                  full_output=True, x_i_est=x_i_est)
    else:
        soln = dew_p_gamma_phi(
            t, p, y_i, tc_i, pc_i, af_omega_i,
            alpha_tr, epsilon, sigma, psi, omega,
            sec_j, nu_ij, unifac_data_dict,
            max_it, tol=tol, x_i_est=x_i_est)
    return soln


def dew_t(t, p, y_i, tc_i, pc_i, af_omega_i,
          alpha_tr, epsilon, sigma, psi, omega,
          max_it, tol=tol, x_i_est=None, print_iterations=False):
    """Dew temperature determination

        Determines dew temperature at given pressure and vapor composition. Lee Kessler method.

        ref. Smith, Joseph Mauk ; Ness, Hendrick C. Van ; Abbott, Michael M.: \
        Introduction to chemical engineering thermodynamics. New York: McGraw-Hill, 2005.\
        (Fig. 14.9)

        ref. e.V., VDI: VDI-Wärmeatlas. Wiesbaden: Springer Berlin Heidelberg, 2013. \
        (D5.1. Abb.6.)

        :param t: temperature / K (known)
        :param p: pressure / bar (estimate)
        :param y_i: vapor composition (known)
        :param tc_i: critical temperatures / K
        :param pc_i: critical pressures / bar
        :param af_omega_i: Pitzer-factors
        :param alpha_tr: CEOS alpha(Tr_i, af_omega_i) function
        :param epsilon: CEOS epsilon
        :param sigma: CEOS sigma
        :param psi: CEOS psi
        :param omega: CEOS omega
        :param max_it: maximum iterations
        :param tol: tolerance of method
        :param x_i_est: optional liquid composition (estimate)
        :param print_iterations: optionally print Lee-Kessler iterations
        :return: dictionary with 'x_i', 'y_i', 'phi_l', 'phi_v', 'k_i', 'sum_ki_xi', 'p', 'success'
        """
    def obj_fun(p_var): return dew_point_step_l_k(
        t, p_var, y_i, tc_i, pc_i, af_omega_i,
        alpha_tr, epsilon, sigma, psi, omega,
        max_it, tol=tol, x_i_est=x_i_est)
    soln = secant_ls_3p(obj_fun, p, tol=tol, x_1=1.001 * p,
                        restriction=lambda p_val: p_val > 0,
                        print_iterations=print_iterations)
    p = soln['x']
    soln = dew_point_step_l_k(t, p, y_i, tc_i, pc_i, af_omega_i,
                              alpha_tr, epsilon, sigma, psi, omega, max_it,
                              full_output=True, x_i_est=x_i_est)
    return soln


def dew_point_step_l_k(t, p, y_i, tc_i, pc_i, af_omega_i,
                       alpha_tr, epsilon, sigma, psi, omega,
                       max_it, tol=tol, full_output=False, x_i_est=None):
    soln_v = phi(t, p, y_i, tc_i, pc_i, af_omega_i, 'v',
                alpha_tr, epsilon, sigma, psi, omega)
    phi_v = soln_v['phi']
    if x_i_est is not None:
        phi_l = phi(t, p, x_i_est, tc_i, pc_i, af_omega_i, 'l',
                    alpha_tr, epsilon, sigma, psi, omega)['phi']
    else:
        soln_l = phi(t, p, y_i, tc_i, pc_i, af_omega_i, 'l',
                    alpha_tr, epsilon, sigma, psi, omega)
        phi_l = soln_l['phi']
    k_i = phi_l / phi_v
    sum_yi_over_ki = sum(y_i / k_i)
    x_i = y_i / k_i / sum_yi_over_ki
    stop = False
    i = 0
    success = True
    while not stop:
        sum_yi_over_ki_k_minus_1 = sum_yi_over_ki
        soln_l = phi(t, p, x_i, tc_i, pc_i, af_omega_i, 'l',
                     alpha_tr, epsilon, sigma, psi, omega)
        phi_l = soln_l['phi']
        k_i = phi_l / phi_v
        sum_yi_over_ki = sum(y_i / k_i)
        x_i = y_i / k_i / sum_yi_over_ki
        if abs((sum_yi_over_ki - sum_yi_over_ki_k_minus_1) /
               sum_yi_over_ki_k_minus_1) <= tol:
            stop = True
        i += 1
        if i >= max_it:
            stop = True
            success = False
    if full_output:
        z_l = soln_l['z']
        z_v = soln_v['z']
        soln = dict()
        for item in ['x_i', 'y_i', 'phi_l', 'phi_v', 'z_l', 'z_v', 'k_i', 'sum_yi_over_ki', 'p', 't', 'success']:
            soln[item] = locals().get(item)
        return soln
    else:
        return 1 - sum_yi_over_ki


def dew_p_gamma_phi(t, p, y_i, tc_i, pc_i, af_omega_i,
                        alpha_tr, epsilon, sigma, psi, omega,
                        sec_j, nu_ij, unifac_data_dict,
                        max_it, tol=tol, full_output=False, x_i_est=None):
    y_i = asarray(y_i)
    phi_coef_fun_i = ones(y_i.size)
    gamma_i = ones(y_i.size)
    soln = p_i_sat_ceos(
        t, p, tc_i, pc_i, af_omega_i,
        alpha_tr, epsilon, sigma, psi, omega, max_it=max_it, tol=tol)
    p_i_sat = soln['p']
    phi_i_sat = soln['phi_v']
    z_l_i = soln['z_l']

    p = 1 / sum(y_i * phi_coef_fun_i / gamma_i / p_i_sat)
    x_i = y_i * phi_coef_fun_i / gamma_i / p_i_sat * p
    x_i = x_i / sum(x_i)

    gamma_i = gamma_u(t, x_i, sec_j, nu_ij, unifac_data_dict)

    p = 1 / sum(y_i * phi_coef_fun_i / gamma_i / p_i_sat)

    for n_it in range(max_it):
        p_old = p
        phi_i_v = phi(t, p, y_i, tc_i, pc_i, af_omega_i, 'v',
                      alpha_tr, epsilon, sigma, psi, omega)['phi']
        v_l_i = z_l_i * r * t / p
        poynting_i = exp(-v_l_i * (p - p_i_sat) / (r * t))
        phi_coef_fun_i = phi_i_v / phi_i_sat * poynting_i

        for n_it_inner in range(max_it):
            gamma_i_old = gamma_i
            x_i = y_i * phi_coef_fun_i / gamma_i / p_i_sat * p
            x_i = x_i / sum(x_i)
            gamma_i = gamma_u(t, x_i, sec_j, nu_ij, unifac_data_dict)
            delta_gamma = gamma_i - gamma_i_old
            mag_delta_gamma = sqrt(delta_gamma.dot(delta_gamma))
            success = mag_delta_gamma <= tol
            if success:
                break

        p = 1 / sum(y_i * phi_coef_fun_i / gamma_i / p_i_sat)
        delta_p = p - p_old
        success = abs(delta_p) <= tol
        if success:
            break

    k_i = gamma_i * p_i_sat / phi_coef_fun_i / p
    phi_v = phi_i_v
    soln = dict()
    for item in ['t', 'p', 'success', 'n_it', 'gamma_i', 'phi_coef_fun_i',
                 'phi_v', 'phi_i_sat', 'p_i_sat', 'z_l_i', 'v_l_i', 'poynting_i',
                 'y_i', 'x_i', 'k_i']:
        soln[item] = locals().get(item)
    return soln


def p_est(t, p, x_i, tc_i, pc_i, af_omega_i, max_it, tol=tol):
    x_i = asarray(x_i)
    tr_i = t / tc_i
    a_i = psi * alpha_tr(tr_i, af_omega_i) * r ** 2 * tc_i ** 2 / pc_i
    b_i = omega * r * tc_i / pc_i
    q_i = a_i / (b_i * r * t)
    a_ij = sqrt(outer(a_i, a_i))

    # Variablen, die von der Flüssigkeit-Zusammensetzung abhängig sind
    b = sum(x_i * b_i)
    a = x_i.dot(a_ij).dot(x_i).item()
    q = a / (b * r * t)

    rho_lim = 1 / b
    rho_lb = 0.0001 * rho_lim
    v = 1 / rho_lb
    dp_dv_at_rho_lb = -r * t / (v - b) ** 2 + a / ((v + epsilon * b) * (v + sigma * b)) * (
            1 / (v + epsilon * b) + 1 / (v + sigma * b)
    )
    d2p_dv2_at_rho_lb = 2 * r * t / (v - b) ** 3 - a / ((v + epsilon * b) * (v + sigma * b)) * (
            1 / (v + epsilon * b) ** 2 + 1 / (v + sigma * b) ** 2
    ) - a / ((v + epsilon * b) * (v + sigma * b)) ** 2 * (
            1 + (v + epsilon * b) / (v + sigma * b) + 1 + (v + sigma * b) / (v + epsilon * b)
    )

    dp_drho_at_rho_lb = - 1/rho_lb**2 * dp_dv_at_rho_lb
    d2p_drho_at_rho_lb = 1/rho_lb**4 * d2p_dv2_at_rho_lb + dp_dv_at_rho_lb * 2 / rho_lb**3
    
    if d2p_drho_at_rho_lb > 0:
        # no inflection point, curve is monotonic
        pass
    else:
        # find inflection point
        p0 = 2 * a * b ** 5 * epsilon ** 2 + 2 * a * b ** 5 * epsilon * sigma + \
             2 * a * b ** 5 * sigma ** 2 + 2 * b ** 6 * epsilon ** 3 * r * t * sigma ** 3
        p1 = -3 * a * b ** 4 * epsilon ** 2 - 3 * a * b ** 4 * epsilon * sigma + \
             3 * a * b ** 4 * epsilon - 3 * a * b ** 4 * sigma ** 2 + \
             3 * a * b ** 4 * sigma + 3 * b ** 5 * epsilon ** 3 * r * t * sigma ** 2 + \
             3 * b ** 5 * epsilon ** 3 * r * t * sigma ** 2 + \
             3 * b ** 5 * epsilon ** 2 * r * t * sigma ** 3
        p2 = 6 * a * b ** 3 * epsilon ** 2 + 6 * a * b ** 3 * epsilon * sigma - \
             18 * a * b ** 3 * epsilon + 6 * a * b ** 3 * sigma ** 2 - \
             18 * a * b ** 3 * sigma + 6 * a * b ** 3 + \
             6 * b ** 4 * epsilon ** 3 * r * t * sigma + \
             18 * b ** 4 * epsilon ** 2 * r * t * sigma ** 2 + \
             6 * b ** 4 * epsilon * r * t * sigma ** 3
        p3 = -2 * a * b ** 2 * epsilon ** 2 - 2 * a * b ** 2 * epsilon * sigma + \
             18 * a * b ** 2 * epsilon - 2 * a * b ** 2 * sigma ** 2 + \
             18 * a * b ** 2 * sigma - 18 * a * b ** 2 + \
             2 * b ** 3 * epsilon ** 3 * r * t + \
             18 * b ** 3 * epsilon ** 2 * r * t * sigma + \
             18 * b ** 3 * epsilon * r * t * sigma ** 2 + 2 * b ** 3 * r * t * sigma ** 3
        p4 = -6 * a * b * epsilon - 6 * a * b * sigma + 18 * a * b + \
             6 * b ** 2 * epsilon ** 2 * r * t + \
             18 * b ** 2 * epsilon * r * t * sigma + 6 * b ** 2 * r * t * sigma ** 2
        p5 = -6 * a + 6 * b * epsilon * r * t + 6 * b * r * t * sigma
        p6 = 2 * r * t

        v_inf_roots = zroots([p0, p1, p2, p3, p4, p5, p6])

        v = v_inf_roots[v_inf_roots.imag == 0].real
        dp_dv = -r * t / (v - b) ** 2 + a / ((v + epsilon * b) * (v + sigma * b)) * (
                1 / (v + epsilon * b) + 1 / (v + sigma * b)
        )
        d2p_dv2 = 2 * r * t / (v - b) ** 3 - a / ((v + epsilon * b) * (v + sigma * b)) * (
                1 / (v + epsilon * b) ** 2 + 1 / (v + sigma * b) ** 2
        ) - a / ((v + epsilon * b) * (v + sigma * b)) ** 2 * (
                          1 + (v + epsilon * b) / (v + sigma * b) + 1 + (v + sigma * b) / (v + epsilon * b)
                  )
        if len(v) == 2 and all(dp_dv < 0):
            # above pseudocritical temperature, no extrema
            v_inf = v[dp_dv == max(dp_dv)].item()
            d2p_dv2 = d2p_dv2[dp_dv == max(dp_dv)].item()
            dp_dv = dp_dv[dp_dv == max(dp_dv)].item()
        elif len(v) == 0:
            # above Boyle-temperature
            pass
        elif len(v) == 1:
            v_inf = v.item()
            d2p_dv2 = d2p_dv2.item()
            dp_dv = dp_dv.item()
        elif len(v) == 2:
            v_inf = v[dp_dv > 0].item()
            d2p_dv2 = d2p_dv2[dp_dv > 0].item()
            dp_dv = dp_dv[dp_dv > 0].item()

        p_rho_inf = r * t / (v_inf - b) - a / ((v_inf + epsilon * b) * (v_inf + sigma * b))
        z_inf = p * v_inf / (r * t)

        beta = b * p_rho_inf / (r * t)

        a1 = beta * (epsilon + sigma) - beta - 1
        a2 = q * beta + epsilon * sigma * beta ** 2 \
             - beta * (epsilon + sigma) * (1 + beta)
        a3 = -(epsilon * sigma * beta ** 2 * (1 + beta) +
               q * beta ** 2)

        soln = solve_cubic([1, a1, a2, a3])
        z_p_inf = array(soln['roots'])
        v_p_inf = z_p_inf * r * t / p_rho_inf
        z_l_p_inf = z_p_inf[-1][0]
        z_v_p_inf = z_p_inf[0][0]
        v_l_p_inf = v_p_inf[-1][0]
        v_v_p_inf = v_p_inf[0][0]
        #if abs((p - p_old)/p) <= -tol:
        #    break

        # find local min and max (extrema)
        p0 = a * b ** 3 * epsilon + a * b ** 3 * sigma - b ** 4 * epsilon ** 2 * r * sigma ** 2 * t
        p1 = -2 * a * b ** 2 * epsilon - 2 * a * b ** 2 * sigma + \
             2 * a * b ** 2 - 2 * b ** 3 * epsilon ** 2 * r * sigma * t - \
             2 * b ** 3 * epsilon * r * sigma ** 2 * t
        p2 = a * b * epsilon + a * b * sigma - 4 * a * b - b ** 2 * epsilon ** 2 * r * t - \
             4 * b ** 2 * epsilon * r * sigma * t - b ** 2 * r * sigma ** 2 * t
        p3 = 2 * a - 2 * b * epsilon * r * t - 2 * b * r * sigma * t
        p4 = -r * t

        # v_roots = zroots([p0, p1, p2, p3, p4])
        soln = solve_quartic([p4, p3, p2, p1, p0])
        v_roots = array(soln['roots'])
        re_v_roots = v_roots[v_roots[:, 1] == 0][:, 0]
        v = re_v_roots[re_v_roots > b]

        # mechanical critical point
        z_mc = -1 / 3 * ((epsilon + sigma) * omega - omega - 1)
        p_mc = sum(x_i * pc_i)
        t_mc = x_i.dot(sqrt(outer(tc_i, tc_i))).dot(x_i).item()
        rho_mc = p_mc / (r * t_mc * z_mc)
        v_mc = 1 / rho_mc

        if t < t_mc and len(v) < 2:
            pass
        elif t > t_mc and len(v) < 2:
            # single real root, vapor-like
            beta = b * p / (r * t)
            a1 = beta * (epsilon + sigma) - beta - 1
            a2 = q * beta + epsilon * sigma * beta ** 2 \
                 - beta * (epsilon + sigma) * (1 + beta)
            a3 = -(epsilon * sigma * beta ** 2 * (1 + beta) +
                   q * beta ** 2)
            soln = solve_cubic([1, a1, a2, a3])
            z_v = soln['roots'][0][0]
            v_v = z_v * r * t / p
            rho_v = 1 / v_v

            # pseudo liquid density
            dp_dv_at_rho_mc = -r * t / (v_mc - b) ** 2 + a / ((v_mc + epsilon * b) * (v_mc + sigma * b)) * (
                    1 / (v_mc + epsilon * b) + 1 / (v_mc + sigma * b)
            )
            dp_drho_at_rho_mc = -v_mc**2 * dp_dv_at_rho_mc
            c1 = dp_drho_at_rho_mc * (rho_mc - 0.7 * rho_mc)
            c0 = p_mc - c1 * log(rho_mc - 0.7 * rho_mc)
            rho_l = exp((p - c0)/c1) + 0.7 * rho_mc
            v_l = 1 / rho_l
            z_l = p * v_l / (r * t)
            rho_l = 1 / v_l

        else:
            v_l = min(v)
            v_v = max(v)
            z_l = p * v_l / (r * t)
            z_v = p * v_v / (r * t)
            rho_l = 1 / v_l
            rho_v = 1 / v_v

            p_min_l = r * t / (v_l - b) - a / ((v_l + epsilon * b) * (v_l + sigma * b))
            p_max_v = r * t / (v_v - b) - a / ((v_v + epsilon * b) * (v_v + sigma * b))

            if p <= p_min_l:
                p = p_min_l
            elif p >= p_max_v:
                p = p_max_v
    for item in ['p', 'p_rho_inf', 'z_rho_inf', 'v_rho_inf', 'p_min_l', 'p_max_v',
                 'z_l', 'z_v', 'v_l', 'v_v', 'rho_l', 'rho_v']:
        soln[item] = locals().get(item)
    return soln

def z_phase(t, p, z_i, tc_i, pc_i, af_omega_i, phase, alpha_tr, epsilon, sigma, psi, omega, tol=tol):
    z_i = asarray(z_i)
    tr_i = t / tc_i
    a_i = psi * alpha_tr(tr_i, af_omega_i) * r ** 2 * tc_i ** 2 / pc_i
    b_i = omega * r * tc_i / pc_i
    q_i = a_i / (b_i * r * t)
    a_ij = sqrt(outer(a_i, a_i))

    # composition-dependent variables
    b = sum(z_i * b_i)
    a = z_i.dot(a_ij).dot(z_i).item()
    q = a / (b * r * t)

    # mechanical critical point
    z_mc = -1 / 3 * ((epsilon + sigma) * omega - omega - 1)
    p_mc = sum(z_i * pc_i)
    t_mc = z_i.dot(sqrt(outer(tc_i, tc_i))).dot(z_i).item()
    rho_mc = p_mc / (r * t_mc * z_mc)
    v_mc = 1 / rho_mc

    # solve f(z)=0
    beta = b * p / (r * t)
    a1 = beta * (epsilon + sigma) - beta - 1
    a2 = q * beta + epsilon * sigma * beta ** 2 \
         - beta * (epsilon + sigma) * (1 + beta)
    a3 = -(epsilon * sigma * beta ** 2 * (1 + beta) +
           q * beta ** 2)
    soln = solve_cubic([1, a1, a2, a3])
    roots_z = array(soln['roots'])
    disc_z = soln['disc']

    # solve S+U=0 (real root = real part of complex root)
    # locus matches p-rho inflection point at low density
    q0 = (9 * (epsilon + sigma) * epsilon * sigma + 18 * epsilon * sigma +
          - 2 * (epsilon + sigma) ** 3 - 3 * (epsilon + sigma) ** 2 +
          3 * (epsilon + sigma) + 2) * (b / (r * t)) ** 3
    q1 = (18 * epsilon * sigma - 2 * (epsilon + sigma) ** 2 +
          6 * (epsilon + sigma) + 6) * (b / (r * t)) ** 2 + (
                 9 * (epsilon + sigma) + 18) * a * b / (r * t) ** 3
    q2 = (3 * (epsilon + sigma) + 6) * b / (r * t) - 9 * a / (r * t) ** 2
    q3 = 2
    if q0 == 0:
        # reduces to quadratic (SRK)
        disc = q2 ** 2 - 4 * q1 * q3
        if disc >= 0:
            roots_p = array([
                [-q2 / q1 / 2 + sqrt((q2 / q1) ** 2 / 4 - q3 / q1), 0],
                [-q2 / q1 / 2 - sqrt((q2 / q1) ** 2 / 4 - q3 / q1), 0]
            ])
            p_low = min(roots_p[:, 0])
            p_cross = max(roots_p[:, 0])
        elif disc < 0:
            roots_p = array([
                [-q2 / q1 / 2, sqrt(-(q2 / q1) ** 2 / 4 + q3 / q1)],
                [-q2 / q1 / 2, -sqrt(-(q2 / q1) ** 2 / 4 + q3 / q1)]
            ])
            p_low = min(roots_p[:, 0])
            p_cross = max(roots_p[:, 0])
    else:
        # cubic
        soln = solve_cubic([q0, q1, q2, q3])
        roots_p = array(soln['roots'])

    re_roots_p = roots_p[roots_p[:, 1] == 0]
    n_positive_roots_p = len(re_roots_p[re_roots_p > 0])

    if n_positive_roots_p <= 1:
        p_high = roots_p[0, 0]
    elif n_positive_roots_p > 1:
        if n_positive_roots_p == 3:
            p_high = roots_p[0, 0]
            p_cross = roots_p[1, 0]
            p_low = roots_p[-1, 0]
        elif n_positive_roots_p == 2:
            p_cross = roots_p[0, 0]
            p_low = roots_p[-1, 0]
        # solve f(z_low)=0
        beta_low = b * p_low / (r * t)
        a1_low = beta_low * (epsilon + sigma) - beta_low - 1
        a2_low = q * beta_low + epsilon * sigma * beta_low ** 2 \
                 - beta_low * (epsilon + sigma) * (1 + beta_low)
        a3_low = -(epsilon * sigma * beta_low ** 2 * (1 + beta_low) +
                   q * beta_low ** 2)
        soln_z_low = solve_cubic([1, a1_low, a2_low, a3_low])
        roots_z_low = array(soln_z_low['roots'])
        disc_z_low = soln_z_low['disc']
        if phase == 'l' and disc_z_low <= 0:
            # 3 real roots
            z_low = roots_z_low[-1][0]
        elif phase == 'l' and disc_z_low > 0:
            # 1 real, 2 complex roots
            z_low = roots_z_low[0][0]
        elif phase == 'v':
            # 3 real roots or 1 real, 2 complex roots,
            # at any rate index 0
            z_low = roots_z_low[0][0]
        v_low = z_low * r * t / (p_low)
        rho_low = 1 / v_low
        dp_dv_at_rho_low = -r * t / (v_low - b) ** 2 + a / (
                (v_low + epsilon * b) * (v_low + sigma * b)) * (
                                   1 / (v_low + epsilon * b) + 1 / (v_low + sigma * b)
                           )
        dp_drho_at_rho_low = -v_low ** 2 * dp_dv_at_rho_low

    if phase == 'l':
        if t < t_mc or n_positive_roots_p > 1:
            if p > p_low:
                # liquid density
                if disc_z <= 0:
                    z = roots_z[-1, 0]
                elif disc_z > 0:
                    z = roots_z[0, 0]
                v = z * r * t / p
                rho = 1 / v
            elif p <= p_low:
                # pseudo liquid density - eq. 36, 40, 41
                # FIXME: extrapolate when (rho_low - 0.7 * rho_mc) < 0 ==>> complex root
                if rho_low - 0.7 * rho_mc <= 0 or dp_drho_at_rho_low < 0:
                    # use linear interpolation
                    rho = rho_low + (rho_low - 0.7 * rho_mc) / dp_drho_at_rho_low
                    rho = (p - p_low) * (rho_low - 0.7 * rho_mc) + rho_low
                else:
                    c1 = dp_drho_at_rho_low * (rho_low - 0.7 * rho_mc)
                    c0 = p_low - c1 * log(rho_low - 0.7 * rho_mc)
                    rho = 0.7 * rho_mc + exp((p - c0) / c1)
                v = 1 / rho
                z = p * v / (r * t)
        elif t >= t_mc or n_positive_roots_p <= 1:
            p_mc_bound = r * t / (v_mc - b) - a / ((v_mc + epsilon * b) * (v_mc + sigma * b))
            if p > p_mc_bound:
                # liquid density
                z = roots_z[0, 0]
                v = z * r * t / p
                rho = 1 / v
            elif p <= p_mc_bound:
                # pseudo liquid density - eq. 36, 38, 39
                dp_dv_at_rho_mc = -r * t / (v_mc - b) ** 2 + a / (
                        (v_mc + epsilon * b) * (v_mc + sigma * b)) * (
                                           1 / (v_mc + epsilon * b) + 1 / (v_mc + sigma * b)
                                   )
                dp_drho_at_rho_mc = -v_mc ** 2 * dp_dv_at_rho_mc

                c1 = dp_drho_at_rho_mc * (rho_mc - 0.7 * rho_mc)
                c0 = p_mc_bound - c1 * log(rho_mc - 0.7 * rho_mc)
                rho = 0.7 * rho_mc + exp((p - c0)/c1)
                v = 1 / rho
                z = p * v / (r * t)
    elif phase == 'v':
        if t < t_mc or n_positive_roots_p > 1:
            if p > p_low:
                # pseudo vapor density - eq. 24, 28, 29, 30
                if dp_drho_at_rho_low < 0.1 * r * t:
                    dp_drho_at_rho_low = 0.1 * r * t
                rho0 = (rho_low + 1.4 * rho_mc) / 2
                rho1 = p_low * (
                        (rho_low - 1.4 * rho_mc) + p_low / dp_drho_at_rho_low)
                rho2 = -p_low ** 2 * (
                        (rho_low - 1.4 * rho_mc) / 2 + p_low / dp_drho_at_rho_low)

                rho = rho0 + rho1 / p + rho2 / p ** 2
                v = 1 / rho
                z = p * v / (r * t)
            elif p <= p_low:
                # vapor density
                z = roots_z[0, 0]
                v = z * r * t / p
                rho = 1 / v
        elif t >= t_mc or n_positive_roots_p <= 0:
            # vapor density
            z = roots_z[0, 0]
            v = z * r * t / p
            rho = 1 / v
    soln = dict()
    for item in ['z', 'rho', 'v',
                 'p_low', 'p_mc_bound', 'p_high', 'p_cross']:
        soln[item] = locals().get(item)
    return soln

def zs_1998():
    use_pr_eos()
    global r
    r = 8.3145 # Pa m^3 / (mol K)
    tc_i = array([305.32, 540.2])
    pc_i = array([48.71, 27.35]) * 1e5 # Pa
    af_omega_i = array([0.099, 0.35])
    max_it = 100
    markers = plt.Line2D.filled_markers
    p_range = linspace(1e-1, 140, 100) * 1e5
    plot1 = plt.subplot2grid([2, 2], [1, 0], rowspan=1, colspan=1)
    plot2 = plt.subplot2grid([2, 2], [1, 1], rowspan=1, colspan=1)
    plot3 = plt.subplot2grid([2, 2], [0, 0], rowspan=1, colspan=1)
    plot4 = plt.subplot2grid([2, 2], [0, 1], rowspan=1, colspan=1)
    x = 0.5
    z_i = array([x, 1 - x])
    for t in[420, 500]:
        v_plot = []
        p_plot = []
        rho_plot = []
        z_plot = []
        z_complex = []
        p_complex = []

        rho_l_phase = []
        rho_v_phase = []
        p_v_phase = []
        for p in p_range:
            tr_i = t / tc_i
            a_i = psi * alpha_tr(tr_i, af_omega_i) * r ** 2 * tc_i ** 2 / pc_i
            b_i = omega * r * tc_i / pc_i
            beta_i = b_i * p / (r * t)
            q_i = a_i / (b_i * r * t)
            a_ij = sqrt(outer(a_i, a_i))

            # Variablen, die von der Flüssigkeit-Zusammensetzung abhängig sind
            b = sum(z_i * b_i)
            a = z_i.dot(a_ij).dot(z_i)
            beta = b * p / (r * t)
            q = a / (b * r * t)
            a_mp_i = -a + 2 * a_ij.dot(z_i)  # partielles molares a_i
            b_mp_i = b_i  # partielles molares b_i
            q_mp_i = q * (1 + a_mp_i / a - b_i / b)  # partielles molares q_i

            a1 = beta * (epsilon + sigma) - beta - 1
            a2 = q * beta + epsilon * sigma * beta ** 2 \
                 - beta * (epsilon + sigma) * (1 + beta)
            a3 = -(epsilon * sigma * beta ** 2 * (1 + beta) +
                   q * beta ** 2)

            soln = solve_cubic([1, a1, a2, a3])
            roots, disc = soln['roots'], soln['disc']
            re_roots = array([roots[0][0], roots[1][0], roots[2][0]])

            if disc <= 0 and all(re_roots >= 0):
                # 3 real roots. smallest ist liq. largest is gas.
                z_l = re_roots[0]
                z_mid = re_roots[1]
                z_v = re_roots[2]
                v_l = z_l * r * t / p
                v_mid = z_mid * r * t / p
                v_v = z_v * r * t / p
                p_plot += [p, p, p]
                v_plot += [v_l, v_mid, v_v]
                rho_plot += [1 / v_l, 1 / v_mid, 1 / v_v]
                z_plot += [z_l, z_mid, z_v]
            elif disc > 0:
                # one real root, 2 complex. First root is the real one.
                z = re_roots[0]
                v = z * r * t / abs(p)
                p_plot += [p]
                v_plot += [v]
                rho_plot += [1 / v]
                z_plot += [z]
                z_complex += [re_roots[1]]
                p_complex += [p]

        for p in linspace(1e-4, max(p_range), 30):
            rho_l_phase += [
                z_phase(t, p, z_i, tc_i, pc_i, af_omega_i, 'l', alpha_tr, epsilon, sigma, psi, omega, tol)['rho']]
            rho_v_phase += [
                z_phase(t, p, z_i, tc_i, pc_i, af_omega_i, 'v', alpha_tr, epsilon, sigma, psi, omega, tol)['rho']]
            p_v_phase += [p]

        plot1.semilogx(v_plot, p_plot, markers[randint(0, len(markers))],
                       label=r'$z_1={:g}, T={:g}K$'.format(z_i[0], t),
                       fillstyle='none')
        rho_line = plot2.plot(rho_plot, p_plot, markers[randint(0, len(markers))],
                              label=r'$z_1={:g}, T={:g}K$'.format(z_i[0], t),
                              fillstyle='none')
        current_color = plt.get(rho_line[0], 'color')
        current_marker = plt.get(rho_line[0], 'marker')
        plot4.plot(rho_l_phase, p_v_phase, current_marker, markeredgewidth=0.5,
                   color=current_color, markersize=4, fillstyle='bottom', linestyle='none',
                   label=r'$L: x_1={:g}$'.format(z_i[0]))
        plot4.plot(rho_v_phase, p_v_phase, current_marker, markeredgewidth=0.25,
                   color=current_color, markersize=4, fillstyle='none', linestyle='--',
                   label=r'$V: y_1={:g}$'.format(z_i[0]))
        plot2.plot(rho_l_phase, p_v_phase, '--',
                   color=current_color, label=r'pseudo-$\rho_L$')
        plot2.plot(rho_v_phase, p_v_phase, ':',
                   color=current_color, label=r'pseudo-$\rho_V$')
        z_line = plot3.plot(z_plot, p_plot, markers[randint(0, len(markers))],
                            label=r'$z_1={:g}, z_2={:g}$'.format(z_i[0], z_i[1]),
                            fillstyle='none')
        current_color = plt.get(z_line[0], 'color')
        current_marker = markers[randint(0, len(markers))]
        plot3.plot(z_complex, p_complex, '.', alpha=0.1,
                   fillstyle='none', color=current_color)

    data = []
    f = open('./data/actual_density.csv')
    for line in f:
        line = f.readline()
        if len(line) > 0:
            data += [[float(x) for x in line.replace(',', '.').replace('\n', '').split(';')]]
    f.close()
    a_data = array(data)
    plot2.plot(a_data[:,0], a_data[:, 1], '-', color='black', label='article')
    for filename in ['pseudo_vapor_density.csv', 'pseudo_liquid_density_420.csv',
                     'pseudo_liquid_density_500.csv', 'actual_density_2.csv']:
        f = open('./data/'+filename)
        data = []
        for line in f:
            line = f.readline()
            if len(line) > 0:
                data += [[float(x) for x in line.replace(',', '.').replace('\n', '').split(';')]]
        f.close()
        a_data = array(data)
        plot2.plot(a_data[:, 0], a_data[:, 1], ':', color='black')


    plot1.axvline(b, linestyle='-')
    plot2.axvline(1 / b, linestyle='-')
    plot4.axvline(1 / b, linestyle='-')
    p_low = z_phase(420, p, z_i, tc_i, pc_i, af_omega_i, 'l', alpha_tr, epsilon, sigma, psi, omega, tol)['p_low']
    plot2.axhline(p_low, linestyle='-.', color='gray', linewidth=0.5, label='$P_{low}$')
    plot4.axhline(p_low, linestyle='-.', color='gray', linewidth=0.5, label='$P_{low}$')
    plot1.set_xlabel(r'$\frac{v}{m^3/mol}$')
    plot1.set_ylabel('p / Pa')
    plot2.set_xlabel(r'$\frac{rho}{mol / m^3}$')
    plot2.set_ylabel('p / Pa')
    plot2.set_title(r'$\rho$' + ', act.')
    plot2.set_ylim(0, max(a_data[:, 1]))
    plot2.ticklabel_format(axis='y', scilimits=[-4,4])
    plot2.legend(fontsize=6)
    plot3.set_xlabel(r'$Z$')
    plot3.set_ylabel('p / Pa')
    plot3.legend(['real root', 'real part of complex root'], fontsize=6)
    plot4.set_xlabel(r'$\frac{rho}{mol / m^3}$')
    plot4.set_ylabel('p / Pa')
    plot4.set_title(r'pseudo-$\rho$' + ', L/V [3]')
    plot4.legend(fontsize=6)
    plot4.set_ylim(0, max(a_data[:, 1]))
    plot4.ticklabel_format(axis='y', scilimits=[-4, 4])
    plt.tight_layout()
    plot1.legend(fontsize=6)


    fig2 = plt.figure()
    ax = plt.axes()
    p_list = linspace(1e-4, 80, 30) * 1e5
    j = 1
    for t in [420, 500]:
        phi_list_l = empty([len(p_list), 2])
        phi_list_v = empty([len(p_list), 2])
        for i, p in enumerate(p_list):
            phi_list_l[i] = phi(t, p, z_i, tc_i, pc_i, af_omega_i, 'l', alpha_tr, epsilon, sigma, psi, omega)['phi']
            phi_list_v[i] = phi(t, p, z_i, tc_i, pc_i, af_omega_i, 'v', alpha_tr, epsilon, sigma, psi, omega)['phi']

        plt.subplot(1, 2, j)
        current_marker = markers[randint(0, len(markers))]
        plt.plot(p_list, log(phi_list_l[:, 0]), current_marker+'-', label=r'$\phi_{C2}^L$')
        plt.plot(p_list, log(phi_list_v[:, 0]), current_marker + '-', label=r'$\phi_{C2}^V$')
        current_marker = markers[randint(0, len(markers))]
        plt.plot(p_list, log(phi_list_l[:, 1]), current_marker + '-', label=r'$\phi_{C7}^L$')
        plt.plot(p_list, log(phi_list_v[:, 1]), current_marker+'-', label=r'$\phi_{C7}^V$')
        plt.xlabel('p / Pa')
        plt.ylabel(r'$log \phi$')
        plt.title('T={:g}K'.format(t))
        if j == 1:
            plt.axvline(z_phase(t, p, z_i, tc_i, pc_i, af_omega_i, 'l', alpha_tr, epsilon, sigma, psi, omega)['p_low'],
                        linestyle='--', label='$P_{low}$')
            plt.ylim(-3, 3)
        else:
            plt.ylim(-2, 4)
        plt.legend()
        j += 1
        ax = plt.gca().ticklabel_format(axis='x', scilimits=[-4,4])
    plt.tight_layout()
    plt.show()



def isot_flash(t, p, x_i, y_i, z_i, tc_i, pc_i, af_omega_i,
               alpha_tr, epsilon, sigma, psi, omega):
    soln_l = phi(t, p, x_i, tc_i, pc_i, af_omega_i, 'l', alpha_tr, epsilon, sigma, psi, omega)
    soln_v = phi(t, p, y_i, tc_i, pc_i, af_omega_i, 'v', alpha_tr, epsilon, sigma, psi, omega)
    k_i = soln_l['phi'] / soln_v['phi']
    soln_v_f = secant_ls_3p(
        lambda v_f: sum(z_i * (1 - k_i) / (1 + v_f * (k_i - 1))), 0.5, tol=1e-10,
        f_prime=lambda v_f: -sum(z_i * (k_i - 1) ** 2 / (1 + v_f * (k_i - 1)) ** 2),
        restriction=lambda v_f_val: 0 <= v_f_val and v_f_val <= 1.0,
        print_iterations=False)
    v_f = soln_v_f['x']
    x_i = z_i / (1 + v_f * (k_i - 1))
    y_i = x_i * k_i / sum(x_i * k_i)

    # Normalize

    x_i = x_i / sum(x_i)
    soln = dict()
    for item in ['soln_l', 'soln_v', 'soln_v_f',
                 'k_i', 'v_f', 'x_i', 'y_i']:
        soln[item] = locals().get(item)
    return soln


def isot_flash_solve(t, p, z_i, tc_i, pc_i, af_omega_i, max_it=20,
                     x_i=None, y_i=None):
    if not x_i:
        x_i = ones(len(z_i)) / len(z_i)
    if not y_i:
        y_i = ones(len(z_i)) / len(z_i)
    for i in range(max_it):
        soln = isot_flash(t, p, x_i, y_i, z_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega)
        y_i = soln['y_i']
        x_i = soln['x_i']
        v_f = soln['v_f']
        k_i = soln['k_i']
    return soln


def pt_flash(t, p, z_i, tc_i, pc_i, af_omega_i,
             alpha_tr, epsilon, sigma, psi, omega,
             sec_j=None, nu_ij=None, unifac_data_dict=None,
             max_it=100, tol=tol, p_est=None):
    if p_est is None:
        # take estimate for 2-phase region
        p0 = p
    else:
        p0 = p_est
    # FIXME: check if bubl_p at all possible (by now, no error when converging to k=1)
    bubl_p_soln = bubl_p(t, p0, z_i, tc_i, pc_i, af_omega_i,
                         alpha_tr, epsilon, sigma, psi, omega,
                         sec_j, nu_ij, unifac_data_dict,
                         max_it=max_it, tol=tol)
    dew_p_soln = dew_p(t, p0, z_i, tc_i, pc_i, af_omega_i,
                       alpha_tr, epsilon, sigma, psi, omega,
                       sec_j, nu_ij, unifac_data_dict,
                       max_it=max_it, tol=tol)
    p_bubl, p_dew = bubl_p_soln['p'], dew_p_soln['p']
    phi_bubl, phi_dew = bubl_p_soln['phi_v'], dew_p_soln['phi_v']
    k_i_bubl, k_i_dew = bubl_p_soln['k_i'], dew_p_soln['k_i']
    if sec_j is None:
        # phi-phi formulation: k-values directly from eos
        pass
    else:
        # gamma-phi-formulation
        gamma_i_bubl, gamma_i_dew = bubl_p_soln['gamma_i'], dew_p_soln['gamma_i']
        poynting_i_bubl, poynting_i_dew = bubl_p_soln['poynting_i'], dew_p_soln['poynting_i']
        phi_i_sat = dew_p_soln['phi_i_sat']  # equal to bubl_p_soln['phi_i_sat']
        p_i_sat = bubl_p_soln['p_i_sat']
    if  p_dew < p and p < p_bubl:
        if sec_j is None:
            # phi-phi
            pass
        else:
            # 0 < v_f < 1
            # gamma-phi-formulation - svn eq. 14.1
            # $y_i \Phi_i P = x_i \gamma_i P_i^{sat}$
            gamma_i = gamma_i_dew + (gamma_i_bubl - gamma_i_dew) * (p - p_dew) / (p_bubl - p_dew)
            poynting_i = poynting_i_dew + (poynting_i_bubl - poynting_i_dew) * (p - p_dew) / (p_bubl - p_dew)
        phi_i_v = phi_dew + (phi_bubl - phi_dew) * (p - p_dew) / (p_bubl - p_dew)
        k_i = k_i_dew + (k_i_bubl - k_i_dew) * (p - p_dew) / (p_bubl - p_dew)
        v_f = 1 + (0 - 1) * (p - p_dew) / (p_bubl - p_dew)
        y_i = z_i
        x_i = z_i / (1 + v_f * (k_i - 1))
        x_i = x_i / sum(x_i)
        for n_it in range(max_it):
            y_i_old = y_i
            x_i_old = x_i
            v_f_old = v_f
            if sec_j is None:
                phi_i_l = phi(t, p, x_i, tc_i, pc_i, af_omega_i, 'l',
                          alpha_tr, epsilon, sigma, psi, omega)['phi']
                k_i = phi_i_l / phi_i_v
            else:
                phi_coef_fun_i = phi_i_v / phi_i_sat * poynting_i
                k_i = gamma_i * p_i_sat / (phi_coef_fun_i * p)
            soln_v_f = secant_ls_3p(
                lambda v_f: sum(z_i * (1 - k_i) / (1 + v_f * (k_i - 1))), v_f, tol=1e-10,
                f_prime=lambda v_f: -sum(z_i * (k_i - 1)**2 / (1 + v_f * (k_i - 1))**2),
                restriction=lambda v_f_val: 0 <= v_f_val and v_f_val <= 1,
                print_iterations=False)
            # soln_v_f = nr_ls(
            #     v_f, lambda v_f: sum(z_i * (1 - k_i) / (1 + v_f * (k_i - 1))),
            #     lambda v_f: -sum(z_i * (k_i - 1) ** 2 / (1 + v_f * (k_i - 1)) ** 2),
            #     tol=1e-10, inner_loop_condition=lambda v_f_val: v_f_val >= 0,
            #     notify_status_func=notify_status_func)
            v_f = soln_v_f['x']
            x_i = z_i / (1 + v_f * (k_i - 1))
            y_i = k_i * x_i
            delta_x_i = x_i - x_i_old
            delta_y_i = y_i - y_i_old
            delta_v_f = v_f - v_f_old
            mag_delta_x_i = sqrt(delta_x_i.dot(delta_x_i))
            mag_delta_y_i = sqrt(delta_y_i.dot(delta_y_i))
            mag_delta_v_f = sqrt(delta_v_f**2)
            success = mag_delta_x_i <= tol and mag_delta_y_i <=tol and mag_delta_v_f <= tol
            if success:
                break
            if sec_j is None:
                # phi-phi
                pass
            else:
                # gamma-phi
                gamma_i = gamma_u(t, x_i, sec_j, nu_ij, unifac_data_dict)
            phi_i_v = phi(t, p, y_i, tc_i, pc_i, af_omega_i, 'v',
                          alpha_tr, epsilon, sigma, psi, omega)['phi']
    elif p_dew >= p:
        v_f = 1.0
    elif p_bubl <= p:
        v_f = 0.0
    soln = dict()
    for item in ['t', 'p', 'k_i', 'v_f', 'z_i', 'x_i', 'y_i', 'gamma_i', 'phi_coef_fun_i',
                 'dew_p_soln', 'bubl_p_soln', 'p_dew', 'p_bubl', 'poynting', 'n_it',
                 'mag_delta_x_i', 'mag_delta_y_i', 'mag_delta_v_f']:
        soln[item] = locals().get(item)
    return soln


def vdi_atlas():
    use_pr_eos()
    print(
        'ref. VDI Wärmeatlas H2 Dampfdruck bei -256.6K: ' +
        '{:.4g}'.format(
            p_sat(-256.6 + 273.15, -0.216, 33.19, 13.13).x.item() * 1000
        ) + ' mbar. (Literaturwert 250mbar)'
    )
    p_new = p_i_sat_ceos(-256.6 + 273.15, 10, 33.19, 13.13, -0.216, alpha_tr, epsilon, sigma, psi, omega, max_it=100,
                         tol=tol)
    p_new = bubl_p(-256.6 + 273.15, 1, 1.0, 33.19, 13.13, -0.216,
                   alpha_tr, epsilon, sigma, psi, omega,
                   max_it=100, tol=tol,print_iterations=True)['p'].item()
    soln = secant_ls_3p(lambda p_var:
                 phi(-256.6 + 273.15, p_var, 1, 33.19, 13.13, -0.216, 'l',
                     alpha_tr, epsilon, sigma, psi, omega
                     )['phi'].item() -
                 phi(-256.6 + 273.15, p_var, 1, 33.19, 13.13, -0.216, 'v',
                             alpha_tr, epsilon, sigma, psi, omega)['phi'].item()
                 , 0.7, tol=tol, x_1=1.001 * 0.7,
                 restriction=lambda p_val: p_val > 0,
                 print_iterations=True)
    phi_sat = phi(-256.6 + 273.15, soln['x'], 1, 33.19, 13.13, -0.216, 'v',
                             alpha_tr, epsilon, sigma, psi, omega)

    t = 273.15 + linspace(-260, 400, 50)
    p_sat_vals = empty([len(t), len(pc)])
    p_sat_vals_ceos = empty([len(t), len(pc)]) * nan
    p0 = 1.0 * ones(len(pc))
    for i in range(len(t)):
        p_sat_vals[i, :] = 10**(
                ant_a - ant_b / (t[i] - 273.15 + ant_c)
        ) * 1 / 760 * 101325 / 1e5  # bar
        for j in range(len(pc)):
            if t[i] >= tc[j]:
                pass  # skip
            else:
                p_min = p_est(t[i], p0[j], 1.0, tc[j], pc[j], omega_af[j], 100, tol)['p_min_l']
                if p_min < -1000:
                    print(bubl_point_step_l_k(t[i], p0[j], 1.0, tc[j], pc[j], omega_af[j],
                                              alpha_tr, epsilon, sigma, psi, omega,
                                              max_it=100, tol=1e-10, full_output=True, y_i_est=1.0))
                else:
                    # p_sat_vals_ceos[i, j] = bubl_p(
                    #     t[i], p0[j], 1.0, tc[j], pc[j], omega_af[j],
                    #     alpha_tr, epsilon, sigma, psi, omega,
                    #     max_it=100, tol=1e-10, print_iterations=False)['p'].item()
                    p_sat_vals_ceos[i, j] = p_i_sat_ceos(t[i], p0[j], tc[j], pc[j], omega_af[j], alpha_tr, epsilon,
                                                         sigma, psi, omega, max_it=100, tol=1e-10)['p']
                    p0[j] = p_sat_vals_ceos[i, j]

    lines = plt.plot(t, p_sat_vals)
    lines_2 = plt.plot(t, p_sat_vals_ceos, 'x', fillstyle='none')
    for i in range(len(lines)):
        lines_2[i].set_color(lines[i].get_color())
        lines[i].set_label(labels[i])
    plt.xlabel('T / K')
    plt.ylabel('$p_{i}^{Sat}$ / bar')
    plt.ylim([0, 75])
    plt.legend()

    t = 273.15 - 256.6
    tc_i = 33.19
    pc_i = 13.13
    af_omega_i = -0.216
    z_i = asarray(1.0)
    p_min = p_est(t, 1e-3, z_i, tc_i, pc_i, af_omega_i, 100, tol)['p_min_l']

    phi(-256.6 + 273.15, 0.2620861427179638, 1, 33.19, 13.13, -0.216, 'l',
        alpha_tr, epsilon, sigma, psi, omega)
    p_i_sat_ceos(-256.6 + 273.15, 0.2620861427179638, 33.19, 13.13, -0.216, alpha_tr, epsilon, sigma, psi, omega,
                 max_it=100, tol=tol)
    p_range = concatenate([linspace(-71, 0.001, 10), linspace(0.001, 10, 20)])

    markers = plt.Line2D.filled_markers
    fig2 = plt.figure()
    plot1 = plt.subplot2grid([2, 2], [1, 0], rowspan=1, colspan=1)
    plot2 = plt.subplot2grid([2, 2], [1, 1], rowspan=1, colspan=1)
    plot3 = plt.subplot2grid([2, 2], [0, 0], rowspan=1, colspan=1)
    plot4 = plt.subplot2grid([2, 2], [0, 1], rowspan=1, colspan=1)

    v_plot = []
    p_plot = []
    rho_plot = []
    z_plot = []
    z_complex = []
    p_complex = []

    rho_l_phase = []
    rho_v_phase = []
    p_v_phase = []
    for p in p_range:
        tr_i = t / tc_i
        a_i = psi * alpha_tr(tr_i, af_omega_i) * r ** 2 * tc_i ** 2 / pc_i
        b_i = omega * r * tc_i / pc_i
        beta_i = b_i * p / (r * t)
        q_i = a_i / (b_i * r * t)
        a_ij = sqrt(outer(a_i, a_i))

        # Variablen, die von der Flüssigkeit-Zusammensetzung abhängig sind
        b = sum(z_i * b_i)
        a = z_i.dot(a_ij).dot(z_i).item()
        beta = b * p / (r * t)
        q = a / (b * r * t)
        a_mp_i = -a + 2 * a_ij.dot(z_i)  # partielles molares a_i
        b_mp_i = b_i  # partielles molares b_i
        q_mp_i = q * (1 + a_mp_i / a - b_i / b)  # partielles molares q_i

        a1 = beta * (epsilon + sigma) - beta - 1
        a2 = q * beta + epsilon * sigma * beta ** 2 \
             - beta * (epsilon + sigma) * (1 + beta)
        a3 = -(epsilon * sigma * beta ** 2 * (1 + beta) +
               q * beta ** 2)

        soln = solve_cubic([1, a1, a2, a3])
        roots, disc = soln['roots'], soln['disc']
        re_roots = array([roots[0][0], roots[1][0], roots[2][0]])

        if disc <= 0:
            # 3 real roots. smallest ist liq. largest is gas.
            z_l = re_roots[0]
            z_mid = re_roots[1]
            z_v = re_roots[2]
            v_l = z_l * r * t / p
            v_mid = z_mid * r * t / p
            v_v = z_v * r * t / p
            p_plot += [p, p, p]
            v_plot += [v_l, v_mid, v_v]
            rho_plot += [1 / v_l, 1 / v_mid, 1 / v_v]
            z_plot += [z_l, z_mid, z_v]
        elif disc > 0:
            # one real root, 2 complex. First root is the real one.
            z = re_roots[0]
            v = z * r * t / abs(p)
            p_plot += [p]
            v_plot += [v]
            rho_plot += [1 / v]
            z_plot += [z]
            z_complex += [re_roots[1]]
            p_complex += [p]

    for p in linspace(1e-4, max(p_range), 30):
        rho_l_phase += [
            z_phase(t, p, z_i, tc_i, pc_i, af_omega_i, 'l', alpha_tr, epsilon, sigma, psi, omega, tol)['rho']]
        rho_v_phase += [
            z_phase(t, p, z_i, tc_i, pc_i, af_omega_i, 'v', alpha_tr, epsilon, sigma, psi, omega, tol)['rho']]
        p_v_phase += [p]

    current_marker = markers[randint(0, len(markers))]
    plot1.axvline(b, linestyle='--')
    plot2.axvline(1 / b, linestyle='--')
    plot4.axvline(1 / b, linestyle='--')
    v_line = plot1.semilogx(v_plot, p_plot, current_marker,
                            label=r'$z_1={:g}$'.format(z_i),
                            fillstyle='none')
    current_marker = plt.get(v_line[0], 'marker')
    current_color = plt.get(v_line[0], 'color')
    plot2.plot(rho_plot, p_plot, markers[randint(0, len(markers))],
               label=r'$z_1={:g}$'.format(z_i),
               fillstyle='none')
    plot4.plot(rho_l_phase, p_v_phase, current_marker, markeredgewidth=0.5,
               color=current_color, markersize=4, fillstyle='bottom', linestyle='none',
               label=r'$L: x_1={:g}$'.format(z_i))
    plot4.plot(rho_v_phase, p_v_phase, current_marker, markeredgewidth=0.25,
                   color=current_color, markersize=4, fillstyle='none', linestyle='--',
                   label=r'$V: y_1={:g}$'.format(z_i))

    plot3.plot(z_plot, p_plot, current_marker,
               label=r'$z_1={:g}$'.format(z_i),
               fillstyle='bottom', linestyle='none')
    plot3.plot(z_complex, p_complex, current_marker, markersize=4, linestyle='--',
               fillstyle='none', color=current_color, markeredgewidth=0.25, linewidth=0.5)
    p_low = z_phase(t, p, z_i, tc_i, pc_i, af_omega_i, 'l', alpha_tr, epsilon, sigma, psi, omega, tol)['p_low']
    plot1.axhline(p_low, linestyle='-.', color='gray', linewidth=0.5, label='$P_{low}$')
    plot2.axhline(p_low, linestyle='-.', color='gray', linewidth=0.5, label='$P_{low}$')
    plot3.axhline(p_low, linestyle='-.', color='gray', linewidth=0.5, label='$P_{low}$')
    plot4.axhline(p_low, linestyle='-.', color='gray', linewidth=0.5, label='$P_{low}$')
    plot1.set_xlabel(r'$\frac{v}{cm^3/mol}$')
    plot1.set_ylabel('p / bar')
    plot1.legend(fontsize=6)
    plot2.set_xlabel(r'$\frac{rho}{mol / cm^3}$')
    plot2.set_ylabel('p / bar')
    plot2.set_title(r'$\rho$' + ', act.')
    plot3.set_xlabel(r'$Z$')
    plot3.set_ylabel('p / bar')
    plot3.legend(['real root', 'real part of complex root'], fontsize=6)
    plot4.set_xlabel(r'$\frac{rho}{mol / cm^3}$')
    plot4.set_ylabel('p / bar')
    plot4.set_title(r'pseudo-$\rho$' + ', L/V [3]')
    plot4.legend(fontsize=6)
    plt.tight_layout()
    plt.show()


def svn_14_1():
    x_i = array([0.4, 0.6])
    tc_i = array([126.2, 190.6])
    pc_i = array([34., 45.99])
    af_omega_i = array([0.038, 0.012])

    mm_i = zeros(2)
    state1 = State(200, 30, x_i, mm_i, tc_i, pc_i, af_omega_i, 'srk')
    print(state1.solve())



def svn_14_2():
    use_pr_eos()
    x_i = array([0.2, 0.8])
    y_i = array([0.2, 0.8])  # Est
    tc_i = array([190.6, 425.1])
    pc_i = array([45.99, 37.96])
    af_omega_i = array([0.012, 0.200])
    ant_a = array([3.7687, 3.93266])
    ant_b = array([395.744, 935.773])
    ant_c = array([266.681, 238.789])
    max_it = 100
    print(bubl_point_step_l_k(
        310.92, 30, x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
        max_it=max_it,full_output=True, y_i_est=y_i))
    y_i = bubl_point_step_l_k(
        310.92, 30, x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
        max_it=max_it, full_output=True, y_i_est=y_i)['y_i']
    for i in range(5):
        soln = bubl_point_step_l_k(
            310.92, 30, x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
            max_it=max_it, full_output=True, y_i_est=y_i)
        y_i = soln['y_i']
        k_i = soln['k_i']
        print(y_i)
        print(1 - sum(y_i))
        print(sum(k_i * x_i))
    soln = bubl_p(
        310.92, 1., x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
        max_it=max_it, tol=1e-10, y_i_est=y_i)
    print(soln)
    print(dew_point_step_l_k(
        310.92, soln['p'], soln['y_i'], tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
        max_it=max_it))
    print(dew_p(
        310.92, 30, soln['y_i'], tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
        max_it=max_it, tol=1e-10, x_i_est=x_i, print_iterations=True))
    print(bubl_point_step_l_k(
        310.92, soln['p'], x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
        max_it=max_it, full_output=True, y_i_est=y_i))

    x = linspace(0.0, 0.8, 50)
    y = empty_like(x)
    p_v = empty_like(x)

    y_dew = linspace(0.0, 0.8789, 50)
    x_dew = y.copy()
    p_v_dew = p_v.copy()

    x_i = array([x[0], 1 - x[0]])
    y_i_est = array([x[0], 1 - x[0]])

    y_i_dew = array([y_dew[0], 1 - y_dew[0]])
    x_i_est_dew = array([y_dew[0], 1 - y_dew[0]])

    p_v0 = 1.0
    p_v_0_dew = 1.0
    for i in range(len(x)):
        x_i = array([x[i], 1 - x[i]])
        soln = bubl_p(310.92, p_v0, x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
                      max_it=max_it, tol=1e-10, y_i_est=y_i_est)
        p_v[i] = soln['p']
        y[i] = soln['y_i'][0]
        p_v0 = p_v[i]
        y_i_est = soln['y_i']


        y_i_dew = array([y_dew[i], 1 - y_dew[i]])
        soln_dew = dew_p(310.92, p_v_0_dew, y_i_dew, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
                         max_it=max_it, tol=1e-10, x_i_est=x_i_est_dew)
        p_v_dew[i] = soln_dew['p']
        x_dew[i] = soln_dew['x_i'][0]
        p_v_0_dew = p_v_dew[i]
        x_i_est_dew = soln_dew['x_i']
    line1 = plt.plot(x, p_v, label=r'$x_1(L)$ bubl_p')
    line2 = plt.plot(y, p_v, label=r'$y_1(V)$ bubl_p')
    plt.plot(x_dew, p_v_dew, 'o', fillstyle='none',
             color=line1[0].get_color(), label=r'$x_1(L)$ dew_p')
    plt.plot(y_dew, p_v_dew, 'o', fillstyle='none',
             color=line2[0].get_color(), label=r'$y_1(L)$ dew_p')
    plt.ylabel('p / bar')
    plt.xlabel(r'$x_1 , y_1$')
    plt.legend()
    plt.show()


def svn_14_2_behchmark():
    use_srk_eos()
    x_i = array([0.2, 0.8])
    y_i = array([0.2, 0.8])  # Est
    tc_i = array([190.6, 425.1])
    pc_i = array([45.99, 37.96])
    af_omega_i = array([0.012, 0.200])
    max_it = 100
    print(bubl_point_step_l_k(
        310.92, 30, x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
        max_it=max_it,full_output=True, y_i_est=y_i))
    y_i = bubl_point_step_l_k(
        310.92, 30, x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
        max_it=max_it, full_output=True, y_i_est=y_i)['y_i']
    for i in range(5):
        soln = bubl_point_step_l_k(
            310.92, 30, x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
            max_it=max_it, full_output=True, y_i_est=y_i)
        y_i = soln['y_i']
        k_i = soln['k_i']
        print(y_i)
        print(1 - sum(y_i))
        print(sum(k_i * x_i))
    soln = bubl_p(
        310.92, 1., x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
        max_it=max_it, tol=1e-10, y_i_est=y_i)
    print(soln)
    print(bubl_point_step_l_k(
        310.92, soln['p'], x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
        max_it=max_it, full_output=True, y_i_est=y_i))

    x = linspace(0.0, 0.8, 50)
    y = empty_like(x)
    x_i = array([x[0], 1 - x[0]])
    y_i_est = array([x[0], 1 - x[0]])
    p_v = empty_like(x)
    p_v0 = 1.0
    for i in range(len(x)):
        x_i = array([x[i], 1 - x[i]])
        soln = bubl_p(310.92, p_v0, x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
                      max_it=max_it, tol=1e-10, y_i_est=y_i_est)
        p_v[i] = soln['p']
        y[i] = soln['y_i'][0]
        p_v0 = p_v[i]
        y_i_est = soln['y_i']
    line1 = plt.plot(x, p_v, label=r'$x_1(L)$ bubl_p')
    line2 = plt.plot(y, p_v, label=r'$y_1(V)$ bubl_p')
    plt.ylabel('p / bar')
    plt.xlabel(r'$x_1 , y_1$')
    plt.legend()
    plt.show()


def svn_fig_14_8():
    # plot fig 14.8
    use_srk_eos()
    x_i = array([0.2, 0.8])
    y_i = array([0.2, 0.8])  # Est
    tc_i = array([190.6, 425.1])
    pc_i = array([45.99, 37.96])
    af_omega_i = array([0.012, 0.200])
    ant_a = array([3.7687, 3.93266])
    ant_b = array([395.744, 935.773])
    ant_c = array([266.681, 238.789])
    max_it = 100
    markers = plt.Line2D.filled_markers
    p_range = linspace(0.1, 140, 40)
    plot1 = plt.subplot2grid([2, 2], [1, 0], rowspan=1, colspan=1)
    plot2 = plt.subplot2grid([2, 2], [1, 1], rowspan=1, colspan=1)
    plot3 = plt.subplot2grid([2, 2], [0, 0], rowspan=1, colspan=1)
    plot4 = plt.subplot2grid([2, 2], [0, 1], rowspan=1, colspan=1)
    for x in linspace(0.0, 1.0, 10 + 1):
        z_i = array([x, 1 - x])
        v_plot = []
        p_plot = []
        rho_plot = []
        z_plot = []
        z_complex = []
        p_complex = []

        rho_l_phase = []
        rho_v_phase = []
        p_v_phase = []
        t = 310.92
        for p in p_range:
            tr_i = t / tc_i
            a_i = psi * alpha_tr(tr_i, af_omega_i) * r ** 2 * tc_i ** 2 / pc_i
            b_i = omega * r * tc_i / pc_i
            beta_i = b_i * p / (r * t)
            q_i = a_i / (b_i * r * t)
            a_ij = sqrt(outer(a_i, a_i))

            # Variablen, die von der Flüssigkeit-Zusammensetzung abhängig sind
            b = sum(z_i * b_i)
            a = z_i.dot(a_ij).dot(z_i)
            beta = b * p / (r * t)
            q = a / (b * r * t)
            a_mp_i = -a + 2 * a_ij.dot(z_i)  # partielles molares a_i
            b_mp_i = b_i  # partielles molares b_i
            q_mp_i = q * (1 + a_mp_i / a - b_i / b)  # partielles molares q_i

            a1 = beta * (epsilon + sigma) - beta - 1
            a2 = q * beta + epsilon * sigma * beta ** 2 \
                 - beta * (epsilon + sigma) * (1 + beta)
            a3 = -(epsilon * sigma * beta ** 2 * (1 + beta) +
                   q * beta ** 2)

            soln = solve_cubic([1, a1, a2, a3])
            roots, disc = soln['roots'], soln['disc']
            re_roots = array([roots[0][0], roots[1][0], roots[2][0]])

            if disc <= 0 and all(re_roots >= 0):
                # 3 real roots. smallest ist liq. largest is gas.
                z_l = re_roots[0]
                z_mid = re_roots[1]
                z_v = re_roots[2]
                v_l = z_l * r * t / p
                v_mid = z_mid * r * t / p
                v_v = z_v * r * t / p
                p_plot += [p, p, p]
                v_plot += [v_l, v_mid, v_v]
                rho_plot += [1 / v_l, 1 / v_mid, 1 / v_v]
                z_plot += [z_l, z_mid, z_v]
                # z_complex += [re_roots[1]]
                # p_complex += [p]
            elif disc > 0:
                # one real root, 2 complex. First root is the real one.
                z = re_roots[0]
                v = z * r * t / abs(p)
                p_plot += [p]
                v_plot += [v]
                rho_plot += [1 / v]
                z_plot += [z]
                z_complex += [re_roots[1]]
                p_complex += [p]

        for p in linspace(1e-4, max(p_range), 30):
            rho_l_phase += [
                z_phase(t, p, z_i, tc_i, pc_i, af_omega_i, 'l', alpha_tr, epsilon, sigma, psi, omega, tol)['rho']]
            rho_v_phase += [
                z_phase(t, p, z_i, tc_i, pc_i, af_omega_i, 'v', alpha_tr, epsilon, sigma, psi, omega, tol)['rho']]
            p_v_phase += [p]

        current_marker = markers[randint(0, len(markers))]
        plot1.axvline(b, linestyle='--')
        if x < 0.3:
            plot2.axvline(1 / b, linestyle='--')
            plot4.axvline(1 / b, linestyle='--')
        v_line = plot1.semilogx(v_plot, p_plot, current_marker,
                                label=r'$z_1={:g}$'.format(z_i[0], z_i[1]),
                                fillstyle='none')
        current_marker = plt.get(v_line[0], 'marker')
        current_color = plt.get(v_line[0], 'color')
        plot2.plot(rho_plot, p_plot, markers[randint(0, len(markers))],
                   label=r'$z_1={:g}, z_2={:g}$'.format(z_i[0], z_i[1]),
                   fillstyle='none')
        if round(x, 1) in [0.0, 0.3, 0.6, 0.9]:
            plot4.plot(rho_l_phase, p_v_phase, current_marker, markeredgewidth=0.5,
                       color=current_color, markersize=4, fillstyle='bottom', linestyle='none',
                       label=r'$L: x_1={:g}$'.format(z_i[0]))
            plot4.plot(rho_v_phase, p_v_phase, current_marker, markeredgewidth=0.25,
                       color=current_color, markersize=4, fillstyle='none', linestyle='--',
                       label=r'$V: y_1={:g}$'.format(z_i[0]))

        plot3.plot(z_plot, p_plot, current_marker,
                   label=r'$z_1={:g}, z_2={:g}$'.format(z_i[0], z_i[1]),
                   fillstyle='bottom', linestyle='none')
        plot3.plot(z_complex, p_complex, current_marker, markersize=4, linestyle='--',
                   fillstyle='none', color=current_color, markeredgewidth=0.25, linewidth=0.5)
        if x in [0.4, 0.5, 0.6, 0.7]:
            p_est(t, p, z_i, tc_i, pc_i, af_omega_i, max_it, tol)
    plot1.set_xlabel(r'$\frac{v}{cm^3/mol}$')
    plot1.set_ylabel('p / bar')
    plot1.legend(fontsize=6)
    plot2.set_xlabel(r'$\frac{rho}{mol / cm^3}$')
    plot2.set_ylabel('p / bar')
    plot2.set_title(r'$\rho$' + ', act.')
    plot3.set_xlabel(r'$Z$')
    plot3.set_ylabel('p / bar')
    plot3.legend(['real root', 'real part of complex root'], fontsize=6)
    plot4.set_xlabel(r'$\frac{rho}{mol / cm^3}$')
    plot4.set_ylabel('p / bar')
    plot4.set_title(r'pseudo-$\rho$' + ', L/V [3]')
    plot4.legend(fontsize=6)
    plt.tight_layout()
    plt.show()


def svn_tab_14_1_2():
    use_pr_eos()
    p = 1.01325 # bar
    max_it = 100
    tol = 1e-10
    tc_i = array([507.6 , 513.92, 532.79, 562.05])
    pc_i = array([30.35, 61.32, 37.84, 48.98])
    af_omega_i = array([0.3  , 0.649, 0.227, 0.21 ])
    z_i = array([0.162, 0.068, 0.656, 0.114])
    # unifac secondary groups and coefficients by component
    sec_i = array([1, 2, 3, 9, 14])
    nu_ji = array([[2, 4, 0, 0, 0], [1, 1, 0, 0, 1], [1, 4, 1, 0, 0], [0, 0, 0, 6, 0]])
    unifac_data_dict = setup_unifac_data()

    soln = bubl_p(334.82, 1.0, z_i, tc_i, pc_i, af_omega_i,
                  alpha_tr, epsilon, sigma, psi, omega,
                  sec_i, nu_ji, unifac_data_dict,
                  max_it, tol, print_iterations=True)
    soln = bubl_t(273.15, p, z_i, tc_i, pc_i, af_omega_i,
                  alpha_tr, epsilon, sigma, psi, omega,
                  sec_i, nu_ji, unifac_data_dict,
                  max_it, 1e-10, print_iterations=True)
    y_i = soln['y_i']
    k_i = soln['k_i']
    t = soln['t']
    p_i_sat = soln['p_i_sat']
    n_it = soln['n_it']
    phi_coef_fun_i = soln['phi_coef_fun_i']

    unifac_data_dict = setup_unifac_data()
    gamma_j = gamma_u(t, z_i, sec_i, nu_ji, unifac_data_dict)
    print('\n'*2)
    print('SVN Table 14.1 - n-hexane (1) / ethanol (2) / methylcyclopentane (3) /' +
          'benzene (4) at {:0.2f} K'.format(t))

    print('i\tx_i\t\ty_i\t\tp_i_sat\tphi\t\tgamma\tK_i')
    for i in range(len(z_i)):
        print(('{:d}'+'\t{:.4g}'*6).format(
            i, z_i[i], y_i[i], p_i_sat[i], phi_coef_fun_i[i], gamma_j[i], k_i[i]))
    print('p: {:0.3f} bar\tT(calc): {:0.3f} K\tIterations:{:d}'.format(p, t, n_it))
    print('\n'*2)

    z_i = array([0.250, 0.400, 0.200, 0.150])
    soln = bubl_t(273.15, p, z_i, tc_i, pc_i, af_omega_i,
                  alpha_tr, epsilon, sigma, psi, omega,
                  sec_i, nu_ji, unifac_data_dict,
                  max_it, 1e-10, print_iterations=True)
    t = soln['t']
    soln = pt_flash(334.152/334.85*334.15, p, z_i, tc_i, pc_i, af_omega_i,
                    alpha_tr, epsilon, sigma, psi, omega,
                    sec_i, nu_ji, unifac_data_dict,
                    max_it=max_it, tol=tol)
    y_i = soln['y_i']
    x_i = soln['x_i']
    z_i = soln['z_i']
    k_i = soln['k_i']
    v_f = soln['v_f']
    t = soln['t']
    p = soln['p']
    n_it = soln['n_it']
    phi_coef_fun_i = soln['phi_coef_fun_i']

    print('SVN Table 14.2 - n-hexane (1) / ethanol (2) / methylcyclopentane (3) /' +
          'benzene (4) at {:0.4g} bar and {:0.4g} K'.format(p, t))
    print('i\tz_i\t\tx_i\t\ty_i\t\tK_i\t\tPhi_i\t\tgamma_i')
    for i in range(len(z_i)):
        print(('{:d}' + '\t{:.4f}' * 6).format(
            i, z_i[i], x_i[i], y_i[i], k_i[i], phi_coef_fun_i[i], gamma_j[i]))
    print('p: {:0.3f} bar\tT(calc): {:0.3f} K\tIterations:{:d}'.format(p, t, n_it))
    print('\n' * 2)


def pat_ue_03_flash():
    use_pr_eos()
    n = array([
        205.66,
        14377.78,
        1489.88,
        854.75,
        1348.86,
        2496.13,
        0
    ])

    z_i = n / sum(n)
    x_i = 1 / len(n) * ones(len(n))
    y_i = 1 / len(n) * ones(len(n))
    tc_i = tc
    pc_i = pc
    af_omega_i = omega_af
    t = 60 + 273.15
    p = 50.

    # soln = isot_flash(t, p, x_i, y_i, z_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega)
    # p_i_sat_ceos(t, 1.0, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega)
    soln = pt_flash(t, p, z_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega, tol=1e-10)
    y_i = soln['y_i']
    x_i = soln['x_i']
    v_f = soln['v_f']
    k_i = soln['k_i']
    print('v_f: ')
    print(v_f)
    print('k_i: ')
    print(k_i)
    print('l_i: ')
    print(sum(n) * (1 - v_f) * x_i)
    print('v_i: ')
    print(sum(n) * v_f * y_i)
    print('f_i: ')
    print(sum(n) * (1 - v_f) * x_i + sum(n) * v_f * y_i)


def isot_flash_seader_4_1():
    use_pr_eos()
    n = array([
        10,
        20,
        30,
        40
    ], dtype=float)
    z_i = n / sum(n)
    x_i = 1 / len(n) * ones(len(n))
    y_i = 1 / len(n) * ones(len(n))
    tc_i = array([
        369.82,
        425.13,
        469.66,
        507.79
    ])
    pc_i = array([
        42.48,
        37.96,
        33.69,
        30.42
    ])
    af_omega_i = array([
        0.152,
        0.201,
        0.252,
        0.300
    ])
    t = (200 - 32) * 5 / 9 + 273.15
    p = 6.895
    for i in range(10):
        soln = isot_flash(t, p, x_i, y_i, z_i, tc_i, pc_i, af_omega_i,
                          alpha_tr, epsilon, sigma, psi, omega)
        x_i = soln['x_i']
        y_i = soln['y_i']
        v_f = soln['v_f']
        k_i = soln['k_i']
        print(x_i)
        print(y_i)
        print(v_f)
        print(k_i)


def pat_ue_03_vollstaendig(rlv, print_output=False):
    # Als Funktion des Rücklaufverhältnises.
    use_pr_eos()

    # log
    old_stdout = sys.stdout
    log_file = open('output.log', 'w')
    # sys.stdout = log_file

    p = 50.  # bar
    temp = 273.15 + 220.  # K
    t_flash = 273.15 + 60  # K
    t0_ref = 298.15  # K
    r = 8.314  # J/(mol K)
    # rlv = 0.2  # Rücklaufverhältnis
    # rlv = 0.606089894 # Rücklaufverhältnis für 350kmol/h in der Flüssigkeit

    namen = ['CO', 'H2', 'CO2', 'H2O', 'CH3OH', 'N2']

    n0co = 750.  # kmol/h
    n0h2 = 5625.  # kmol/h
    n0co2 = 750.  # kmol/h
    n0h2o = 375.  # kmol/h
    n0ch3oh = 0.  # kmol/h
    n0n2 = 500.  # kmol/h

    ne = array([n0co, n0h2, n0co2, n0h2o, n0ch3oh, n0n2])

    nuij = array([[-1, -2, 0, 0, +1, 0],
                     [0, -3, -1, +1, +1, 0],
                     [-1, +1, +1, -1, 0, 0]]).T

    h_298 = array(
        [-110.541, 0., -393.505, -241.826, -201.167, 0.]) * 1000  # J/mol

    g_298 = array([-169.474, -38.962, -457.240, -
                      298.164, -272.667, -57.128]) * 1000  # J/mol

    # Kritische Parameter Tc, Pc, omega(azentrischer Faktor)
    tc = array([
        132.86, 33.19, 304.13, 647.10, 513.38, 126.19
    ])  # K

    pc = array([
        34.98, 13.15, 73.77, 220.64, 82.16, 33.96
    ])  # bar

    omega_af = array(
        [0.050, -0.219, 0.224, 0.344, 0.563, 0.037]
    )

    # Berechne delta Cp(T) mit Temperaturfunktionen für ideale Gase (SVN).

    # Koeffizienten für Cp(T)/R = A + B*T + C*T^2 + D*T^-2, T[=]K
    # Nach rechts hin: A, B, C, D
    # Nach unten hin: CO, H2, CO2, H2O, CH3OH, N2
    cp_coefs = array([z for z in [
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
            r * cp_coefs[:, 0] * t * log(t / t0_ref) -
            r * cp_coefs[:, 1] * t ** 2 * (1 - t0_ref / t) -
            r * cp_coefs[:, 2] / 2. * t ** 3 * (1 - (t0_ref / t) ** 2) +
            r * cp_coefs[:, 3] / 2. * 1 / t * (1 - (t / t0_ref) ** 2)
        )  # J/mol

    def k(t, g_t):
        delta_g_t = nuij.T.dot(g_t)
        return exp(-delta_g_t / (r * t))

    delta_gr_298 = nuij.T.dot(g_298)

    delta_hr_298 = nuij.T.dot(h_298)

    cp_493 = cp(493.15)  # J/(mol K)
    h_493 = h(493.15)  # J/mol
    g_493 = g(493.15, h_493)  # J/mol
    k_493 = k(493.15, g_493)  # []

    # Lösung des einfacheren Falls in schwierigerem Fall einwenden.
    def fun(x_vec):
        n2co = x_vec[0]
        n2h2 = x_vec[1]
        n2co2 = x_vec[2]
        n2h2o = x_vec[3]
        n2ch3oh = x_vec[4]
        n2n2 = x_vec[5]
        xi1 = x_vec[6]
        # Folgende Werte können benutzt sein, um entweder drei Reaktionen
        # oder eine Reaktion zu behandeln.
        xi2 = 0.
        xi3 = 0.
        t2 = 0.
        if len(x_vec) > 8:
            xi2 = x_vec[7]
            xi3 = x_vec[8]
            t2 = x_vec[9]
        elif len(x_vec) == 8:
            t2 = x_vec[7]

        # Stoffströme am Ausgang des Reaktors
        n2 = array([n2co, n2h2, n2co2, n2h2o, n2ch3oh, n2n2])
        n2_t = sum(n2)
        # Stoffströme am Austritt des Systems (Gas)
        # n = array([nco, nh2, nco2, nh2o, nch3oh, nn2])
        # Stoffströme am Austritt des Systems (Flüssigkeit)
        # nl = n0 - n
        # Stoffströme im Rücklaufstrom
        # nr = rlv * n
        # Reaktionslaufzahlen
        xi = array([xi1, xi2, xi3])

        h_0 = h(t_feed)
        cp_0 = cp(t_feed)
        cp_t2 = cp(t2)
        h_t2 = h(t2)
        g_t2 = g(t2, h_t2)
        k_t2 = k(t2, g_t2)

        # phi_l, phi_v, k_i. Lösung des isothermischen Verdampfers
        z_i = n2 / sum(n2)
        x_i = 1 / len(n2) * ones(len(n2))
        y_i = 1 / len(n2) * ones(len(n2))
        for i in range(10):
            soln = isot_flash(t_flash, p, x_i, y_i, z_i, tc, pc, omega_af, alpha_tr, epsilon, sigma, psi, omega)
            y_i = soln['y_i']
            x_i = soln['x_i']
            v_f = soln['v_f']
            k_i_verteilung = soln['k_i']
            # print('k_i: ')
            # print(k_i_verteilung)
            # print('l_i: ')
            # print(n2_t * (1 - v_f) * x_i)
            # print('v_i: ')
            # print(n2_t * v_f * y_i)

        nv = (n2_t * v_f) * y_i
        nvco = nv[0]
        nvh2 = nv[1]
        nvco2 = nv[2]
        nvh2o = nv[3]
        nvch3oh = nv[4]
        nvn2 = nv[5]

        delta_h_t2 = nuij.T.dot(h_t2)  # J/mol

        f1 = -n2co + rlv * nvco + n0co - xi1 + 0 - xi3
        f2 = -n2h2 + rlv * nvh2 + n0h2 - 2 * xi1 - 3 * xi2 + xi3
        f3 = -n2co2 + rlv * nvco2 + n0co2 + 0 - xi2 + xi3
        f4 = -n2h2o + rlv * nvh2o + n0h2o + 0 + xi2 - xi3
        f5 = -n2ch3oh + rlv * nvch3oh + n0ch3oh + xi1 + xi2 - 0
        f6 = -n2n2 + rlv * nvn2 + n0n2 + 0
        f7 = -k_t2[0] * (n2co * n2h2 ** 2) + \
            n2ch3oh * (p / 1.) ** -2 * (n2_t) ** -(-2)
        f8 = -k_t2[1] * (n2co2 * n2h2 ** 3) + \
            n2ch3oh * n2h2o * (p / 1.) ** -2 * (n2_t) ** -(-2)
        f9 = -k_t2[2] * (n2co * n2h2o) + \
            n2co2 * n2h2 * (p / 1.) ** 0 * (n2_t) ** -0
        f10 = sum(
            multiply(ne + rlv * nv, (h_0 - h_298)) -
            multiply(n2, (h_t2 - h_298))) + dot(xi, -delta_h_t2)

        res = [f1, f2, f3, f4, f5, f6, f7, f10]

        if len(x_vec) > 8:
            res = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
        elif len(x_vec) == 8:
            # res = [f1, f2, f3, f4, f5, f6, f7, f10]
            pass

        return res

    # Schlechte Anfangswerte können mit dem vereinfachten Fall
    # einer einzigen Reaktion gelöst werden:
    n0 = ne
    t0 = 493.15  # K
    t_feed = 493.15  # K

    xi0 = zeros([1, 1])

    x0 = append(n0, xi0)
    x0 = append(x0, [t0])

    sol = optimize.root(fun, x0)
    print(sol)

    # Mit der Lösung des vereinfachten Falls als Anfangswerte erreicht
    # man Konvergenz des vollständigen Problems.
    xi0 = array([sol.x[-2], 0., 0.])
    t0 = sol.x[-1]
    n0 = sol.x[:6]

    x0 = append(n0, xi0)
    x0 = append(x0, [t0])

    sol = optimize.root(fun, x0)

    print(sol)

    n2 = sol.x[:6]  # kmol/h
    n2_t = sum(n2)  # kmol/h
    xi = sol.x[6:-1]  # kmol/h
    t2 = sol.x[-1]  # K

    h_0 = h(t_feed)  # J/mol
    cp_0 = cp(t_feed)  # J/(mol K)
    cp_t2 = cp(t2)  # J/(mol K)
    h_t2 = h(t2)  # J/mol
    g_t2 = g(t2, h_t2)  # J/mol
    k_t2 = k(t2, g_t2)  # []

    h_t_flash = h(t_flash)  # J/mol

    delta_h_t2 = nuij.T.dot(h_t2)  # J/mol

    # phi_l, phi_v, k_i. Lösung des isothermischen Verdampfers
    z_i = n2 / sum(n2)
    x_i = 1 / len(n2) * ones(len(n2))
    y_i = 1 / len(n2) * ones(len(n2))
    for i in range(10):
        soln = isot_flash(t_flash, p, x_i, y_i, z_i, tc, pc, omega_af, alpha_tr, epsilon, sigma, psi, omega)
        y_i = soln['y_i']
        x_i = soln['x_i']
        v_f = soln['v_f']
        k_i_verteilung = soln['k_i']
    nv = (n2_t * v_f) * y_i  # kmol/h
    nl = (n2_t * (1 - v_f)) * x_i  # kmol/h
    nr = nv * rlv  # kmol/h
    npr = nv * (1 - rlv)  # kmol/h
    nmischer = ne + rlv * nv  # kmol/h

    tmischung = optimize.root(
        lambda t: sum(
            multiply(rlv * nv, (h(t_flash) - h_298)) +
            multiply(ne, (h_0 - h_298)) -
            multiply(ne + rlv * nv, (h(t) - h_298))
        ), (493.15 + t_flash) / 2
    ).x  # K

    q_f_heiz = sum(
        multiply(ne + rlv * nv, (h_0 - h_298)) -
        multiply(ne + rlv * nv, (h(tmischung) - h_298)))  # kJ/h

    q_a_reak = sum(
        multiply(ne + rlv * nv, (h_0 - h_298)) -
        multiply(n2, (h(t2) - h_298))) + \
        dot(xi, -delta_h_t2)  # kJ/h

    q_g_kueh = sum(
        multiply(n2, (h(t_flash) - h_298)) -
        multiply(n2, (h_t2 - h_298)))  # kJ/h

    print('========================================')
    print('MISCHER')
    print('========================================')
    print('Zulauf des gesammten Prozesses:')
    for i in range(len(namen)):
        print(namen[i] + ': ' + '{:0.16g}'.format(ne[i]) + ' kmol/h')
    print('n: ' + '{:0.16g}'.format(sum(ne)) + ' kmol/h')
    print('T: ' + '{:g}'.format(t_feed) + ' K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{:g}'.format(
        sum(multiply(ne, h(t_feed)))) + ' kJ/h')
    print('\n')
    print('Rücklaufstrom bei Rücklaufverhältnis ' +
          str(rlv) + ' :')
    for i in range(len(namen)):
        print(namen[i] + ': ' + '{:0.16g}'.format(nr[i]) + ' kmol/h')
    print('n: ' + '{:0.16g}'.format(sum(nr)) + ' kmol/h')
    print('T: ' + '{:g}'.format(t_flash) + ' K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{:g}'.format(
        sum(multiply(nr, h_t_flash))) + ' kJ/h')
    print('\n')
    print('Erzeugnis des Mischers:')
    for i in range(len(namen)):
        print(namen[i] + ': ' + '{:0.16g}'.format(nmischer[i]) + ' kmol/h')
    print('n: ' + '{:0.16g}'.format(sum(nmischer)) + ' kmol/h')
    print('T: ' + '{:g}'.format(tmischung.item()) + ' K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{:g}'.format(
        sum(multiply(ne + rlv * nv, h(tmischung)))) + ' kJ/h')
    print('\n\n')

    print('========================================')
    print('F-HEIZ')
    print('========================================')
    print('Aufheizung des Eingangsstroms:')
    print('Q: ' + '{:g}'.format(q_f_heiz) + ' kJ/h')
    print('\n\n')

    print('========================================')
    print('R-MeOH')
    print('========================================')
    print('Aufgeheizter Eingangsstrom am Reaktor:')
    for i in range(len(namen)):
        print(namen[i] + ': ' + '{:0.16g}'.format(nmischer[i]) + ' kmol/h')
    print('n: ' + '{:0.16g}'.format(sum(nmischer)) + ' kmol/h')
    print('T: ' + '{:g}'.format(t_feed) + ' K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{:g}'.format(
        sum(multiply(ne + rlv * nv, h_493))) + ' kJ/h')
    print('\n')

    print('Reaktionslaufzahlen des adiabatisch betriebenen Reaktors:')
    for i in range(len(nuij.T)):
        print('xi_' + str(i) + ': ' + '{:g}'.format(xi[i]))
    print('Q: ' + '{:g}'.format(q_a_reak) + ' kJ/h ~ 0')
    print('\n')

    print('Erzeugnis des Reaktors:')
    for i in range(len(namen)):
        print(namen[i] + ': ' + '{:0.16g}'.format(n2[i]) + ' kmol/h')
    print('n: ' + '{:0.16g}'.format(sum(n2)) + ' kmol/h')
    print('T: ' + '{:g}'.format(t2) + ' K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{:g}'.format(
        sum(multiply(n2, h_t2))) + ' kJ/h')
    print('\n\n')

    print('========================================')
    print('KUEHLER')
    print('========================================')
    print('Abkühlungsleistung des Ausgangsstroms:')
    print('Q: ' + '{:g}'.format(q_g_kueh) + ' kJ/h')
    print('\n\n')

    print('========================================')
    print('FLASH')
    print('========================================')
    print('Abgekühlter Strom nach Reaktorkühler:')
    for i in range(len(namen)):
        print(namen[i] + ': ' + '{:0.16g}'.format(n2[i]) + ' kmol/h')
    print('n: ' + '{:0.16g}'.format(sum(n2)) + ' kmol/h')
    for i in range(len(namen)):
        print('z(' + namen[i] + ')' + ': ' +
              '{:0.16g}'.format(n2[i] / sum(n2)))
    print('T: ' + '{:g}'.format(t2) + ' K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{:g}'.format(
        sum(multiply(n2, h_t2))) + ' kJ/h')
    print('\n\n')

    print('Dampf/Flüssigkeit Verhältnis am Flash:')
    print('V/F: ' + '{:0.16g}'.format(v_f.item()) + ' kJ/h')
    print('\n\n')

    print('Produkt Flüssigkeit:')
    for i in range(len(namen)):
        print(namen[i] + ': ' + '{:0.16g}'.format(nl[i]) + ' kmol/h')
    print('n: ' + '{:0.16g}'.format(sum(nl)) + ' kmol/h')
    for i in range(len(namen)):
        print('x(' + namen[i] + ')' + ': ' +
              '{:0.16g}'.format(nl[i] / sum(nl)))
    print('T: ' + '{:g}'.format(t_flash) + ' K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{:g}'.format(
        sum(multiply(nl, h_t_flash))) + ' kJ/h')
    print('\n\n')

    print('Dampf aus Verdampfer:')
    for i in range(len(namen)):
        print(namen[i] + ': ' + '{:0.16g}'.format(nv[i]) + ' kmol/h')
    print('n: ' + '{:0.16g}'.format(sum(nv)) + ' kmol/h')
    for i in range(len(namen)):
        print('y(' + namen[i] + ')' + ': ' +
              '{:0.16g}'.format(nv[i] / sum(nv)))
    print('T: ' + '{:g}'.format(t_flash) + ' K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{:g}'.format(
        sum(multiply(nv, h_t_flash))) + ' kJ/h')
    print('\n\n')

    print('Dampf-Ablauf:')
    for i in range(len(namen)):
        print(namen[i] + ': ' + '{:0.16g}'.format(npr[i]) + ' kmol/h')
    print('n: ' + '{:0.16g}'.format(sum(npr)) + ' kmol/h')
    for i in range(len(namen)):
        print('y(' + namen[i] + ')' + ': ' +
              '{:0.16g}'.format(npr[i] / sum(npr)))
    print('T: ' + '{:g}'.format(t_flash) + ' K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{:g}'.format(
        sum(multiply(npr, h_t_flash))) + ' kJ/h')
    print('\n\n')

    print('Rücklaufstrom bei Rücklaufverhältnis ' +
          str(rlv) + ' :')
    for i in range(len(namen)):
        print(namen[i] + ': ' + '{:0.16g}'.format(nr[i]) + ' kmol/h')
    print('n: ' + '{:0.16g}'.format(sum(nr)) + ' kmol/h')
    print('T: ' + '{:g}'.format(t_flash) + ' K')
    print('p: ' + '{:g}'.format(p) + ' bar')
    print('H: ' + '{:g}'.format(
        sum(multiply(nr, h_t_flash))) + ' kJ/h')
    print('\n\n')

    print('Umsatz (CO): ' + '{:g}'.format(
        (ne[0] - nl[0] - nv[0] * (1 - rlv)) / ne[0]
    ))

    print('Umsatz (CO2): ' + '{:g}'.format(
        (ne[2] - nl[2] - nv[2] * (1 - rlv)) / ne[2]
    ))

    print('Ausbeute (CH3OH/CO, Flüssigkeit): ' + '{:g}'.format(
        (nl[-2] + 0 * nv[-2] * (1 - rlv) - ne[-2]) /
        (ne[0] - nl[0] - nv[0] * (1 - rlv))
    ))

    print('Ausbeute (CH3OH/CO2, Flüssigkeit): ' + '{:g}'.format(
        (nl[-2] + 0 * nv[-2] * (1 - rlv) - ne[-2]) /
        (ne[2] - nl[2] - nv[2] * (1 - rlv))
    ))

    sys.stdout = old_stdout
    log_file.close()

    log_file = open('output.log', 'r')

    if print_output:
        print(log_file.read())

    log_file.close()

    return nl[-2]  # Methanol in der Flüssigkeit


def interaction_pairs(list, i=0, j=0, perms=[]):
    # recursively produce pairs of indexes for interaction parameters
    if i + 1 >= len(list) and j + 1 >= len(list):
        return perms + [[list[i], list[j]]]
    elif j + 1 < len(list):
        return interaction_pairs(list, i, j + 1, perms + [[list[i], list[j]]])
    elif i + 1 < len(list):
        return interaction_pairs(list, i + 1, 0, perms + [[list[i], list[j]]])


def ppo_ex_8_12():
    t = 307
    x_j = array([0.047, 1 - 0.047])
    sec_j = array([1, 2, 18])
    nu_ij = array([[1, 0, 1], [2, 3, 0]])
    unifac_data_dict = setup_unifac_data()
    gamma_j = gamma_u(t, x_j, sec_j, nu_ij, unifac_data_dict)
    print('PPO example 8-12 - acetone (1) / n-pentane (2) at 307K')
    print('gamma_1: {:0.4f}\tgamma_2: {:0.4f}'.format(*gamma_j))
    print('\n')


def svn_h_1():
    t = 308.15
    x_j = array([0.4, 1 - 0.4])
    sec_j = array([1, 2, 32])
    nu_ij = array([[2, 1, 1], [2, 5, 0]])
    unifac_data_dict = setup_unifac_data()
    gamma_j = gamma_u(t, x_j, sec_j, nu_ij, unifac_data_dict)
    print('SVN Example H.1 - diethylamine (1) / n-heptane (2) at 308.15K')
    print('gamma_1: {:0.4f}\tgamma_2: {:0.4f}'.format(*gamma_j))
    print('\n')


def fredenslund_t_6():
    t = 318
    x_j = array([0.8869, 0.0991, 1 - 0.8869 - 0.0991])
    sec_j = array([1, 2, 9, 40])
    nu_ij = array([[0, 0, 0, 1], [0, 0, 6, 0], [2, 5, 0, 0]])
    unifac_data_dict = setup_unifac_data()
    gamma_j = gamma_u(t, x_j, sec_j, nu_ij, unifac_data_dict)
    print('Fredenslund 1975 Table 6 - acetonitrile (1) / benzene (2) / n-heptane (3) at 318K')
    print('x_1: {:0.4f}\tx_2: {:0.4f}\tx_3: {:0.4f}'.format(*x_j))
    print('gamma_1: {:0.4f}\tgamma_2: {:0.4f}\tgamma_3: {:0.4f}'.format(*gamma_j))
    print('\n')


def gamma_u(t, x_j, sec_j, nu_ij, unifac_data_dict):
    """Activity coefficients by UNIFAC

    Determines gamma_j(T) at temperature T by UNIFAC method.

    :param t: temperature / K
    :param x_j: composition
    :param sec_j: secondary groups / subgroups
    :param nu_ij: coefficients in secondary groups per component
    :param unifac_data_dict: dict with table columns subgroups_k, main_groups_of_k, r_k, q_k and
    interaction parameters table with A_ij on lower diagonal, A_ji on upper diagonal (nxn matrix)
    :return: gamma_j

    Example
    --------

    Fredenslund 1975 Table 6 - acetonitrile (1) / benzene (2) / n-heptane (3) at 318K

    >> t = 318

    >> x_j = array([0.8869, 0.0991, 1 - 0.8869 - 0.0991])

    >> sec_j = array([1, 2, 10, 41])

    >> nu_ij = array([[0, 0, 0, 1], [0, 0, 6, 0], [2, 5, 0, 0]])

    >> gamma_j = gamma_unifac(t, x_j, sec_j, nu_ij)

    >> print(gamma_j)

    [ 1.01793099  2.27445733 17.53230898]
    """
    subroups_k = unifac_data_dict['subgroups_k']
    main_groups_of_k = unifac_data_dict['main_groups_of_k']
    r_k = unifac_data_dict['r_k']
    q_k = unifac_data_dict['q_k']
    unifac_interaction_params = unifac_data_dict['unifac_interaction_params']

    indexes_j = [argwhere(x == subroups_k).item() for x in sec_j]
    q_m = q_k[indexes_j]
    r_m = r_k[indexes_j]
    groups = main_groups_of_k[indexes_j]
    pairs = interaction_pairs(groups - 1)

    r_j = nu_ij.dot(r_m)
    q_j = nu_ij.dot(q_m)
    phi_j = r_j * x_j / r_j.dot(x_j)
    theta_j = q_j * x_j / q_j.dot(x_j)
    z = 10
    i_j = z / 2 * (r_j - q_j) - (r_j - 1)
    ln_gamma_c_j = log(phi_j / x_j) + z / 2 * q_j * log(
        theta_j / phi_j) + i_j - phi_j / x_j * x_j.dot(i_j)

    a_mn = zeros([len(groups), len(groups)])
    counter = 0
    for i in range(len(groups)):
        for j in range(len(groups)):
            a_mn[i, j] = unifac_interaction_params[
                pairs[counter][0], pairs[counter][1]
            ]
            counter += 1
    psi_mn = exp(-a_mn / t)
    sum_nu_ij = sum(nu_ij, 1).reshape(len(nu_ij), 1)
    # residual activity coefficients in reference solution: molecules of type i only
    x_m = nu_ij / sum_nu_ij
    sum_x_m_q_m = sum(x_m * q_m, 1).reshape(len(nu_ij), 1)
    theta_m = x_m * q_m / sum_x_m_q_m
    theta_m.dot(psi_mn)
    ln_gamma_k_i = q_m * (
            1 - log(theta_m.dot(psi_mn)) - (
                theta_m / theta_m.dot(psi_mn)).dot(psi_mn.T)
    )
    # group residual activity coefficients
    x_k = x_j.dot(nu_ij)/x_j.dot(sum(nu_ij, 1))
    theta_k = x_k * q_m / x_k.dot(q_m)
    ln_gamma_k = q_m * (
            1 - log(theta_k.dot(psi_mn)) - (
                theta_k / theta_k.dot(psi_mn)).dot(psi_mn.T)
    )
    ln_gamma_r_j = sum(nu_ij * (ln_gamma_k - ln_gamma_k_i), 1)
    gamma_j = exp(ln_gamma_c_j + ln_gamma_r_j)
    return gamma_j


def setup_unifac_data():
    f = open('./data/unifac_list_of_interaction_parameters.csv', 'r')
    sep_char = f.readline().split('=')[-1].replace('\n', '')
    col_names = f.readline().split(sep_char)
    table_data = []
    for line in f:
        table_data += [line.split(sep_char)]
    f.close()
    data = array(table_data)
    indexes_i = array(data[:, 0], dtype=int)
    indexes_j = array(data[:, 1], dtype=int)
    a_ij = array(data[:, 2], dtype=float)
    a_ji = array(data[:, 3], dtype=float)
    unifac_interaction_params = zeros([len(indexes_i), len(indexes_j)])
    for row_no in range(len(data)):
        unifac_interaction_params[
            indexes_i[row_no] - 1, indexes_j[row_no] - 1
        ] = a_ij[row_no]
        unifac_interaction_params[
            indexes_j[row_no] - 1, indexes_i[row_no] - 1
        ] = a_ji[row_no]

    f = open('./data/unifac_sub_groups_surfaces_and_volumes.csv', 'r')
    sep_char = f.readline().split('=')[-1].replace('\n', '')
    col_names = f.readline().split(sep_char)
    table_data = []
    for line in f:
        table_data += [line.split(sep_char)]
    f.close()
    unifac_subgroup_data = array(table_data)
    subgroups_k = array(unifac_subgroup_data[:, 0], dtype=int)
    main_group_names_of_k = array(unifac_subgroup_data[:, 1], dtype=str)
    main_groups_of_k = array(
        [re.match('\[([0-9]*)\]', x).groups()[0] for x in
         array(unifac_subgroup_data[:, 2], dtype=str)],
        dtype=int)
    main_group_names_of_k = array(
        [re.match('\[[0-9]*\](.*)', x).groups()[0] for x in
         array(unifac_subgroup_data[:, 2], dtype=str)])
    r_k = array(unifac_subgroup_data[:, 3], dtype=float)
    q_k = array(unifac_subgroup_data[:, 4], dtype=float)

    unifac_data_dict = dict()
    for var in ['subgroups_k', 'r_k', 'q_k', 'main_groups_of_k',
                'main_group_names_of_k', 'unifac_interaction_params']:
        unifac_data_dict[var] = locals()[var]
    return unifac_data_dict

# vdi_atlas()
# svn_14_1()
# svn_fig_14_8()
# svn_14_2()
# svn_14_2_behchmark()
# zs_1998()
# ppo_ex_8_12()
# svn_h_1()
# fredenslund_t_6()
# svn_tab_14_1_2()
# pat_ue_03_flash()
# isot_flash_seader_4_1()
# pat_ue_03_vollstaendig(0.2)
# optimize.root(
#    lambda rlv: 350.0 - pat_ue_03_vollstaendig(rlv),
#    0.4
# )
# pat_ue_03_vollstaendig(0.64137041)
# pat_ue_03_vollstaendig(0.633)
# Die Lösung ist zwischen 0,65 (346kmol/h) und
# 0,70 (400kmol/h), aber in jenem Bereich entsteht ein
# Stabilitätsproblem
# pat_ue_03_vollstaendig(0.65, True)
