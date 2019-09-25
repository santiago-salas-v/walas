import sys
import re
from poly_3_4 import solve_cubic, solve_quartic
from numerik import secant_ls_3p, nr_ls, line_search
from poly_n import zroots
from numpy import array, append, zeros, abs, ones, empty_like, empty, argwhere, unique, asarray, concatenate
from numpy import sqrt, outer, sum, log, exp, multiply, diag, sign
from numpy import linspace, dot, nan, finfo, isnan, isinf
from numpy.random import randint
from scipy import optimize

r = 8.314 * 10. ** 6 / 10. ** 5  # bar cm^3/(mol K)
tol = finfo(float).eps

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
    epsilon = 0.
    sigma = 1.
    omega = 0.08664
    psi = 0.42748

    def alpha_tr(tr, af_omega): return \
        (tr) ** (-1 / 2.)


    return alpha_tr, epsilon, sigma, psi, omega


# Dampfdruck nach VDI-Wärmeatlas
def p_sat_func(psat, t, af_omega, tc, pc, alpha_tr, epsilon, sigma, psi, omega, full_output=False):
    tr = t / tc
    pr = psat / pc
    beta_i = omega * pr / tr
    q_i = psi * alpha_tr(tr, af_omega) / (omega * tr)

    z_l = optimize.root(
        lambda z: -z + beta_i +
        (z + epsilon * beta_i) * (z + sigma * beta_i) * (
            (1 + beta_i - z) / (q_i * beta_i)
        ), beta_i
    ).x.item()
    z_v = optimize.root(
        lambda z: -z + 1 + beta_i -
        q_i * beta_i * (z - beta_i) / (
            (z + epsilon * beta_i) * (z + sigma * beta_i)),
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


def p_sat(t, af_omega, tc, pc, alpha_tr, epsilon, sigma, psi, omega):
    def fs(psat): return p_sat_func(psat, t, af_omega, tc, pc,
                                    alpha_tr, epsilon, sigma, psi, omega)

    # Das Levenberg-Marquardt Algorithmus weist wenigere Sprunge auf.
    return optimize.root(fs, 1., method='lm')


def p_i_sat_ceos(t, p, tc_i, pc_i, af_omega_i,
                 alpha_tr, epsilon, sigma, psi, omega,
                 max_it=100, tol=tol):
    n_comps = asarray(tc_i).size
    p_sat_list = empty(n_comps)
    success = empty(n_comps, dtype=bool)
    n_fev = zeros(n_comps, dtype=int)
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
            n_fev[i] += first_step_soln['n_fev']
            success[i] = first_step_soln['success']
        if not success[i]:
            p_sat_list[i] = nan
        else:
            p0_it = first_step_soln['p']
            inv_slope = 1 / first_step_soln['ddisc_dp']
            p1_it = p0_it * (1 + sign(-inv_slope) * 0.01)
            if first_step_soln['n_fev'] == 1:
                # initial slope not necessarily convergent, direct it toward descending disc
                second_step_soln = approach_pt_i_sat_ceos(
                    t, p1_it, tc, pc, af_omega, alpha_tr, epsilon, sigma, psi, omega,
                    p_or_t='p', max_it=max_it, tol=tol)
                disc_0 = first_step_soln['disc']
                disc_1 = second_step_soln['disc']
                n_fev[i] += second_step_soln['n_fev']
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
            n_fev[i] += soln_temp['iterations'] + soln_temp['total_backtracks']
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
        n_fev[i] += first_step_soln['n_fev']
        if pc < p or not first_step_soln['success']:
            # no p_sat if supercritical
            t_sat_list[i] = nan
        else:
            t0_it = first_step_soln['t']
            inv_slope = 1 / first_step_soln['ddisc_dt']
            t1_it = t0_it * (1 + sign(-inv_slope) * 0.001)
            if first_step_soln['n_fev'] == 1:
                # initial slope not necessarily convergent, direct it toward descending disc
                second_step_soln = approach_pt_i_sat_ceos(
                    t1_it, p, tc, pc, af_omega, alpha_tr, epsilon, sigma, psi, omega,
                    p_or_t='t', max_it=max_it, tol=tol)
                disc_0 = first_step_soln['disc']
                disc_1 = second_step_soln['disc']
                n_fev[i] += second_step_soln['n_fev']
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
            n_fev[i] += soln_temp['iterations'] + soln_temp['total_backtracks']
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


def z_non_sat(t, p, x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega):
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
        lambda z: -z + 1 + beta -
        q * beta * (z - beta) / (
            (z + epsilon * beta) * (z + sigma * beta)),
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
        soln = secant_ls_3p(obj_fun, p, tol=tol, x_1=1.01 * p,
                            restriction=lambda p_val: p_val > 0,
                            print_iterations=print_iterations)
        p = soln['x']
        n_fev = soln['iterations'] + soln['total_backtracks']
        soln = bubl_point_step_l_k(t, p, x_i, tc_i, pc_i, af_omega_i,
                                   alpha_tr, epsilon, sigma, psi, omega, max_it,
                                   full_output=True, y_i_est=y_i_est)
        n_fev += soln['n_fev']
        soln['n_fev'] = n_fev
    else:
        soln = bubl_p_gamma_phi(
            t, p, x_i, tc_i, pc_i, af_omega_i,
            alpha_tr, epsilon, sigma, psi, omega,
            sec_j, nu_ij, unifac_data_dict,
            max_it, tol=tol, y_i_est=y_i_est)
        n_fev = soln['n_it']
        soln['n_fev'] = n_fev
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
        n_fev = soln['iterations'] + soln['total_backtracks']
        soln = bubl_point_step_l_k(t, p, x_i, tc_i, pc_i, af_omega_i,
                                   alpha_tr, epsilon, sigma, psi, omega, max_it,
                                   full_output=True, y_i_est=y_i_est)
        n_fev += soln['n_fev']
        soln['n_fev'] = n_fev
    else:
        soln = bubl_t_gamma_phi(
            t, p, x_i, tc_i, pc_i, af_omega_i,
            alpha_tr, epsilon, sigma, psi, omega,
            sec_j, nu_ij, unifac_data_dict,
            max_it, tol=tol, y_i_est=y_i_est)
        n_fev = soln['n_it']
        soln['n_fev'] = n_fev
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
        n_fev = 1 + i
        soln = dict()
        for item in ['x_i', 'y_i', 'phi_l', 'phi_v', 'z_l', 'z_v', 'k_i', 'sum_ki_xi',
                     'p', 't', 'success', 'n_fev']:
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
        soln = secant_ls_3p(obj_fun, p, tol=tol, x_1=0.99 * p,
                            restriction=lambda p_val: p_val > 0,
                            print_iterations=print_iterations)
        p = soln['x']
        n_fev = soln['iterations'] + soln['total_backtracks']
        soln = dew_point_step_l_k(t, p, y_i, tc_i, pc_i, af_omega_i,
                                  alpha_tr, epsilon, sigma, psi, omega, max_it,
                                  full_output=True, x_i_est=x_i_est)
        n_fev += soln['n_fev']
        soln['n_fev'] = n_fev
    else:
        soln = dew_p_gamma_phi(
            t, p, y_i, tc_i, pc_i, af_omega_i,
            alpha_tr, epsilon, sigma, psi, omega,
            sec_j, nu_ij, unifac_data_dict,
            max_it, tol=tol, x_i_est=x_i_est)
        n_fev = soln['n_it']
        soln['n_fev'] = n_fev
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
    n_fev = soln['iterations'] + soln['total_backtracks']
    soln = dew_point_step_l_k(t, p, y_i, tc_i, pc_i, af_omega_i,
                              alpha_tr, epsilon, sigma, psi, omega, max_it,
                              full_output=True, x_i_est=x_i_est)
    n_fev += soln['n_fev']
    soln['n_fev'] = n_fev
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
        n_fev = 1 + i
        soln = dict()
        for item in ['x_i', 'y_i', 'phi_l', 'phi_v', 'z_l', 'z_v', 'k_i',
                     'sum_yi_over_ki', 'p', 't',
                     'n_fev','success']:
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
    n_fev = soln['n_fev']
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
    n_fev += n_it
    soln = dict()
    for item in ['t', 'p', 'success', 'n_it', 'gamma_i', 'phi_coef_fun_i',
                 'phi_v', 'phi_i_sat', 'p_i_sat', 'z_l_i', 'v_l_i', 'poynting_i',
                 'y_i', 'x_i', 'k_i']:
        soln[item] = locals().get(item)
    return soln


def p_est(t, p, x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega, max_it, tol=tol):
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
    iterations = soln_v_f['iterations']
    total_backtracks = soln_v_f['total_backtracks']
    soln = dict()
    for item in ['soln_l', 'soln_v', 'soln_v_f',
                 'k_i', 'v_f', 'x_i', 'y_i',
                 'iterations', 'total_backtracks']:
        soln[item] = locals().get(item)
    return soln


def isot_flash_solve(t, p, z_i, tc_i, pc_i, af_omega_i,
                     alpha_tr, epsilon, sigma, psi, omega, max_it=20,
                     x_i=None, y_i=None):
    if not x_i:
        x_i = ones(len(z_i)) / len(z_i)
    if not y_i:
        y_i = ones(len(z_i)) / len(z_i)
    for i in range(max_it):
        soln = isot_flash(t, p, x_i, y_i, z_i, tc_i, pc_i, af_omega_i,
                          alpha_tr, epsilon, sigma, psi, omega)
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
    # FIXME: bubl_p / dew_p might be off converging toward k_i=1
    bubl_p_soln = bubl_p(t, p0, z_i, tc_i, pc_i, af_omega_i,
                         alpha_tr, epsilon, sigma, psi, omega,
                         sec_j, nu_ij, unifac_data_dict,
                         max_it=max_it, tol=tol)
    dew_p_soln = dew_p(t, p0, z_i, tc_i, pc_i, af_omega_i,
                       alpha_tr, epsilon, sigma, psi, omega,
                       sec_j, nu_ij, unifac_data_dict,
                       max_it=max_it, tol=tol)
    n_fev = bubl_p_soln['n_fev'] + dew_p_soln['n_fev']
    total_backtracks = 0
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
            f = sum(z_i * (k_i - 1) / (1 + v_f * (k_i - 1)))
            df_dv_f = -sum(z_i * (k_i - 1) ** 2 / (1 + v_f * (k_i - 1)) ** 2)
            v_f = v_f - 1 / df_dv_f * f
            if not 0 < v_f < 1:
                ls_approach = line_search(
                    lambda v_f: sum(z_i * (k_i - 1) / (1 + v_f * (k_i - 1))),
                    lambda v_f: -sum(z_i * (k_i - 1) ** 2 / (1 + v_f * (k_i - 1)) ** 2),
                    v_f_old, additional_restrictions=lambda v_f: 0 < v_f < 1)
                v_f = ls_approach['x_2']
                total_backtracks += ls_approach['backtrack_count']
            x_i = z_i / (1 + v_f * (k_i - 1))
            y_i = k_i * x_i
            x_i = x_i / sum(x_i)
            y_i = y_i / sum(y_i)
            delta_x_i = x_i - x_i_old
            delta_y_i = y_i - y_i_old
            delta_v_f = v_f - v_f_old
            sum_rr = sum(z_i * (k_i - 1) / (1 + v_f * (k_i - 1)))
            # criterion as vector sum of distance to convergence
            criterion = sqrt(sum_rr**2 + delta_x_i.dot(delta_x_i) +
                             delta_y_i.dot(delta_y_i) + delta_v_f**2)
            success = criterion <= tol
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
    iterations = n_it
    soln = dict()
    for item in ['t', 'p', 'k_i', 'v_f', 'z_i', 'x_i', 'y_i', 'gamma_i', 'phi_coef_fun_i',
                 'dew_p_soln', 'bubl_p_soln', 'p_dew', 'p_bubl', 'poynting', 'n_it',
                 'mag_delta_x_i', 'mag_delta_y_i', 'mag_delta_v_f', 'sum_rr', 'criterion',
                 'iterations', 'total_backtracks']:
        soln[item] = locals().get(item)
    return soln


def interaction_pairs(list, i=0, j=0, perms=[]):
    # recursively produce pairs of indexes for interaction parameters
    if i + 1 >= len(list) and j + 1 >= len(list):
        return perms + [[list[i], list[j]]]
    elif j + 1 < len(list):
        return interaction_pairs(list, i, j + 1, perms + [[list[i], list[j]]])
    elif i + 1 < len(list):
        return interaction_pairs(list, i + 1, 0, perms + [[list[i], list[j]]])


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