import sys

from matplotlib import pyplot as plt
from numpy import array, zeros, ones, empty, log, append, linspace, sqrt, exp, sum
from numpy import finfo, nan, concatenate, asarray, empty_like, dot, outer, multiply
from numpy.random import randint
from scipy import optimize

from numerik import secant_ls_3p
from poly_3_4 import solve_cubic
from setup_results_log import setup_log_file
from z_l_v import bubl_p, dew_p, bubl_t, p_sat, p_i_sat_ceos, phi, p_est, z_phase
from z_l_v import setup_unifac_data, gamma_u, bubl_point_step_l_k, dew_point_step_l_k, State
from z_l_v import use_pr_eos, use_srk_eos, isot_flash, pt_flash

r = 8.314 * 10. ** 6 / 10. ** 5  # bar cm^3/(mol K)
rlv = 0.8  # Rücklaufverhältnis
t_flash = 273.16 + 60  # K
tol = finfo(float).eps
setup_log_file('log_z_l_v.log', with_console=False)

# Nach unten hin: CO, H2, CO2, H2O, CH3OH, N2, CH4

# Kritische Parameter Tc, Pc, omega(azentrischer Faktor)
tc = array([132.85, 32.98, 304.12, 647.14, 512.64, 126.2, 190.56])  # K
pc = array([34.94, 12.93, 73.74, 220.64, 80.92, 34.0, 45.992])  # bar
omega_af = array([0.045, -0.217, 0.225, 0.344, 0.565, 0.037, 0.011])
ant_a = array([6.72828, 6.14858, 9.81367, 8.05573, 8.08404, 6.72531, 6.84377])
ant_b = array([295.2279, 80.948, 1340.9768, 1723.6425, 1580.4585, 285.5727, 435.4534])
ant_c = array([268.243, 277.53, 271.883, 233.08, 239.096, 270.09, 271.361])
labels = ['CO', 'H2', 'CO2', 'H2O', 'CH3OH', 'N2', 'CH4']


def vdi_atlas():
    alpha_tr, epsilon, sigma, psi, omega = use_pr_eos()
    print(
        'ref. VDI Wärmeatlas H2 Dampfdruck bei -256.6K: ' +
        '{:.4g}'.format(
            p_sat(-256.6 + 273.15, -0.216, 33.19, 13.13,
                  alpha_tr, epsilon, sigma, psi, omega).x.item() * 1000
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
                     )['phi_i'].item() -
                 phi(-256.6 + 273.15, p_var, 1, 33.19, 13.13, -0.216, 'v',
                             alpha_tr, epsilon, sigma, psi, omega)['phi_i'].item()
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
                p_min = p_est(t[i], p0[j], 1.0, tc[j], pc[j], omega_af[j],
                              alpha_tr, epsilon, sigma, psi, omega, 100, tol)['p_min_l']
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
    p_min = p_est(t, 1e-3, z_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega,
                  100, tol)['p_min_l']

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
    


def svn_14_1():
    x_i = array([0.4, 0.6])
    tc_i = array([126.2, 190.6])
    pc_i = array([34., 45.99])
    af_omega_i = array([0.038, 0.012])

    mm_i = zeros(2)
    state1 = State(200, 30, x_i, mm_i, tc_i, pc_i, af_omega_i, 'srk')
    print(state1.solve())



def svn_14_2():
    alpha_tr, epsilon, sigma, psi, omega = use_pr_eos()
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
    y = empty_like(x) * nan
    p_v = empty_like(x) * nan

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

        if p_v[i] > 140:
            break
    line1 = plt.plot(x, p_v, label=r'$x_1(L)$ bubl_p')
    line2 = plt.plot(y, p_v, label=r'$y_1(V)$ bubl_p')
    plt.plot(x_dew, p_v_dew, 'o', fillstyle='none',
             color=line1[0].get_color(), label=r'$x_1(L)$ dew_p')
    plt.plot(y_dew, p_v_dew, 'o', fillstyle='none',
             color=line2[0].get_color(), label=r'$y_1(L)$ dew_p')
    plt.ylabel('p / bar')
    plt.xlabel(r'$x_1 , y_1$')
    plt.legend()
    


def svn_14_2_behchmark():
    alpha_tr, epsilon, sigma, psi, omega = use_srk_eos()
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
    


def zs_1998():
    alpha_tr, epsilon, sigma, psi, omega = use_pr_eos()
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
                z_phase(t, p, z_i, tc_i, pc_i, af_omega_i, 'l', alpha_tr, epsilon, sigma, psi, omega, tol, r)['rho']]
            rho_v_phase += [
                z_phase(t, p, z_i, tc_i, pc_i, af_omega_i, 'v', alpha_tr, epsilon, sigma, psi, omega, tol, r)['rho']]
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
        lines = plot2.plot(a_data[:, 0], a_data[:, 1], ':', color='black')
    lines[0].set_label(r'$pseudo-\rho_{article}$')

    plot1.axvline(b, linestyle='-')
    plot2.axvline(1 / b, linestyle='-')
    plot4.axvline(1 / b, linestyle='-')
    p_low = z_phase(420, p, z_i, tc_i, pc_i, af_omega_i, 'l', alpha_tr, epsilon, sigma, psi, omega, tol)['p_low']
    plot1.axhline(p_low, linestyle='-.', color='gray', linewidth=0.5, label='$P_{low}$')
    plot2.axhline(p_low, linestyle='-.', color='gray', linewidth=0.5)
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
            phi_list_l[i] = phi(t, p, z_i, tc_i, pc_i, af_omega_i, 'l', alpha_tr, epsilon, sigma, psi, omega)['phi_i']
            phi_list_v[i] = phi(t, p, z_i, tc_i, pc_i, af_omega_i, 'v', alpha_tr, epsilon, sigma, psi, omega)['phi_i']

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
    


def svn_fig_14_8():
    # plot fig 14.8
    alpha_tr, epsilon, sigma, psi, omega = use_srk_eos()
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
            p_est(t, p, z_i, tc_i, pc_i, af_omega_i,
                  alpha_tr, epsilon, sigma, psi, omega, max_it, tol)
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
    


def svn_tab_14_1_2():
    alpha_tr, epsilon, sigma, psi, omega = use_pr_eos()
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
    soln = pt_flash(334.152 / 334.85 * 334.15, p, z_i, tc_i, pc_i, af_omega_i,
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
    criterion = soln['criterion']

    print('SVN Table 14.2 - n-hexane (1) / ethanol (2) / methylcyclopentane (3) /' +
          'benzene (4) at {:0.4g} bar and {:0.4g} K'.format(p, t))
    print('i\tz_i\t\tx_i\t\ty_i\t\tK_i\t\tPhi_i\t\tgamma_i')
    for i in range(len(z_i)):
        print(('{:d}' + '\t{:.4f}' * 6).format(
            i, z_i[i], x_i[i], y_i[i], k_i[i], phi_coef_fun_i[i], gamma_j[i]))
    print(('p: {:0.3f} bar\tT(calc): {:0.3f} K\tv_f: {:0.4g}'+
          '\titerations:{:d}\tcriterion:{:0.4g}').format(
        p, t, v_f, n_it, criterion))
    print('\n' * 2)


def pat_ue_03_flash():
    alpha_tr, epsilon, sigma, psi, omega = use_pr_eos()
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
    alpha_tr, epsilon, sigma, psi, omega = use_pr_eos()
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
    alpha_tr, epsilon, sigma, psi, omega = use_pr_eos()

    # log
    old_stdout = sys.stdout
    log_file = open('output.log', 'w')
    sys.stdout = log_file

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


vdi_atlas()
# svn_14_1()
plt.figure()
svn_fig_14_8()
plt.figure()
svn_14_2()
plt.figure()
svn_14_2_behchmark()
plt.figure()
zs_1998()
ppo_ex_8_12()
svn_h_1()
fredenslund_t_6()
svn_tab_14_1_2()
pat_ue_03_flash()
isot_flash_seader_4_1()
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
#pat_ue_03_vollstaendig(0.65, True)

plt.show()