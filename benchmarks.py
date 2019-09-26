from numpy import array, ones, sqrt, nan
from z_l_v import isot_flash, pt_flash, use_pr_eos, p_i_sat_ceos
from timeit import timeit

t = 293.15
p = 180.0
z_i = array([
    1.74525844326899369775524e-10,
    0.716562622816228844691011,
    2.62271986917900010078784e-11,
    0.0187246062565056513304906,
    0.0264853274626257462198708,
    0.152871431923119555085222,
    0.00222550285133812312393653,
    8.26713153902460841359341e-24,
    0.0831305084894291140829026])
tc = array([
    132.86, 33.19, 304.13,
    647.10, 190.56, 405.50,
    150.69, 154.60, 126.19
])  # K
pc = array([
    34.98, 13.15, 73.77,
    220.64, 45.99, 113.59,
    48.63, 50.46, 33.96
])  # bar
omega_af = array([
    0.050, -0.219, 0.224,
    0.344, 0.011, 0.256,
    -0.002, 0.022, 0.037
])

alpha_tr, epsilon, sigma, psi, omega = use_pr_eos()


def benchmark_isot_flash():
    # FIXME: times to 0.9 s (brute force better than sofisticated)
    x_i = 1 / len(z_i) * ones(len(z_i))
    y_i = z_i
    v_f = 1.
    iterations = 0
    backtracks = 0
    criterion = nan
    for i in range(10):
        x_i_old = x_i
        y_i_old = y_i
        v_f_old = v_f
        soln = isot_flash(t, p, x_i, y_i, z_i, tc, pc, omega_af, alpha_tr, epsilon, sigma, psi, omega)
        y_i = soln['y_i']
        x_i = soln['x_i']
        v_f = soln['v_f']
        k_i = soln['k_i']
        delta_x_i = x_i - x_i_old
        delta_y_i = y_i - y_i_old
        delta_v_f = v_f - v_f_old
        sum_rr = sum(z_i * (k_i - 1) / (1 + v_f * (k_i - 1)))
        criterion = sqrt(sum_rr ** 2 + delta_x_i.dot(delta_x_i) +
                         delta_y_i.dot(delta_y_i) + delta_v_f ** 2)
        iterations += soln['iterations']
        backtracks += soln['total_backtracks']
        if abs(v_f - v_f_old) < 1e-12:
            break
    print(('v_f: {:0.4f}\tcriterion: {:0.4e}\t'+
           'iterations+backtracks: {:d}\tx_i_max: {:0.4f}'
           ).format(v_f, criterion, iterations + backtracks,
                    x_i[x_i == max(x_i)].item()))


def benchmark_pt_flash():
    soln = p_i_sat_ceos(t, p, tc, pc, omega_af, alpha_tr, epsilon, sigma, psi, omega, tol=1e-10)
    n_fev = sum(soln['n_fev'])
    pisat = array([soln['p'][i] for i in range(len(z_i)) if soln['success'][i]])
    zisat = array([z_i[i] for i in range(len(z_i)) if soln['success'][i]])
    p_est = sum(pisat * zisat / sum(zisat))
    soln = pt_flash(t, p, z_i, tc, pc, omega_af,
                    alpha_tr, epsilon, sigma, psi, omega, tol=1e-10, p_est_0=p_est)
    iterations = soln['iterations']
    backtracks = soln['total_backtracks']
    v_f = soln['v_f']
    k_i = soln['k_i']
    x_i = soln['x_i']
    criterion = soln['criterion']

    print(('v_f: {:0.4f}\tcriterion: {:0.4e}\t' +
           'iterations: {:d}\tn_fev: {:d}\ttotal_backtracks: {:d}\tx_i_max: {:0.4f}'
           ).format(v_f, criterion,  iterations, n_fev, backtracks,
                    x_i[x_i == max(x_i)].item()))

# benchmark_isot_flash()
# benchmark_pt_flash()
time_1 = timeit(stmt='benchmark_pt_flash()', globals=globals(), number=10)
# print(time_1)
# time_2 = timeit(stmt='benchmark_isot_flash()', globals=globals(), number=10)
# print(time_2)