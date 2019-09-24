"""Global convergence Newton Raphson

Jr., J. E. Dennis ; Schnabel, Robert B.: Numerical Methods for Unconstrained
Optimization and Nonlinear Equations. Philadelphia: SIAM, 1996.

"""
import numpy as np
from numerik import line_search, gauss_elimination
from setup_results_log import setup_log_file, notify_status_func, log_line
import logging


eps = np.finfo(float).eps
np.set_printoptions(linewidth=200)
setup_log_file('dennis_schabel_global_nr.log', with_console=True)


def f(x):
    x1, x2 = x
    f1 = x1**2 + x2**2 - 2
    f2 = np.exp(x1 - 1) + x2**3 - 2
    fun = np.array([f1, f2])
    logging.debug('x=' + str(x))
    logging.debug('f(x)=' + str(fun))
    return fun


def j(x):
    x1, x2 = x
    j_11 = 2 * x1
    j_21 = np.exp(x1 - 1)
    j_12 = 2 * x2
    j_22 = 3 * x2**2
    jac = np.array([[j_11, j_12], [j_21, j_22]])
    logging.debug('j(x)=' + str(jac.tolist()))
    return jac


def log_var_by_name(var_name):
    val = globals()[var_name]
    if np.size(val) > 1:
        log_line(var_name + '= ' + str(globals()[var_name].tolist()))
    else:
        log_line(var_name + '= ' + str(globals()[var_name]))


alpha = 1e-4
x_0 = np.array([2., 0.5])
f_0 = f(x_0)
j_0 = j(x_0)
inv_j_val = np.linalg.matrix_power(j_0, -1)
# alt. to inv(J): solution of system J(X) Y = -F(X)
s_0_n = gauss_elimination(j_0, -f_0)
log_var_by_name('inv_j_val')
log_var_by_name('s_0_n')
lambda_0 = 1.0
lambda_k = lambda_0
x_plus_1 = x_0 + lambda_k * s_0_n
f_plus_1 = f(x_plus_1)
f_plus_minus_f_0 = (f_plus_1 - f_0).dot(f_plus_1 - f_0)
log_var_by_name('x_plus_1')
log_var_by_name('f_plus_1')
# in book: [small] f(x+): f(x+)=1/2*sum_i(f_i(x)**2)
#      vs: [caps ] F(x+)
# in Burden-Faires: g(x)
g_0 = 1 / 2. * f_0.dot(f_0)
g_plus_1 = 1 / 2. * f_plus_1.dot(f_plus_1)
g_plus_minus_g_0 = g_plus_1 - g_0
nabla_f_t_s_0_n = -f_0.dot(f_0)
# criterion g(x_k+1) < g(x_k) + alpha * lambda_k * nabla_f_t_s_k_n
satisfactory = g_plus_minus_g_0 < alpha * lambda_k * nabla_f_t_s_0_n
log_var_by_name('g_0')
log_var_by_name('g_plus_1')
log_var_by_name('g_plus_minus_g_0')
log_var_by_name('satisfactory')
log_var_by_name('nabla_f_t_s_0_n')
# quadratic backtrack
lambda_1 = -nabla_f_t_s_0_n / (2 * (g_plus_1 - g_0 - nabla_f_t_s_0_n))
log_var_by_name('lambda_0')
log_var_by_name('lambda_1')
# guaranteed: lambda_1 < 0.5
if lambda_1 < 0.1:
    lambda_1 = 0.1
lambda_k = lambda_1
x_plus_2 = x_0 + lambda_k * s_0_n
f_plus_2 = f(x_plus_2)
g_plus_2 = 1 / 2. * f_plus_2.dot(f_plus_2)
g_plus_minus_g_0 = g_plus_2 - g_0
# criterion g(x_k+1) < g(x_k) + alpha * lambda_k * nabla_f_t_s_k_n
satisfactory = g_plus_minus_g_0 < alpha * lambda_k * nabla_f_t_s_0_n
log_var_by_name('x_plus_2')
log_var_by_name('f_plus_2')
log_var_by_name('g_plus_2')
log_var_by_name('g_plus_minus_g_0')
log_var_by_name('satisfactory')
# cubic backtrack
a, b = 1 / (lambda_1 - lambda_0) * np.array(
    [[+1 / lambda_1**2, - 1 / lambda_0**2],
     [- lambda_0 / lambda_1**2, +lambda_1 / lambda_0**2]]
).dot(np.array(
    [[g_plus_2 - g_0 - nabla_f_t_s_0_n * lambda_1],
     [g_plus_1 - g_0 - nabla_f_t_s_0_n * lambda_0]]))
lambda_2 = (-b + np.sqrt(b**2 - 3 * a * nabla_f_t_s_0_n)) / (3 * a)
log_var_by_name('a')
log_var_by_name('b')
log_var_by_name('lambda_2')
if lambda_2 > 1 / 2 * lambda_1:
    lambda_2 = 1 / 2 * lambda_1
lambda_k = lambda_2
x_plus_3 = x_0 + lambda_k * s_0_n
f_plus_3 = f(x_plus_3)
g_plus_3 = 1 / 2. * f_plus_3.dot(f_plus_3)
g_plus_minus_g_0 = g_plus_3 - g_0
# criterion g(x_k+1) < g(x_k) + alpha * lambda_k * nabla_f_t_s_k_n
satisfactory = g_plus_minus_g_0 < alpha * lambda_k * nabla_f_t_s_0_n
log_var_by_name('x_plus_3')
log_var_by_name('x_plus_3')
log_var_by_name('f_plus_3')
log_var_by_name('g_plus_3')
log_var_by_name('g_plus_minus_g_0')
log_var_by_name('satisfactory')
# cubic backtrack
a, b = 1 / (lambda_2 - lambda_1) * np.array(
    [[+1 / lambda_2**2, - 1 / lambda_1**2],
     [- lambda_1 / lambda_2**2, +lambda_2 / lambda_1**2]]
).dot(np.array(
    [[g_plus_3 - g_0 - nabla_f_t_s_0_n * lambda_2],
     [g_plus_2 - g_0 - nabla_f_t_s_0_n * lambda_1]]))
lambda_3 = (-b + np.sqrt(b**2 - 3 * a * nabla_f_t_s_0_n)) / (3 * a)
log_var_by_name('lambda_3')
lambda_3_used = all(
    [lambda_3 > 1 / 10 * lambda_2, lambda_3 < 1 / 2 * lambda_2])
log_var_by_name('lambda_3_used')
if lambda_3_used:
    lambda_k = lambda_3
x_plus_4 = x_0 + lambda_k * s_0_n
f_plus_4 = f(x_plus_4)
g_plus_4 = 1 / 2. * f_plus_4.dot(f_plus_4)
g_plus_minus_g_0 = g_plus_4 - g_0
ineq_rhs_criterion = g_0 + alpha * lambda_k * nabla_f_t_s_0_n
# criterion g(x_k+1) < g(x_k) + alpha * lambda_k * nabla_f_t_s_k_n
satisfactory = g_plus_minus_g_0 < alpha * lambda_k * nabla_f_t_s_0_n
log_var_by_name('x_plus_4')
log_var_by_name('f_plus_4')
log_var_by_name('g_plus_4')
log_var_by_name('ineq_rhs_criterion')
log_var_by_name('g_plus_minus_g_0')
log_var_by_name('satisfactory')
x_1 = x_plus_4


f_1 = f_plus_4
j_1 = j(x_1)
s_1_n = gauss_elimination(j_1, -f_1)
log_var_by_name('s_1_n')
lambda_0 = 1.0
lambda_k = lambda_0
x_plus_1 = x_1 + lambda_k * s_1_n
f_plus_1 = f(x_plus_1)
log_var_by_name('x_plus_1')
log_var_by_name('f_plus_1')
# in book: [small] f(x+): f(x+)=1/2*sum_i(f_i(x)**2)
#      vs: [caps ] F(x+)
# in Burden-Faires: g(x)
g_1 = 1 / 2. * f_1.dot(f_1)
g_plus_1 = 1 / 2. * f_plus_1.dot(f_plus_1)
nabla_f_t_s_1_n = -f_1.dot(f_1)
# criterion g(x_k+1) < g(x_k) + alpha * lambda_k * nabla_f_t_s_k_n
ineq_rhs_criterion = g_1 + alpha * lambda_k * nabla_f_t_s_1_n
satisfactory = g_plus_1 < ineq_rhs_criterion
log_var_by_name('g_1')
log_var_by_name('g_plus_1')
log_var_by_name('ineq_rhs_criterion')
log_var_by_name('satisfactory')
log_var_by_name('nabla_f_t_s_1_n')
# quadratic backtrack
lambda_1 = -nabla_f_t_s_1_n / (2 * (g_plus_1 - g_0 - nabla_f_t_s_1_n))
log_var_by_name('lambda_0')
log_var_by_name('lambda_1')
# guaranteed: lambda_1 < 0.5
if lambda_1 < 0.1:
    lambda_1 = 0.1
lambda_k = lambda_1
x_plus_2 = x_1 + lambda_k * s_1_n
f_plus_2 = f(x_plus_2)
g_plus_2 = 1 / 2. * f_plus_2.dot(f_plus_2)
# criterion g(x_k+1) < g(x_k) + alpha * lambda_k * nabla_f_t_s_k_n
ineq_rhs_criterion = g_1 + alpha * lambda_k * nabla_f_t_s_1_n
satisfactory = g_plus_2 < ineq_rhs_criterion
log_var_by_name('x_plus_2')
log_var_by_name('f_plus_2')
log_var_by_name('g_plus_2')
log_var_by_name('ineq_rhs_criterion')
log_var_by_name('satisfactory')
x_k = x_plus_2

f_k = f_plus_2
j_k = j(x_k)
s_k_n = gauss_elimination(j_k, -f_k)
log_var_by_name('s_k_n')
lambda_0 = 1.0
lambda_k = lambda_0
x_k_plus_1 = x_k + lambda_k * s_k_n
f_k_plus_1 = f(x_k_plus_1)
f_k_plus_1_minus_f_k = (f_k_plus_1 - f_k).dot(f_k_plus_1 - f_k)
log_var_by_name('x_k_plus_1')
log_var_by_name('f_k_plus_1')
# in book: [small] f(x+): f(x+)=1/2*sum_i(f_i(x)**2)
#      vs: [caps ] F(x+)
# in Burden-Faires: g(x)
g_k = 1 / 2. * f_k.dot(f_k)
g_k_plus_1 = 1 / 2. * f_k_plus_1.dot(f_k_plus_1)
g_k_plus_1_minus_g_k = g_k_plus_1 - g_k
nabla_f_t_s_k_n = -f_k.dot(f_k)
# criterion g(x_k+1) < g(x_k) + alpha * lambda_k * nabla_f_t_s_k_n
ineq_rhs_criterion = g_k + alpha * lambda_k * nabla_f_t_s_k_n
satisfactory = g_k_plus_1 < ineq_rhs_criterion
log_var_by_name('g_k')
log_var_by_name('g_k_plus_1')
log_var_by_name('ineq_rhs_criterion')
log_var_by_name('satisfactory')


print('')
x_0 = np.array([2., 0.5])
x_1 = x_0
for iteration in range(7):
    soln = line_search(f, j, x_1)
    x_1 = soln['x_2']
    print(x_1)
