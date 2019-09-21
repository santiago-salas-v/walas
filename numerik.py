from numpy import array, zeros, ones, eye, argmax, finfo, copy, empty_like, zeros_like, empty
from numpy import sqrt, log, exp, nan, isnan, isinf, size, asarray


def lrpd(a):
    """
    LR = PDA factorization Method (LU)

    Dahmen W., Reusken A.; Numerik fuer Ingenieure und
    Naturwissenschaeftler; Springer S. 79

    :param a: array N x M
    :return: l, r, p, d, dlr
    """
    n = a.shape[0]
    m = a.shape[1]
    p = eye(n, dtype=float)
    d = eye(n, dtype=float)
    dlr = zeros([n, m], dtype=float)
    l = eye(n, m, dtype=float)
    r = zeros([n, m], dtype=float)
    indexes_r = list([item for item in range(n)])
    for i in range(n):
        d[i, i] = 1 / sum(abs(a[i, :m]))
        # scaling
        dlr[i] = d[i, i] * a[i]  # dlr is just for storage of both l and r
    for j in range(n):
        indexes_r[j] = j + argmax(abs(dlr[j:, j]))
        # columns pivoting
        dlr[[j, indexes_r[j]]] = dlr[[indexes_r[j], j]]
        p[[j, indexes_r[j]]] = p[[indexes_r[j], j]]
        for i in range(j + 1, n):
            # new entries in L
            dlr[i, j] = dlr[i, j] / dlr[j, j]
            for k in range(j + 1, m):
                # new entries in R
                dlr[i, k] = dlr[i, k] - dlr[i, j] * dlr[j, k]
    for i in range(m):
        l[i + 1:n, i] = dlr[i + 1:n, i]
    for i in range(n):
        r[i, i:m] = dlr[i, i:m]
    return l, r, p, d, dlr


def rref(r):
    """
    Reduced row echelon form of upper triangular matrix r

    :param r: array N x M
    :return: array N x M
    """
    n = r.shape[0]
    m = r.shape[1]
    r_form = copy(r)
    # denominators for diagonal 1
    den = [0.0] * n
    for i in range(n - 1, 0 - 1, -1):
        for j in range(m):
            if i > 0 and den[i] == 0.0 and abs(r[i, j]) > finfo(float).eps:
                # first non-zero element in last rows
                den[i] = r[i, j]
                for k in range(i - 1, 0 - 1, -1):
                    num = r[k, j]
                    r_form[k, :] = r_form[k, :] - num / den[i] * r_form[i, :]
            elif i == 0 and j == 0:
                den[i] = r[i, j]
            if abs(r_form[i, j]) > 0 and abs(
                    r_form[i, j]) <= finfo(float).eps:
                # Either way make eps 0, avoid propagation of error.
                r_form[i, j] = 0.0
    for i in range(n):
        r_form[i] = r_form[i] / den[i]
    return r_form


def ref(a):
    """
    Reduced echelon form of matrix a

    :param a: array N x M
    :return: array N x M
    """
    n = a.shape[0]
    m = a.shape[1]
    p = eye(n, dtype=float)
    d = eye(n, dtype=float)
    dr = zeros([n, m], dtype=float)
    r = zeros([n, m], dtype=float)
    indexes_r = list([item for item in range(n)])
    shift = 0  # diagonal !=0, no shift
    for i in range(n):
        d[i, i] = 1 / sum(abs(a[i, :m]))
        # scaling
        dr[i] = d[i, i] * a[i]  # dr is just for storage r
    for j in range(n):
        while all(abs(dr[j:, j + shift]) == 0) and j + shift < dr.shape[1] - 1:
            # shift from diagonal
            shift += 1
        if j + shift >= dr.shape[1] - 1:
            # rows below all 0
            break
        indexes_r[j] = j + argmax(abs(dr[j:, j + shift]))
        # columns pivoting
        dr[[j, indexes_r[j]]] = dr[[indexes_r[j], j]]
        p[[j, indexes_r[j]]] = p[[indexes_r[j], j]]
        # pivot for row j
        piv_j = dr[j, j + shift]
        for i in range(j + 1, n):
            # numerator for row i
            num_i = dr[i, j + shift]
            # k for every element of the row up to diagonal
            for k in range(m - 1, j - 1, -1):
                # new entries in R
                dr[i, k] = dr[i, k] - num_i / piv_j * dr[j, k]
    for i in range(n):
        r[i, i:m] = dr[i, i:m]
    return dr, p, d


def gauss_elimination(a, b):
    """
    Gauss elimination by LR factorization.

    Solution of the system Ax = b (LRx=PDb) by forward substitution of
    Ly = Pb, followed by backward substitution of Rx = y

    :param a: array n X n
    :param b: array n X 1
    """
    n = a.shape[0]
    x = zeros_like(b, dtype=float)
    y = zeros_like(b, dtype=float)
    l, r, p, d, da = lrpd(a)
    pdb = (p.dot(d).dot(b))
    sum_lik_xk = 0.
    sum_lik_yk = 0.
    for j in range(0, n, +1):
        # Forward substitution Ly = PDb
        for k in range(0, j, +1):
            sum_lik_yk = sum_lik_yk + l[j, k] * y[k]
        y[j] = (pdb[j] - sum_lik_yk) / 1.0
        sum_lik_yk = 0
    for j in range(n - 1, -1, -1):
        # Backward substitution Rx = y
        for k in range(n - 1, j, -1):
            sum_lik_xk = sum_lik_xk + r[j, k] * x[k]
        x[j] = (y[j] - sum_lik_xk) / r[j, j]
        sum_lik_xk = 0
    return x


# noinspection PyAugmentAssignment
def nr_ls(x0, f, j, tol, max_it, inner_loop_condition,
          notify_status_func, process_func_handle,
          alpha=1e-4):
    """
    Newton method: G(x) = J(x)^-1 * F(x)

    :param x0: initial estimate. array of length N.
    :param f: function. Returns array same size as x0, length N.
    :param j: jacobian. Returns squared array array N X N.
    :param tol: absolute tolerance.
    :param max_it: maximum number of iterations.
    :param inner_loop_condition: inner loop condition. Function to activate line search.
    :param notify_status_func: logging function.
    :param process_func_handle: function handle to exit process.
    :param alpha: line search parameter.
    :return: progress_k, stop, outer_it_k,\
        inner_it_j, lambda_ls, accum_step,\
        x, diff, f_val, lambda_ls * y
    """
    x = x0
    outer_it_k = 0
    inner_it_j = 0
    j_val = j(x)
    f_val = f(x)
    y = empty_like(f_val)
    magnitude_f = sqrt(f_val**2)
    diff = nan
    if size(x) > 1:
        y = ones(len(x)).T * tol / (sqrt(len(x)) * tol)
        magnitude_f = sqrt((f_val.dot(f_val)).item())
        diff = empty(len(x))
        diff.fill(nan)
    elif size(x) == 1:
        pass  # defined above
    # Line search variable lambda
    lambda_ls = 1.0
    accum_step = 0.0
    # Machine precision
    machine_eps = finfo(float).eps
    # For progress bar, use exp scale to compensate for quadratic convergence
    progress_factor = 1 / (1 - 10. ** (5 * (2 - 1))) * \
        log(0.1)  # prog=0.1, x=10^-5 & tol=10^-10
    progress_k = exp((-magnitude_f + tol) / tol * progress_factor) * 100.
    stop = magnitude_f < tol  # stop if already fulfilling condition
    stop = stop or isnan(j_val).any() or isnan(f_val).any()
    divergent = False
    no_gradient_change_once = False
    # Non-functional status notification
    if notify_status_func is not None:
        g_val = 1 / 2. * scalar_prod(f_val, f_val)
        g_prime_t_s = -scalar_prod(f_val, f_val)
        descent = alpha * lambda_ls * g_prime_t_s
        g_max = g_val + descent
        notify_status_func(progress_k, stop, outer_it_k,
                           inner_it_j, lambda_ls, accum_step,
                           x, diff, f_val, j_val, 1.0 * y,
                           inner_it_j, g_min=g_val, g1=g_max)
    # End non-functional notification
    while outer_it_k <= max_it and not stop:
        outer_it_k += 1
        x_k_m_1 = x
        progress_k_m_1 = progress_k
        # First attempt without backtracking. Continue line search
        # until satisfactory descent.
        x, f_val, g_val, y, j_val, magnitude_f, g_max, ls_it, \
            lambda_ls, overall_stop, accum_step = \
            line_search(
                f, j, x, known_f_c=f_val, known_j_c=j_val,
                max_iter=max_it, alpha=alpha, tol=tol,
                additional_restrictions=inner_loop_condition,
                notify_status_func=notify_status_func,
                outer_it=outer_it_k, accum_step=accum_step)
        inner_it_j = ls_it
        diff = x - x_k_m_1
        if overall_stop:
            stop = True  # Procedure successful
        else:
            # For progress use exp scale to compensate for quadratic
            # convergence
            progress_k = exp(
                (-magnitude_f + tol) / tol * progress_factor) * 100.
            # continue if gradient still changing
            gradient_change = abs(g_val - g_max) > finfo(float).eps
            no_gradient_change_twice = no_gradient_change_once and not gradient_change
            no_gradient_change_once = no_gradient_change_once or not gradient_change
            min_g_reached = g_val <= machine_eps
            # conditions to stop prematurely: sum of squares reaches machine precision, or
            # no progress and no gradient change
            premature_stop = progress_k == progress_k_m_1 and \
                no_gradient_change_twice or \
                min_g_reached
            if isnan(magnitude_f) or isinf(magnitude_f):
                stop = True  # Divergent method
                divergent = True
                progress_k = 0.0
            else:
                pass
            if premature_stop:
                # Non-functional gui processing
                process_func_handle()
                stop = True
                # End non-functional processing
                # if form.progress_var.wasCanceled():
                # stop = True
    if stop and not divergent:
        progress_k = 100.0
    elif divergent:
        progress_k = 0.0
    return progress_k, stop, outer_it_k,\
        inner_it_j, lambda_ls, accum_step,\
        x, diff, f_val, lambda_ls * y


def sdm(x0, f, j, tol, notify_status_func, inner_loop_condition=None):
    """
    Steepest descent method.
    Burden, Richard L. / Faires, J. Douglas / Burden, Annette M. (2015):
    Numerical Analysis. Clifton Park, NY (Cengage Learning).

    :param inner_loop_condition: line search condition
    :param x0: initial estimate. Array length N.
    :param f: function. Returns array length N.
    :param j: jacobian. Returns array N X N.
    :param tol: absolute tolerance.
    :param notify_status_func: logging function.
    :return: x, array with reduced gradient to tol level.
    """
    def g(x_var):
        f_val = asarray(f(x_var))
        return f_val.T.dot(f_val)
    x = x0
    k = 1
    stop = False
    max_it = 1000
    while k < max_it and not stop:
        x_n_m_1 = x
        f_x = asarray(f(x))
        j_x = asarray(j(x))
        z = asarray(2 * j_x.dot(f_x))
        z0 = asarray(sqrt(z.T.dot(z)))
        if z0 == 0:
            # Zero gradient
            # stop = True
            break
        z = z / z0
        alpha1 = 0
        alpha3 = 1
        if inner_loop_condition is not None:
            # external restriction for alpha 3 (backtracking)
            inner_it_j = 0
            while inner_it_j <= max_it and \
                    not inner_loop_condition(x - alpha3 * z):
                inner_it_j += 1
                alpha3 = alpha3 / 2.0
        g1 = g(x - alpha1 * z)
        g3 = g(x - alpha3 * z)
        while g3 >= g1 and alpha3 > tol / 2.0:
            alpha3 = alpha3 / 2.0
            g3 = g(x - alpha3 * z)
        alpha2 = alpha3 / 2.0
        g2 = g(x - alpha2 * z)
        """
        (Note: Newton’s forward divided-difference formula is used to find
        the quadratic P(α) = g1 + h1α + h3α(α − α2) that interpolates
        h(α) at α = 0, α = α2, α = α3.)
        """
        h1 = (g2 - g1) / alpha2
        h2 = (g3 - g2) / (alpha3 - alpha2)
        h3 = (h2 - h1) / alpha3
        # (The critical point of P occurs at α0.)
        alpha0 = 0.5 * (alpha2 - h1 / h3)
        g0 = g(x - alpha0 * z)
        if g0 < g3:
            alpha = alpha0
            g_min = g0
        else:
            alpha = alpha3
            g_min = g3
        x = x - alpha * z
        g_min_minus_g1 = g_min - g1
        if abs(g_min_minus_g1) < tol:
            stop = True  # Procedure successful

        # Non-functional status notification
        diff = x - x_n_m_1
        outer_it_k = k
        inner_it_j, accum_step, lambda_ls = k, k, nan
        progress_k = (-g_min_minus_g1 / tol) * 100.
        notify_status_func(progress_k, stop, outer_it_k,
                           inner_it_j, lambda_ls, accum_step,
                           x, diff, f_x, j_x, z,
                           nan, g_min=g_min, g1=g1)
        # End non-functional notification

        k += 1
    return x


def line_search(fun, jac, x_c, known_f_c=None, known_j_c=None,
                max_iter=50, alpha=1e-4, tol=1e-8,
                additional_restrictions=None, notify_status_func=None,
                outer_it=1, accum_step=0):
    """Line search algorithm

    Jr., J. E. Dennis ; Schnabel, Robert B.: Numerical Methods for Unconstrained
    Optimization and Nonlinear Equations. Philadelphia: SIAM, 1996.

    Returns x+, f(x+), g(x+), s_0_n, j(x+) such that x+ = x_c + lambda s_0_n satisfies \
        f(x+) <= f(xc) + alpha lambda (nabla f(x_0))^T s_0^N

    :param fun: function
    :param jac: jacobian
    :param known_f_c: if already calculated, j_c
    :param known_j_c: if already calculated, j_c
    :param x_c: initial estimate for x+
    :param max_iter: maximum iterations
    :param alpha: stringence constant in (0,1/2)
    :param tol: step tolerance
    :param additional_restrictions: optional restrictions imposed on x. Line search when False.
    :param notify_status_func: optional function to log result
    :param accum_step: optional accumulated step count
    :param outer_it: optional initial iteration count
    :return: x+, f(x+), g(x+), s_0_n, j(x+), backtrack_count, lambda_ls, \
        magnitude_f, outer_it_stop, accum_step
    """
    if known_f_c is not None:
        f_0 = known_f_c
    else:
        f_0 = fun(x_c)
    if known_j_c is not None:
        j_0 = known_j_c
    else:
        j_0 = jac(x_c)

    # p in the Newton-direction: $s_N = -J(x_c)^{-1} F(x_c)$
    s_0_n = solve_ax_equal_b(j_0, -f_0)
    # relative length of p as calculated in the stopping routine
    if size(x_c) == 1:
        rellength = abs(s_0_n / x_c)
    else:
        rellength = max(abs(s_0_n) / max(abs(x_c)))
    # minimum allowable step length
    lambda_min = tol / rellength
    # FIXME: lambda_min can sometimes be lower than tol / rellength
    lambda_min = finfo(float).eps * lambda_min
    # $\nabla f(x_c)^T s^N = -F(x_c)^T F(x_c)$
    # initslope: expressed (p344) $g^T p$ as gradient . direction
    g_0 = 1 / 2. * scalar_prod(f_0, f_0)
    g_prime_t_s = -scalar_prod(f_0, f_0)

    # first attempt full Newton step
    lambda_ls = 1.0

    # init other variables
    backtrack_count = 0
    lambda_temp = lambda_ls
    lambda_prev = lambda_ls
    x_2 = empty_like(x_c)
    f_2 = empty_like(f_0)
    g_2 = empty_like(g_0)
    g_1 = empty_like(g_0)
    magnitude_f = empty_like(lambda_ls)
    g_max = empty_like(lambda_ls)
    progress_factor = 1 / (1 - 10. ** (5 * (2 - 1))) * \
        log(0.1)  # prog=0.1, x=10^-5 & tol=10^-10
    stop = False
    outer_it_stop = False
    while backtrack_count < max_iter and not stop:
        x_2 = x_c + lambda_ls * s_0_n
        f_2 = fun(x_2)
        g_2 = 1 / 2. * scalar_prod(f_2, f_2)
        descent = alpha * lambda_ls * g_prime_t_s
        g_max = g_0 + descent
        nan_result = isnan(f_2).any()
        # add current lambda to accumulated steps
        accum_step += lambda_ls
        while nan_result and lambda_ls >= lambda_min:
            # handle case in which f_2 throws nan as rough line search
            # Non-functional status notification
            if notify_status_func is not None:
                diff = (lambda_ls - lambda_prev) * s_0_n
                inner_it_j = backtrack_count
                magnitude_f = sqrt(2 * g_2)
                progress_k = exp(
                    (-magnitude_f + tol) / tol * progress_factor) * 100.
                notify_status_func(
                    progress_k,
                    outer_it_stop and stop,
                    outer_it,
                    inner_it_j,
                    lambda_ls,
                    accum_step,
                    x_2,
                    diff,
                    f_2,
                    j_0,
                    lambda_ls * s_0_n,
                    backtrack_count,
                    g_min=g_2,
                    g1=g_max)
            # End non-functional notification
            accum_step -= lambda_ls
            backtrack_count += 1
            lambda_ls = 0.1 * lambda_ls
            x_1 = x_c + lambda_ls * s_0_n
            f_1 = fun(x_1)
            lambda_prev = lambda_ls
            lambda_ls = 0.5 * lambda_ls
            x_2 = x_c + lambda_ls * s_0_n
            f_2 = fun(x_2)
            g_2 = 1 / 2. * scalar_prod(f_2, f_2)
            nan_result = isnan(f_1).any() or isnan(f_2).any()
            accum_step += lambda_ls
        satisfactory = g_2 <= g_max
        stop = satisfactory
        magnitude_f = sqrt(2 * g_2)
        outer_it_stop = magnitude_f < tol
        if additional_restrictions is not None:
            stop = satisfactory and additional_restrictions(x_2)
        if lambda_ls < lambda_min:
            # satisfactory x_2 cannot be found sufficiently distinct from x_c
            outer_it_stop = True
            stop = True
            pass
        # Non-functional status notification
        if notify_status_func is not None:
            diff = (lambda_ls - lambda_prev) * s_0_n
            inner_it_j = backtrack_count
            progress_k = exp(
                (-magnitude_f + tol) / tol * progress_factor) * 100.
            notify_status_func(progress_k, outer_it_stop and stop, outer_it,
                               inner_it_j, lambda_ls, accum_step,
                               x_2, diff, f_2, j_0, lambda_ls * s_0_n,
                               backtrack_count, g_min=g_2, g1=g_max)
        # End non-functional notification
        if not stop:
            # reduce lambda
            # backtrack accumulated steps in current lambda,
            # then reduce lambda once more
            accum_step -= lambda_ls
            backtrack_count += 1
            if lambda_ls == 1:
                # first backtrack: quadratic fit
                lambda_temp = -g_prime_t_s / (
                    2 * (g_2 - g_0 - g_prime_t_s)
                )
            elif lambda_ls < 1:
                # subsequent backtracks: cubic fit
                a, b = 1 / (lambda_ls - lambda_prev) * array(
                    [[+1 / lambda_ls ** 2, - 1 / lambda_prev ** 2],
                     [- lambda_prev / lambda_ls ** 2, +lambda_ls / lambda_prev ** 2]]
                ).dot(array(
                    [[g_2 - g_0 - g_prime_t_s * lambda_ls],
                     [g_1 - g_0 - g_prime_t_s * lambda_prev]]))
                a = a.item()
                b = b.item()
                disc = b ** 2 - 3 * a * g_prime_t_s
                if a == 0:
                    # actually quadratic
                    lambda_temp = -g_prime_t_s / (2 * b)
                else:
                    # legitimate cubic
                    lambda_temp = (-b + sqrt(disc)) / (3 * a)
                if lambda_temp > 1 / 2 * lambda_ls:
                    lambda_temp = 1 / 2 * lambda_ls
            lambda_prev = lambda_ls
            g_1 = g_2
            if lambda_temp <= 0.1 * lambda_ls:
                lambda_ls = 0.1 * lambda_ls
            else:
                lambda_ls = lambda_temp
    j_2 = jac(x_2)
    return x_2, f_2, g_2, s_0_n, j_2, magnitude_f, g_max,\
        backtrack_count, lambda_ls, outer_it_stop, accum_step


def scalar_prod(factor_a, factor_b):
    if size(factor_b) > 1:
        return factor_a.dot(factor_b)
    elif size(factor_b) == 1:
        return factor_a * factor_b


def solve_ax_equal_b(factor_a, term_b):
    if size(term_b) > 1:
        return gauss_elimination(factor_a, term_b)
    elif size(term_b) == 1:
        return 1 / factor_a * term_b

# noinspection PyUnboundLocalVariable


def secant_ls_3p(y, x_0, tol, x_1=None, f_prime=None,
                 max_it=100, alpha=1e-4, restriction=None, print_iterations=False):
    """3 point secant method with line search algorithm (single dimension)

        Tiruneh, Ababu Teklemariam. "A modified three-point Secant method with improved rate and
        characteristics of convergence." arXiv preprint arXiv:1902.09058 (2019).

        solves f(x) = 0 using the 3 point secant method (quadratic order), with additional line
        search for improved convergence. Function is evaluated in x_0 and x_1.
        First step is ordinary secant method, subsequent steps are 3-point secant method.
        Backtracking ("line search") ensures that the step decreases the optimization function.

        :param y: function
        :param x_0: point 1 for evaluation
        :param tol: tolerance for method
        :param max_it: maximum iterations
        :param alpha: stringency constant in (0,1/2)
        :param x_1: point 2 for evaluation (required if no derivative supplied)
        :param f_prime: derivative (optional)
        :param restriction: optional restrictions imposed on x. Line search when False.
        :param print_iterations: optionally print iteration results
        :return: dict with keys: ['x', 'f', 'x_list', 'f_list',
                 'iterations', 'total_backtracks', 'steps', 'success']
        """
    x_k = x_0
    y_k = y(x_k)
    g_k = 1 / 2 * y_k**2
    g_prime_k = - y_k**2
    # initialize to prevent
    x_k_minus_1 = 0
    x_k_minus_2 = 0
    y_k_minus_1 = 0
    y_k_minus_2 = 0
    # relative length of p as calculated in the stopping routine
    p = x_1 - x_0
    rellength = abs(p / x_k)
    lambda_min = tol / rellength
    accum_step = 0.0
    total_backtracks = 0
    backtrackcount = 0
    success = True
    steps = []
    y_list = []
    x_list = []
    for j in range(max_it):
        y_list += [y_k]
        x_list += [x_k]
        steps += [accum_step]
        if abs(y_k) <= tol:
            break
        if f_prime is not None:
            inv_slope = - 1 / f_prime(x_k)
            p = -inv_slope * y_k

            x_k_minus_1 = x_k
            y_k_minus_1 = y_k
            g_k_minus_1 = g_k
            g_prime_k_minus_1 = g_prime_k

        elif j == 0:
            inv_slope = -(x_1 - x_0) / y_k
            p = -inv_slope * y_k

            x_k_minus_1 = x_k
            y_k_minus_1 = y_k
            g_k_minus_1 = g_k
            g_prime_k_minus_1 = g_prime_k

        elif j == 1:
            inv_slope = (x_k - x_k_minus_1) / (y_k - y_k_minus_1)
            p = -inv_slope * y_k

            x_k_minus_2 = x_k_minus_1
            x_k_minus_1 = x_k

            y_k_minus_2 = y_k_minus_1
            y_k_minus_1 = y_k
            g_k_minus_1 = g_k
            g_prime_k_minus_1 = g_prime_k
        else:
            p = (x_k_minus_2 - x_k) - y_k_minus_2 * (
                y_k - y_k_minus_1
            ) / ((y_k - y_k_minus_2) / (x_k - x_k_minus_2) * (
                y_k - y_k_minus_1) - y_k * (
                (y_k - y_k_minus_2) / (x_k - x_k_minus_2) -
                    (y_k_minus_1 - y_k_minus_2) / (x_k_minus_1 - x_k_minus_2)
            )
            )

            x_k_minus_2 = x_k_minus_1
            x_k_minus_1 = x_k

            y_k_minus_2 = y_k_minus_1
            y_k_minus_1 = y_k
            g_k_minus_1 = g_k
            g_prime_k_minus_1 = g_prime_k

        stop = False
        lambda_ls = 1.0
        total_backtracks += backtrackcount
        backtrackcount = 0
        g_0 = g_k_minus_1
        g_prime_0 = g_prime_k_minus_1
        f_0 = y_k_minus_1
        while not stop and j + backtrackcount <= max_it + 1:
            if j + backtrackcount + 1 >= max_it:
                success = False
            # backtracking, line search - numerical recipes 3ed
            accum_step += lambda_ls
            if lambda_ls <= lambda_min:
                # opposite direction
                p = -p
                lambda_ls = 1.0
                stop = False

            x_2 = x_k + lambda_ls * p
            if restriction is not None and not restriction(x_2):
                # restriction not fulfilled - cannot evaluate func.
                x_2 = x_k
            else:
                f_2 = y(x_2)
                g_2 = 1 / 2 * f_2**2
            descent = alpha * lambda_ls * g_prime_0
            g_max = g_0 + descent

            satisfactory = g_2 <= g_max
            stop = satisfactory or lambda_ls <= lambda_min
            if print_iterations:
                print(
                    ('{:d}-{:d}:\t x_2: {:.4f}\ty_2: {:.4g}' +
                     '\tx_1: {:.4g}' +
                     '\tg_0: {:.4e}\tg_2: {:.4e}\tg_max: {:.4e}\tlambda: {:.2g}\t p: {:2g}').format(
                        j, backtrackcount, x_2, f_2, x_k_minus_1, g_0, g_2, g_max, lambda_ls, p
                    )
                )

            if not stop:
                # backtrack - reduce lambda
                x_list += [x_2]
                y_list += [f_2]
                steps += [accum_step]
                backtrackcount += 1
                accum_step -= lambda_ls
                if lambda_ls == 1:
                    # first backtrack quadratic fit
                    lambda_temp = -g_prime_0 / (
                        2 * (g_2 - g_0 - g_prime_0)
                    )
                elif lambda_ls < 1:
                    # subsequent backtracks cubic fit
                    a, b = 1 / (lambda_ls - lambda_prev) * array(
                        [[+1 / lambda_ls**2, -1 / lambda_prev**2],
                         [-lambda_prev / lambda_ls**2,
                          +lambda_ls / lambda_prev**2]]
                    ).dot(array(
                        [[g_2 - g_0 - g_prime_0 * lambda_ls],
                         [g_1 - g_0 - g_prime_0 * lambda_prev]]
                    ))
                    a, b = a.item(), b.item()
                    disc = b**2 - 3 * a * g_prime_0
                    if a == 0:
                        # actually quadratic
                        lambda_temp = - g_prime_0 / (2 * b)
                    else:
                        # legitimate cubic
                        lambda_temp = (-b + sqrt(disc)) / (3 * a)
                    if lambda_temp > 1 / 2 * lambda_ls:
                        lambda_temp = 1 / 2 * lambda_ls
                lambda_prev = lambda_ls
                g_1 = g_2
                if lambda_temp <= 0.1 * lambda_ls:
                    lambda_ls = 0.1 * lambda_ls
                else:
                    lambda_ls = lambda_temp
            x_k_plus_1 = x_2
            y_k_plus_1 = f_2
            g_k_plus_1 = g_2

        x_k = x_k_plus_1
        y_k = y_k_plus_1
        g_k = g_k_plus_1
        g_prime_k = - y_k ** 2
        if abs(p) <= tol or isnan(p):
            # avoid 1/0 division
            success = False
            break
    success = success and j <= max_it - 1
    x = x_k
    f = y_k
    f_list = y_list
    iterations = j
    if backtrackcount > 0:
        total_backtracks += backtrackcount
    soln = dict()
    for item in ['x', 'f', 'x_list', 'f_list',
                 'iterations', 'total_backtracks', 'steps', 'success']:
        soln[item] = locals().get(item)
    return soln
