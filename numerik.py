import numpy as np


def lrpd(a):
    """L,R,P,D,DA from LR = PDA factorization
    Method: Dahmen W., Reusken A.; Numerik fuer Ingenieure und Naturwissenschaeftler; Springer S. 79
    :param a: numpy.matrix NxM
    """
    n = a.shape[0]
    m = a.shape[1]
    p = np.matrix(np.eye(n, dtype=float))
    d = np.matrix(np.eye(n, dtype=float))
    dlr = np.matrix(np.zeros([n, m], dtype=float))
    l = np.matrix(np.eye(n, m, dtype=float))
    r = np.matrix(np.zeros([n, m], dtype=float))
    indexes_r = [item for item in range(n)]
    for i in range(n):
        d[i, i] = 1 / sum(abs(a[i, :m]))
        # scaling
        dlr[i] = d[i, i] * a[i]  # dlr is just for storage of both l and r
    for j in range(n):
        indexes_r[j] = j + np.argmax(abs(dlr[j:, j]))
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
    :param r: numpy.matrix NxM
    :return: numpy.matrix NxM
    """
    n = r.shape[0]
    m = r.shape[1]
    r_form = np.matrix(np.copy(r))
    # denominators for diagonal 1
    den = [0.0 for i in range(n)]
    for i in range(n - 1, 0 - 1, -1):
        for j in range(m):
            if i > 0 and den[i] == 0.0 and abs(r[i, j]) > np.finfo(float).eps:
                # first non-zero element in last rows
                den[i] = r[i, j]
                for k in range(i - 1, 0 - 1, -1):
                    num = r[k, j]
                    r_form[k, :] = r_form[k, :] - num / den[i] * r_form[i, :]
            elif i == 0 and j == 0:
                den[i] = r[i, j]
            if abs(r_form[i, j]) > 0 and abs(
                    r_form[i, j]) <= np.finfo(float).eps:
                # Either way make eps 0, avoid propagation of error.
                r_form[i, j] = 0.0
    for i in range(n):
        r_form[i] = r_form[i] / den[i]
    return r_form


def ref(a):
    """
    Reduced echelon form of matrix a
    :param a: numpy.matrix NxM
    :return: numpy.matrix NxM
    """
    n = a.shape[0]
    m = a.shape[1]
    p = np.matrix(np.eye(n, dtype=float))
    d = np.matrix(np.eye(n, dtype=float))
    dr = np.matrix(np.zeros([n, m], dtype=float))
    r = np.matrix(np.zeros([n, m], dtype=float))
    indexes_r = [item for item in range(n)]
    shift = 0  # diagonal !=0, no shift
    for i in range(n):
        d[i, i] = 1 / sum(abs(a[i, :m]))
        # scaling
        dr[i] = d[i, i] * a[i]  # dr is just for storage r
    for j in range(n):
        while all(abs(dr[j:, j + shift]) == 0):
            # shift from diagonal
            shift += 1
        indexes_r[j] = j + np.argmax(abs(dr[j:, j + shift]))
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
    """Solution of the system Ax = b (LRx=PDb) through forward substitution of Ly = Pb, and then
    backward substitution of Rx = y
    :param a: numpy.matrix n X n
    :param b: numpy.matrix n X 1
    """
    n = a.shape[0]
    x = np.matrix(np.zeros_like(b, dtype=float))
    y = np.matrix(np.zeros_like(b, dtype=float))
    l, r, p, d, da = lrpd(a)
    pdb = p * d * b
    sum_lik_xk = 0.
    sum_lik_yk = 0.
    for j in range(0, n, +1):
        # Forward substitution Ly = PDb
        for k in range(0, j, +1):
            sum_lik_yk = sum_lik_yk + l[j, k] * y[k]
        y[j] = (pdb[j, 0] - sum_lik_yk) / 1.0
        sum_lik_yk = 0
    for j in range(n - 1, -1, -1):
        # Backward substitution Rx = y
        for k in range(n - 1, j, -1):
            sum_lik_xk = sum_lik_xk + r[j, k] * x[k]
        x[j] = (y[j, 0] - sum_lik_xk) / r[j, j]
        sum_lik_xk = 0
    return x


# noinspection PyAugmentAssignment
def nr_ls(x0, f, j, tol, max_it, inner_loop_condition,
          notify_status_func, method_loops, process_func_handle):
    x = x0
    # Newton method: G(x) = J(x)^-1 * F(x)
    outer_it_k = 0
    inner_it_j = 0
    j_val = j(x)
    f_val = f(x)
    y = np.matrix(np.ones(len(x))).T * tol / (np.sqrt(len(x)) * tol)
    magnitude_f = np.sqrt((f_val.T * f_val).item())
    # Line search variable lambda
    lambda_ls = 0.0
    accum_step = 0.0
    # For progress bar, use log scale to compensate for quadratic convergence
    log10_to_o_max_magnitude_f = np.log10(tol / magnitude_f)
    progress_k = (1.0 - np.log10(tol / magnitude_f) /
                  log10_to_o_max_magnitude_f) * 100.0
    diff = np.matrix(np.empty([len(x), 1]))
    diff.fill(np.nan)
    stop = False
    divergent = False
    # Non-functional status notification
    notify_status_func(progress_k, stop, outer_it_k,
                       inner_it_j, lambda_ls, accum_step,
                       x, diff, f_val, j_val, lambda_ls * y,
                       method_loops)
    # End non-functional notification
    while outer_it_k <= max_it and not stop:
        outer_it_k += 1
        method_loops[1] += 1
        inner_it_j = 0
        lambda_ls = 1.0
        accum_step += lambda_ls
        x_k_m_1 = x
        progress_k_m_1 = progress_k
        y = gauss_elimination(j_val, -f_val)
        # First attempt without backtracking
        x = x + lambda_ls * y
        diff = x - x_k_m_1
        j_val = j(x)
        f_val = f(x)
        magnitude_f = np.sqrt((f_val.T * f_val).item())
        if magnitude_f < tol and inner_loop_condition(x):
            stop = True  # Procedure successful
        else:
            # Non-functional status notification
            notify_status_func(progress_k, stop, outer_it_k,
                               inner_it_j, lambda_ls, accum_step,
                               x, diff, f_val, j_val, lambda_ls * y,
                               method_loops)
            # End non-functional notification

            # For progress use log scale to compensate for quadratic
            # convergence
            progress_k = (1.0 - np.log10(tol / magnitude_f) /
                          log10_to_o_max_magnitude_f) * 100.0
            if np.isnan(magnitude_f) or np.isinf(magnitude_f):
                # TODO: Re-implement steepest descent
                stop = True  # Divergent method
                divergent = True
                progress_k = 0.0
            else:
                pass
            if progress_k == progress_k_m_1:
                # Non-functional gui processing
                process_func_handle()
                # End non-functional processing
                # if form.progress_var.wasCanceled():
                # stop = True
        while inner_it_j <= max_it and \
                not inner_loop_condition(x) and \
                not stop:
            # Backtrack if any conc < 0. Line search method.
            # Ref. http://dx.doi.org/10.1016/j.compchemeng.2013.06.013
            inner_it_j += 1
            lambda_ls = lambda_ls / 2.0
            accum_step += -lambda_ls
            x = x_k_m_1
            progress_k = progress_k_m_1
            x = x + lambda_ls * y
            diff = x - x_k_m_1
            j_val = j(x)
            f_val = f(x)
            # Non-functional status notification
            notify_status_func(progress_k, stop, outer_it_k,
                               inner_it_j, lambda_ls, accum_step,
                               x, diff, f_val, j_val, lambda_ls * y,
                               method_loops)
            # End non-functional notification
            method_loops[0] += 1
    if stop and not divergent:
        progress_k = 100.0
    elif divergent:
        progress_k = 0.0
    # Non-functional status notification
    notify_status_func(progress_k, stop, outer_it_k,
                       inner_it_j, lambda_ls, accum_step,
                       x, diff, f_val, j_val, lambda_ls * y,
                       method_loops)
    # End non-functional notification
    return progress_k, stop, outer_it_k,\
        inner_it_j, lambda_ls, accum_step,\
        x, diff, f_val, lambda_ls * y,\
        method_loops


a = """
1	0	1	0	1	0	0	0	0
1	0	2	1	0	0	0	2	0
0	0	0	0	0	1	0	0	0
0	2	0	2	4	3	0	0	2
0	0	0	0	0	0	1	0	0
"""

a = np.array(eval(
    '[' + a.replace('\t', ',').replace('\n', '],[')[2:-2] + ']'
), dtype=float)

dr, p, d = ref(a)

print(np.array(dr))
