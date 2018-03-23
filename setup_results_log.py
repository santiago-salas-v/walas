import numpy as np
import logging
import os


def setup_log_file(file_name, with_console=True):
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=os.path.join('./logs', file_name),
        filemode='w')
    if with_console:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(console)


def notify_status_func(progress_k, stop_value, k,
                       j_it_backtrack, lambda_ls, accum_step,
                       x, diff, f_val, j_val, lambda_ls_y,
                       method_loops, g_min=np.nan, g1=np.nan):
    y = lambda_ls_y
    if np.size(x) > 1:
        diff_str = str(np.sqrt(diff.T.dot(diff)))
        x_str = '[' + ','.join(map(str, x)) + ']'
        f_val_str = '[' + ','.join(map(str, f_val)) + ']'
        f_mag_str = str(np.sqrt(f_val.T.dot(f_val)))
        j_val_str = ','.join(map(str, j_val.tolist()))
        y_str = '[' + str(np.sqrt(y.T.dot(y))) + ']'
    else:
        diff_str = str(diff)
        x_str = str(x)
        f_val_str = str(f_val)
        f_mag_str = str(np.sqrt(f_val**2))
        j_val_str = str(j_val)
        y_str = str(y)

    pr_str = ';k=' + str(k) + \
        ';backtrack=' + str(j_it_backtrack) + \
        ';lambda_ls=' + str(lambda_ls) + \
        ';accum_step=' + str(accum_step) + \
        ';stop=' + str(stop_value) + \
        ';X=' + x_str + \
        ';||X(k)-X(k-1)||=' + diff_str + \
        ';f(X)=' + f_val_str + \
        ';||f(X)||=' + f_mag_str + \
        ';j(X)=' + j_val_str + \
        ';Y=' + y_str + \
        ';||Y||=' + y_str + \
        ';g=' + str(g_min) + \
        ';|g-g1|=' + str(abs(g_min - g1))
    logging.debug(pr_str)


def log_line(line):
    logging.debug(line)
