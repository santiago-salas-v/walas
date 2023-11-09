from numpy import array, lexsort, pi, cos, arccos, log10


def solve_cubic(abcd):
    """ solve cubic polynomial - Tartaglia-Cardano

    ref. Polyanin, Manzhirov Handbook of Mathematics for engineers
    and scientists

    a*x^3+b*x^2+c*x+d=0

    params:
    abc: list [a,b,c,d] of parameters

    returns:
    dict with {'roots': [x1, x2, x3], 'disc': disc} 
    where roots are pairs [Re(xi), Im(xi)]
    """
    a, b, c, d = abcd
    # transform to incomplete eq y^3+py+q=0
    # substitution  x = y-b/(3a)
    p = -1 / 3 * (b / a)**2 + c / a
    q = 2 / 27 * (b / a)**3 - 1 / 3 * b * c / a**2 + d / a

    # roots of the incomplete eq
    # as [Re(yi), Im(yi)]
    disc = (p / 3)**3 + (q / 2)**2

    if disc < 0:
        # 3 real roots
        re_y1 = 2 * (-p / 3)**(1 / 2) * cos(
            1 / 3 * arccos(-q / 2 / (-p / 3)**(3 / 2)) + 0 * 2 * pi / 3)
        re_y2 = 2 * (-p / 3)**(1 / 2) * cos(
            1 / 3 * arccos(-q / 2 / (-p / 3)**(3 / 2)) + 1 * 2 * pi / 3)
        re_y3 = 2 * (-p / 3)**(1 / 2) * cos(
            1 / 3 * arccos(-q / 2 / (-p / 3)**(3 / 2)) + 2 * 2 * pi / 3)
        im_y1, im_y2, im_y3 = 0, 0, 0
    elif disc >=0:

        if -q / 2 + (disc)**(1 / 2) < 0:
            # avoid complex bv by negative root
            au = -(+q / 2 - (disc)**(1 / 2))**(1 / 3)
        else:
            au = (-q / 2 + (disc) ** (1 / 2)) ** (1 / 3)
        if -q / 2 - (disc)**(1 / 2) < 0:
            # avoid complex bv by negative root
            bv = -(+q / 2 + (disc)**(1 / 2))**(1 / 3)
        else:
            bv = (-q / 2 - (disc) ** (1 / 2)) ** (1 / 3)

        if disc > 0:
            # 1 real root, 2 complex roots
            re_y1 = au + bv
            re_y2 = -1 / 2 * (au + bv)
            re_y3 = -1 / 2 * (au + bv)
            im_y1 = 0
            im_y2 = +(3)**(1 / 2) / 2 * (au - bv)
            im_y3 = -(3)**(1 / 2) / 2 * (au - bv)
        elif disc == 0:
            # one real root, and two real roots of multiplicity 2
            re_y1 = au + bv
            re_y2 = -1 / 2 * (au + bv)
            re_y3 = -1 / 2 * (au + bv)
            im_y1, im_y2, im_y3 = 0, 0, 0
    # sort the roots by real (descending), then imaginary part (0 first)
    y_real_parts = array([re_y1, re_y2, re_y3])
    y_imag_parts = array([im_y1, im_y2, im_y3])
    positions = lexsort([-y_real_parts, y_imag_parts, abs(y_imag_parts)])
    # roots of the complete equation through substitution xk=yk-b/(3a)
    x_list = []
    for i in positions:
        x_list += [[y_real_parts[i] - b / 3 / a, y_imag_parts[i]]]
    return dict([['roots', x_list], ['disc', disc], ['p', p], ['q', q]])


def solve_quartic(abcde):
    """ solve quartic polynomial - Ferrari

    ref. Bewersdorff Algebra für Einsteiger 5. Auflage

    a*x^4+b*x^3+c*x^2+d*x+e=0

    params:
    abc: list [a,b,c,d,e] of parameters

    returns:
    x1, x2, x3, x4 as pairs  [Re(xi), Im(xi)]
    """
    a, b, c, d, e = abcde
    # transform to biquadratic eq y^4+py^2+qy+r=0
    # substitution  x = y-b/(4a)
    p = (8 * a * c - 3 * b**2) / (8 * a**2)
    q = (b**3 - 4 * a * b * c + 8 * a**2 * d) / (8 * a**3)
    r = (16 * a * b**2 * c + 256 * a**3 * e - 3 *
         b**4 - 64 * a**2 * b * d) / (256 * a**4)

    # solve cubic resolvent z^3-p/2z^2-rz+pr/2-q^2/8=0
    z = solve_cubic([1, -p / 2, -r, p * r / 2 - q**2 / 8])['roots']
    # chose any root (here the real one)
    # z = z[1][0]+z[1][1]*1j
    z = complex(z[0][0])  # +z[0][1]*1j

    if -q / 2 < 0:
        s = -1
    else:
        s = +1

    # roots to biquadratic
    y1 = 1 / 2 * (2 * z - p)**(1 / 2) + (
        -1 / 2 * z - 1 / 4 * p + s * (z**2 - r)**(1 / 2))**(1 / 2)
    y2 = 1 / 2 * (2 * z - p)**(1 / 2) - (
        -1 / 2 * z - 1 / 4 * p + s * (z**2 - r)**(1 / 2))**(1 / 2)
    # print([-q/2, s*(2*z-p)**(1/2)*(z**2-r)**(1/2)])
    y3 = -1 / 2 * (2 * z - p)**(1 / 2) + (
        -1 / 2 * z - 1 / 4 * p - s * (z**2 - r)**(1 / 2))**(1 / 2)
    y4 = -1 / 2 * (2 * z - p)**(1 / 2) - (
        -1 / 2 * z - 1 / 4 * p - s * (z**2 - r)**(1 / 2))**(1 / 2)

    # sort the roots by real (descending), then imaginary part (0 first)
    y_real_parts = array([yi.real for yi in [y1, y2, y3, y4]])
    y_imag_parts = array([yi.imag for yi in [y1, y2, y3, y4]])
    positions = lexsort([-y_real_parts, y_imag_parts, abs(y_imag_parts)])
    # roots of the complete equation by substitution xk=yk-b/(4a)
    x_list = []
    for i in positions:
        x_list += [[y_real_parts[i] - b / (4 * a), y_imag_parts[i]]]
    return dict([['roots', x_list]])


def solve_scaled(coefs):
    ck = [abs(x) for x in coefs]
    v = log10(max(ck) / min(ck))
    print(v)
    sij = []
    s = 1
    for i in range(len(ck)):
        for j in range(len(ck)):
            if j > i:
                print('i: {:d} j: {:d}'.format(i, j))
                sij += [(ck[i] / ck[j])**(1 / (j - i))]
                print('s*=' + '{:0.20f}'.format(sij[-1]))
                cksk = [sij[-1]**(len(ck) - 1 - k) * ck[k]
                        for k in range(len(ck))]
                vs = log10(max(cksk) / min(cksk))
                print('log10(M/m)={:0.15f}'.format(vs) + '\n')
                if vs < v:
                    v = vs
                    s = sij[-1]

    print('s*={:0.4f}'.format(s) + '\n coeffs:')
    coefs2 = [coefs[i] * s**(len(ck) - 1 - i) for i in range(len(ck))]
    print(coefs2)
    print('soln:')
    if len(coefs) == 4:
        sx = solve_cubic(coefs2)
    elif len(coefs) == 5:
        sx = solve_quartic(coefs2)
    x = [(x[0] + x[1] * 1j) * s for x in sx]
    print(x)
    p = 0
    for soln in sx:
        sxi = soln[0] + soln[1] * 1j
        x = sxi * s
        for i in range(len(coefs)):
            p += coefs[i] * (x)**(len(coefs) - 1 - i)
    print('err:')
    print(p)
    print('')
