from numpy import array, exp, real, imag, empty, finfo

# Ref. Press, William H., et al. "Numerical recipes in C++." The art of scientific computing 2 (2007): 1002.

def zroots(a, polish=False):
    eps = 1.0e-14 # a small number
    m = len(a)-1
    roots = empty(len(a)-1, dtype=complex)
    # copy coefficients for successful deflation
    ad = a.copy()
    for j in range(m-1, 0-1, -1):
        x = 0.0 # start at zero to favor convergence to
        # smallest remaining root, and return the root.
        ad_v = empty(j+2, dtype=complex)
        for jj in range(0, j+2):
            ad_v[jj] = ad[jj]
        ad_v, x, its = laguerre(ad_v, x)
        if(abs(imag(x)) <= 2.0 * eps * abs(real(x))):
            x = real(x) + 0.0*1j
        roots[j] = x
        b = ad[j + 1]
        for jj in range(j, 0-1, -1):
            c = ad[jj]
            ad[jj] = b
            b = x * b + c
    return roots

def laguerre(a, x):
    ad_v = a
    mr = 8
    mt = 10
    maxit = mt * mr
    eps = finfo(float).eps
    # EPS here: estimated fractional roundoff error

    # try to break (rare) limit cycles with
    # mr different fractional values, once every mt steps,
    # for maxit total allowed iterations
    frac = [0.0,0.5,0.25,0.75,0.13,0.38,0.62,0.88,1.0]
    m = len(a) - 1
    for iter in range(1, maxit+1):
        # loop over iterations up to allowed maximum
        its = iter
        b = a[m]
        err = abs(b)
        d = f = 0.0
        abx = abs(x)
        for j in range(m-1, 0-1, -1):
            # efficient computation of the polynomial
            # and its first two derivatives. f stores P''/2
            f = x * f + d
            d = x * d + b
            b = x * b + a[j]
            err = abs(b) + abx * err

        # estimate of roundoff error in evaluating 
        # polynomial
        err *= eps
        if abs(b) <= err: return ad_v, x, its  # we are on the root
        # the generic case: use Laguerre's formula
        g = d/b
        g2 = g**2
        h = g2 - 2.0 * f/b
        sq = (float(m-1) * (float(m)*h - g2))**(1/2)
        gp = g + sq
        gm = g -sq
        abp = abs(gp)
        abm = abs(gm)
        if abp < abm: gp = gm
        if max(abp, abm) > 0.0:
            dx = float(m) / gp
        else:
            # equivalent to polar(1+abx, iter)
            dx = (1+abx)*exp(iter*1j)
        x1 = x - dx
        if x == x1:
            print('converged')
            return adv_v, its  # converged
        if iter % mt != 0:
            x = x1
        else:
            x -= frac[int(iter/mt)] * dx

    print('not converged')
    raise Exception("too many iterations in laguerre")
    # very unusual: can occurr only for complex roots.
    # try a different starting guess.
    return ad_v, x, its
    