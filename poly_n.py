from numpy import array, exp, empty, finfo, zeros, maximum, emath

# Ref. Press, William H., et al. "Numerical recipes in C++." The art of scientific computing 2 (2007): 1002.

def zroots(a, polish=False):
    a=array(a,ndmin=2)
    N,n=a.shape # N rows, poly order n
    assert (a[:,-1]!=0).all() # poly has order n-1
    eps = finfo(float).eps # a small number
    m = n-1
    roots = empty([N,m], dtype=complex)
    # copy coefficients for successful deflation
    ad = a.astype(complex).copy()
    for j in range(m-1, 0-1, -1):
        x = zeros(N,dtype=complex) # start at zero to favor convergence to
        # smallest remaining root, and return the root.
        ad_v, x, its = laguerre(ad[:,:j+2], x)
        idx=(abs(x.imag) <= 2.0 * eps * abs(x.real)) # near-real roots
        x[idx]=x[idx].real
        roots[:,j] = x
        # deflate, synthetic division
        b = ad[:,j+1]
        for jj in range(j, 0-1, -1):
            c = ad[:,jj].copy()
            ad[:,jj] = b
            b = x * b + c
    return roots

def laguerre(a, x):
    a=array(a,ndmin=2)
    N,n=a.shape # N rows, n cols
    ad_v = a
    mr = 8
    mt = 10*2
    maxit = mt * mr
    eps = finfo(float).eps
    # EPS here: estimated fractional roundoff error

    # try to break (rare) limit cycles with
    # mr different fractional values, once every mt steps,
    # for maxit total allowed iterations
    frac = [0.0,0.5,0.25,0.75,0.13,0.38,0.62,0.88,1.0]
    m = n-1
    dx=empty(N,dtype=complex) # init. dx
    for iter in range(1, maxit+1):
        # loop over iterations up to allowed maximum
        its = iter
        b = a[:,m]
        err = abs(b)
        d = f = zeros(N)
        abx = abs(x)
        for j in range(m-1, 0-1, -1):
            # efficient computation of the polynomial
            # and its first two derivatives. f stores P''/2
            f = x * f + d
            d = x * d + b
            b = x * b + a[:,j]
            err = abs(b) + abx * err

        # estimate of roundoff error in evaluating 
        # polynomial
        err *= eps
        if (abs(b) <= err).all(): return ad_v, x, its  # we are on the root
        # the generic case: use Laguerre's formula
        g = d/b # exclude rows that converged already (b=0)
        g2 = g**2
        h = g2 - 2.0 * f/b
        sq = emath.sqrt((m-1) * (m*h - g2))
        gp = g + sq
        gm = g -sq
        abp = abs(gp)
        abm = abs(gm)
        gp[abp<abm]=gm[abp<abm]
        idx=maximum(abp,abm)>0
        dx[idx]=float(m)/gp[idx]
        dx[~idx]=(1+abx[~idx])*exp(iter*1j) # equivalent to polar(1+abx, iter)
        x1 = x - dx
        if (dx == 0).all():
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


print(zroots([[1,2,3,4,5],[5,6,7,8,9]]))

a=array([[-2.4e+01, -2.3e+01, -2.2e+01, -2.1e+01, -2.0e+01],
       [-1.9e+01, -1.8e+01, -1.7e+01, -1.6e+01, -1.5e+01],
       [-1.4e+01, -1.3e+01, -1.2e+01, -1.1e+01, -1.0e+01],
       [-9.0e+00, -8.0e+00, -7.0e+00, -6.0e+00, -5.0e+00],
       [-4.0e+00, -3.0e+00, -2.0e+00, -1.0e+00,  1.0e-13],
       [ 1.0e+00,  2.0e+00,  3.0e+00,  4.0e+00,  5.0e+00],
       [ 6.0e+00,  7.0e+00,  8.0e+00,  9.0e+00,  1.0e+01],
       [ 1.1e+01,  1.2e+01,  1.3e+01,  1.4e+01,  1.5e+01],
       [ 1.6e+01,  1.7e+01,  1.8e+01,  1.9e+01,  2.0e+01],
       [ 2.1e+01,  2.2e+01,  2.3e+01,  2.4e+01,  2.5e+01],
       [ 2.6e+01,  2.7e+01,  2.8e+01,  2.9e+01,  3.0e+01],
       [ 3.1e+01,  3.2e+01,  3.3e+01,  3.4e+01,  3.5e+01],
       [ 3.6e+01,  3.7e+01,  3.8e+01,  3.9e+01,  4.0e+01],
       [ 4.1e+01,  4.2e+01,  4.3e+01,  4.4e+01,  4.5e+01],
       [ 4.6e+01,  4.7e+01,  4.8e+01,  4.9e+01,  5.0e+01]])

print(zroots(a))
