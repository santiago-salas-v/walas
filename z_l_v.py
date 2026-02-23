import re

from numpy import array, zeros, abs, ones, empty, argwhere, asarray
from numpy import nan, finfo
from numpy import sqrt, outer, sum, log, exp, diag, sign, diagonal, maximum
from scipy import optimize
from scipy.optimize import fsolve

from numerik import secant_ls_3p, line_search, bisection
from poly_3_4 import solve_cubic, solve_quartic
from poly_n import zroots
from numpy import minimum

r_def = 8.3145 * 10. ** 6 / 10. ** 5  # bar cm^3/(mol K)
eps=finfo(float).eps
abs_tol=eps
inf=1/finfo(float).eps


class State:
    def __init__(self, t, p, z_i, mm_i, tc_i, pc_i, af_omega_i, eos_name='pr',
                 max_it=100, tol=1e-10, sec_j=None, nu_ij=None, unifac_data_dict=None):
        z_i = array(z_i)
        mm_i = array(mm_i)
        tc_i = array(tc_i)
        pc_i = array(pc_i)
        af_omega_i = array(af_omega_i)
        if sec_j is not None:
            sec_j = array(sec_j)
            nu_ij = array(nu_ij)
        self.eos = Eos(t, p, z_i, mm_i, tc_i, pc_i, af_omega_i, eos_name, max_it, tol, r=r_def,
                       sec_j=sec_j, nu_ij=nu_ij, unifac_data_dict=unifac_data_dict)
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
        self.z = 0  # init
        self.sec_j = sec_j
        self.nu_ij = nu_ij
        self.unifac_data_dict = unifac_data_dict
        if sec_j is not None:
            self.sec_j = sec_j
            self.nu_ij = nu_ij
            if unifac_data_dict is None:
                self.unifac_data_dict = setup_unifac_data()
        self.bubl_p = p
        self.bubl_t = t
        self.dew_p = p
        self.dew_t = t
        self.v_f = 1
        self.v_l = nan
        self.v_v = nan
        self.phi_l = ones(len(z_i))
        self.phi_v = ones(len(z_i))
        self.solve()

    def set_t(self, t):
        self.eos.t = t
        self.solve()

    def set_p(self, p):
        self.eos.p = p
        self.solve()

    def get_z(self):
        return self.z

    def solve(self):
        self.eos.pt_flash()
        self.bubl_p = self.eos.bubl_p_p
        self.dew_p = self.eos.dew_p_p
        self.v_f = self.eos.v_f
        self.phi_v = self.eos.phi_v
        self.phi_l = self.eos.phi_l
        self.v_l = self.eos.v_l
        self.v_v = self.eos.v_v


class Eos:
    def __init__(self, t, p, z_i, mm_i, tc_i, pc_i, af_omega_i, eos_name,
                 max_it=100, tol=finfo(float).eps, r=r_def,
                 sec_j=None, nu_ij=None, unifac_data_dict=None):
        self.eos = eos_name
        if eos_name == 'pr':
            epsilon = 1 - sqrt(2)
            sigma = 1 + sqrt(2)
            omega = 0.07780
            psi = 0.45724
            m = 0.37464 + 1.54226 * af_omega_i - 0.26992 * af_omega_i ** 2

            def alpha_tr(tr, af_omega):
                return (1 + (
                        0.37464 + 1.54226 * af_omega - 0.26992 * af_omega ** 2
                        ) * (1 - tr ** (1 / 2.))) ** 2

            def dalpha_dt(tr, af_omega):
                alpha_i = self.alpha_tr(tr, af_omega)
                m_i = self.m
                return - alpha_i ** (1 / 2.) * m_i / sqrt(tr) / tc_i
        elif eos_name == 'srk':
            epsilon = 0.
            sigma = 1.
            omega = 0.08664
            psi = 0.42748
            m = 0.480 + 1.574 * af_omega_i - 0.176 * af_omega_i ** 2

            def alpha_tr(tr, af_omega):
                return (1 + (
                        0.480 + 1.574 * af_omega - 0.176 * af_omega ** 2
                        ) * (1 - tr ** (1 / 2.))) ** 2

            def dalpha_dt(tr, af_omega):
                alpha_i = self.alpha_tr(tr, af_omega)
                m_i = self.m
                return - alpha_i ** (1 / 2.) * m_i / sqrt(tr) / tc_i
        elif eos_name == 'rk':
            epsilon = 0.
            sigma = 1.
            omega = 0.08664
            psi = 0.42748
            m = 1.0

            def alpha_tr(tr, _):
                return tr ** (-1 / 2.)

            def dalpha_dt(tr, af_omega):
                alpha_i = self.alpha_tr(tr, af_omega)
                m_i = self.m
                return - 1 / 2 * tr**(-3 / 2) / tc_i
        else:
            # vdw
            epsilon = 0
            sigma = 0
            omega = 1 / 8
            psi = 27 / 64
            m = 0

            def alpha_tr(_, __):
                return 1

            def dalpha_dt(_, __):
                return 0
        self.epsilon = epsilon
        self.sigma = sigma
        self.omega = omega
        self.psi = psi
        self.m = m
        self.alpha_tr = alpha_tr
        self.max_it = max_it
        self.tol = tol
        self.r = r
        self.sec_j = sec_j
        self.nu_ij = nu_ij
        self.unifac_data_dict = unifac_data_dict

        self.z_i = z_i
        self.tc_i = tc_i
        self.pc_i = pc_i
        self.af_omega_i = af_omega_i
        self.t = t
        self.p = p
        self.z = 0
        self.bubl_p_p = p
        self.dew_p_p = p
        self.v_f = 1
        self.phi_l = ones(len(z_i))
        self.phi_v = ones(len(z_i))
        self.k_i = ones(len(z_i))
        self.v_l = nan
        self.v_v = nan

    def bubl_p(self):
        soln = bubl_p(
            self.t, self.p, self.z_i, self.tc_i, self.pc_i, self.af_omega_i,
            self.alpha_tr, self.epsilon, self.sigma, self.psi, self.omega,
            self.max_it)
        return soln

    def pt_flash(self):
        r = r_def
        # mechanical critical point
        z_mc = -1 / 3 * ((self.epsilon + self.sigma)
                         * self.omega - self.omega - 1)
        p_mc = sum(self.z_i * self.pc_i)
        t_mc = self.z_i.dot(sqrt(
            outer(self.tc_i, self.tc_i))).dot(
            self.z_i).item()
        if self.t > t_mc:
            # no saturation point possible, supercritical mixture
            self.bubl_p_p = None
            self.dew_p_p = None
            self.v_f = 1
            soln_v = phi(self.t, self.p, self.z_i, self.tc_i, self.pc_i, self.af_omega_i,
                         self.alpha_tr, self.epsilon, self.sigma, self.psi, self.omega)
            self.phi_v = soln_v['phi_i_v']
            self.phi_l = None
            self.v_l = None
            self.v_v = soln_v['v_v']
            return soln_v
        else:
            soln_p_sat = p_i_sat_ceos(self.t, self.p, self.tc_i, self.pc_i, self.af_omega_i,
                                      self.alpha_tr, self.epsilon, self.sigma, self.psi, self.omega,
                                      tol=self.tol)
            pisat = array([soln_p_sat['p'][i] for i in range(
                len(self.z_i)) if soln_p_sat['success'][i]])
            zisat = array([self.z_i[i] for i in range(
                len(self.z_i)) if soln_p_sat['success'][i]])
            p_est_0 = sum(pisat * zisat / sum(zisat))
            soln = pt_flash(self.t, self.p, self.z_i, self.tc_i, self.pc_i, self.af_omega_i,
                            self.alpha_tr, self.epsilon, self.sigma, self.psi, self.omega,
                            sec_j=self.sec_j, nu_ij=self.nu_ij, unifac_data_dict=self.unifac_data_dict,
                            max_it=self.max_it, tol=self.tol, p_est_0=p_est_0)
            self.bubl_p_p = soln['bubl_p_soln']['p']
            self.dew_p_p = soln['dew_p_soln']['p']
            self.v_f = soln['v_f']
            if self.v_f == 0:
                # liquid
                soln_phi_l = phi(self.t, self.p, self.z_i, self.tc_i, self.pc_i, self.af_omega_i,
                                 self.alpha_tr, self.epsilon, self.sigma, self.psi, self.omega)
                self.phi_l = soln_phi_l['phi_i_l']
                self.phi_v = None
                self.v_v = soln_phi_l['v']
                self.v_l = None
            elif self.v_f == 1:
                # vapor
                soln_phi_v = phi(self.t, self.p, self.z_i, self.tc_i, self.pc_i, self.af_omega_i,
                                 self.alpha_tr, self.epsilon, self.sigma, self.psi, self.omega)
                self.phi_l = None
                self.phi_v = soln_phi_v['phi_i_v']
                self.v_l = None
                self.v_v = soln_phi_v['v_v']
            else:
                self.phi_l = soln['phi_i_l']
                self.phi_v = soln['phi_i_v']
                self.v_v = soln['soln_phi_v']['v']
                self.v_l = soln['soln_phi_l']['v']
            return soln

def mid(a,b,c):
    #  simplified helper function for the median of 3 arguments
    return maximum(minimum(a,b),minimum(maximum(a,b),c))

def use_pr_eos():
    # PR (1976) params
    epsilon=1-sqrt(2)
    sigma=1+sqrt(2)
    omega=0.07780
    psi=0.45724
    zc=0.3070
    def alpha_tr(tr, af_omega,return_deriv=False):
        #alpha0=tr**-0.171813*exp(0.125283*(1-tr**1.77634))
        #alpha1=tr**-0.607352*exp(0.511614*(1-tr**2.20517))
        #return alpha0+w*(alpha0-alpha1)
        alpha=(1+(0.37464+1.54226*af_omega-0.26992*af_omega**2)*(1-tr**(1/2.)))**2
        if not return_deriv:
            return alpha
        else:
            t_dalpha_dt=-sqrt(alpha_tr(tr,af_omega))*(0.3746+1.54226*w-0.26992*w**2)*tr**(1/2)
            return alpha, t_dalpha_dt
    def rho_lims(tc,pc,r=r_def):
        # absolute boundaries are points of singularity in the EOS p=RT/(v-b)-th/((v-(-d/2+sqrt((d/2)^2-e)))*((v-(-d/2-sqrt((d/2)^2-e)))))
        b=omega*r*tc/pc
        rho_lo=-1/((1+sqrt(2))*b) # negative root of 1-sqrt(2) and 1+sqrt(2) is the pole of the second term
        rho_hi=1/b # pole of the first term
        return rho_lo,rho_hi

    return alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims


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

    return alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims


def use_srk_eos_simple_alpha():
    # SRK (1972) params
    # SRK without acentric factor alpha(Tr) = Tr^(-1/2)
    epsilon = 0.
    sigma = 1.
    omega = 0.08664
    psi = 0.42748

    def alpha_tr(tr, _): return \
        tr ** (-1 / 2.)

    return alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims


# Dampfdruck nach VDI-Wärmeatlas
def p_sat_func(psat, t, af_omega, tc, pc, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, full_output=False):
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


def p_sat(t, af_omega, tc, pc, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims):
    def fs(psat): return p_sat_func(psat, t, af_omega, tc, pc, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)

    # Das Levenberg-Marquardt Algorithmus weist wenigere Sprunge auf.
    return optimize.root(fs, 1., method='lm')


def p_i_sat_ceos(t, p, tc_i, pc_i, af_omega_i,
                 alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
                 max_it=100, tol=abs_tol, k=0.95, rt_factor=0.1, r=r_def):
    """saturation pressure of **individual** components by ceos"""
    # ensure shape is possible
    t,p,tc_i,pc_i,af_omega_i=[array(x,ndmin=1) for x in [t,p,tc_i,pc_i,af_omega_i]] 

    n_comps = tc_i.size
    p_sat = empty([t.shape[0],n_comps])
    success = empty([t.shape[0],n_comps], dtype=bool)
    n_fev = 0

    tr_i=outer(t,1/tc_i)
    pr_i=outer(p,1/pc_i)
    tr_i=outer(t,1/tc_i)
    pr_i=outer(p,1/pc_i)
    a=psi*alpha_tr(tr_i,af_omega_i)*r**2*tc_i**2/pc_i # individual
    b=omega*r*tc_i/pc_i#*ones(tr_i.shape) # individual
    q=a/(b*r*tr_i*tc_i) # individual
    rho_mc=pc_i/(zc*r*tc_i)#*ones(tr_i.shape) # individual
    rho_lo,rho_hi=rho_lims(tc_i,pc_i) #[x*ones(tr_i.shape) for x in rho_lims(tc_i,pc_i)]
    beta=b*pr_i*pc_i/(r*tr_i*tc_i)

    # since secant_ls_3p has poor performance near zero, use bisection
    def dp_drho_t(rho,t):
        # limiting density value for which 2 phases cease to coexist, limit of mechanical stability
        return (-r*t/(1/rho-b)**2+a*(1/(1/rho+epsilon*b)+1/(1/rho+sigma*b))/((1/rho+epsilon*b)*(1/rho+sigma*b)))*-(1/rho**2)

    # exclude supercritical temperature cases (no p_sat)
    idx_sc=(tr_i>1)
    t=(tr_i*tc_i)[~idx_sc]
    p=(pr_i*pc_i)[~idx_sc]
    q=q[~idx_sc]
    a=a[~idx_sc]
    beta=beta[~idx_sc]
    b=array([b for _ in range(tr_i.shape[0])])[~idx_sc]
    rho_mc=array([rho_mc for _ in range(tr_i.shape[0])])[~idx_sc]
    rho_hi=array([rho_hi for _ in range(tr_i.shape[0])])[~idx_sc]
    rho_lo=array([rho_lo for _ in range(tr_i.shape[0])])[~idx_sc]
    tc_i_ravel=array([tc_i for _ in range(tr_i.shape[0])])[~idx_sc]
    pc_i_ravel=array([pc_i for _ in range(tr_i.shape[0])])[~idx_sc]
    af_omega_i_ravel=array([af_omega_i for _ in range(tr_i.shape[0])])[~idx_sc]

    # find local min and max (extrema)
    p0=a*b**3*epsilon+a*b**3*sigma-b**4*epsilon**2*r*sigma**2*t
    p1=-2*a*b**2*epsilon-2*a*b**2*sigma+2*a*b**2-\
    2*b**3*epsilon**2*r*sigma*t-2*b**3*epsilon*r*sigma**2*t
    p2=a*b*epsilon+a*b*sigma-4*a*b-b**2*epsilon**2*r*t-\
    4*b**2*epsilon*r*sigma*t-b**2*r*sigma**2*t
    p3=2*a-2*b*epsilon*r*t-2*b*r*sigma*t
    p4=-r*t

    soln = solve_quartic([p4, p3, p2, p1, p0])
    v_roots = soln['roots']
    v_roots = soln['roots']
    v=zeros([v_roots.shape[1],2])
    idx=(v_roots.imag==0).T & array([v_roots[j]>b for j in range(v_roots.shape[0])]).T
    v[:,0]=(v_roots.T*idx).real.max(axis=1)
    v[:,1]=(v_roots.T*idx+(~idx)*inf).real.min(axis=1) # mask 0 to get positive min # FIXME: case v_roots.T*idx==0 yields min=0, not positive minimum
    rho_v_eos,rho_l_eos=1/v[:,0],1/v[:,1]
    rho_v_omega,rho_l_omega=rho_v_eos,rho_l_eos
    rho_l_bound=mid(rho_mc,rho_l_omega,rho_hi)
    rho_v_bound=mid(rho_lo,rho_v_omega,k*rho_mc)
    
    # extrapolating functions - l
    p_bound_l=r*t/(1/rho_l_bound-b)-a/((1/rho_l_bound+epsilon*b)*(1/rho_l_bound+sigma*b)) # p_bound directly with EOS at rho_bound_l 
    dp_drho_t_bound=dp_drho_t(rho_l_bound,t)
    B_L=dp_drho_t_bound*(rho_l_bound-0.7*rho_mc)
    A_L=p_bound_l-B_L*log(rho_l_bound-0.7*rho_mc)
    rho_l_extrap=minimum(exp((p-A_L)/B_L)+0.7*rho_mc,rho_hi)
    rho_l=mid(rho_l_eos,rho_l_bound,rho_l_extrap)
    p_calc=r*t/(1/rho_l-b)-a/((1/rho_l+epsilon*b)*(1/rho_l+sigma*b)) # eos evaluated at rho_l

    # extrapolating functions - v
    dp_drho_t_omega=0.1*r*t # used for v-extrapolation func only
    p_bound_v=r*t/(1/rho_v_bound-b)-a/((1/rho_v_bound+epsilon*b)*(1/rho_v_bound+sigma*b)) # p_bound directly with EOS at rho_bound_l 
    dp_drho_t_bound=dp_drho_t(rho_v_bound,t)
    A_V=1/p_bound_v
    B_V=-dp_drho_t_bound/p_bound_v**2
    C_V=-abs(A_V+1/2*B_V*(rho_mc-rho_v_bound))/(1/2*(rho_mc-rho_v_bound))**2
    rho_v_extrap=mid(0,rho_hi,rho_v_bound+minimum(0,p-p_bound_v)/dp_drho_t_omega+(-B_V-sqrt(B_V**2-4*C_V*maximum(0,A_V-1/(p))))/(2*C_V)+maximum(0,t-tc_i_ravel)*maximum(0,dp_drho_t_bound-dp_drho_t_omega)*(rho_v_eos-rho_v_bound)) # eq. (29) of paper
    rho_v=mid(rho_v_eos,rho_v_bound,rho_v_extrap)

    pr_i_lo_bis=eps*beta*r*t*rho_v_bound/pc_i_ravel # some very low reduced pressure to fulfill sign>0 for bisection
    pr_i_hi_bis=p_bound_v*(1-eps*10)/pc_i_ravel # k=0.95 is too small to make all zero_fun <0, but k=1 causes some nans. Ok between (1-1e-10) and (1-eps)

    #n=100;from matplotlib import pyplot as plt;from numpy import linspace;fig,ax=plt.subplots();x=linspace(pr_i_lo_bis,1e-14,n);y=array([phi_sat_ceos(tr_i,v1,tc_i,pc_i,af_omega_i,alpha_tr,epsilon,sigma,psi,omega,zc,rho_lims)['zero_fun'] for v1 in x]).squeeze();print(x[[0,1,-2,-1]],y[[0,1,-2-1]]);ax.plot(x.squeeze(),y,'o',linestyle='none');x=linspace(1e-14,pr_i_hi_bis,n);y=array([phi_sat_ceos(tr_i,v1,tc_i,pc_i,af_omega_i,alpha_tr,epsilon,sigma,psi,omega,zc,rho_lims)['zero_fun'] for v1 in x]).squeeze();ax.plot(x.squeeze(),y,'.',linestyle='none');print(x[[0,1,-2,-1]],y[[0,1,-2-1]]);ax.axhline(0,color='black');plt.show()
    soln=bisection(lambda x:phi_sat_ceos(tr_i[~idx_sc],x,tc_i_ravel,pc_i_ravel,af_omega_i_ravel,alpha_tr,epsilon,sigma,psi,omega,zc,rho_lims)['zero_fun'],pr_i_hi_bis,pr_i_lo_bis,full_output=True,tol=1e-7)
    
    pr_i[~idx_sc]=soln['c'] # unravel (return to expected format)
    pr_i[idx_sc]=nan
    n_fev=soln['n_fev']
    # get final values
    soln_out=phi_sat_ceos(tr_i,pr_i,tc_i,pc_i,af_omega_i,alpha_tr,epsilon,sigma,psi,omega,zc,rho_lims)
    success=soln_out['success']
    z_l,z_v=soln_out['z_l'],soln_out['z_v']
    v_l,v_v=z_l*r*tr_i*tc_i/(pr_i*pc_i),z_v*r*tr_i*tc_i/(pr_i*pc_i)
    rho_l,rho_v=1/v_l,1/v_v
    return {item:locals().get(item) for item in ['pr_i', 'tr_i','n_fev', 'rho_l', 'rho_v', 'v_l', 'v_v']}|{
    item:soln_out.get(item)  for item in ['success', 'zero_fun', 'z_l', 'z_v', 'phi_l', 'phi_v']}

def t_i_sat_ceos(t, p, tc_i, pc_i, af_omega_i,
                 alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
                 max_it=100, tol=abs_tol):
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
            t, p, tc, pc, af_omega, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
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
                # initial slope not necessarily convergent, direct it toward
                # descending disc
                second_step_soln = approach_pt_i_sat_ceos(
                    t1_it, p, tc, pc, af_omega, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
                    p_or_t='t', max_it=max_it, tol=tol)
                disc_0 = first_step_soln['disc']
                disc_1 = second_step_soln['disc']
                n_fev[i] += second_step_soln['n_fev']
                inv_slope = -(t1_it - t0_it) / (disc_1 - disc_0)
                t1_it = t0_it * (1 + sign(-inv_slope) * 0.001)
            soln_temp = secant_ls_3p(
                lambda t_var: phi_sat_ceos(
                    t_var, p, tc, pc, af_omega,
                    alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)['zero_fun'],
                t0_it, tol, x_1=t1_it,
                restriction=lambda t_var: t_var > 0 and phi_sat_ceos(
                    t_var, p, tc, pc, af_omega, alpha_tr, epsilon,
                    sigma, psi, omega)['success'],
                print_iterations=False
            )
            n_fev[i] += soln_temp['iterations'] + soln_temp['total_backtracks']
            t_sat_list[i] = soln_temp['x']
            soln_temp = phi_sat_ceos(t_sat_list[i], p, tc, pc, af_omega,
                                     alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)
            success[i] = soln_temp['success']
            zero_fun[i] = soln_temp['zero_fun']
            z_l[i] = soln_temp['z_l']
            z_v[i] = soln_temp['z_v']
            phi_l[i] = soln_temp['phi_l']
            phi_v[i] = soln_temp['phi_v']

    t = t_sat_list
    soln = dict()
    for item in ['p', 't', 'success', 'n_fev',
                 'zero_fun', 'z_l', 'z_v', 'phi_l', 'phi_v']:
        soln[item] = locals().get(item)
    return soln


def approach_pt_i_sat_ceos(tr_i, pr_i, tc_i, pc_i, af_omega_i,
                           alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
                           p_or_t='p', max_it=100, tol=abs_tol, r=r_def):
    """
    Find Tr or Pr such that discriminant becomes negative (two-phase region)
    """
    b_i = omega * r * tc_i / pc_i
    n_fev = 0
    idx_sc=(tc_i<tr_i*tc_i) if p_or_t=='p' else (pc_i<pr_i*pc_i) 
    success=idx_sc.copy() # short circuit if p_sat or t_sat supercritical
    f=zeros(tr_i.shape)
    while n_fev in range(max_it):
        n_fev+=1
        a_i=psi*alpha_tr(tr_i,af_omega_i)*r**2*tc_i**2/pc_i
        beta=b_i*pr_i*pc_i/(r*tr_i*tc_i)
        q=a_i/(b_i*r*tr_i*tc_i)
        a1=beta*(epsilon+sigma)-beta-1
        a2=q*beta+epsilon*sigma*beta**2-beta*(epsilon+sigma)*(1+beta)
        a3=-(epsilon*sigma*beta**2*(1+beta)+q*beta**2)
        # partial derivative of discriminant with respect to p
        da1_dp=1/(pr_i*pc_i)*beta*(epsilon+sigma-1)
        da2_dp=1/(pr_i*pc_i)*(q*beta+2*epsilon*sigma*beta**2-beta*(epsilon+sigma)*(1+2*beta))
        da3_dp=1/(pr_i*pc_i)*(-(epsilon*sigma*beta**2*(2+3*beta)+2*q*beta**2))
        disc=1/27*(-1/3*a1**2+a2)**3+1/4*(2/27*a1**3-1/3*a1*a2+a3)**2
        ddisc_dp=1/9*(-1/3*a1**2+a2)**2*(-2/3*a1*da1_dp+da2_dp)+\
        1/2*(2/27*a1**3-1/3*a1*a2+a3)*(2/9*a1**2*da1_dp-1/3*(a1*da2_dp+a2*da1_dp)+da3_dp)
        # partial derivative of discriminant with respect to t
        da1_dt=-1/(tr_i*tc_i)*beta*(epsilon+sigma-1)
        da2_dt=-2/(tr_i*tc_i)*(q*beta+epsilon*sigma*beta**2-beta*(epsilon+sigma)*(1/2+beta))
        da3_dt=-1/(tr_i*tc_i)*(-(epsilon*sigma*beta**2*(2+3*beta)+3*q*beta**2))
        ddisc_dt=1/9*(-1/3*a1**2+a2)**2*(-2/3*a1*da1_dt+da2_dt)+\
        1/2*(2/27*a1**3-1/3*a1*a2+a3)*(2/9*a1**2*da1_dt-1/3*(a1*da2_dt+a2*da1_dt)+da3_dt)
        f[success] = 0 # only process if not converged and subcritical
        f[~success] = disc[~success] + tol
        if p_or_t == 'p':
            df_dp = ddisc_dp
            inv_slope = 1 / df_dp
            pr_i -= inv_slope * f / pc_i
        elif p_or_t == 't':
            df_dt = ddisc_dt
            inv_slope = 1 / df_dt
            tr_i -= inv_slope * f / tc_i
        success[~idx_sc]=(disc <= -tol/100)[~idx_sc]
        if success[~idx_sc].all():
            break
        #print(n_fev,pc_i*pr_i,tc_i*tr_i,disc,inv_slope)
    p=(~idx_sc)*(pr_i*pc_i)
    t=(~idx_sc)*(tr_i*tc_i)
    success=(~idx_sc) & success
    soln = dict()
    for item in ['p', 't', 'success', 'n_fev',
                 'ddisc_dt', 'ddisc_dp', 'f', 'disc', 'idx_sc']:
        soln[item] = locals().get(item)
    return soln


def phi_sat_ceos(tr_i, pr_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, r=r_def):
    a_i = psi * alpha_tr(tr_i, af_omega_i) * r ** 2 * tc_i ** 2 / pc_i
    b_i = omega * r * tc_i / pc_i
    beta = b_i * pr_i*pc_i / (r * tr_i*tc_i)
    q = a_i / (b_i * r * tr_i*tc_i)
    a0 = ones(beta.shape)
    a1 = beta * (epsilon + sigma) - beta - 1
    a2 = q * beta + epsilon * sigma * beta ** 2 \
        - beta * (epsilon + sigma) * (1 + beta)
    a3 = -(epsilon * sigma * beta ** 2 * (1 + beta) +
           q * beta ** 2)

    soln = solve_cubic(array([a0, a1, a2, a3]))
    disc = soln['disc']
    success = disc <= 0 # only succeeds if two-region (three real roots)
    z_l = soln['roots'][2].real # l: smallest root
    z_v = soln['roots'][0].real # v: largest real root
    z_roots = soln['roots']

    if epsilon != sigma:
        # vdw: epsilon = sigma
        i_int_l=1/(sigma-epsilon)*log((z_l+sigma*beta)/(z_l+epsilon*beta))
        i_int_v=1/(sigma-epsilon)*log((z_v+sigma*beta)/(z_v+epsilon*beta))
    elif epsilon == sigma:
        # only vdw
        i_int_l=beta/(z_l+epsilon*beta)
        i_int_v=beta/(z_v+epsilon*beta)
    ln_phi_l,ln_phi_v=zeros(i_int_l.shape),zeros(i_int_l.shape)
    idx=(z_l-beta>0)
    ln_phi_l[idx]=+z_l[idx]-1-log(z_l[idx]-beta[idx])-q[idx]*i_int_l[idx]
    idx=(z_l-beta<=0)
    ln_phi_l[idx]=+z_l[idx]-1+log((z_l[idx]+epsilon*beta[idx])*(z_l[idx]+sigma*beta[idx])*(1+beta[idx]-z_l[idx])/(q[idx]*beta[idx]))-q[idx]*i_int_l[idx] # symm. log crossing 0 int (1/abs(x))dx
    ln_phi_v = + z_v - 1 - log(z_v - beta) - q * i_int_v
    phi_l = exp(ln_phi_l)
    phi_v = exp(ln_phi_v)
    zero_fun = -ln_phi_v + ln_phi_l 
    #zero_fun = phi_v/phi_l-1 # K_i=1

    soln = dict()
    for item in ['z_l', 'z_v', 'phi_l', 'phi_v', 'success', 'zero_fun', 'disc']:
        soln[item] = locals().get(item)
    return soln


def z_non_sat(t, p, x_i, tc_i, pc_i, af_omega_i,
              alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, r=r_def):
    tr_i = t / tc_i
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
        lambda z_val: -z_val + 1 + beta -
        q * beta * (z_val - beta) / (
            (z_val + epsilon * beta) * (z_val + sigma * beta)),
        1.0).x
    i_int = 1 / (sigma - epsilon) * \
        log((z + sigma * beta) / (z + epsilon * beta))
    ln_phi_i = b_i / b * (z - 1) - log(z - beta) - q_mp_i * i_int
    phi_i = exp(ln_phi_i)

    soln = dict()
    for item in ['a_i', 'b_i', 'b', 'a', 'q',
                 'a_mp_i', 'b_mp_i', 'q_mp_i',
                 'beta', 'z', 'i_int', 'ln_phi_i', 'phi_i']:
        soln[item] = locals().get(item)
    return soln


def phi(t, p, z_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, r=r_def):
    # ensure shape is possible
    t,p,tc_i,pc_i,af_omega_i=[array(x,ndmin=1) for x in [t,p,tc_i,pc_i,af_omega_i]] 
    z_i=array(z_i,ndmin=2)

    tr_i=outer(t,1/tc_i)
    pr_i=outer(p,1/pc_i)
    a_i=psi*alpha_tr(tr_i,af_omega_i)*r**2*tc_i**2/pc_i
    b_i=omega*r*tc_i/pc_i
    beta_i=b_i*pr_i*pc_i/(r*tr_i*tc_i)
    q_i=a_i/(b_i*r*tr_i*tc_i)
    a_ij=array([[sqrt(a_i[:,i]*a_i[:,j]) for i in range(tc_i.shape[0])] for j in range(tc_i.shape[0])]).T
    tc_ij=sqrt(outer(tc_i,tc_i))

    # composition-dependent variables
    b=z_i.dot(b_i)
    a=diagonal(diagonal(z_i.dot(a_ij).dot(z_i.T)))
    beta=b*p/(r*t)
    q=a/(b*r*t)
    a_mp_i=(-a+2*diagonal(z_i.dot(a_ij))).T
    b_mp_i=b_i  # partielles molares b_i
    q_mp_i=array([q*(1+a_mp_i[:,j]/a-b_i[j]/b) for j in range(tc_i.shape[0])]).T # partielles molares q_i

    phase_soln = z_phase(t, p, z_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, abs_tol)
    z_l,z_v=phase_soln['z_l'],phase_soln['z_v']
    v_l,v_v=phase_soln['v_l'],phase_soln['v_v']
    rho_l,rho_v=phase_soln['rho_l'],phase_soln['rho_v']
    i_int = 0  # init
    if epsilon != sigma:
        # vdw: epsilon = sigma
        i_int_l=1/(sigma-epsilon)*log((z_l+sigma*beta)/(z_l+epsilon*beta))
        i_int_v=1/(sigma-epsilon)*log((z_v+sigma*beta)/(z_v+epsilon*beta))
    elif epsilon == sigma:
        # only vdw
        i_int_l=beta/(z_l+epsilon*beta)
        i_int_v=beta/(z_v+epsilon*beta)
    # $G^R/(RT) = Z - 1 - ln(1-\rho b) - ln(Z) - q I$
    # and $\beta = \rho b Z$
    ln_phi_i_l,ln_phi_i_v=zeros(q_mp_i.shape),zeros(q_mp_i.shape)
    idx=(z_l-beta>0)
    ln_phi_i_l[idx,:]=array([
        b_i[j]/b[idx]*(z_l[idx]-1)-log(z_l[idx]-beta[idx])-q_mp_i[idx,j]*i_int_l[idx] 
        for j in range(z_i.shape[1])]).T
    ln_phi_i_v[idx,:]=array([
        b_i[j]/b[idx]*(z_v[idx]-1)-log(z_v[idx]-beta[idx])-q_mp_i[idx,j]*i_int_v[idx] 
        for j in range(z_i.shape[1])]).T
    idx=(z_l-beta<=0)
    ln_phi_i_l[idx,:]=array([
        b_i[j]/b[idx]*(z_l[idx]-1)-log(beta[idx]/z_l[idx]**2-1/z_l[idx])-q_mp_i[idx,j]*i_int_l[idx]
        for j in range(z_i.shape[1])]).T
    ln_phi_i_v[idx,:]=array([
        b_i[j]/b[idx]*(z_v[idx]-1)-log(beta[idx]/z_v[idx]**2-1/z_v[idx])-q_mp_i[idx,j]*i_int_v[idx]
        for j in range(z_i.shape[1])]).T

    phi_i_calc_l=exp(ln_phi_i_l)
    p_calc_l=r*t/(v_l-b)-a/((v_l+epsilon*b)*(v_l+sigma*b))
    # correction for pseudoproperties phi (Matthias et al. 1984)
    phi_i_l=array([phi_i_calc_l[:,j]*p_calc_l/p for j in range(z_i.shape[1])]).T
    ln_phi_i_l=log(phi_i_l)
    phi_i_v=exp(ln_phi_i_v)

    return {item:locals().get(item) for item in ['a_i','b_i','b','a','q',
    'a_mp_i','b_mp_i','q_mp_i','beta','z_l','z_v','v_l','v_v','rho_l','rho_v','i_int_l','i_int_v','ln_phi_i_l','ln_phi_i_v','phi_i_l','phi_i_v']}


def bubl_p(t, p, x_i, tc_i, pc_i, af_omega_i,
           alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
           sec_j=None, nu_ij=None, unifac_data_dict=None,
           max_it=100, tol=abs_tol, y_i_est=None, print_iterations=False):
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
        :param sec_j: unifac secondary group / subgroup (optional)
        :param nu_ij: coefficients in unifac subgroups per component (optional)
        :param unifac_data_dict: dict with unifac data (optional)
        :param max_it: maximum iterations
        :param tol: tolerance of method
        :param y_i_est: optional vapor composition (estimate)
        :param print_iterations: optionally print Lee-Kessler iterations
        :return: dictionary with 'x_i', 'y_i', 'phi_l', 'phi_v', 'k_i', 'sum_ki_xi', 'p', 'success'
        """
    if sec_j is None:
        def obj_fun(p_var): return bubl_point_step_l_k(
            t, p_var, x_i, tc_i, pc_i, af_omega_i,
            alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
            max_it, tol=tol, y_i_est=y_i_est)
        soln = secant_ls_3p(obj_fun, p, tol=tol, x_1=1.01 * p,
                            restriction=lambda p_val: p_val > 0,
                            print_iterations=print_iterations)
        p = soln['x']
        n_fev = soln['iterations'] + soln['total_backtracks']
        soln = bubl_point_step_l_k(t, p, x_i, tc_i, pc_i, af_omega_i,
                                   alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, max_it,
                                   full_output=True, y_i_est=y_i_est)
        n_fev += soln['n_fev']
        soln['n_fev'] = n_fev
    else:
        soln = bubl_p_gamma_phi(
            t, p, x_i, tc_i, pc_i, af_omega_i,
            alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
            sec_j, nu_ij, unifac_data_dict,
            max_it, tol=tol, y_i_est=y_i_est)
        n_fev = soln['n_it']
        soln['n_fev'] = n_fev
    return soln


def bubl_t(t, p, x_i, tc_i, pc_i, af_omega_i,
           alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
           sec_j=None, nu_ij=None, unifac_data_dict=None,
           max_it=100, tol=abs_tol, y_i_est=None, print_iterations=False):
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
        :param sec_j: unifac secondary group / subgroup (optional)
        :param nu_ij: coefficients in unifac subgroups per component (optional)
        :param unifac_data_dict: dict with unifac data (optional)
        :param max_it: maximum iterations
        :param tol: tolerance of method
        :param y_i_est: optional vapor composition (estimate)
        :param print_iterations: optionally print Lee-Kessler iterations
        :return: dictionary with 'x_i', 'y_i', 'phi_l', 'phi_v', 'k_i', 'sum_ki_xi', 'p', 'success'
        """
    if sec_j is None:
        def obj_fun(t_var): return bubl_point_step_l_k(
            t_var, p, x_i, tc_i, pc_i, af_omega_i,
            alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
            max_it, tol=tol, y_i_est=y_i_est)
        soln = secant_ls_3p(obj_fun, t, tol=tol, x_1=1.001 * t,
                            restriction=lambda t_val: t_val > 0,
                            print_iterations=print_iterations)
        t = soln['x']
        n_fev = soln['iterations'] + soln['total_backtracks']
        soln = bubl_point_step_l_k(t, p, x_i, tc_i, pc_i, af_omega_i,
                                   alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, max_it,
                                   full_output=True, y_i_est=y_i_est)
        n_fev += soln['n_fev']
        soln['n_fev'] = n_fev
    else:
        soln = bubl_t_gamma_phi(
            t, p, x_i, tc_i, pc_i, af_omega_i,
            alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
            sec_j, nu_ij, unifac_data_dict,
            max_it, tol=tol, y_i_est=y_i_est)
        n_fev = soln['n_it']
        soln['n_fev'] = n_fev
    return soln


def bubl_point_step_l_k(t, p, x_i, tc_i, pc_i, af_omega_i,
                        alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
                        max_it, tol=abs_tol, full_output=False, y_i_est=None):
    soln_l = phi(t, p, x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)
    phi_l = soln_l['phi_i_l']
    if y_i_est is not None:
        y_i_est = asarray(y_i_est)
        soln_v = phi(t, p, y_i_est, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)
        phi_v = soln_v['phi_i_v']
    else:
        phi_v = soln_l['phi_i_v']
    k_i = phi_l / phi_v
    sum_ki_xi = sum(k_i * x_i)
    y_i = k_i * x_i / sum_ki_xi
    stop = False
    i = 0
    success = True
    while not stop:
        sum_ki_xi_k_minus_1 = sum_ki_xi
        soln_v = phi(t, p, y_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)
        phi_v = soln_v['phi_i_v']
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
        z_l = soln_l['z_l']
        z_v = soln_v['z_v']
        n_fev = 1 + i
        soln = dict()
        for item in ['x_i', 'y_i', 'phi_l', 'phi_v', 'z_l', 'z_v', 'k_i', 'sum_ki_xi',
                     'p', 't', 'success', 'n_fev']:
            soln[item] = locals().get(item)
        return soln
    else:
        return 1 - sum_ki_xi


def bubl_t_gamma_phi(t, p, x_i, tc_i, pc_i, af_omega_i,
                     alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
                     sec_j, nu_ij, unifac_data_dict,
                     max_it, tol=abs_tol, full_output=False, y_i_est=None, r=r_def):
    x_i = asarray(x_i)
    soln = t_i_sat_ceos(t, p, tc_i, pc_i, af_omega_i,
                        alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, max_it=max_it, tol=tol)
    t_i_sat = soln['t']
    t = sum(x_i * t_i_sat)
    gamma_i = gamma_u(t, x_i, sec_j, nu_ij, unifac_data_dict)
    phi_coef_fun_i = ones(x_i.size)
    soln = p_i_sat_ceos(t, p, tc_i, pc_i, af_omega_i,
                        alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, max_it=max_it, tol=tol)
    p_i_sat = soln['p']
    phi_i_v = soln['phi_v']  # init
    for n_it in range(max_it):
        # find closest to mean value, use as key component j for vapor pressure
        # ratios
        j = abs(p_i_sat - p_i_sat.mean()).argmin()
        p_j_sat = p_i_sat[j]
        p_j_sat = p / sum(x_i * gamma_i / phi_coef_fun_i * p_i_sat / p_j_sat)
        # with sat. pressure of key component determine sat. temperature
        t_old = t
        t = t_i_sat_ceos(
            t, p_j_sat, tc_i[j], pc_i[j], af_omega_i[j],
            alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, max_it=max_it, tol=tol
        )['t'].item()

        delta_t = t - t_old
        success = abs(delta_t) <= tol
        if success:
            break

        soln = p_i_sat_ceos(t, p, tc_i, pc_i, af_omega_i,
                            alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, max_it=max_it, tol=tol)
        p_i_sat = soln['p']
        phi_i_sat = soln['phi_v']
        z_l_i = soln['z_l']
        v_l_i = z_l_i * r * t / p
        y_i = x_i * gamma_i * p_i_sat / phi_coef_fun_i / p
        poynting_i = exp(-v_l_i * (p - p_i_sat) / (r * t))
        phi_i_v = phi(t, p, y_i, tc_i, pc_i, af_omega_i,
                      alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)['phi_i_v']

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
                     alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
                     sec_j, nu_ij, unifac_data_dict,
                     max_it, tol=abs_tol, full_output=False, y_i_est=None, r=r_def):
    x_i = asarray(x_i)
    gamma_i = gamma_u(t, x_i, sec_j, nu_ij, unifac_data_dict)
    soln = p_i_sat_ceos(t, p, tc_i, pc_i, af_omega_i,
                        alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, max_it=max_it, tol=tol)
    p_i_sat = soln['p']
    phi_i_sat = soln['phi_v']
    z_l_i = soln['z_l']

    phi_coef_fun_i = ones(x_i.size)
    phi_i_v = phi_i_sat  # init

    for n_it in range(max_it):
        p_old = p
        p = sum(x_i * gamma_i * p_i_sat / phi_coef_fun_i)
        delta_p = p - p_old
        y_i = x_i * gamma_i * p_i_sat / phi_coef_fun_i / p
        y_i = y_i / sum(y_i)
        phi_i_v = phi(t, p, y_i, tc_i, pc_i, af_omega_i,
                      alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)['phi_i_v']
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
          alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
          sec_j=None, nu_ij=None, unifac_data_dict=None,
          max_it=100, tol=abs_tol, x_i_est=None, print_iterations=False):
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
        :param sec_j: unifac secondary group / subgroup (optional)
        :param nu_ij: coefficients in unifac subgroups per component (optional)
        :param unifac_data_dict: dict with unifac data (optional)
        :param max_it: maximum iterations
        :param tol: tolerance of method
        :param x_i_est: optional liquid composition (estimate)
        :param print_iterations: optionally print Lee-Kessler iterations
        :return: dictionary with 'x_i', 'y_i', 'phi_l', 'phi_v', 'k_i', 'sum_ki_xi', 'p', 'success'
        """
    if sec_j is None:
        def obj_fun(p_var): return dew_point_step_l_k(
            t, p_var, y_i, tc_i, pc_i, af_omega_i,
            alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
            max_it, tol=tol, x_i_est=x_i_est)
        soln = secant_ls_3p(obj_fun, p, tol=tol, x_1=0.99 * p,
                            restriction=lambda p_val: p_val > 0,
                            print_iterations=print_iterations)
        p = soln['x']
        n_fev = soln['iterations'] + soln['total_backtracks']
        soln = dew_point_step_l_k(t, p, y_i, tc_i, pc_i, af_omega_i,
                                  alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, max_it,
                                  full_output=True, x_i_est=x_i_est)
        n_fev += soln['n_fev']
        soln['n_fev'] = n_fev
    else:
        soln = dew_p_gamma_phi(
            t, p, y_i, tc_i, pc_i, af_omega_i,
            alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
            sec_j, nu_ij, unifac_data_dict,
            max_it, tol=tol, x_i_est=x_i_est)
        n_fev = soln['n_it']
        soln['n_fev'] = n_fev
    return soln


def dew_t(t, p, y_i, tc_i, pc_i, af_omega_i,
          alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
          max_it, tol=abs_tol, x_i_est=None, print_iterations=False):
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
        alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
        max_it, tol=tol, x_i_est=x_i_est)
    soln = secant_ls_3p(obj_fun, p, tol=tol, x_1=1.001 * p,
                        restriction=lambda p_val: p_val > 0,
                        print_iterations=print_iterations)
    p = soln['x']
    n_fev = soln['iterations'] + soln['total_backtracks']
    soln = dew_point_step_l_k(t, p, y_i, tc_i, pc_i, af_omega_i,
                              alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, max_it,
                              full_output=True, x_i_est=x_i_est)
    n_fev += soln['n_fev']
    soln['n_fev'] = n_fev
    return soln


def dew_point_step_l_k(t, p, y_i, tc_i, pc_i, af_omega_i,
                       alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
                       max_it, tol=abs_tol, full_output=False, x_i_est=None):
    soln_v = phi(t, p, y_i, tc_i, pc_i, af_omega_i,
                 alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)
    phi_v = soln_v['phi_i_v']
    if x_i_est is not None:
        phi_l = phi(t, p, x_i_est, tc_i, pc_i, af_omega_i,
                    alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)['phi_i_l']
    else:
        phi_l = soln_v['phi_i_v']
    k_i = phi_l / phi_v
    sum_yi_over_ki = sum(y_i / k_i)
    x_i = y_i / k_i / sum_yi_over_ki
    stop = False
    i = 0
    success = True
    soln_l = dict()  # init
    while not stop:
        sum_yi_over_ki_k_minus_1 = sum_yi_over_ki
        soln_l = phi(t, p, x_i, tc_i, pc_i, af_omega_i,
                     alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)
        phi_l = soln_l['phi_i_l']
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
        z_l = soln_l['z_l']
        z_v = soln_v['z_v']
        n_fev = 1 + i
        soln = dict()
        for item in ['x_i', 'y_i', 'phi_l', 'phi_v', 'z_l', 'z_v', 'k_i',
                     'sum_yi_over_ki', 'p', 't',
                     'n_fev', 'success']:
            soln[item] = locals().get(item)
        return soln
    else:
        return 1 - sum_yi_over_ki


def dew_p_gamma_phi(t, p, y_i, tc_i, pc_i, af_omega_i,
                    alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
                    sec_j, nu_ij, unifac_data_dict,
                    max_it, tol=abs_tol, full_output=False, x_i_est=None, r=r_def):
    y_i = asarray(y_i)
    phi_coef_fun_i = ones(y_i.size)
    gamma_i = ones(y_i.size)
    soln = p_i_sat_ceos(
        t, p, tc_i, pc_i, af_omega_i,
        alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, max_it=max_it, tol=tol)
    n_fev = soln['n_fev']
    p_i_sat = soln['p']
    phi_i_sat = soln['phi_v']
    z_l_i = soln['z_l']
    phi_i_v = phi_i_sat  # init

    p = 1 / sum(y_i * phi_coef_fun_i / gamma_i / p_i_sat)
    x_i = y_i * phi_coef_fun_i / gamma_i / p_i_sat * p
    x_i = x_i / sum(x_i)

    gamma_i = gamma_u(t, x_i, sec_j, nu_ij, unifac_data_dict)

    p = 1 / sum(y_i * phi_coef_fun_i / gamma_i / p_i_sat)
    n_it = 0  # init
    for n_it in range(max_it):
        p_old = p
        phi_i_v = phi(t, p, y_i, tc_i, pc_i, af_omega_i,
                      alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)['phi_i_v']
        v_l_i = z_l_i * r * t / p
        poynting_i = exp(-v_l_i * (p - p_i_sat) / (r * t))
        phi_coef_fun_i = phi_i_v / phi_i_sat * poynting_i

        for n_it_inner in range(max_it):
            gamma_i_old = gamma_i
            x_i = y_i * phi_coef_fun_i / gamma_i / p_i_sat * p
            x_i = x_i / sum(x_i)
            gamma_i = gamma_u(t, x_i, sec_j, nu_ij, unifac_data_dict)
            delta_gamma = gamma_i - gamma_i_old
            mag_delta_gamma = sqrt(delta_gamma.dot(delta_gamma.T))
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


def p_est(t, p, x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, max_it, tol=abs_tol, r=r_def):
    # ensure shape is possible
    t,p,tc_i,pc_i,af_omega_i=[array(x,ndmin=1) for x in [t,p,tc_i,pc_i,af_omega_i]] 
    x_i=array(x_i,ndmin=2)

    x_i = asarray(x_i)
    tr_i=outer(t,1/tc_i)
    pr_i=outer(p,1/pc_i)
    a_i=psi*alpha_tr(tr_i,af_omega_i)*r**2*tc_i**2/pc_i
    b_i=omega*r*tc_i/pc_i
    q_i=a_i/(b_i*r*tr_i*tc_i)
    a_ij=array([[sqrt(a_i[:,i]*a_i[:,j]) for i in range(tc_i.shape[0])] for j in range(tc_i.shape[0])]).T
    tc_ij=sqrt(outer(tc_i,tc_i))

    # mechanical critical point
    z_mc = -1 / 3 * ((epsilon + sigma) * omega - omega - 1)
    p_mc = x_i.dot(pc_i)
    t_mc = diagonal(x_i.dot(tc_ij).dot(x_i.T))
    rho_mc = p_mc / (r * t_mc * z_mc)
    v_mc = 1 / rho_mc

    # composition-dependent variables
    b = x_i.dot(b_i)
    a=diagonal(diagonal(x_i.dot(a_ij).dot(x_i.T)))
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
        1 + (v + epsilon * b) / (v + sigma * b) +
        1 + (v + sigma * b) / (v + epsilon * b)
    )

    dp_drho_at_rho_lb = - 1 / rho_lb**2 * dp_dv_at_rho_lb
    d2p_drho_at_rho_lb = 1 / rho_lb**4 * \
        d2p_dv2_at_rho_lb + dp_dv_at_rho_lb * 2 / rho_lb**3

    # no inflection point, curve is monotonic
    idx=(d2p_drho_at_rho_lb>0)
    # find inflection point
    idx=(d2p_drho_at_rho_lb<=0) 

    # find inflection point
    p0=2*a*b**5*epsilon**2+2*a*b**5*epsilon*sigma+\
    2*a*b**5*sigma**2+2*b**6*epsilon**3*r*t*sigma**3
    p1=-3*a*b**4*epsilon**2-3*a*b**4*epsilon*sigma+\
    3*a*b**4*epsilon-3*a*b**4*sigma**2+\
    3*a*b**4*sigma+3*b**5*epsilon**3*r*t*sigma**2+\
    3*b**5*epsilon**3*r*t*sigma**2+\
    3*b**5*epsilon**2*r*t*sigma**3
    p2=6*a*b**3*epsilon**2+6*a*b**3*epsilon*sigma-\
    18*a*b**3*epsilon+6*a*b**3*sigma**2-\
    18*a*b**3*sigma+6*a*b**3+\
    6*b**4*epsilon**3*r*t*sigma+\
    18*b**4*epsilon**2*r*t*sigma**2+\
    6*b**4*epsilon*r*t*sigma**3
    p3=-2*a*b**2*epsilon**2-2*a*b**2*epsilon*sigma+\
    18*a*b**2*epsilon-2*a*b**2*sigma**2+\
    18*a*b**2*sigma-18*a*b**2+\
    2*b**3*epsilon**3*r*t+\
    18*b**3*epsilon**2*r*t*sigma+\
    18*b**3*epsilon*r*t*sigma**2+2*b**3*r*t*sigma**3
    p4=-6*a*b*epsilon-6*a*b*sigma+18*a*b+\
    6*b**2*epsilon**2*r*t+\
    18*b**2*epsilon*r*t*sigma+6*b**2*r*t*sigma**2
    p5=-6*a+6*b*epsilon*r*t+6*b*r*t*sigma
    p6=2*r*t

    v_inf_roots = zroots(array([p0, p1, p2, p3, p4, p5, p6]).T)

    v = v_inf_roots
    dp_dv=array([-r*t/(v[:,j]-b)**2+a/((v[:,j]+epsilon*b)*(v[:,j]+sigma*b))*(
        1/(v[:,j]+epsilon*b)+1/(v[:,j]+sigma*b)
    ) for j in range(v.shape[1])]).T
    d2p_dv2=array([2*r*t/(v[:,j]-b)**3-a/((v[:,j]+epsilon*b)*(v[:,j]+sigma*b))*(
        1/(v[:,j]+epsilon*b)**2+1/(v[:,j]+sigma*b)**2
        )-a/((v[:,j]+epsilon*b)*(v[:,j]+sigma*b))**2*(
        1+(v[:,j]+epsilon*b)/(v[:,j]+sigma*b)+
        1+(v[:,j]+sigma*b)/(v[:,j]+epsilon*b)
    ) for j in range(v.shape[1])]).T
    v_inf=zeros(v.shape[0])
    dp_dv_real=zeros(v.shape[0])
    d2p_dv2_real=zeros(v.shape[0])
    # single real root with dp_dv<0
    idx=((v.imag==0) & (dp_dv<0)).sum(axis=1)==1
    v_inf[idx] = (v[idx]*((v.imag==0) & (dp_dv<0))[idx]).sum(axis=1).real
    d2p_dv2_real[idx] = (d2p_dv2[idx]*((v.imag==0) & (dp_dv<0))[idx]).sum(axis=1).real
    dp_dv_real[idx] = (dp_dv[idx]*((v.imag==0) & (dp_dv<0))[idx]).sum(axis=1).real
    # above pseudocritical temperature, no extrema
    idx=((v.imag==0) & (dp_dv<0)).sum(axis=1)==2
    v_inf[idx] = (v[idx]*(v.imag==0)[idx]).sum(axis=1).real
    d2p_dv2_real[idx] = (d2p_dv2[idx]*(v.imag==0)[idx]).sum(axis=1).real
    dp_dv_real[idx] = (dp_dv[idx]*(v.imag==0)[idx]).sum(axis=1).real
    # above Boyle-temperature
    idx=((v.imag==0) & (dp_dv<0)).sum(axis=1)<1
    v_inf[idx]=0
    p_rho_inf = r * t / (v_inf - b) - a / \
        ((v_inf + epsilon * b) * (v_inf + sigma * b))
    z_inf = p * v_inf / (r * t)

    beta = b * p_rho_inf / (r * t)

    a0 = ones(q.shape)
    a1 = beta * (epsilon + sigma) - beta - 1
    a2 = q * beta + epsilon * sigma * beta ** 2 \
        - beta * (epsilon + sigma) * (1 + beta)
    a3 = -(epsilon * sigma * beta ** 2 * (1 + beta) +
           q * beta ** 2)

    soln = solve_cubic([a0, a1, a2, a3])
    z_p_inf = array(soln['roots'])
    z_l_p_inf = z_p_inf[2].real # l: smaller real root
    z_v_p_inf = z_p_inf[0].real # v: larger real root
    v_l_p_inf = z_l_p_inf * r * t / p_rho_inf
    v_v_p_inf = z_v_p_inf * r * t / p_rho_inf

    # find local min and max (extrema)
    p0=a*b**3*epsilon+a*b**3*sigma-b**4*epsilon**2*r*sigma**2*t
    p1=-2*a*b**2*epsilon-2*a*b**2*sigma+2*a*b**2-\
    2*b**3*epsilon**2*r*sigma*t-2*b**3*epsilon*r*sigma**2*t
    p2=a*b*epsilon+a*b*sigma-4*a*b-b**2*epsilon**2*r*t-\
    4*b**2*epsilon*r*sigma*t-b**2*r*sigma**2*t
    p3=2*a-2*b*epsilon*r*t-2*b*r*sigma*t
    p4=-r*t

    # v_roots = zroots([p0, p1, p2, p3, p4])
    soln = solve_quartic([p4, p3, p2, p1, p0])
    v_roots = soln['roots']
    v=zeros([v_roots.shape[1],2])
    idx=(v_roots.imag==0).T & array([v_roots[j]>b for j in range(v_roots.shape[0])]).T
    v[:,0]=(v_roots.T*idx).real.max(axis=1)
    v[:,1]=(v_roots.T*idx).real.min(axis=1) # FIXME: case v_roots.T*idx==0 yields min=0, not positive minimum: see p_sat_ceos, add ~idx*1/eps?

    v_l = v.min(axis=1)
    v_v = v.max(axis=1)
    z_l = p * v_l / (r * t)
    z_v = p * v_v / (r * t)
    rho_l = 1 / v_l
    rho_v = 1 / v_v

    p_min_l = r * t / (v_l - b) - a / \
        ((v_l + epsilon * b) * (v_l + sigma * b))
    p_max_v = r * t / (v_v - b) - a / \
        ((v_v + epsilon * b) * (v_v + sigma * b))

    idx=t>t_mc
    if idx.any():
        # single real root, vapor-like
        beta[idx] = b[idx] * p[idx] / (r * t[idx])
        a0 = ones(t[idx].shape)
        a1 = beta[idx] * (epsilon + sigma) - beta[idx] - 1
        a2 = q[idx] * beta[idx] + epsilon * sigma * beta[idx] ** 2 \
            - beta[idx] * (epsilon + sigma) * (1 + beta[idx])
        a3 = -(epsilon * sigma * beta[idx] ** 2 * (1 + beta[idx]) +
               q[idx] * beta[idx] ** 2)
        soln = solve_cubic([a0, a1, a2, a3])
        z_v[idx] = soln['roots'][0].real # largest real
        v_v[idx] = z_v[idx] * r * t[idx] / p[idx]
        rho_v[idx] = 1 / v_v[idx]

        # pseudo liquid density
        dp_dv_at_rho_mc = -r * t[idx] / (v_mc[idx] - b[idx]) ** 2 + a[idx] / (
            (v_mc[idx] + epsilon * b[idx]) * (v_mc[idx] + sigma * b[idx])) * (
            1 / (v_mc[idx] + epsilon * b[idx]) + 1 / (v_mc[idx] + sigma * b[idx])
        )
        dp_drho_at_rho_mc = -v_mc[idx]**2 * dp_dv_at_rho_mc
        c1 = dp_drho_at_rho_mc * (rho_mc[idx] - 0.7 * rho_mc[idx])
        c0 = p_mc[idx] - c1 * log(rho_mc[idx] - 0.7 * rho_mc[idx])
        rho_l[idx] = exp((p[idx] - c0) / c1) + 0.7 * rho_mc[idx]
        v_l[idx] = 1 / rho_l[idx]
        z_l[idx] = p[idx] * v_l[idx] / (r * t[idx])
        rho_l[idx] = 1 / v_l[idx]

    for item in ['p', 'p_rho_inf', 'z_rho_inf', 'v_rho_inf', 'p_min_l', 'p_max_v',
                 'z_l', 'z_v', 'v_l', 'v_v', 'rho_l', 'rho_v']:
        soln[item] = locals().get(item)
    return soln


def z_phase(t, p, z_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, tol=abs_tol, r=r_def):
    # ensure shape is possible
    t,p,tc_i,pc_i,af_omega_i=[array(x,ndmin=1) for x in [t,p,tc_i,pc_i,af_omega_i]] 
    z_i=array(z_i,ndmin=2)

    tr_i=outer(t,1/tc_i)
    pr_i=outer(p,1/pc_i)
    a_i=psi*alpha_tr(tr_i,af_omega_i)*r**2*tc_i**2/pc_i
    b_i=omega*r*tc_i/pc_i
    q_i=a_i/(b_i*r*tr_i*tc_i)
    a_ij=array([[sqrt(a_i[:,i]*a_i[:,j]) for i in range(tc_i.shape[0])] for j in range(tc_i.shape[0])]).T
    tc_ij=sqrt(outer(tc_i,tc_i))

    # composition-dependent variables
    b = z_i.dot(b_i)
    a=diagonal(diagonal(z_i.dot(a_ij).dot(z_i.T)))
    q = a / (b * r * t)

    # mechanical critical point
    z_mc = -1 / 3 * ((epsilon + sigma) * omega - omega - 1)
    p_mc = z_i.dot(pc_i)
    t_mc = diagonal(z_i.dot(tc_ij).dot(z_i.T))
    rho_mc = p_mc / (r * t_mc * z_mc)
    v_mc = 1 / rho_mc
    p_mc_bound=r*t/(v_mc-b)-a/((v_mc+epsilon*b)*(v_mc+sigma*b))
    dp_drho_at_rho_mc=(-r*t/(v_mc-b)**2+a/((v_mc+epsilon*b)*(v_mc+sigma*b))*(
        1/(v_mc+epsilon*b)+1/(v_mc+sigma*b)))*(-v_mc**2)

    # solve f(z)=0
    beta = b * p / (r * t)
    a0=ones(t.shape)
    a1=beta*(epsilon+sigma)-beta-1
    a2=q*beta+epsilon*sigma*beta**2-beta*(epsilon+sigma)*(1+beta)
    a3=-(epsilon*sigma*beta**2*(1+beta)+q*beta**2)
    soln = solve_cubic([a0, a1, a2, a3])
    roots_z, disc_z = soln['roots'], soln['disc']

    # solve S+U=0 (real root = real part of complex root)
    # locus matches p-rho inflection point at low density
    q0=(9*(epsilon+sigma)*epsilon*sigma+18*epsilon*sigma+-2*(epsilon+sigma)**3-3*(epsilon+sigma)**2+3*(epsilon+sigma)+2)*(b/(r*t))**3
    q1=(18*epsilon*sigma-2*(epsilon+sigma)**2+6*(epsilon+sigma)+6)*(b/(r*t))**2+(9*(epsilon+sigma)+18)*a*b/(r*t)**3
    q2=(3*(epsilon+sigma)+6)*b/(r*t)-9*a/(r*t)**2
    q3=2*ones(p.shape)

    roots_p=zeros([p.shape[0],3])*1j  # init
    disc_p=q2**2-4*q1*q3
    idx=((q0==0) & (disc_p>=0)) # reduces to quadratic (SRK), 2 X real
    roots_p[idx,:2]=array([-(q2/q1)[idx]/2+sqrt(disc_p[idx])/(2*q1[idx]),-(q2/q1)[idx]/2-sqrt(disc_p[idx])/(2*q1[idx])]).T
    idx=((q0==0) & (disc_p<0)) # reduces to quadratic (SRK), 2 X complex
    roots_p[idx,:2]=array([-(q2/q1)[idx]/2+sqrt(-disc_p[idx])/(2*q1[idx])*1j,-(q2/q1)[idx]/2-sqrt(-disc_p[idx])/(2*q1[idx])*1j]).T
    #p_low = roots_p.real.min(axis=1)
    #p_cross = roots_p.real.max(axis=1)

    idx=(q0!=0) # full cubic
    roots_p[idx,:]=solve_cubic([x[idx] for x in [q0,q1,q2,q3]])['roots'].reshape([sum(idx),3])

    n_positive_roots_p = ((roots_p.imag==0) & (roots_p>0)).sum(axis=1)

    z_l,z_l_low,z_v,z_v_low = zeros(p.shape), zeros(p.shape), zeros(p.shape), zeros(p.shape)
    rho_l_low,rho_v_low = nan*ones(p.shape), nan*ones(p.shape)
    p_high,p_cross,p_low=zeros(p.shape),zeros(p.shape),ones(p.shape)*finfo(float).eps
    # roots are sorted
    p_high[n_positive_roots_p<=1]=roots_p[n_positive_roots_p<=1,0].real
    p_high[n_positive_roots_p==3]=roots_p[n_positive_roots_p==3,0].real
    p_cross[n_positive_roots_p==3]=roots_p[n_positive_roots_p==3,1].real
    p_low[n_positive_roots_p==3]=roots_p[n_positive_roots_p==3,2].real
    p_cross[n_positive_roots_p==2]=roots_p[n_positive_roots_p==2,0].real
    p_low[n_positive_roots_p==2]=roots_p[n_positive_roots_p==2,2].real
    # solve f(z_low)=0
    beta_low = b * p_low / (r * t)
    a0_low=ones(beta_low.shape)
    a1_low=beta_low*(epsilon+sigma)-beta_low-1
    a2_low=q*beta_low+epsilon*sigma*beta_low**2-beta_low*(epsilon+sigma)*(1+beta_low)
    a3_low=-(epsilon*sigma*beta_low**2*(1+beta_low)+q*beta_low**2)
    soln_z_low = solve_cubic([a0_low, a1_low, a2_low, a3_low])
    roots_z_low = soln_z_low['roots']
    disc_z_low = soln_z_low['disc']
    idx=(disc_z_low<=0) # 3 real roots
    z_l_low[idx]=roots_z_low[2,idx].real # smallest is liq
    idx=(disc_z_low>0) # 1 real, 2 complex
    z_l_low[idx]=roots_z_low[0,idx].real
    z_v_low=roots_z_low[0,:].real # at any rate largest real root

    # phase: L
    v_l_low=z_l_low*r*t/p_low
    rho_l_low=1/v_l_low
    dp_drho_at_rho_l_low=(-r*t/(v_l_low-b)**2+a/((v_l_low+epsilon*b)*(v_l_low+sigma*b))*(
        1/(v_l_low+epsilon*b)+1/(v_l_low+sigma*b)))*(-v_l_low**2)
    # liquid pseudo-density single real fallback
    idx=((t>=t_mc) | (n_positive_roots_p<=1)) & (p>p_mc_bound)
    z_l[idx]=roots_z[0,idx].real
    # liquid pseudo-density - eq. 36, 38, 39 fallback
    idx=((t>=t_mc) | (n_positive_roots_p<=1)) & (p<=p_mc_bound)
    c1=dp_drho_at_rho_mc[idx]*(rho_mc-0.7*rho_mc)[idx]
    c0=p_mc_bound[idx]-c1*log((rho_mc-0.7*rho_mc)[idx])
    rho_l=0.7*rho_mc[idx]+exp((p[idx]-c0)/c1)
    z_l[idx]=p[idx]*1/rho_l/(r*t[idx])
    # liquid density smallest real
    idx=((t<t_mc) | (n_positive_roots_p>1)) & (p>p_low) & (disc_z<=0)
    z_l[idx]=roots_z[2,idx].real
    # liquid density only real
    idx=((t<t_mc) | (n_positive_roots_p>1)) & (p>p_low) & (disc_z>0)
    z_l[idx]=roots_z[0,idx].real
    # liquid pseudo-density complex
    idx=((t<t_mc) | (n_positive_roots_p>1)) & (p<=p_low) & (
        (rho_l_low-0.7*rho_mc<=0) | (dp_drho_at_rho_l_low<0))
    rho_l=rho_l_low[idx]+(rho_l_low-0.7*rho_mc)[idx]/dp_drho_at_rho_l_low[idx]
    rho_l=(p-p_low)[idx]*(rho_l_low-0.7*rho_mc)[idx]+rho_l_low[idx]
    z_l[idx]=p[idx]*1/rho_l/(r*t[idx])
    # liquid pseudo-density real: extrapolating function l - eq. 36, 38, 39
    idx=((t<t_mc) | (n_positive_roots_p>1)) & (p<=p_low) & (
        (rho_l_low-0.7*rho_mc>0) & (dp_drho_at_rho_l_low>=0))
    c1=dp_drho_at_rho_l_low[idx]*(rho_l_low-0.7*rho_mc)[idx]
    c0=p_low[idx]-c1*log((rho_l_low-0.7*rho_mc)[idx])
    rho_l=0.7*rho_mc[idx]+exp((p[idx]-c0)/c1)
    z_l[idx]=p[idx]*1/rho_l/(r*t[idx])
    rho_l=p/(r*t*z_l)
    v_l=1/rho_l

    # phase: V
    v_v_low=z_v_low*r*t/p_low
    rho_v_low=1/v_v_low
    dp_drho_at_rho_v_low=(-r*t/(v_v_low-b)**2+a/((v_v_low+epsilon*b)*(v_v_low+sigma*b))*(
        1/(v_v_low+epsilon*b)+1/(v_v_low+sigma*b)))*(-v_v_low**2)
    dp_drho_at_rho_v_low=maximum(dp_drho_at_rho_v_low,0.1*r*t)
    # vapor density fallback
    idx=((t>=t_mc) | (n_positive_roots_p<=0))
    z_v[idx]=roots_z[0,idx].real
    # vapor density single real
    idx=((t<t_mc) | (n_positive_roots_p>1)) & (p<=p_low)
    z_v[idx]=roots_z[0,idx].real
    # vapor pseudo-density - eq. 24, 28, 29, 30
    idx=((t<t_mc) | (n_positive_roots_p>1)) & (p>p_low)
    rho0=(rho_v_low+1.4*rho_mc)/2
    rho1=p_low*((rho_v_low-1.4*rho_mc)+p_low/dp_drho_at_rho_v_low)
    rho2=-p_low**2*((rho_v_low-1.4*rho_mc)/2+p_low/dp_drho_at_rho_v_low)
    rho_v=rho0+rho1/p+rho2/p**2
    z_v[idx]=p[idx]*1/rho_v[idx]/(r*t[idx])
    rho_v=p/(r*t*z_v)
    v_v=1/rho_v

    return {item:locals().get(item) for item in ['z_l','z_v', 'rho_l','rho_v', 'v_l','v_v', 'p_low', 'p_mc_bound', 'p_high', 'p_cross']}

def isot_flash(t, p, x_i, y_i, z_i, tc_i, pc_i, af_omega_i,
               alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims):
    soln_l = phi(t, p, x_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)
    soln_v = phi(t, p, y_i, tc_i, pc_i, af_omega_i, alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)
    k_i = soln_l['phi_i_l'] / soln_v['phi_i_v']
    soln_v_f = secant_ls_3p(
        lambda v_f_var: sum(z_i * (1 - k_i) / (1 + v_f_var * (k_i - 1))),
        0.5, tol=1e-10,
        f_prime=lambda v_f_var: -sum(z_i * (k_i - 1) **
                                     2 / (1 + v_f_var * (k_i - 1)) ** 2),
        restriction=lambda v_f_val: 0 <= v_f_val <= 1.0,
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
                     alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims, max_it=20,
                     x_i=None, y_i=None):
    if not x_i:
        x_i = ones(len(z_i)) / len(z_i)
    if not y_i:
        y_i = ones(len(z_i)) / len(z_i)
    soln = dict()  # init
    for i in range(max_it):
        soln = isot_flash(t, p, x_i, y_i, z_i, tc_i, pc_i, af_omega_i,
                          alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)
        y_i = soln['y_i']
        x_i = soln['x_i']
    return soln


def pt_flash(t, p, z_i, tc_i, pc_i, af_omega_i,
             alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
             sec_j=None, nu_ij=None, unifac_data_dict=None,
             max_it=100, tol=abs_tol, p_est_0=None):
    if p_est_0 is None:
        # take estimate for 2-phase region
        p0 = p
    else:
        p0 = p_est_0
    # FIXME: bubl_p / dew_p might be off converging toward k_i=1
    bubl_p_soln = bubl_p(t, p0, z_i, tc_i, pc_i, af_omega_i,
                         alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
                         sec_j, nu_ij, unifac_data_dict,
                         max_it=max_it, tol=tol)
    dew_p_soln = dew_p(t, p0, z_i, tc_i, pc_i, af_omega_i,
                       alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims,
                       sec_j, nu_ij, unifac_data_dict,
                       max_it=max_it, tol=tol)
    n_fev = bubl_p_soln['n_fev'] + dew_p_soln['n_fev']
    total_backtracks = 0
    p_bubl, p_dew = bubl_p_soln['p'], dew_p_soln['p']
    phi_bubl, phi_dew = bubl_p_soln['phi_v'], dew_p_soln['phi_v']
    k_i_bubl, k_i_dew = bubl_p_soln['k_i'], dew_p_soln['k_i']
    n_it = 0
    if p_dew < p < p_bubl:
        # 0 < v_f < 1
        if sec_j is not None:
            # gamma-phi-formulation - svn eq. 14.1
            # $y_i \Phi_i P = x_i \gamma_i P_i^{sat}$
            p_i_sat = bubl_p_soln['p_i_sat']
            # equal to bubl_p_soln['phi_i_sat']
            phi_i_sat = dew_p_soln['phi_i_sat']
            gamma_i_bubl, gamma_i_dew = bubl_p_soln['gamma_i'], dew_p_soln['gamma_i']
            poynting_i_bubl, poynting_i_dew = bubl_p_soln['poynting_i'], dew_p_soln['poynting_i']
            gamma_i = gamma_i_dew + \
                (gamma_i_bubl - gamma_i_dew) * (p - p_dew) / (p_bubl - p_dew)
            poynting_i = poynting_i_dew + \
                (poynting_i_bubl - poynting_i_dew) * \
                (p - p_dew) / (p_bubl - p_dew)
        else:
            # phi-phi formulation: k-values directly from eos
            p_i_sat = None  # not needed
            phi_i_sat = dew_p_soln['phi_v']  # not needed
            poynting_i = ones(len(z_i))  # not needed
            gamma_i = ones(len(z_i))  # not needed
        phi_i_v = phi_dew + (phi_bubl - phi_dew) * \
            (p - p_dew) / (p_bubl - p_dew)
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
                soln_phi_l = phi(t, p, x_i, tc_i, pc_i, af_omega_i,
                              alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)
                phi_i_l = soln_phi_l['phi_i_l']
                k_i = phi_i_l / phi_i_v
            else:
                phi_coef_fun_i = phi_i_v / phi_i_sat * poynting_i
                k_i = gamma_i * p_i_sat / (phi_coef_fun_i * p)
            f = sum(z_i * (k_i - 1) / (1 + v_f * (k_i - 1)))
            df_dv_f = -sum(z_i * (k_i - 1) ** 2 / (1 + v_f * (k_i - 1)) ** 2)
            v_f = v_f - 1 / df_dv_f * f
            if not 0 < v_f < 1:
                ls_approach = line_search(
                    lambda v_f_var: sum(
                        z_i * (k_i - 1) / (1 + v_f_var * (k_i - 1))),
                    lambda v_f_var: -sum(z_i * (k_i - 1) ** 2 /
                                         (1 + v_f_var * (k_i - 1)) ** 2),
                    v_f_old, additional_restrictions=lambda v_f_var: 0 < v_f_var < 1)
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
            criterion = sqrt(sum_rr**2 + delta_x_i.dot(delta_x_i.T) +
                             delta_y_i.dot(delta_y_i.T) + delta_v_f**2)
            success = criterion <= tol
            if success:
                break
            if sec_j is None:
                # phi-phi
                pass
            else:
                # gamma-phi
                gamma_i = gamma_u(t, x_i, sec_j, nu_ij, unifac_data_dict)
            soln_phi_v = phi(t, p, y_i, tc_i, pc_i, af_omega_i,
                          alpha_tr, epsilon, sigma, psi, omega, zc, rho_lims)
            phi_i_v = soln_phi_v['phi_i_v']
    elif p_dew >= p:
        v_f = 1.0
    elif p_bubl <= p:
        v_f = 0.0
    iterations = n_it
    soln = dict()
    for item in ['t', 'p', 'k_i', 'v_f', 'z_i', 'x_i', 'y_i', 'gamma_i', 'phi_coef_fun_i',
                 'dew_p_soln', 'bubl_p_soln', 'p_dew', 'p_bubl', 'poynting', 'n_it',
                 'mag_delta_x_i', 'mag_delta_y_i', 'mag_delta_v_f', 'sum_rr', 'criterion',
                 'iterations', 'total_backtracks', 'phi_i_l', 'phi_i_v', 'soln_phi_l', 'soln_phi_v']:
        soln[item] = locals().get(item)
    return soln


def interaction_pairs(elements, i=0, j=0, perms=None):
    # recursively produce pairs of indexes for interaction parameters
    if perms is None:
        perms = []
    if i + 1 >= len(elements) and j + 1 >= len(elements):
        return perms + [[elements[i], elements[j]]]
    elif j + 1 < len(elements):
        return interaction_pairs(
            elements, i, j + 1, perms + [[elements[i], elements[j]]])
    elif i + 1 < len(elements):
        return interaction_pairs(
            elements, i + 1, 0, perms + [[elements[i], elements[j]]])


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
    phi_j = r_j * x_j / r_j.dot(x_j.T)
    theta_j = q_j * x_j / q_j.dot(x_j.T)
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
    # residual activity coefficients in reference solution: molecules of type
    # i only
    x_m = nu_ij / sum_nu_ij
    sum_x_m_q_m = sum(x_m * q_m, 1).reshape(len(nu_ij), 1)
    theta_m = x_m * q_m / sum_x_m_q_m
    theta_m.dot(psi_mn)
    ln_gamma_k_i = q_m * (
        1 - log(theta_m.dot(psi_mn)) - (
            theta_m / theta_m.dot(psi_mn)).dot(psi_mn.T)
    )
    # group residual activity coefficients
    x_k = x_j.dot(nu_ij) / x_j.dot(sum(nu_ij, 1))
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
        [re.match(r'\[([0-9]*)\]', x).groups()[0] for x in
         array(unifac_subgroup_data[:, 2], dtype=str)],
        dtype=int)
    main_group_names_of_k = array(
        [re.match(r'\[[0-9]*\](.*)', x).groups()[0] for x in
         array(unifac_subgroup_data[:, 2], dtype=str)])
    r_k = array(unifac_subgroup_data[:, 3], dtype=float)
    q_k = array(unifac_subgroup_data[:, 4], dtype=float)

    unifac_data_dict = dict()
    for var in ['subgroups_k', 'r_k', 'q_k', 'main_groups_of_k',
                'main_group_names_of_k', 'unifac_interaction_params']:
        unifac_data_dict[var] = locals()[var]
    return unifac_data_dict
