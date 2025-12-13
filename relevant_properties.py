# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from numpy import array,outer,exp,sqrt,pi,concatenate,log,roots,linspace,loadtxt,asarray,cos,ones,abs,emath,sign
from scipy.sparse import coo_array
from scipy.optimize import least_squares
from numpy.linalg import inv # one linear system
from matplotlib import pyplot as plt
from tabulate import tabulate
import ipdb
from numpy import finfo
eps=finfo(float).eps
R=8.31446261815324 # J/mol/K
k_B=1.380649e-23 # J/K
N_A=6.02214076e+23 # 1/mol
R=k_B*N_A


components_list=['H2','CH4','CO2','CO','H2O','N2','O2','Ar','He','C6H6','C6H12','C6H14'] # order of components
sigma_lj=array([2.827,3.758,3.941,3.69,2.641,3.798,3.467,3.542,2.551,5.349,6.182,5.949]) # Lennard Jones sigma, Angström 
epsilon_lj_ov_k=array([59.7,148.6,195.2,91.7,809.1,71.4,106.7,93.3,10.22,412.3,297.1,399.3]) # Lennard Jones epsilon/Boltztmann k, K
M=array([0.00201588,0.01604246,0.0440095,0.0280101,0.01801528,0.0280134,0.031998,0.039948,0.0040026,0.078114,0.084162,0.086178]) # kg/mol
# coefficients of NASA polynomials from Burcat, A., & Ruscic, B. (2001). Third millennium ideal gas and condensed phase thermochemical database for combustion. Technion-Israel Institute of Technology.
a1_a7_low=array([x.split('\t') for x in """2,34433112	0,007980521	-1,95E-05	2,02E-08	-7,38E-12	-917,935173	0,683010238
5,14825732	-0,013700241	4,94E-05	-4,92E-08	1,70E-11	-10245,3222	-4,63322726
2,356813	0,00898413	-7,12E-06	2,46E-09	-1,43E-13	-48371,971	9,9009035
3,5795335	-0,000610354	1,02E-06	9,07E-10	-9,04E-13	-14344,086	3,5084093
4,1986352	-0,002036402	6,52E-06	-5,49E-09	1,77E-12	-30293,726	-0,84900901
3,53100528	-0,000123661	-5,03E-07	2,44E-09	-1,41E-12	-1046,97628	2,96747038
3,78245636	-0,002996734	9,85E-06	-9,68E-09	3,24E-12	-1063,94356	3,65767573
2,5	0	0,00E+00	0,00E+00	0,00E+00	-745,375	4,37967491
2,5	0	0,00E+00	0,00E+00	0,00E+00	-745,375	0,928723974
0,504818632	0,018502064	7,38E-05	-1,18E-07	5,07E-11	8552,47913	21,6412893
4,04357527	-0,006196083	1,77E-04	-2,23E-07	8,64E-11	-16920,3544	8,52527441
9,87121167	-0,00936699	1,70E-04	-2,15E-07	8,45E-11	-23718,5495	-12,4999353
""".replace(',','.').split('\n') if len(x)>0],dtype=float) 
a1_a7_high=array([x.split('\t') for x in """2,93286575	0,000826608	-1,46E-07	1,54E-11	-6,89E-16	-813,065581	-1,02432865
1,911786	0,00960268	-3,38E-06	5,39E-10	-3,19E-14	-10099,2136	8,48241861
4,6365111	0,002741457	-9,96E-07	1,60E-10	-9,16E-15	-49024,904	-1,9348955
3,0484859	0,001351728	-4,86E-07	7,89E-11	-4,70E-15	-14266,117	6,0170977
2,6770389	0,002973182	-7,74E-07	9,44E-11	-4,27E-15	-29885,894	6,88255
2,95257637	0,0013969	-4,93E-07	7,86E-11	-4,61E-15	-923,948688	5,87188762
3,66096065	0,000656366	-1,41E-07	2,06E-11	-1,30E-15	-1215,97718	3,41536279
2,5	0	0,00E+00	0,00E+00	0,00E+00	-745,375	4,37967491
2,5	0	0,00E+00	0,00E+00	0,00E+00	-745,375	0,928723974
11,0809576	0,020717675	-7,52E-06	1,22E-09	-7,36E-14	4306,41035	-40,041331
13,214597	0,035824343	-1,32E-05	2,17E-09	-1,32E-13	-22809,2102	-55,3518322
19,5158086	0,026775394	-7,50E-06	1,20E-09	-7,52E-14	-29436,2466	-77,4895497
""".replace(',','.').split('\n') if len(x)>0],dtype=float)


Cp_R_coefs_200_1000_K=a1_a7_low[:,:4+1] # for Cp function, coefficients from a1 to a5 are applicable
Cp_R_coefs_1000_6000_K=a1_a7_high[:,:4+1] # for Cp function, coefficients from a1 to a5 are applicable
h_ig_coefs_200_1000_K=array([[1/1,1/2,1/3,1/4,1/5,1] for _ in range(a1_a7_low.shape[0])])*a1_a7_low[:,:6]
h_ig_coefs_1000_6000_K=array([[1/1,1/2,1/3,1/4,1/5,1] for _ in range(a1_a7_low.shape[0])])*a1_a7_high[:,:6]
s_ig_coefs_200_1000_K=array([[1,1,1/2,1/3,1/4,0,1] for _ in range(a1_a7_low.shape[0])])*a1_a7_low[:,:7]
s_ig_coefs_1000_6000_K=array([[1,1,1/2,1/3,1/4,0,1] for _ in range(a1_a7_low.shape[0])])*a1_a7_high[:,:7]


def mu(T):
    T_=outer(T,1/epsilon_lj_ov_k)
    MT=outer(T,M*1000)
    omega_mu=1.16145*T_**(-0.14874)+0.52487*exp(-0.77320*T_)+2.16178*exp(-2.43787*T_)
    return 5/16*sqrt(k_B/(pi*1000*N_A))*1/1e-10**2*sqrt(MT)/(sigma_lj**2*omega_mu)  # Pa s


def D_ij(T,p):
    """Returns binary diffusion coefficients for p in Pa, T in K."""
    eps_lj_ov_k_ij=sqrt(outer(epsilon_lj_ov_k,epsilon_lj_ov_k))
    sigma_lj_ij=1/2*(sigma_lj.reshape([len(sigma_lj),1])+sigma_lj.reshape([1,len(sigma_lj)]))
    M_ij=1/(1/2*(1/M.reshape([1,len(M)])+1/M.reshape([len(M),1])))
    Tstar=T/eps_lj_ov_k_ij
    omega_D=1.06036*Tstar**(-0.15610)+0.19300*exp(-0.47635*Tstar)+1.03587*exp(-1.52996*Tstar)+1.76474*exp(-3.89411*Tstar)
    c1=3/(16)*sqrt(4/pi)*(1.3806e-23)**(3/2)*(6.0221e23)**(1/2)*1000**(1/2)*100**2/(1e5*1e-10**2) # 0.0026633 cm^2/s
    return c1*T**(3/2)/((p/1e5)*M_ij**(1/2)*sigma_lj_ij**2*omega_D)*100**(-2)  # m^2/s


def cp_ig(T):
    result=((200<=T)&(T<=1000))*R*Cp_R_coefs_200_1000_K.dot(pow(T,array([[0],[1],[2],[3],[4]],dtype=float)))+\
            ((1000<T)&(T<=6000))*R*Cp_R_coefs_1000_6000_K.dot(pow(T,array([[0],[1],[2],[3],[4]],dtype=float)))
    return result.T # ensure row dimension is T


def h_ig(T):
    result=((200<=T)&(T<=1000))*R*T*h_ig_coefs_200_1000_K.dot(pow(T,array([[0],[1],[2],[3],[4],[-1]],dtype=float)))+\
            ((1000<T)&(T<=6000))*R*T*h_ig_coefs_1000_6000_K.dot(pow(T,array([[0],[1],[2],[3],[4],[-1]],dtype=float)))
    return result.T # ensure row dimension is T


def s_ig(T):
    result=((200<=T)&(T<=1000))*R*s_ig_coefs_200_1000_K.dot(concatenate([log(array(T,ndmin=2)),pow(T,array([[1],[2],[3],[4],[0],[0]],dtype=float))]))+\
            ((1000<T)&(T<=6000))*R*s_ig_coefs_1000_6000_K.dot(concatenate([log(array(T,ndmin=2)),pow(T,array([[1],[2],[3],[4],[0],[0]],dtype=float))]))
    return result.T # ensure row dimension is T


def cp_mid(T0,T):
    result=(
        ((200<=T)&(T<=1000))*R*h_ig_coefs_200_1000_K+((1000<T)&(T<=6000))*R*T*h_ig_coefs_1000_6000_K).dot(
            array([sum([(T**(i-j)*T0**j) for j in range(i,0-1,-1)]) for i in range(4+1)]))
    return result # ensure row dimension is T


def transp(T,x,p=101325):
    # mixture properties according to Bird R. B., Stweart W. E., Lightfoot E. N. (2002). Transport phenomena. 2nd ed. John Wiley & Sons. New York. S. 26, 276, 864
    T=array(T,ndmin=1)
    x=array(x,ndmin=2)
    viscosity=mu(T)
    heat_capacity=cp_ig(T)
    conductivity=(heat_capacity+5/4*R)*viscosity/M
    if T.shape[0]<=1:
        phi=(1/sqrt(8)*(1+outer(M,1/M))**(-1/2)*(1+outer(viscosity,1/viscosity)**(1/2)*outer(1/M,M)**(1/4))**2).T
        viscosity_mix=((x*viscosity)/x.dot(phi)).sum(axis=1) # semiempirical mixing rule
        conductivity_mix=((x*conductivity)/x.dot(phi)).sum(axis=1) # semiempirical mixing rule
    else:
        phi=array([[1/sqrt(8)*(1+M[i]/M[j])**(-1/2)*(1+(viscosity[:,i]/viscosity[:,j])**(1/2)*(M[j]/M[i])**(1/4))**2 for i in range(M.shape[0])] for j in range(M.shape[0])]).T
        phi=phi.reshape([phi.shape[0]*phi.shape[1],phi.shape[2]])
        cols=array(range(x.shape[0]*x.shape[1]))
        rows=array(x.shape[1]*[[j for j in range(x.shape[0])]]).T.ravel()
        x_t=coo_array((x.ravel(),(rows,cols)),shape=[len(T),len(T)*len(M)],dtype=float)
        viscosity_mix=((x*viscosity)/x_t.dot(phi)).sum(axis=1) # semiempirical mixing rule
        conductivity_mix=((x*conductivity)/x_t.dot(phi)).sum(axis=1) # semiempirical mixing rule
    heat_capacity_mix=(heat_capacity*x).sum(axis=1)
    density_mix=(p/R/T*(M*x).sum(axis=1))
    mol_mass_mix=(M*x).sum(axis=1)
    Pr=viscosity_mix*(heat_capacity_mix/mol_mass_mix)/conductivity_mix
    return viscosity_mix, conductivity_mix, heat_capacity_mix, density_mix, mol_mass_mix, Pr


T=1000. # K
viscosity=mu(T).flatten()
density=101325/8.3145/T*M
heat_capacity=cp_ig(T).flatten()
conductivity=(heat_capacity+5/4*R)*viscosity/M


x=array([0,0,0,0,0,0.79,0.21,0,0,0,0,0])
viscosity_air,conductivity_air,heat_capacity_air,density_air,mol_mass_air,Pr=transp(T,x)


x=array([0.4,0,0,0,0.6,0,0,0,0,0,0,0])
viscosity_1,conductivity_1,heat_capacity_1,density_1,mol_mass_1,Pr=transp(T,x)


x_mass=array([0.2,0.5,0.05,0.05,0.2,0,0,0,0,0,0,0]) # ref. CH4 with mass fractions {0.2, 0.5, 0.05, 0.05, 0.2, 0.0} for {"Hydrogen", "Methane", "Carbondioxide", "Carbonmonoxide", "Water", "Nitrogen"}
x_2=x_mass/M/sum(x_mass/M)
viscosity_2,conductivity_2,heat_capacity_2,density_2,mol_mass_2,Pr=transp(T,x_2)


print(f'viscosity, density, heat capacity, conductivity at {T}K')
print('\t\tviscosity/(Pa s)\tdensity/(kg/m^3)\tCp heat cap./(J/mol/K)\tconductivity/(W/m/K)\tmol mass/(kg/mol)')
for i in range(len(M)):
    print('{:s}\t\t{:0.16g}\t{:0.16g}\t{:16.16g}\t{:16.16g}\t{:16.16g}'.format(components_list[i],viscosity[i],density[i],heat_capacity[i],conductivity[i],M[i]))
print('{:s}\t{:0.16g}\t{:0.16g}\t{:0.16g}\t{:0.16g}\t{:16.16g}'.format('N2:O2 79:21',viscosity_air.item(),density_air.item(),heat_capacity_air.item(),conductivity_air.item(),mol_mass_air.item()))
print('{:s}\t{:0.16g}\t{:0.16g}\t{:0.16g}\t{:0.16g}\t{:16.16g}'.format('H2O:H2 60:40',viscosity_1.item(),density_1.item(),heat_capacity_1.item(),conductivity_1.item(),mol_mass_1.item()))
print('{:s}\t{:0.16g}\t{:0.16g}\t{:0.16g}\t{:0.16g}\t{:16.16g}'.format('H2:..:H2O '+':'.join(['{:d}'.format(int(round(x*100,0))) for x in x if round(x*100,0)!=0]),viscosity_2.item(),density_2.item(),heat_capacity_2.item(),conductivity_2.item(),mol_mass_2.item()))


print('\n')
print('Example 23 from ref. [4] Stephan, P. u. a. (2013). VDI-Wärmeatlas. 11. bearb. und erw. Aufl. Berlin, Heidelberg: Springer-Verlag. S. 652.')
print('Benzene:Argon 25:75 mol, 100.6 °C, 1 bar')
T=273.15+100.6 # K
p=1e5 # Pa



x_3=array([0,0,0,0,0,0,0,0.75,0,0.25,0,0]) # ref. CH4 with mass fractions {0.2, 0.5, 0.05, 0.05, 0.2, 0.0} for {"Hydrogen", "Methane", "Carbondioxide", "Carbonmonoxide", "Water", "Nitrogen"}
viscosity_3,conductivity_3,heat_capacity_3,density_3,mol_mass_3,Pr=transp(T,x_3)


conductivity_3_2=(heat_capacity_3+5/4*R)*viscosity_3/sum(x_3*M)
print('\t\tviscosity/(Pa s)\tdensity/(kg/m^3)\tCp heat cap./(J/mol/K)\tconductivity/(W/m/K)')
for i in range(len(M)):
    print('{:s}\t\t{:0.16g}\t{:0.16g}\t{:16.16g}\t{:16.16g}'.format(components_list[i],viscosity[i],density[i],heat_capacity[i],conductivity[i]))
print('{:s}\t{:0.16g}\t{:0.16g}\t{:0.16g}\t{:0.16g}'.format('N2:O2 79:21',viscosity_air.item(),density_air.item(),heat_capacity_air.item(),conductivity_air.item()))
print('{:s}\t{:0.16g}\t{:0.16g}\t{:0.16g}\t{:0.16g}'.format('H2O:H2 95:5',viscosity_1.item(),density_1.item(),heat_capacity_1.item(),conductivity_1.item()))
print('{:s}\t{:0.16g}\t{:0.16g}\t{:0.16g}\t{:0.16g}'.format(':'.join([components_list[i] for i in range(len(components_list)) if x_2[i]!=0])+' '+':'.join(['{:d}'.format(int(round(x_2[i]*100,0))) for i in range(len(components_list)) if round(x_2[i]*100,0)!=0]),viscosity_2.item(),density_2.item(),heat_capacity_2.item(),conductivity_2.item()))
print('{:s}\t{:0.16g}\t{:0.16g}\t{:0.16g}\t{:0.16g}'.format(':'.join([components_list[i] for i in range(len(components_list)) if   x_3[i]!=0])+' '+':'.join(['{:d}'.format(int(round(  x_3[i]*100,0))) for i in range(len(components_list)) if round(  x_3[i]*100,0)!=0]),viscosity_3.item(),density_3.item(),heat_capacity_3.item(),conductivity_3.item()))
print('{:s}\t{:0.16g}\t{:0.16g}\t{:0.16g}\t{:0.16g} {:s}'.format(':'.join([components_list[i] for i in range(len(components_list)) if   x_3[i]!=0])+' '+':'.join(['{:d}'.format(int(round(  x_3[i]*100,0))) for i in range(len(components_list)) if round(  x_3[i]*100,0)!=0]),viscosity_3.item(),density_3.item(),heat_capacity_3.item(),conductivity_3_2.item(),'<<-- as (Cp+(5/4)R)mu/M'))


print('')
x=array([[0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0.79,0.21,0,0,0,0,0],[0,0,0,0,0,0.79,0.21,0,0,0,0,0],[0,0,0,0,0,0.79,0.21,0,0,0,0,0],[0,0,0.046,0,0.044,0.75,0.16,0,0,0,0,0]])
T=array([25,25,25,45,250])+273.15
p=array([5,1,1,1,1]).T*1e5
viscosity_mix,conductivity_mix,heat_capacity_mix,density_mix,mol_mass_mix,Pr=transp(T,x,p)
print('{:30s}{:8s}{:8s}{:20s}{:20s}{:20s}{:20s}'.format('comp.','T/°C','p/bar','viscosity/(Pa s)','density/(kg/m^3)','Cp heat cap./(J/mol/K)','conductivity/(W/m/K)'))
for i in range(x.shape[0]):
    print('{:30s}{:<8.5g}{:<8.5g}{:20.14}{:20.14}{:20.14}{:20.14}'.format(':'.join([components_list[j]+'({:d})'.format(int(round(x*100,0))) for j,x in enumerate(x[i,:]) if round(x*100,0)!=0]),T[i]-273.15,p[i]/1e5,viscosity_mix[i],density_mix[i],heat_capacity_mix[i],conductivity_mix[i]))



# %%
#Properties of gases and liquids p. 145 (eq. 4-11.2), 243 (Table 7-4), 777. (Section D)
#No. Formula Name CAS # Eq. #   A/A/Tc B/B/a C/C/b Tc/c to/d n/Pc E F      Pvpmin,bar  Tmin,K Pvpmax,bar   Tmax,K
#440 H2O water 7732-18-5    1   5.11564 1687.537 230.17                     0.01        273.20 16           473.20
#                           3   647.300 -7.77224 1.45684 -2.71942* -1.41336* 0.01 273.20 221 647.30
# Eq. # 1 : log_10(Pvp)=A-B/(T+C-273.15) (Antoine)
# Eq. # 3 : ln(Pvp/Pc)=(Tc/T)*(a*tau+b*tau^1.5+c*tau^2.5+d*tau^5), tau=(1-t/tc) Note: for water only the last two terms are c*tau^3+d*tau^6 (Wagner)
#No. Formula Name CAS # Mol. Wt. Tfp,K Tb,K Tc,K Pc,bar Vc,cm3/mol Zc=PcVc/RTc Omega
#440 H2O water 7732-18-5 18.015 273.15 373.15 647.14 220.64 55.95 0.229 0.344
tc=647.30 # K
pc=220.64e5 # bar
omega=0.344 # - acentric f.
pg=lambda t: 10**(5.11564-1687.537/(t+230.17-273.15)) # Antoine, water, t in K, psat in bar
psi=lambda t: 2.303*1687.537/tc*(t/tc/(t/tc+(230.17-273.15)/tc))**2 # Mod. Clausius-Clapeyron - Antoine, t in K, psi=-dlnpvr/(1/Tr)

pg=lambda t: pc*exp(tc/t*(-7.77224*(1-t/tc)+1.45684*(1-t/tc)**1.5-2.71942*(1-t/tc)**3-1.41336*(1-t/tc)**6)) # Wagner, t in K, psat in Pa
psi=lambda t: (-(-7.77224)+1.45684*(1-t/tc)**0.5*(0.5*(1-t/tc)-1.5)+(-2.71942)*(1-t/tc)**2*(2*(1-t/tc)-3)+(-1.41336)*(1-t/tc)**5*(5*(1-t/tc)-6)) # Mod. Clausius-Clapeyron - Wagner, t in K, psi=-dlnpvr/(1/Tr)
vs=lambda t: R*tc/pc*0.229**(1+(1-t/tc)**(2/7)) # Rackett, t in K, Vsat in m^3/mol
dhlv_rtc_pitzer=lambda t:(7.08*(1-t/tc)**0.354+10.95*0.229*(1-t/tc)**0.456) # Pitzer (7-9.4)

def poly_3_vec(p):
    """
    vectorized p(3) roots for (n by 3) p
    p[0]x^3+p[1]x^2+p[2]x^1+p[3]x^0=0
    """
    a3,a2,a1,a0=p
    p_coef=-1/3*(a2/a3)**2+a1/a3
    q_coef=2/27*(a2/a3)**3-1/3*a2*a1/a3**2+a0/a3
    # disc>0 -> 1 real 2 complex
    # disc<0 -> 3 real distinct
    # disc=0 -> 1 real 2 real of multiplicity 2
    disc=(p_coef/3)**3+(q_coef/2)**2 
    au=(sign(-q_coef/2+emath.sqrt(disc))*emath.power(abs(-q_coef/2+emath.sqrt(disc)),1/3)).real # only used where disc>=0
    bv=(sign(-q_coef/2-emath.sqrt(disc))*emath.power(abs(-q_coef/2-emath.sqrt(disc)),1/3)).real # only used where disc>=0
    re_z1=-a2/(3*a3)+(disc<0)*(
        2*sqrt(abs(-p_coef/3))*cos(1/3*emath.arccos(-q_coef/2/emath.power(-p_coef/3+(p_coef==0)*eps/3,3/2))+0*2*pi/3)
    )+(disc>0)*(
        au+bv
    )+((disc==0)|(abs(disc)<eps))*(
        au+bv
    )
    re_z2=-a2/(3*a3)+(disc<0)*(
        2*sqrt(abs(-p_coef/3))*cos(1/3*emath.arccos(-q_coef/2/emath.power(-p_coef/3+(p_coef==0)*eps/3,3/2))+1*2*pi/3)
    )+(disc>0)*(
        -1/2*(au+bv)
    )+((disc==0)|(abs(disc)<eps))*(
        -1/2*(au+bv)
    )
    re_z3=-a2/(3*a3)+(disc<0)*(
        2*sqrt(abs(-p_coef/3))*cos(1/3*emath.arccos(-q_coef/2/emath.power(-p_coef/3+(p_coef==0)*eps/3,3/2))+2*2*pi/3)
    )+(disc>0)*(
        -1/2*(au+bv)
    )+((disc==0)|(abs(disc)<eps))*(
        -1/2*(au+bv)
    )
    im_z1=(disc<0)*(0)+(disc>0)*(0)+((disc==0)|(abs(disc)<eps))*(0)
    im_z2=(disc<0)*(0)+(disc>0)*(+sqrt(3)/2*(au-bv)
                                 )+((disc==0)|(abs(disc)<eps))*(0)
    im_z3=(disc<0)*(0)+(disc>0)*(-sqrt(3)/2*(au-bv)
                                 )+((disc==0)|(abs(disc)<eps))*(0)
    z_roots=array([re_z1+im_z1*1j,re_z2+im_z2*1j,re_z3+im_z3*1j])
    #ipdb.set_trace()
    if (z_roots.imag==0).all():
        return z_roots.real
    else:
        return z_roots

def z(p,t,tc,pc,w,zc=0,eq='PR',c1=0,c2=0,c3=0,N=0):
    t=array(t,ndmin=1)
    p=array(p,ndmin=1)
    if eq=='RK':
        zc=0.3333 # z_mc
        apc_rtc2=0.42748 # Omega=aPc/(RTc)^2 -> SVN psi
        bpcrtc=0.08664 # bPc/(RTc) <-> SVN omega
        dpcrtc=0.08664 # dPc/(RTc)
        e_pcrtc2=0 # e*(Pc/(RTc))^2
        alpha=1/(t/tc)**(1/2)
        t_dalpha_dt=-1/2*1/(t/tc)**(1/2)

    elif eq=='RKS':
        zc=0.3333 # z_mc
        apc_rtc2=0.42188 # Omega=aPc/(RTc)^2 -> SVN psi
        bpcrtc=0.08333 # bPc/(RTc)
        dpcrtc=0.08333 # dPc/(RTc)
        e_pcrtc2=0.001736 # e*(Pc/(RTc))^2
        alpha=(1+(0.4998+1.5928*w-0.19563*w**2+0.025*w**3)*(1-(t/tc)**(1/2)))**2
        t_dalpha_dt=-sqrt(alpha)*(0.4998+1.5928*w-0.19563*w**2+0.025*w**3)*(t/tc)**(1/2)

    elif eq=='PT':
        zc=0.2695 # z_mc
        omega_b_roots=roots([1,(2-3*zc),3*zc**2,-zc**3])
        omega_b=min(omega_b_roots[omega_b_roots>=0])
        omega_a=3*zc**2+3*(1-2*zc)*omega_b+omega_b**2+1-3*zc
        omega_c=1-3*zc
        apc_rtc2=omega_a # Omega=aPc/(RTc)^2 -> psi
        bpcrtc=omega_b # bPc/(RTc)
        cpcrtc=omega_c # cPc/(RTc)
        dpcrtc=bpcrtc+cpcrtc # dPc/(RTc)
        e_pcrtc2=-bpcrtc*cpcrtc # e*(Pc/(RTc))^2
        alpha=1+c1*(t/tc-1)+c2*(sqrt(t/tc)-1)+c3*((t/tc)**N-1)
        t_dalpha_dt=c1*t/tc+c2*(1/2)*(t/tc)**(1/2)+c3*N*(t/tc)**N

    else: #eq=='PR':
        zc=0.3070 # z_mc
        apc_rtc2=0.45724 # Omega=aPc/(RTc)^2 -> psi
        bpcrtc=0.0778 # bPc/(RTc)
        dpcrtc=0.15559 # dPc/(RTc)
        e_pcrtc2=-0.006053 # e*(Pc/(RTc))^2
        alpha=(1+(0.3746+1.54226*w-0.26992*w**2)*(1-(t/tc)**(1/2)))**2
        #alpha0=(t/tc)**-0.171813*exp(0.125283*(1-(t/tc)**1.77634))
        #alpha1=(t/tc)**-0.607352*exp(0.511614*(1-(t/tc)**2.20517))
        #alpha=alpha0+w*(alpha0-alpha1)
        t_dalpha_dt=-sqrt(alpha)*(0.3746+1.54226*w-0.26992*w**2)*(t/tc)**(1/2)

    bprime=bpcrtc*(p/pc)/(t/tc) # B'=bp/(RT) <-> SVN beta
    dprime=dpcrtc*(p/pc)/(t/tc) # d'=dp/(RT) <-> 
    tprime=apc_rtc2*(p/pc)/(t/tc)**2*alpha # theta*p/(RT)^2=(a*alpha)*p/(RT)^2=(aPc/(RTc^2))*(RTc)^2/*alphaPc*p/(RT)*alpha
    eprime=e_pcrtc2*(p/pc)**2/(t/tc)**2
    etaprime=bprime
    svn_q=apc_rtc2/bpcrtc/(t/tc)**(3/2) # only ref.
    svn_beta=bprime # only ref.
    a3=ones(t.shape)
    a2=dprime-bprime-1
    a1=tprime+eprime-dprime*(bprime+1)
    a0=-(eprime*(bprime+1)+tprime*etaprime)

    z_roots=poly_3_vec([a3,a2,a1,a0])
    #z_roots=roots([a3,a2,a1,a0])
    z_min=bpcrtc*(p/pc)/(t/tc)
    z_l=z_roots.min(axis=0)
    z_v=z_roots.max(axis=0)
    v_l=z_l*R*t/p
    v_v=(z_v*R*t/p).real


    a=apc_rtc2*(R*tc)**2/pc
    b=bpcrtc*R*tc/pc
    d=dpcrtc*R*tc/pc
    e=e_pcrtc2*(R*tc/pc)**2
    v=v_v
    h_r_rt=+(a*alpha-a*t_dalpha_dt)/(R*t*(d**2-4*e)**(1/2))*log((2*v+d-(d**2-4*e)**(1/2))/(2*v+d+(d**2-4*e)**(1/2)))-1+z_v # typo in minus sign? (PGL)
    s_r_r=-a*t_dalpha_dt/(R*t*(d**2-4*e)**(1/2))*log((2*v+d-(d**2-4*e)**(1/2))/(2*v+d+(d**2-4*e)**(1/2)))+log(z_v*(1-b/v))
    ln_phi=(a*alpha+a*t_dalpha_dt)/(R*t*(d**2-4*e)**(1/2))*log((2*v+d-(d**2-4*e)**(1/2))/(2*v+d+(d**2-4*e)**(1/2)))-log(z_v*(1-b/v))-(1-z_v)
    return z_roots,bpcrtc*(p/pc)/(t/tc),h_r_rt

#z(1e5,linspace(273.15+-100,273.15+100),tc,pc,omega,eq='PT',c1=0.60462,c2=-2.56713)
z_1,z_min,h_r_rt=z(1e5,273.15+99.6,tc,pc,omega,eq='PT',c1=0.60462,c2=-2.56713)
z_1=z_1[z_1>=z_min]
delta_zlv=max(z_1)-min(z_1)
dh_lv=psi(273.15+99.6)*R*tc/0.018/1000*delta_zlv
print('\n\nTest deltaH_LV=',dh_lv,'J/kg (99.6 °C)')

hl=h_ig(273.15+99.6)[:,4]-psi(273.15+99.6)*R*tc*(1-vs(273.15+99.6)*1e5/(R*273.15+99.6))
print('\n\nTest Wagner+Rackett: hl=',hl.item(),'J/mol (99.6 °C)')

hl=h_ig(273.15+99.6)[:,4]-psi(273.15+99.6)*R*tc*delta_zlv
print('\n\nTest Wagner+zlv: hl=',hl.item(),'J/mol (99.6 °C)')

t=linspace(273.15,273.15+373.946,50)
p=pg(t)
dh_lv_r=psi(t)*R*tc*(1-vs(t)*1e5/(R*t))

z_1,z_min,h_r_rt=z(p,t,tc,pc,omega,eq='PT',c1=0.60462,c2=-2.56713)
delta_zlv=z_1.max(axis=0)-z_1.min(axis=0)

dh_lv_z=psi(t)*R*tc*delta_zlv

dh_lv_p=dhlv_rtc_pitzer(t)*R*tc

t_t,p_t,hl_t,hv_t=loadtxt('steam_table_vdi_wa.csv',usecols=[0,1,4,5],skiprows=2,dtype=float).T
dh_lv_t=hv_t-hl_t

delta_zlv_triple=array([max(x[0][x[0]>=x[1]])-min(x[0][x[0]>=x[1]]) for x in [z(611.657,273.16,tc,pc,omega,eq='PT',c1=0.60462,c2=-2.56713)]])
dh_lv_z_triple=psi(273.16)*R*tc*delta_zlv_triple
dh_lv_r_triple=psi(273.16)*R*tc*(1-vs(273.16)*611.657/(R*273.16))
dh_lv_p_triple=dhlv_rtc_pitzer(273.16)*R*tc

hl_ref_r=h_ig(273.16)[:,4]-dh_lv_r_triple # water at triple point (273.16K) https://iapws.org/public/documents/UWTF-/IF97-Rev.pdf
hl_ref_z=h_ig(273.16)[:,4]-dh_lv_z_triple # water at triple point (273.16K) https://iapws.org/public/documents/UWTF-/IF97-Rev.pdf
hl_ref_p=h_ig(273.16)[:,4]-dh_lv_p_triple # water at triple point (273.16K) https://iapws.org/public/documents/UWTF-/IF97-Rev.pdf
hl_r=h_ig(t)[:,4]-dh_lv_r-hl_ref_r
hl_z=h_ig(t)[:,4]-dh_lv_z-hl_ref_z
hl_p=h_ig(t)[:,4]-dh_lv_p-hl_ref_p

hv_r=h_ig(t)[:,4]-hl_ref_r
hv_z=h_ig(t)[:,4]-hl_ref_z
hv_p=h_ig(t)[:,4]-hl_ref_p

def h_lv_h2o(t,p=None):
    t=array(t,ndmin=1)
    if p is None:
        p=pg(t)
    p=array(p,ndmin=1)
    z_1,z_min,h_r_rt=z(p,t,tc,pc,omega,eq='PT',c1=0.60462,c2=-2.56713)
    delta_zlv=z_1.max(axis=0)-z_1.min(axis=0)
    dh_lv_z=psi(t)*R*tc*delta_zlv
    # note: referencing only liquid to IF97 conflicts due to gas not so referenced
    return {'l':h_ig(t)[:,4]+h_r_rt*R*t-dh_lv_z-hl_ref_z,
            'v':h_ig(t)[:,4]+h_r_rt*R*t-hl_ref_z}

def convert_ax(ax):
    x0,x1=ax.get_xlim()
    twiny0.set_xlim((x0+273.15)/tc,(x1+273.15)/tc)
    twiny0.figure.canvas.draw()

fig=plt.figure(figsize=[8,5],layout='constrained')
gs0=fig.add_gridspec(2,2)
ax_list=[]
ax_list+=[fig.add_subplot(gs0[:,0])]
ax_list+=[fig.add_subplot(gs0[0,1])]
ax_list+=[fig.add_subplot(gs0[1,1])]
#fig,ax_list=plt.subplots(1,2,constrained_layout=True,sharex=True)
twiny0=ax_list[0].twiny()
twinx0=ax_list[0].twinx()
#twiny0.spines.bottom.set_position(("axes", -0.75))
ax_list[0].callbacks.connect('ylim_changed',convert_ax)
ax_list[0].plot(t-273.15,dh_lv_r/0.018/1000,label=r'$\Delta z_{LV}=1-\frac{V^{L,Rackett} p}{R T}$')
ax_list[0].plot(t-273.15,dh_lv_z/0.018/1000,label=r'$\Delta z_{LV}=Z^V_{ceos}-Z^L_{ceos}$')
ax_list[0].plot(t-273.15,dh_lv_p/0.018/1000,label=r'Pitzer (7-9.4)')
twinx0.plot(t-273.15,p/1e5,':',color='green',label='$p^{sat}(T)$')
ax_list[0].plot(t_t,dh_lv_t,'o',label='steam table')
ax_list[0].set_ylabel(r'$\Delta H_{LV}$ / $kJ\cdot kg^{-1}$')
ax_list[0].set_xlabel(r'$T$ / $^\circ C$')
twiny0.set_xlabel(r'$T_r$ / -')
twinx0.set_ylabel('$p$ / bar')
ax_list[0].legend(loc='best',handles=ax_list[0].lines+twinx0.lines)

ax_list[1].plot(t-273.15,hl_r/0.018/1000,label=r'$\Delta z_{LV}=1-\frac{V^{L,Rackett} p}{R T}$')
ax_list[1].plot(t-273.15,h_lv_h2o(t)['l']/0.018/1000,label=r'$\Delta z_{LV}=Z^V_{ceos}-Z^L_{ceos}$')
ax_list[1].plot(t-273.15,hl_p/0.018/1000,label=r'Pitzer (7-9.4)')
ax_list[1].plot(t_t,hl_t,'o',label=r'steam table')
ax_list[1].set_ylabel(r'$H_{L}-H_{L,triple}$ / $kJ\cdot kg^{-1}$')
ax_list[1].set_xlabel(r'$T$ / $^\circ C$')

ax_list[2].plot(t-273.15,hv_r/0.018/1000,label=r'$\Delta z_{LV}=1-\frac{V^{L,Rackett} p}{R T}$')
ax_list[2].plot(t-273.15,h_lv_h2o(t)['v']/0.018/1000,label=r'$\Delta z_{LV}=Z^V_{ceos}-Z^L_{ceos}$')
ax_list[2].plot(t-273.15,hv_p/0.018/1000,label=r'Pitzer (7-9.4)')
ax_list[2].plot(t_t,hv_t,'o',label=r'steam table')
ax_list[2].set_ylabel(r'$H_{V}-H_{V,triple}$ / $kJ\cdot kg^{-1}$')
ax_list[2].set_xlabel(r'$T$ / $^\circ C$')
#ax_list[2].legend(loc='lower center',bbox_to_anchor=[0.5,1.05])

fig.show()

# %% [markdown]
# # condensation dehumidification cycle

# %%
# Moran, M.J. et al. 2002. Introduction to Thermal Systems Engineering: Thermodynamics, Fluid Mechanics, and Heat Transfer
# Ex. 10.1
p=14.7*(101325/14.7) # Pa
yw_0=0.7*pg((70-32)*5/9+273.15)/p
yw_0_mass=yw_0*0.018/(yw_0*0.018+(1-yw_0)*0.02885)
w_0=yw_0/(1-yw_0)*0.018/0.02885 # kgw/kgdryair
soln=least_squares(lambda t: 1.0*pg(t)/p-yw_0,(70-32)*5/9+273.15)
yw_1=pg((40-32)*5/9+273.15)/p
w_1=yw_1/(1-yw_1)*0.018/0.02885 # kgw/kgdryair

m=1 # lb
mas=m*(1-yw_0_mass)
# w0/(1-w0)Mas0=w1/(1-w1)Mas1+ml
# Mas0=M*(1-yw0)=M*(1-yw0*0.018/(yw0*0.018+(1-yw0)*0.02885))=Mas1
ml=m*(1-yw_0_mass)*(w_0-w_1) # condensate
print('10.1')
print('a) rel. humidity: ',w_0,'kgw/kg dry air')
if soln.success:
    t_dew=(soln.x-273.15)*9/5+32
    print('b) dew point',t_dew.item(),'°F')
print('c) condensate: ',ml,'lb')

# Ex. 10.3
p=1.01325*1e5 # Pa
v_dot_0=280 # m^3/min
t0=30+273.15 # K
t1=10+273.15 # K
yw_0=0.5*pg(t0)/p
yw_1=pg(t1)/p
yw_0_mass=yw_0*0.018/(yw_0*0.018+(1-yw_0)*0.02885)
m_dot_0=v_dot_0*(p/(R*t0))*(yw_0*0.018+(1-yw_0)*0.02885)
mas_dot=m_dot_0*(1-yw_0_mass)

w_0=yw_0/(1-yw_0)*0.018/0.02885 # kgw/kgdryair
w_1=yw_1/(1-yw_1)*0.018/0.02885 # kgw/kgdryair
ml_dot=mas_dot*(w_0-w_1) # condensate

soln=least_squares(lambda t: 1.0*pg(t)/p-yw_0,t0)
tsat_yw_0=soln.x.item()
qpunkt_sens1=v_dot_0*(p/(R*t0))*(h_ig(t0)-h_ig(tsat_yw_0))[:,4]
delta_zlv=array([max(x[0][x[0]>=x[1]])-min(x[0][x[0]>=x[1]]) for x in [z(p,tsat_yw_0,tc,pc,omega,eq='PT',c1=0.60462,c2=-2.56713)]])
dh_lv_z=psi(tsat_yw_0)*R*tc*delta_zlv
qpunkt_latnt=ml_dot/0.018*dh_lv_z
qpunkt_sens2=ml_dot/0.018*(h_lv_h2o(tsat_yw_0,p)['l']-h_lv_h2o(t1,p)['l'])

hf_in=v_dot_0*(p/(R*t0))*h_ig(t0).dot([0,0,0,0,yw_0,0.79*(1-yw_0),0.21*(1-yw_0),0,0,0,0,0])
hf_outg=(v_dot_0*(p/(R*t0))-ml_dot/0.018)*h_ig(t1).dot([0,0,0,0,yw_1,0.79*(1-yw_1),0.21*(1-yw_1),0,0,0,0,0])
hf_outl=ml_dot/0.018*h_lv_h2o(t1,p)['l']
print('10.3')
print('a) dry air Mas',mas_dot,'kg dry air/min')
print('b)',ml_dot,'kg / min; ',ml_dot/mas_dot,'kg condensate / kg dry air flow')
print('c) sensible heat (1)',qpunkt_sens1.item()/60/1000,'kW; latent',qpunkt_latnt.item()/1000/60,'kW; sens (2)',qpunkt_sens2.item()/60/1000,'kW; total',(qpunkt_sens1+qpunkt_latnt+qpunkt_sens2).item()/1000,'kJ/min')
print('c) total cooling capacity',(hf_in-hf_outg-hf_outl)/1000,'kJ/min')


# Dehumidification cooling for 421.057 kg/h air, 45°C, a) 50% RH, b) 100
p=1.01325*1e5 # Pa
mas_dot=408.43040733366814 # kg/h dry air req.
t0=45+273.15 # K
t1=17.5+273.15 # K
yw_0=0.5*pg(t0)/p
yw_1=pg(t1)/p
yw_0_mass=yw_0*0.018/(yw_0*0.018+(1-yw_0)*0.02885)
yw_1_mass=yw_1*0.018/(yw_1*0.018+(1-yw_1)*0.02885)
w_0=yw_0/(1-yw_0)*0.018/0.02885 # kgw/kgdryair
w_1=yw_1/(1-yw_1)*0.018/0.02885 # kgw/kgdryair
m_dot_0=mas_dot/(1-yw_0_mass) # kg/h wet air
ml_dot=mas_dot*(w_0-w_1) # kg/h condensate
v_dot_0=m_dot_0/(yw_0*0.018+(1-yw_0)*0.02885)*R*t0/p # m3/h wet air

soln=least_squares(lambda t: 1.0*pg(t)/p-yw_1,t0)
tsat_yw_0=soln.x.item()
hf_in=v_dot_0*(p/(R*t0))*h_ig(t0).dot([0,0,0,0,yw_0,0.79*(1-yw_0),0.21*(1-yw_0),0,0,0,0,0])
hf_outg=(v_dot_0*(p/(R*t0))-ml_dot/0.018)*h_ig(t1).dot([0,0,0,0,yw_1,0.79*(1-yw_1),0.21*(1-yw_1),0,0,0,0,0])
hf_outl=ml_dot/0.018*h_lv_h2o(t1,p)['l']
[q_dot_cond]=(hf_in-hf_outg-hf_outl)/3600/1000 # kW
# compare against psychrometry chart with ref. 0°C, 0%RH, 0J /g dry air
dh_0=1/((1-yw_0_mass)*(yw_0*0.018+(1-yw_0)*0.02885))*(h_ig(t0).dot([0,0,0,0,yw_0,0.79*(1-yw_0),0.21*(1-yw_0),0,0,0,0,0])-h_ig(273.15).dot([0,0,0,0,0,0.79,0.21,0,0,0,0,0]))/1000 # J/(g dry air) 
dh_1=1/((1-yw_1_mass)*(yw_1*0.018+(1-yw_1)*0.02885))*(h_ig(t1).dot([0,0,0,0,yw_1,0.79*(1-yw_1),0.21*(1-yw_1),0,0,0,0,0])-h_ig(273.15).dot([0,0,0,0,0,0.79,0.21,0,0,0,0,0]))/1000 # J/(g dry air) 

print('45°C, 50%RH')
print('a) dry air Mas',mas_dot,'kg dry air/h')
print('b)',ml_dot,'kg / h; ',ml_dot/mas_dot,'kg condensate / kg dry air flow')
print('c) total cooling capacity',q_dot_cond,'kW')
print('45°C, 50%RH: w_0=',w_0,'kgw/kg dry air; h=',dh_0.item(),'J /g dry air','... read plot',125,'kJ/kg dry air')
print('17.5°C, 100%RH: w_1=',w_1,'kgw/kg dry air; h=',dh_1.item(),'J /g dry air','... read plot',49,'kJ/kg dry air')
print('from plot: ',mas_dot*(125-49)/3600,'kW')


# %% [markdown]
# # calc. exhaust properties at 100% load, 50% load.
# spec. for 100% load: 75%N2, 4.6% CO2, 4.4% H2O, 16% O2 --> implies humidity at inlet air was 0.85 mol-%.

# %%
Mi=M[:7]
# 100% load
# t0=datetime(2024,11,6,6,41,0).astimezone(timezone('Europe/Berlin'))
# m_dot_fuel_in[abs(dat['Timeloc']-t0).argmin()]*3600
m_dot_air_in=416 # kg/h
m_dot_fuel_in=7 # kg/h
p_sofc=55.93 # kW
T_exhaust=257.4 # °C
y=array([0,0,0.02943767,0,0.05887534,0.76674424,0.14494275])
y_mass=array([0,0,0.04550827,0,0.03723404,0.75429778,0.1629599])
y_mass_spec=array([0,0,0.046,0,0.044,0.75,0.16])
m_dot_air_in_spec,m_dot_fuel_in_spec=411,6.9
print('spec. composition in mol-%',','.join(['{:g} % '.format(x*100)+['H2','CH4','CO2','CO','H2O','N2','O2'][j] for j,x in enumerate(y_mass_spec/Mi/(y_mass_spec/Mi).sum()) if x>0]))# 0.         0.         0.02963675 0.         0.06929559 0.75932668 0.14174098
dew_point=41.3970

# given spec, find out humidity
y_mol_spec=y_mass_spec/Mi/(y_mass_spec/Mi).sum()
[y_mol_h2o_in],[y_mol_ch4_in]=inv(array([[-1,-1],[1,2]])).dot(array([[y_mol_spec[5]/0.79-1],[y_mol_spec[4]]]))
y_mol_in=array([0,y_mol_ch4_in,0,0,y_mol_h2o_in,0.79*(1-y_mol_ch4_in-y_mol_h2o_in),0.21*(1-y_mol_ch4_in-y_mol_h2o_in)])
y_mass_in=y_mol_in*Mi/(y_mol_in*Mi).sum()
m_dot_in=y_mass_in*(m_dot_air_in_spec+m_dot_fuel_in_spec)
print('H2O initial humidity: ',y_mol_h2o_in)
print('CH4 inlet fraction considering humidity: ',y_mol_ch4_in)
print('inlet flows in kg/h: \n', ','.join(['{:g} kg/h '.format(x)+['H2','CH4','CO2','CO','H2O','N2','O2'][j] for j,x in enumerate(m_dot_in) if x>0]))

m_dot_N2_in=Mi[5]*y_mol_spec[5]*(m_dot_air_in_spec/Mi[4]+m_dot_fuel_in_spec/Mi[1])/(1+y_mol_spec[5]*(Mi[5]/Mi[4]*(1+0.21/(1-0.21)*Mi[6]/Mi[5])-(1+0.21/(1-0.21))))
m_dot_O2_in=0.21/(1-0.21)*m_dot_N2_in*Mi[6]/Mi[5]
m_dot_H2O_in=m_dot_air_in_spec-m_dot_N2_in-m_dot_O2_in
m_dot_CH4_in=m_dot_fuel_in_spec
m_dot_in=array([0,m_dot_CH4_in,0,0,m_dot_H2O_in,m_dot_N2_in,m_dot_O2_in])
y_mol_in=m_dot_in/Mi/(m_dot_in/Mi).sum()
y_mass_in=m_dot_in/m_dot_in.sum()
m_dot_air_in_total_dry=m_dot_air_in_spec*(1-y_mass_in[4]/y_mass_in[4:].sum())*(5*32)
m_dot_air_in_total_45deg_c_50pct_rh=m_dot_air_in_spec*(1-y_mass_in[4]/y_mass_in[4:].sum())/(1-yw_0_mass)*(5*32) # 5XHD60X32 4.72-mol%
m_dot_condensate_total=m_dot_air_in_total_dry*(w_0-w_1) # kg/h condensate
m_dot_air_in_total_to_process=m_dot_air_in_total_45deg_c_50pct_rh-m_dot_condensate_total
print('second (preferred) humidity spec to match exact m_dot_in spec for CH4, air: \n',','.join(['{:g} kg/h '.format(x)+['H2','CH4','CO2','CO','H2O','N2','O2'][j] for j,x in enumerate(m_dot_in) if x>0]))
print('--> H2O initial humidity in air (actual): ',y_mol_in[4]/y_mol_in[4:].sum())
print('--> CH4 inlet fraction considering humidity (actual): ',y_mol_in[1])
print('--> dry air flow: ','{:g} kg/h for 1 unit, {:g} kg/h for 160 units'.format(m_dot_air_in_total_dry/5/32,m_dot_air_in_total_dry)) 
print('--> humid air flow 45°C, 50%RH: {:g} kg/h for 1 unit, {:g} kg/h for 160 units '.format(m_dot_air_in_total_45deg_c_50pct_rh/5/32,m_dot_air_in_total_45deg_c_50pct_rh))
print('--> condensate at 17.5 °C: {:g} kg/h; air with 2-mol% to process: {:g} kg/h'.format(m_dot_condensate_total,m_dot_air_in_total_to_process),'\n')

# 50% load
# t0=datetime(2024,11,5,20,26,0).astimezone(timezone('Europe/Berlin'))
# m_dot_fuel_in[abs(dat['Timeloc']-t0).argmin()]*3600
m_dot_air_in=308 # kg/h
m_dot_fuel_in=3.7 # kg/h
p_sofc=29.74 # kW
T_exhaust=153.1 # °C
y=array([0,0,0.02167669,0,0.04335337,0.77287542,0.16209452])
y_mass=array([0,0,0.0333935,0,0.02732196,0.75767666,0.18160788])
#array([0.        , 0.        , 0.02167669, 0.        , 0.04335338, 0.77287542, 0.16209452])
dew_point=33.0698 # °C

# get outlet calc, assuming 0.9683% initial humidity
y_air_in=array([0,0,0,0,y_mol_in[4]/y_mol_in[4:].sum(),0.79*(1-y_mol_in[4]/y_mol_in[4:].sum()),0.21*(1-y_mol_in[4]/y_mol_in[4:].sum())])
y_fuel_in=array([0,1,0,0,0,0,0])
n_in=m_dot_air_in/y_air_in.dot(Mi)*y_air_in+m_dot_fuel_in/Mi[1]*y_fuel_in
y_in=n_in/n_in.sum()
nu_i=array([0,-1,1,0,2,0,-2]) # CH4+2O2 --> CO2+2H2O
y_out=array([(y_in[j]-nu_i[j]/nu_i[1]*y_in[1]*1)/(1-0) for j in range(len(Mi))]).T # 100% CH4 conversion

print('50% load (initial humidity in air {:g} mol-%); y_in='.format(y_mol_in[4]/y_mol_in[4:].sum()*100),','.join(['{:g} % '.format(x*100)+['H2','CH4','CO2','CO','H2O','N2','O2'][j] for j,x in enumerate(y_in) if x>0]))
print('y_out=',','.join(['{:g} % '.format(x*100)+['H2','CH4','CO2','CO','H2O','N2','O2'][j] for j,x in enumerate(y_out) if x>0]))
print('y_out_mass=',','.join(['{:g} % '.format(x*100)+['H2','CH4','CO2','CO','H2O','N2','O2'][j] for j,x in enumerate(y_out*Mi/(y_out*Mi).sum()) if x>0]))

# 4-1 (isob. vaporization at 6°C, 5.35 bar)
t_4=6+273.15 # K
p_4=5.35 # bar
h_4=-560 # kJ/kg
h_1=+520 # kJ/kg
h_vsat_4=h_1 # kJ/kg
h_lsat_4=-740 # kJ/kg
x_lv_4=(h_vsat_4-h_4)/(h_vsat_4-h_lsat_4)
m_dot_nh3=-q_dot_cond*3600/(h_4-h_1) # kg/h

# 1-2 (compression eta=0.8 to 17.2 bar)
s_1=10.25 # kJ/kg
t_1=6+273.15 # K
s_2_isent=s_1
t_2_isent=87+273.15 # K
h_2_isent=665 # kJ/kg
eta_comp=0.8
h_2=h_1+(h_2_isent-h_1)/eta_comp
s_2=10.35 # kJ/kg
t_2=100+273.15 # K
p_2=17.2 # bar
w_dot=m_dot_nh3/3600*(h_1-h_2) # kW

# 2-3 (cooling to 42°C l_sat 12.8 bar)
t_3=42+273.15
h_3=h_4 # kJ/kg (isent. expansion already fixed)
q_dot_evap=m_dot_nh3*(h_3-h_2)/3600 # kW

# 3-4 (expansion to 6°C, 5.35 bar)
t_4=6+273.15 # K
p_4=5.35 # bar

# heat delivered by dry air temperature rise 17.5°C --> 22°C
y_0=array([0,0,0,0,yw_0,0.79*(1-yw_0),0.21*(1-yw_0),0,0,0,0,0])
y_1=array([0,0,0,0,yw_1,0.79*(1-yw_1),0.21*(1-yw_1),0,0,0,0,0])
q_dot_air=(m_dot_0-mas_dot*(w_0-w_1))*(h_ig(273.15+37)-h_ig(273.15+17.5)).dot(y_1)/M.dot(y_1)/1000/3600
h_3_mid=h_2-q_dot_air/(m_dot_nh3/3600)
h_vsat_3=525 # kJ/kg
h_lsat_3=h_3
x_lv_3_mid=(h_vsat_3-h_3_mid)/(h_vsat_3-h_lsat_3)
m_dot_additional_water=(-q_dot_evap-q_dot_air)/((h_lv_h2o(298.15+37,1.01325)['l']-h_lv_h2o(298.15+32,1.01325)['l'])/0.018/1000)*3600 # 42°C -> 32 °C (feed seawater)

# needed additional air flow at 45°C, 50%RH

results_table=[
    ['x_lv_4',x_lv_4,'-'],
    ['q_dot_cond',q_dot_cond,'kW'],
    ['m_dot_nh3',m_dot_nh3,'kg/h'],
    ['h_2',h_2,'kJ/kg'],
    ['w_dot',w_dot,'kW'],
    ['w_dot*5*32',w_dot*5*32,'kW'],
    ['q_dot_evap',q_dot_evap,'kW'],
    ['h_3_mid',h_3_mid,'kJ/kg'],
    ['x_lv_3_mid',x_lv_3_mid,'-'],
    ['q_dot_air',q_dot_air,'kW'],
    ['m_dot_additional_water*32*5',m_dot_additional_water*32*5,'kg/h'],
    ['m_dot_air_in_total_45deg_c_50pct_rh*32*5',m_dot_air_in_total_45deg_c_50pct_rh,'kg/h'],
    ['m_dot_condensate_total',m_dot_condensate_total,'kg/h'],
    ['m_dot_air_in_total_to_process',m_dot_air_in_total_to_process,'kg/h'],
]

print(tabulate(results_table,headers=['','val','units']))

