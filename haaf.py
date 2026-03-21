# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     main_language: python
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (Spyder)
#     language: python3
#     name: python3
# ---

# %% [markdown]
# # base case

# %%
from numpy import array,outer,exp,sqrt,pi,concatenate,log,roots,linspace,loadtxt,asarray,cos,ones,abs,emath,sign,maximum,minimum,interp
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
from tabulate import tabulate

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import FloatSlider,IntSlider


# doi.org/10.1007/978-3-642-82522-4_13
# Haaf, S. (1988). Wärmeübertragung in Luftkühlern. In: Steimle, F., et al. Wärmeaustauscher. Handbuch der Kältetechnik, vol 6 / B. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-82522-4_13
# Kap. 12

R=8.3145 # J/mol/K

def pg(t):
    tc,pc,vc,zc,omega=647.14,220.64,55.95,0.229,0.344
    a,b,c,d=-7.86975,1.90561,-2.30891,-2.06472
    pc=1e5*pc # Pc in Pa
    vc=(1/100)**3*vc # Vc in m3/gmol
    vc=zc*(R*tc)/pc # Vc in m3/gmol (keep consistent with zc)
    tr=t/tc
    return pc*exp(1/tr*(a*(1-tr)+b*(1-tr)**1.5+c*(1-tr)**2.5+d*(1-tr)**5))

def alpha_r_x_alpha_r_l(x,rho_r_l,rho_r_g,alpha_r_g0,alpha_r_l0):
    # doi.org/10.1007/978-3-662-55480-7
    return ((1-x)**0.01*((1-x)**1.5+1.2*x**0.4*(rho_r_l/rho_r_g)**0.37)**-2.2
            +x**0.01*(alpha_r_g0/alpha_r_l0*(1+8*(1-x)**0.7*(rho_r_l/rho_r_g)**0.67))**-2)**-0.5

def diff(x): return concatenate([array([0]),x[1:]-x[:-1]])

def tanh(x): return (exp(x)-exp(-x))/(exp(x)+exp(-x))

# Ref. R22 parameters R22, VDI-WA D3 (S. 470) CHClF2 (Tc = 369,3 K, pc = 4,990 MPa, ρc = 523,8 kg/m3)
t="""
115,73 0,000379 1721,3 0,000034 29,60 332,71 303,11 1,075 0,425 13435,3 4,88 177,01 2,59 81,61 0,800 38,56 1,530
200 16,67 1499,7 0,875 119,22 372,15 252,93 1,064 0,539 544,0 8,37 129,11 5,54 4,48 0,814 23,46 1,809
225 70,91 1430,3 3,371 146,03 384,30 238,27 1,081 0,589 379,4 9,40 116,85 6,68 3,51 0,828 19,27 2,001
232,34 101,33 1409,2 4,704 154,00 387,75 233,75 1,090 0,606 345,4 9,70 113,37 7,05 3,32 0,834 18,08 2,074
250 216,90 1356,3 9,605 173,51 395,71 222,20 1,117 0,655 279,6 10,41 105,22 7,99 2,97 0,854 15,25 2,292
275 528,65 1275,2 22,50 202,17 405,72 203,55 1,174 0,747 211,7 11,44 93,91 9,53 2,65 0,897 11,42 2,742
300 1097,0 1183,4 46,54 232,62 413,50 180,88 1,265 0,885 161,0 12,61 82,64 11,51 2,47 0,970 7,82 3,510
325 2026,4 1073,2 90,19 265,84 417,55 151,71 1,438 1,139 120,1 14,21 71,01 14,52 2,43 1,11 4,51 5,155
350 3442,7 920,1 177,54 304,70 413,70 109,00 1,996 1,956 83,1 17,33 58,30 21,29 2,85 1,59 1,62 11,35
369,3 4990 523,8 523,8 366,90 366,90 nan nan nan nan nan nan nan nan nan nan nan
""" # Ts, K; ps, kPa; ρ′, kg/m3; ρ′′, kg/m3; h′, kJ/kg; h′′, kJ/kg; Δhv, kJ/kg; cp′ , kJ/ (kg K); cp′′, kJ/ (kg K); η′, 10−6 kg/ (m s); η′′, 10−6 kg/ (m s); λ′, 10−3 W/(m K); λ′′, 10−3 W/(m K) Pr′ Pr′′; σ, 10−3 N/m; β, 10−3 /K
datlv=array([x.replace(',','.').replace('−','-').split(' ') for x in t.split('\n') if len(x)>0],dtype=float)*array([1,1e3,1,1,1e3,1e3,1e3,1e3,1e3,1e-6,1e-6,1e-3,1e-3,1,1,1e-3,1e-3])

# F Faktor  Korrelation Nu 
# Blasensieden Nu-Korrelation für Kältemittel abhäingig von der Verdampfungstemperatur to in °C
t="""
- -60 -50 -40 -30 -20 -10 0 +10 
R11 nan nan nan nan 1005 1193 1403 1619 
R12 nan nan 1443 1691 1945 2218 2503 2794 
R13 2789 3207 3646 4099 4553 5006 5508 nan
R13B1 1463 1727 1996 2293 2579 2887 3189 nan
R21 nan nan 792 979 1166 1382 1605 nan
R22 nan nan 1982 2318 2669 3050 3424 3812 
R40 nan nan nan nan 1817 2093 2375 2673 
NH3 1784 2220 2704 3233 3803 4413 5055 5734 
""" 
yf=array([x.replace(',','.').strip().split(' ')[1:] for x in t.split('\n') if len(x)>0],dtype=float)

# process params
p=1e5
v_dot_air_in_total_to_process=5200/3600 # m^3/h
t0=10+273.15
rh_0=0.85
q_dot_cond=9000#7000 # W
tsat_yw_0=least_squares(lambda t: 0.85*pg(t)/p-0.006573/0.018/(0.006573/0.018+(1-0.006573)/0.02885),273.15).x

# air parameters 10°C
eta_l,rho_l,cp_l,lambda_l=14.25e-6*1.23,1.23,1007.0,0.02525 # Luft 10°C
pr_l=eta_l*cp_l/lambda_l
delta_h_lv_h2o=45000.0 # J/mol

# metal parameters Al/Cu tubes
lambda_r=221 # W/m/K

# refrigerant conditions
mm_r=0.08647 # kg/mol
tr1,tr2,tru=0+273.15,6+273.15,30+273.15

# hex params

# 12.3.2 Luftkiihler mit Feuchtigkeitsausscheidung, Kiltemitteleinspritzbetrieb
da=0.0162 # outer tube diameter
tw=0.0008
di=da-2*tw # inner tube diameter
l=0.9 # tube length
tr=1/133.3333 # fin pitch
delta_r=0.0003 # fin thickness
sq,zq=1/20,12#0.028,36 # parallel tube distance, number
sl,zl=1/23.09,6#0.020,50 # serial tube distance, number
zp=2 # refrigerant-side parallel tube number "Anzahl der kältemittelseitig parallelgeschalteten Rohrstänge"
ss=1/20 # Rohrteilung Sechseck

def get_hex(p=p,v_dot_air_in_total_to_process=v_dot_air_in_total_to_process,t0=t0,rh_0=rh_0,q_dot_cond=q_dot_cond,
    eta_l=eta_l,rho_l=rho_l,cp_l=cp_l,lambda_l=lambda_l,pr_l=pr_l,delta_h_lv_h2o=delta_h_lv_h2o,lambda_r=lambda_r,
    tr1=tr1,tr2=tr2,tru=tru,mm_r=mm_r,yf_idx=-3,
    da=da,tw=tw,di=di,l=l,tr=tr,delta_r=delta_r,sq=sq,zq=zq,sl=sl,zl=zl,zp=zp,ss=ss,
    use_alpha_i_r22=False,gamma=57.0/48.0,#2.2/1.6,
    i_it=4,j_it=4,k_it=2):

    yw_0=rh_0*pg(t0)/p
    yw_0_mass=yw_0*0.018/(yw_0*0.018+(1-yw_0)*0.02885)

    di=da-2*tw

    rho_r_u,h_r_u,cp_r_u,eta_r_u,lambda_r_u,pr_r_u=[interp(tru,datlv[:,0],datlv[:,j]) for j in [2,4,7,9,11,13]]
    rho_r_l,h_r_l,cp_r_l,eta_r_l,lambda_r_l,pr_r_l=[interp(tr1,datlv[:,0],datlv[:,j]) for j in [2,4,7,9,11,13]]
    rho_r_g,h_r_g,cp_r_g,eta_r_g,lambda_r_g,pr_r_g=[interp(tr1,datlv[:,0],datlv[:,j]) for j in [3,5,8,10,12,14]]
    rho_r_2,h_r_2,cp_r_2,eta_r_2,lambda_r_2,pr_r_2=[interp(tr2,datlv[:,0],datlv[:,j]) for j in [3,5,8,10,12,14]]
    pr_r_l,pr_r_g,pr_r_2=eta_r_l*cp_r_l/lambda_r_l,eta_r_g*cp_r_g/lambda_r_g,eta_r_2*cp_r_2/lambda_r_2
    sigma_r=interp(tr1,datlv[:,0],datlv[:,15])
    f_factor=interp(tr1-273.15,yf[0,:],yf[yf_idx,:])

    # fin & tube HEX
    delta_t_lm=((t0-tr2)-(tsat_yw_0-tr1))/log((t0-tr2)/(tsat_yw_0-tr1)) # Gegenstrom mit Überhitzung

    #Grenzzustand Gl. (22) Gegenstrom
    th_smg=tsat_yw_0-273.15 # th_ref=0°C

    # doi.org/10.1007/978-3-642-82522-4_13
    sl0=sq*zq*l # 0.54 # Luftseitiger Anströmquerschnitt

    # 1. Kenngrößen des Lamellenrohrbündels
    ag=pi*da*(1-delta_r/tr)*l*zq*zl # Glattrohroberfläche
    ar=(2*sq*sl-pi/2*da**2)*l/tr*zq*zl # Rippenoberfläche
    aa=(2*sq*sl+pi*da*(tr-delta_r-da/2))*l/tr*zq*zl # äußere Oberfläche
    ai=pi*di*l*zq*zl # innere Oberfläche
    v=l*sq*sl*zq*zl # Rohrbündelvolumen
    psi=1-delta_r/tr-pi*da**2/(4*sq*sl)*(1-delta_r/tr) # Hohlraumanteil
    dae=4*v*psi/aa # äquivalenter Durchmesser
    faf=1+0.7/psi**1.5*(sl/sq-0.3)/(sl/sq+0.7)**2 # Rohranordnungsfaktor (fluchtende Rohre)
    sle_sl0=1-da/sq-delta_r/tr*(sq-da)/sq # Verhältnis von engstem Querschnitt zum Anströmquerschnitt
    wl0=(v_dot_air_in_total_to_process)/(l*sq*zq) # Anströmgeschwindigkeit
    wlm=wl0/psi
    wle=wl0/sle_sl0 # Geschwindigkeit am engsten Querschnitt (bei Rippen, nicht Lamellen)
    re_dae=wlm*dae*p/(R*t0)*(yw_0*0.018+(1-yw_0)*0.02885)/eta_l
    nu_dae=0.31*re_dae**0.625*pr_l**(1/3)*(dae/sl)**(1/3) # Erzwungene Konvektion, Gl. 12.42
    alpha_a=nu_dae*lambda_l/dae
    # 2. Rippenwirkungsgrad
    rho_r=1.28*sq/da*(max(sl,sq)/min(sl,sq)-0.2)**0.5 # Rechteckrippe
    rho_s=1.27*sq/da*(ss/sq-0.3)**0.5 # Sechseckrippe
    rho=rho_s
    hw=da/2*(rho-1)*(1+0.35*log(rho)) # wirksame Rippenhöhe
    alpha_r=alpha_a.copy() # Gl. 12.26 einheitlicher alpha_a
    x=(2*alpha_r/(delta_r*lambda_r))**0.5*hw
    eta_r_h=tanh(x)/x  # Rippenwirkungsgrad eta_r=(tl-tr)/(tl-tg) # Gl. 12.27
    # 3. Scheinbarer Wärmeübergangskoeffizient (a außen, s scheinbar)
    alpha_as=alpha_a*(ag/aa+eta_r_h*ar/aa)

    q_dot_cond_list=[array([q_dot_cond])]
    for k_idx in range(k_it):
        # 4. Kältemittelmassenstrom
        h_r_2=h_r_g+cp_r_g*(tr2-tr1) # Überhitzung
        h_r_1=h_r_u # T nach Expansionsventil gegeben
        pr1,pr2,pru=interp(tr1,datlv[:,0],datlv[:,1]),interp(tr1,datlv[:,0],datlv[:,1]),interp(tru,datlv[:,0],datlv[:,1])
        x_lv_1=1-(h_r_1-h_r_l)/(h_r_g-h_r_l) # kg Flüssigkeit/kg Mischung
        m_dot_r_total_to_process=q_dot_cond/(h_r_2-h_r_1) #(m_dot_nh3/3600*n)
        m_dot_air_in_total_to_process=v_dot_air_in_total_to_process*rho_l
        w_dot=gamma/(gamma-1)*R*tr1*((pru/pr1)**((gamma-1)/gamma)-1) * m_dot_r_total_to_process/mm_r
        h_r_3=h_r_2+gamma/(gamma-1)*R*tr1*((pru/pr1)**((gamma-1)/gamma)-1)/mm_r # after comp.
        q_dot_cool=m_dot_r_total_to_process*(h_r_3-h_r_u)
        # 6. Massenstromdichte
        m_dot_r=m_dot_r_total_to_process / zp /(pi/4*di**2) # parallel refrigerant side mass flux
        m_dot_l=m_dot_air_in_total_to_process / aa # air mass flux ref. to outer apparent area
        # Kältemittel-Überhitzungszone
        # 5. Wärmestrom in der Überhitzungszone
        q_dot_h=m_dot_r_total_to_process * (h_r_2-h_r_g) # 0.5°C Überhitzung
        # 7. Wärmeübergangskoeffizient in der Überhitzungszone
        re_r_g=m_dot_r*di/eta_r_g # Gl. 12.79
        nu_i_h=0.0214*(re_r_g**0.8-100)*pr_r_g**0.4*(1+(di/l)**(2/3))
        alpha_i_h=0.0214*lambda_r_g/di*(re_r_g**0.8-100)*pr_r_g**0.4*(1+(di/l)**(2/3)) # Gl. 12.79
        # 8. Wärmedurchgangskoeffizient in der Überhitzungszone
        kh=1/(aa/ai*(1/alpha_i_h+0+0)+1/alpha_as)# Wärmedurchgangskoeffizient in der Überhitzungszone
        # 9. Lufttemperatur am Übergang von Überhitzungs- zu Verdampfungszone
        tlh=t0-q_dot_h/(m_dot_air_in_total_to_process*cp_l) # Lufttemperatur am Übergang von Überhitzung zu Verdampfungszone
        # 10. Mittlerer Temperaturabstand in der Überhitzungszone
        delta_t_lm_h=((t0-tr2)-(tsat_yw_0-tr1))/log((t0-tr2)/(tsat_yw_0-tr1)) # bei Gegenstrom in der Überhitzungszone
        # 11. äußere Oberfläche der Überhitzungszone
        aah=q_dot_h/(kh*delta_t_lm_h)
        # 12. äußere Oberfläche der Verdampfungszone
        aas=aa-aah

        tl_m_s_list=[tlh.copy()]
        for j_idx in range(j_it):
            # 13. Schätzwert für die mittlere Temperatur des Grundrohrs in der Verdampfungszone
            if j_idx==0: tl_m_s,eta_r_s,k_s,alpha_as_s,tr_m_s=tlh.copy(),eta_r_h.copy(),0,0,0
            tg_m_s=tl_m_s-k_s/(alpha_as_s)*(tl_m_s-tr_m_s) if j_idx>0 else 1/2*(tr1+tr2)-1
            # 14. Sättigungsfeuchte bei tg_m_s
            yw_tg_m_s=pg(tg_m_s)/p
            yw_tg_m_s_mass=yw_tg_m_s*0.018/(yw_tg_m_s*0.018+(1-yw_tg_m_s)*0.02885)
            # 15. Verhältnis des gesamten Wärmestroms zum sensiblen Wärmestrom für die Verdampfungszone
            q_dot_g_q_dot_s_g_s=1+delta_h_lv_h2o/0.018*(yw_0_mass-yw_tg_m_s_mass)/(cp_l*(tlh-tg_m_s))
            # 16. Gesamter Wärmeübergangskoeffizient am Grundrohr mit r=0.81
            alpha_g_g_s=alpha_a*q_dot_g_q_dot_s_g_s**0.81 # Gl 12.59, 12.70. G Grundrohg, g gesamt, s, scheinbar
            # 17. Rippentemperatur abschätzen. Fur tL,m,s wird zunachst tl' und für etaR,s wird etaR,h angenommen

            eta_r_s_list=[]
            for i_idx in range(i_it): # 17-21 wiederholen
                tr_m_s=tl_m_s-(tl_m_s-tg_m_s)*eta_r_s
                # 18. Feuchte bei Rippentemperatur
                yw_tr_m_s=pg(tr_m_s)/p
                yw_tr_m_s_mass=yw_tr_m_s*0.018/(yw_tr_m_s*0.018+(1-yw_tr_m_s)*0.02885)
                # 19. Verhältnis des gesamten Wärmestroms zum sensiblen Wärmestrom für die Verdampfungszone
                q_dot_g_q_dot_s_r_s=1+delta_h_lv_h2o/0.018*(yw_0_mass-yw_tr_m_s_mass)/(cp_l*(tlh-tr_m_s))
                # 20. Gesamter Wärmeübergangskoeffizient am Grundrohr mit r=0.81
                alpha_r_g_s=alpha_a*q_dot_g_q_dot_s_r_s**0.81 # Gl 12.59, 12.70. G Grundrohg, g gesamt, s, scheinbar
                # 21. Rippenwirkungsgrad
                alpha_r=alpha_r_g_s.copy() # Gl. 12.26 einheitlicher alpha_a
                x=(2*alpha_r/(delta_r*lambda_r))**0.5*hw
                eta_r_s=tanh(x)/x  # Rippenwirkungsgrad eta_r=(tl-tr)/(tl-tg) # Gl. 12.27
                eta_r_s_list+=[eta_r_s]

            # 22. Richtung der Luftzustandsänderung
            delta_x_l_delta_t_l_s=(yw_0_mass-yw_tg_m_s_mass)/(tlh-tg_m_s)*(1+ar/ag*(yw_0_mass-yw_tr_m_s_mass)/(yw_0_mass-yw_tg_m_s_mass))/(1+ar/ag*(tlh-tr_m_s)/(tlh-tg_m_s))
            # 23. Enthalpieänderung
            delta_h_l_delta_t_l_s=cp_l+delta_x_l_delta_t_l_s*delta_h_lv_h2o/0.018
            # 24. scheinbarer^2 außen-Wärmeübergangskoeffizient
            alpha_as_s=alpha_g_g_s*ag/aa+alpha_r_g_s*eta_r_s*ar/aa
            # 25. auf die Fläche bezogene Wärmestromdichte in der Verdampfungszone
            q_dot_i_s_flux=(q_dot_cond-q_dot_h)/aas*aa/ai
            # 26. Mittlerer Warmeübergangskoeffizient für Blasensieden nach Gl. (4.115) und Konvektionssieden Gl. (4.122)
            rp=1e-6 # m Glatte Rohre
            alpha_r_bs_s=0.0027*f_factor*m_dot_r**0.1*q_dot_i_s_flux**0.56*rp**0.133/di**0.3
            re_r_l,re_r_g=m_dot_r*di/eta_r_l,m_dot_r*di/eta_r_g
            xi_r_l,xi_r_g=(1.8*log(re_r_l)/log(10)-1.5)**(-2),(1.8*log(re_r_g)/log(10)-1.5)**(-2)
            nu_r_l,nu_r_g=xi_r_l/8*re_r_l*pr_r_l/(1+12.7*sqrt(xi_r_l/8)*(pr_r_l**(2/3)-1)),xi_r_g/8*re_r_g*pr_r_g/(1+12.7*sqrt(xi_r_g/8)*(pr_r_g**(2/3)-1))
            alpha_r_l0,alpha_r_g0=nu_r_l*lambda_r_l/di,nu_r_g*lambda_r_g/di
            alpha_r_k_s=alpha_r_l0*alpha_r_x_alpha_r_l(1/2*((1-x_lv_1)+1),rho_r_l,rho_r_g,alpha_r_g0,alpha_r_l0)
            x=linspace((1-x_lv_1),1,100)
            alpha_r_k_s=alpha_r_l0*1/(1-(1-x_lv_1))*sum(alpha_r_x_alpha_r_l(x,rho_r_l,rho_r_g,alpha_r_g0,alpha_r_l0)*diff(x))
            r_kr=0.3e-6
            q_dot_onb_flux=tr1*(rho_r_l-rho_r_g)/((h_r_g-h_r_l)*rho_r_l*rho_r_g)*2*sigma_r*alpha_r_l0/r_kr 
            krit_bs=q_dot_i_s_flux>q_dot_onb_flux # Blasensieden Kriterium
            alpha_i_s=(alpha_r_bs_s**3+alpha_r_k_s**3)**(1/3) if krit_bs else alpha_r_k_s.copy()
            if use_alpha_i_r22: alpha_i_s=0.67*m_dot_r**1.1*(369.3/tr1)**2.6/di**0.3 # only for R22 (lower result)
            # 27. Warmedurchgangskoeffizient für die Verdampfungszone nach Gl. (12,82)
            k_s=(aa/ai*1/alpha_i_s+1/alpha_as_s)**-1
            #aas_erf=(m_dot_l*aa)*delta_h_l_delta_t_l_s/k_s*log(1/(1+(t1-tlh)/(tlh-tg_m_s))) # erforderliche Fläche Aas abschätzen und vergleichen

            # 28. Reibungsdruckabfall im Innenrohr
            xi_r_l=0.3164/re_r_l**0.25
            dp_dl_r_l=xi_r_l*m_dot_r**2/(2*di*rho_r_l) # Gronnerud # Gl. 4.59
            o=rho_r_l/rho_r_g/(eta_r_l/eta_r_g)**0.25-1
            g=9.81 # m/s^2
            fr_r_l=m_dot_r**2/(rho_r_l**2*di*g) # Trägheit zu Schwerkraft
            f_fr=fr_r_l**0.3+0.0055*log(1/fr_r_l)**2 if fr_r_l<1 else 1
            x=1/2*(1+(1-x_lv_1)) # Mittlerer Dampfgehalt
            dp_dl_r_fr=1/(1-(1-x_lv_1))*f_fr*(1/2*(1-(1-x_lv_1))+4/2.8*(1-(1-x_lv_1)**2.8)-4/11*f_fr**0.5*(1-(1-x_lv_1)**11)) # integriert 1/(1-x)*sum(dp_dl_r_fr(x)*dx) von: f_fr*(x+4*(x**1.8-x**10*f_fr**0.5))
            phi_gd=o*dp_dl_r_fr+1
            dp_dl_r_zph=dp_dl_r_l*phi_gd
            dp_r_fr=dp_dl_r_zph*l*zq*zl/zp
            dp_r_u=1.5*dp_dl_r_zph*(pi*sq*zq*zl/zp) # Umlenkungsdruckabfall
            dp_r_b=1/2*(m_dot_r**2/rho_r_g-0**2) # Beschleunigungsdruckabfall
            dp_r=dp_r_fr+dp_r_u+dp_r_b

            # 29. Entsprechender Abfall der Verdampfungstemperatur, Verdampfungstemperatur des Kiiltemittels am Kühlereintritt
            pr2=pr1+dp_r
            delta_t_r_s=((interp(pr2,datlv[:,1],datlv[:,0]))-tr1)

            # 30. Kältemitteltemperatur Eintritt nach Druckkorrektur
            tr_m_s=tr1+delta_t_r_s/2 # T R, m , s aber keine Rippentemperatur sondern Kältemitteltemperatur

            # 31. Mittlere Luftaustrittstemperatur in der Verdampfungszone bei Hälfte der NTU
            ntu_m=k_s*aas/(m_dot_l*aa*delta_h_l_delta_t_l_s)
            tl_m_s_first=tl_m_s.copy()
            tl_m_s=tlh-(tlh-tr_m_s)*(1-exp(-1/2*ntu_m)) # nur Hälfte der NTU
            tl_m_s_list+=[tl_m_s]
            #print('\n\n deviation T_L_m_s: ',tl_m_s-k_s/(alpha_as_s)*(tl_m_s-tr_m_s)-tg_m_s,'K\n\n')

        # 32. Luftaustrittstemperatur
        tl2=tlh-(tlh-tr_m_s)*(1-exp(-ntu_m)) # ganze NTU

        # 33. Wärmestrom in der Verdampfungszone
        q_dot_s=m_dot_air_in_total_to_process * delta_h_l_delta_t_l_s * (tlh-tl2)

        # 34. Gesamt-Wärmestrom
        q_dot_cond=q_dot_s+q_dot_h
        q_dot_cond_list+=[q_dot_cond]

    # 34. Luft-Austrittsfeuchte
    yw_tl2=pg(tl2)/p
    yw_tl2_mass=yw_tl2*0.018/(yw_tl2*0.018+(1-yw_tl2)*0.02885)
    yw_tl2_mass=yw_0_mass-delta_x_l_delta_t_l_s*(tlh-tl2) # deviation of 6%

    # Druckverlust der Luftseite
    xim=10.5*re_dae**(-1/3)*(dae/sl)**(0.6)
    dp_l=xim*zl*(sl/dae)*rho_l*wlm**2/2
    dp_l_feucht=dp_l*1.2

    tl2_degC=tl2-273.15
    tl_m_s_degC=tl_m_s-273.15
    tr_m_s_degC=tr_m_s-273.15
    tr1_degC=tr1-273.15
    tr2_degC=tr2-273.15
    tru_degC=tru-273.15

    results_table=[
        ['n',1,'-'],
        ['x_lv_1',x_lv_1,'-'],
        ['q_dot_cond',q_dot_cond/1000,'kW'],
        ['m_dot_r*Aa',m_dot_r*zp*(pi/4*di**2)*3600,'kg/h'],
        ['h_r_2',h_r_2/1000,'kJ/kg'],
        ['w_dot',w_dot/1000,'kW'],
        ['q_dot_cool',q_dot_cool/1000,'kW'],
        #['x_lv_3_mid',x_lv_3_mid,'-'],
        #['q_dot_air',q_dot_air,'kW'],
        #['m_dot_additional_water',m_dot_additional_water,'kg/h'],
        #['m_dot_air_in_total_32deg_c_85pct_rh',m_dot_air_in_total_32deg_c_85pct_rh,'kg/h'],
        #['m_dot_condensate_total',m_dot_condensate_total,'kg/h'],
        ['m_dot_air_in_total_to_process',m_dot_air_in_total_to_process,'kg/h'],
        ['v_dot_air_in_total_to_process',v_dot_air_in_total_to_process,'m3/h'],
        ['h_r_1',h_r_1/1000,'kJ/kg'],
        ['h_r_g',h_r_g/1000,'kJ/kg'],
        ['h_r_2',h_r_2/1000,'kJ/kg'],
        ['h_r_3',h_r_3/1000,'kJ/kg'],
        ['h_r_u',h_r_u/1000,'kJ/kg'],
        ['l',l,'m'],
        ['eta_l',eta_l,'Pa s'],
        ['rho_l',rho_l,'kg/m3'],
        ['lambda_l',lambda_l,'W/(m K)'],
        ['pr_l',pr_l,'-'],
        ['delta_t_lm',delta_t_lm,'K'],
        ['dae',dae*1000,'mm'],
        ['A_G',ag,'m2'],
        ['A_R',ar,'m2'],
        ['A_a',aa,'m2'],
        ['A_i',ai,'m2'],
        ['psi',psi,'-'],
        ['v',v,'m^3'],
        ['zl',zl,'-'],
        ['zq',zq,'-'],
        ['zp',zp,'-'],
        ['sq*zq',sq*zq,'m'],
        ['sl*zl',sl*zl,'m'],
        ['l',l,'-'],
        ['faf',faf,'-'],
        ['wl0',wl0,'m/s'],
        ['wlm',wlm,'m/s'],
        ['wle',wlm,'m/s'],
        ['Re_dae',re_dae,'-'],
        ['sle_sl0',sle_sl0,'-'],
        ['Nu_dae',nu_dae,'-'],
        ['alpha_a (Luftseitig)',alpha_a,'W/(m^2 K)'],
        ['rho_r',rho_r,'-'],
        ['hw',hw,'m'],
        ['eta_R_h',eta_r_h,'-'],
        ['alpha_as',alpha_as,'W/(m^2 K)'],
        ['m_dot_R',m_dot_r,'kg/s/m^2'],
        ['m_dot_L',m_dot_l,'kg/s/m^2'],
        ['q_dot_h',q_dot_h,'W'],
        ['Pr_R_g',pr_r_g,'-'],
        ['Re_R_g',re_r_g,'-'],
        ['Pr_R_l',pr_r_l,'-'],
        ['Re_R_l',re_r_l,'-'],
        ['Nu_i_h',nu_i_h,'-'],
        ['alpha_i_h',alpha_i_h,'W/(m^2 K)'],
        ['kh',kh,'W/(m^2 K)'],
        ['T_L_h',tlh-273.15,'°C'],
        ['delta_t_lm_h',delta_t_lm_h,'K'],
        ['T_G_m_s first it.',1/2*(tr1+tr2)-1-273.15,'°C'],
        ['T_G_m_s',tg_m_s-273.15,'°C'],
        ['yw_t_g_m_s_mass',yw_tg_m_s_mass,'kg/kg'],
        ['T_R_m_s',tr_m_s-273.15,'°C'],
        ['yw_tr_m_s_mass',yw_tr_m_s_mass,'kg/kg'],
        ['Aah',aah,'m^2'],
        ['Aas',aas,'m^2'],
        ['(Q_dot_g/Q_dot_s)_G_s',q_dot_g_q_dot_s_g_s,'-'],
        ['alpha_G_g_s',alpha_g_g_s,'W/(m^2 K)'],
        ['(Q_dot_g/Q_dot_s)_R_s',q_dot_g_q_dot_s_r_s,'-'],
        ['alpha_R_g_s',alpha_r_g_s,'W/(m^2 K)'],
        ['eta_R_s',eta_r_s,'-'],
        ['delta_x_l_delta_t_l_s',delta_x_l_delta_t_l_s,'1/K'],
        ['delta_h_l_delta_t_l_s',delta_h_l_delta_t_l_s,'J/(kg K)'],
        ['alpha_as_s',alpha_as_s,'W/(m^2 K)'],
        ['q_dot_i_s_flux',q_dot_i_s_flux,'W/m^2'],
        ['q_dot_onb_flux',q_dot_onb_flux,'W/m^2'],
        ['alpha_bs',alpha_r_bs_s,'W/(m^2 K)'],
        ['alpha_r_k_s',alpha_r_k_s,'W/(m^2 K)'],
        ['krit_bs',krit_bs,'-'],
        ['alpha_i_s',alpha_i_s,'W/(m^2 K)'],
        ['k_s',k_s,'W/(m^2 K)'],
        ['xi_R_l',xi_r_l,'-'],
        ['dp_dl_R_l',dp_dl_r_l,'Pa/m'],
        ['O',o,'-'],
        ['Fr_R_l',fr_r_l,'-'],
        ['f_fr',f_fr,'-'],
        ['dp_dl_R_fr',dp_dl_r_fr,'Pa/m'],
        ['phi_gd',phi_gd,'-'],
        ['dp_dl_R_zph',dp_dl_r_zph,'Pa/m'],
        ['dp_R_fr',dp_r_fr/1000,'kPa'],
        ['dp_R_U',dp_r_u/1000,'kPa'],
        ['dp_R_B',dp_r_b/1000,'kPa'],
        ['dp_R',dp_r/1000,'kPa'],
        ['delta_T_R_s',delta_t_r_s,'K'],
        ['T_R_1',tr1-273.15,'°C'],
        #['T_R_1_m_s_corr',tr1_r_m_s_corr-273.15,'°C'],
        ['NTU',ntu_m,'-'],
        ['T_L_m_s_first',tl_m_s_first-273.15,'°C'],
        ['T_L_m_s',tl_m_s-273.15,'°C'],
        #['Aas erforderlich (<<-->>Aas?)',aas_erf,'m^2'],
        #['Aas_erf/Aas',aas_erf/aas,'-'],
        ['T_L_2',tl2-273.15,'°C'],
        ['q_dot_s',q_dot_s/1000,'kW'],
        ['q_dot_cond',q_dot_cond/1000,'kW'],
        ['yw_tl2_mass',yw_tl2_mass,'kg/kg'],
        ['xim',xim,'-'],
        ['dp_l',dp_l,'Pa'],
        ['dp_feucht',dp_l_feucht,'Pa'],
    ]

    return {x:locals()[x] for x in locals().keys()}

results=get_hex(use_alpha_i_r22=True)
results_table=results['results_table']
rho_r_l,rho_r_g,alpha_r_g0,alpha_r_l0,f_fr,x_lv_1=[results[x] for x in 'rho_r_l,rho_r_g,alpha_r_g0,alpha_r_l0,f_fr,x_lv_1'.split(',')]
print('array vars check: '+'\t'.join([x[0] for x in results_table if isinstance(x[1],type(array([])))]))
print(tabulate(results_table,headers=['','val','units']))

fig,ax_list=plt.subplots(1,2,constrained_layout=True)
x=linspace(0,1,100)
ax_list[0].plot(x,alpha_r_x_alpha_r_l(x,rho_r_l,rho_r_g,alpha_r_g0,alpha_r_l0))
ax_list[1].plot(x,(x+4*(x**1.8-x**10*f_fr**0.5)))
x=linspace(0.15,1,100)
ax_list[0].axhline(1/(1-0.15)*sum(alpha_r_x_alpha_r_l(x,rho_r_l,rho_r_g,alpha_r_g0,alpha_r_l0)*diff(x)),color='black')
ax_list[1].axhline(1/(1-0.15)*sum((x+4*(x**1.8-x**10*f_fr**0.5))*diff(x)),color='black')

for j in range(len(ax_list)):
    ax_list[j].axvline(1-x_lv_1,linestyle='--',color='black')
    ax_list[j].axvline(1,linestyle='--',color='black')
    ax_list[j].set_xlabel(r'$\dot{x}$ / -')
ax_list[0].set_ylabel(r'$\alpha_k(z)/\alpha_{l,0}$ / -')
ax_list[1].set_ylabel(r'$f_{Fr}$ / -')

# %% [markdown]
# # base case variations

# %%
vars_out=('dp_r,dp_l,eta_r_s_list,tl_m_s_list,tl2_degC,tl_m_s_degC,tr_m_s_degC,v,q_dot_cond,sl,sq,tr,tr1_degC,tr2_degC,'+
    'q_dot_cool,w_dot,pru,pr1,pr2,q_dot_cond_list,yw_tl2,q_dot_s,q_dot_h,l,wle,x_lv_1').split(',')
k_it=5
interact(lambda da,zp,zl,zq,tr:print(tabulate([x for x in [[key,value.flatten().item() if isinstance(value,type(array([]))) else array(value).flatten().tolist()]
                                   for key,value in get_hex(yf_idx=-1,da=da,zp=zp,zl=zl,sl=sl,zq=zq,sq=sq,
                                                            eta_l=eta_l,rho_l=rho_l,cp_l=cp_l,lambda_l=lambda_l,pr_l=pr_l,delta_h_lv_h2o=delta_h_lv_h2o,lambda_r=lambda_r,
                                                            l=l,tr=tr,delta_r=delta_r,ss=ss,
                                                            v_dot_air_in_total_to_process=v_dot_air_in_total_to_process,t0=t0,rh_0=rh_0,q_dot_cond=q_dot_cond,
                                                            tr1=tr1,tr2=tr2,tru=tru,mm_r=mm_r,
                                                            k_it=k_it).items() if key in 
                                               vars_out
                                              ]])),
         da=FloatSlider(value=da,min=0.009,max=0.049,step=0.001),zp=IntSlider(value=zp,min=1,step=1),
         zl=IntSlider(value=zl,min=1,max=60,step=1),zq=IntSlider(value=zq,min=1,max=50,step=1),
         tr=FloatSlider(value=tr,min=1/500,max=1/50,step=1/500/100),
         k_it=IntSlider(value=k_it,min=1,max=20,step=1),
         #tru=FloatSlider(value=tru,min=273,max=330,step=0.1)
        );

# %% [markdown]
# # case # 2 variations
#
# * Air in 8270 kg/h, 32°C, 85%RH
# * Ammonia in 6°C, 6 bar, 85% liquid, 0.5°C superheat

# %%
v_dot_air_in_total_to_process=20*413.5/1.13/3600 # m^3/h
rh_0=0.85
t0=32+273.15
q_dot_cond=108000#7000 # W

da=0.0162 # outer tube diameter
di=da-2*0.0008 # inner tube diameter
l=0.8 # tube length
tr=1/100#1/72 # 1/155 # fin pitch
delta_r=0.0003 # fin thickness
sq,zq=1/20,20#1/12,12#0.028,36 # parallel tube distance, number
sl,zl=1/20,20#1/18,18#0.020,50 # serial tube distance, number
zp=20 # refrigerant-side parallel tube number "Anzahl der kältemittelseitig parallelgeschalteten Rohrstänge"
ss=1/20 # Rohrteilung Sechseck

# air parameters 32°C 85%RH
eta_l,rho_l,cp_l,lambda_l=1.82273997e-05,1.13490122,1033.38045965,0.02548685 # Luft 32°C
pr_l=eta_l*cp_l/lambda_l
delta_h_lv_h2o=42939.61665608 # J/mol

# Ref. NH3 parameters VDI-WA D3 (S. 470) Ammoniak, NH3 (Tc = 405,5 K, pc = 11,353 MPa, ρc = 234,7 kg/m3)
t="""
195,5 60,9 732,9 0,064 −143,1 1341,2 1484,3 4,202 2,063 559,0 6,84 nan nan nan nan 43,9 nan
239,75 101 682,1 0,886 48,4 1418,2 1369,8 4,448 2,296 256,4 8,05 613,7 17,53 1,86 1,05 33,9 1,76
270 381 642,9 3,087 185,4 1458,8 1273,4 4,599 2,634 176,1 8,96 544,3 21,34 1,49 1,11 26,9 2,17
290 774 614,8 6,074 278,8 1477,8 1199,0 4,722 2,967 142,7 9,58 499,6 24,78 1,35 1,15 22,4 2,63
310 1424 584,5 11,019 375,1 1489,0 1112,9 4,897 3,423 117,5 10,22 454,9 28,67 1,26 1,22 18,0 3,11
330 2421 550,9 18,983 475,6 1490,2 1014,6 5,176 4,078 97,32 10,93 409,8 33,19 1,23 1,34 13,7 4,03
350 3866 512,4 31,334 582,6 1477,9 895,30 5,671 5,125 80,43 11,79 364,3 39,14 1,25 1,54 9,60 5,36
370 5878 465,3 51,729 700,8 1444,7 743,90 6,715 7,214 65,49 13,05 318,7 48,20 1,38 1,95 5,74 8,30
390 8605 399,6 90,224 842,7 1369,9 527,15 10,305 14,192 50,88 15,53 273,4 64,74 1,92 3,40 2,21 17,63
400 10305 344,6 131,09 941,8 1288,8 346,94 22,728 34,924 42,02 18,45 251,2 81,52 3,80 7,90 0,68 41,72
""" # Ts, K; ps, kPa; ρ′, kg/m3; ρ′′, kg/m3; h′, kJ/kg; h′′, kJ/kg; Δhv, kJ/kg; cp′ , kJ/ (kg K); cp′′, kJ/ (kg K); η′, 10−6 kg/ (m s); η′′, 10−6 kg/ (m s); λ′, 10−3 W/(m K); λ′′, 10−3 W/(m K) Pr′ Pr′′; σ, 10−3 N/m; β, 10−3 /K
datlv=array([x.replace(',','.').replace('−','-').split(' ') for x in t.split('\n') if len(x)>0],dtype=float)*array([1,1e3,1,1,1e3,1e3,1e3,1e3,1e3,1e-6,1e-6,1e-3,1e-3,1,1,1e-3,1e-3])

# refrigerant conditions
mm_r=0.017 # kg/mol
tr1,tr2,tru=6+273.15,6.5+273.15,44+273.15

vars_out=('dp_r,dp_l,eta_r_s_list,tl_m_s_list,tl2_degC,tl_m_s_degC,tr_m_s_degC,v,q_dot_cond,sl,sq,tr,tr1_degC,tr2_degC,'+
    'q_dot_cool,w_dot,pru,pr1,pr2,q_dot_cond_list,yw_tl2,q_dot_s,q_dot_h,l,wle,x_lv_1,ntu_m').split(',')
k_it=5 # start with 6 iterations

tr_init=least_squares(lambda tr:(17.48-get_hex(yf_idx=-1,da=da,zp=zp,zl=zl,sl=0*sl+0.8/zl,zq=zq,sq=0*sq+0.8/zq,
                                                            eta_l=eta_l,rho_l=rho_l,cp_l=cp_l,lambda_l=lambda_l,pr_l=pr_l,delta_h_lv_h2o=delta_h_lv_h2o,lambda_r=lambda_r,
                                                            l=l,tr=tr,delta_r=delta_r,ss=ss,
                                                            v_dot_air_in_total_to_process=v_dot_air_in_total_to_process,t0=t0,rh_0=rh_0,q_dot_cond=q_dot_cond,
                                                            tr1=tr1,tr2=tr2,tru=tru,mm_r=mm_r,
                                                            k_it=k_it,gamma=57.0/48.0)['tl2_degC'])**2,tr,gtol=1e-7,verbose=0).x

interact(lambda da,zp,zl,zq,tr:print(tabulate([x for x in [[key,value.flatten().item() if isinstance(value,type(array([]))) else array(value).flatten().tolist()]
                                   for key,value in get_hex(yf_idx=-1,da=da,zp=zp,zl=zl,sl=0*sl+0.8/zl,zq=zq,sq=0*sq+0.8/zq,
                                                            eta_l=eta_l,rho_l=rho_l,cp_l=cp_l,lambda_l=lambda_l,pr_l=pr_l,delta_h_lv_h2o=delta_h_lv_h2o,lambda_r=lambda_r,
                                                            l=l,tr=tr,delta_r=delta_r,ss=ss,
                                                            v_dot_air_in_total_to_process=v_dot_air_in_total_to_process,t0=t0,rh_0=rh_0,q_dot_cond=q_dot_cond,
                                                            tr1=tr1,tr2=tr2,tru=tru,mm_r=mm_r,
                                                            k_it=k_it,gamma=57.0/48.0).items() if key in 
                                               vars_out
                                              ]])),
         da=FloatSlider(value=da,min=0.009,max=0.049,step=0.001),zp=IntSlider(value=zp,min=1,step=1),
         zl=IntSlider(value=zl,min=1,max=60,step=1),zq=IntSlider(value=zq,min=1,max=50,step=1),
         tr=FloatSlider(value=tr_init,min=1/500,max=1/50,step=1/500/100),
         #k_it=IntSlider(value=k_it,min=1,max=20,step=1),
         #tru=FloatSlider(value=tru,min=273,max=330,step=0.1)
        );
