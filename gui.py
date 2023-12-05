import streamlit
from streamlit import write, markdown, sidebar, text_input, data_editor, column_config, session_state
from lxml import etree
from os.path import sep
from pandas import DataFrame, to_numeric, merge, concat
from numpy import loadtxt, array, where, log, empty_like, ones, linspace, exp, log10
from z_l_v import State
import string
from re import search
from os import linesep

r = 8.3145 # Pa m^3/mol
t_ref = 298.15 # K

data_path = sep.join(['data', 'BURCAT_THR.xml'])
template_path = sep.join(['data', 'xsl_stylesheet_burcat.xsl'])
antoine_csv = './data/The-Yaws-Handbook-of-Vapor-Pressure-Second-Edition-Antoine-coefficients.csv'
poling_basic_i_csv = './data/basic_constants_i_properties_of_gases_and_liquids.csv'
poling_basic_ii_csv = './data/basic_constants_ii_properties_of_gases_and_liquids.csv'
poling_cp_l_ig_poly_csv = './data/ig_l_heat_capacities_properties_of_gases_and_liquids.csv'
poling_pv_csv = './data/vapor_pressure_correlations_parameters_clean.csv'
linesep_b = linesep.encode('utf-8')

markdown('# thr')
sidebar.markdown('# thr')
sidebar.write('filter by:')
cas_filter = sidebar.text_input(label='cas', key='cas', placeholder='cas', help='type cas no.')
name_filter = sidebar.text_input(label='name', key='name', placeholder='name', help='type name')
formula_filter = sidebar.text_input(label='formula', key='formula', placeholder='formula', help='type formula')
phase_filter = sidebar.selectbox('phase',['','G','L','S','C'])
sidebar.markdown('# composition')
col1, col2, col3, col4 = sidebar.columns(4)
col1.text('', help='clear non-selected from composition')
col3.text('', help='reset composition to default')
col4.text('', help='plot')
temp_input = col2.number_input(label='T_ref/K', key='T_ref', placeholder='1000 K', value=1000.0, min_value=0.0, max_value=6000.0, help='Reference temperature for diyplay of h_ig, s_ig, g_ig, cp_ig')

def helper_func1(x):
    # helper function for csv reading
    # replace dot in original file for minus in some cases ocr'd like .17.896
    str_with_dot_actually_minus = x
    matches = search(b'(\.(\d+\.\d+))', x)
    if matches:
        str_with_dot_actually_minus = b'-' + matches.groups()[1]
    return str_with_dot_actually_minus.replace(b' ', b'')


def helper_func2(x):
    # helper function 2 for csv reading.
    # Return None when empty string
    if len(x) == 0 or x == linesep_b:
        return b'nan'
    return x.replace(b' ', b'')


def helper_func3(x):
    # helper function 3 for csv reading.
    # replace - for . in cas numbers (Antoine coeff)
    return x.replace(b'.', b'-').decode('utf-8')


def helper_func4(x):
    # helper function 4 for csv reading.
    # replace whitespace ' ' for '' in float numbers
    return x.replace(b' ', b'')

@streamlit.cache_data
def load_data(cas_filter='', name_filter='', formula_filter='', phase_filter=''):
    tree = etree.parse(data_path)
    root = tree.getroot()
    xsl = etree.parse(template_path)
    transformer = etree.XSLT(xsl)
    result = transformer(tree)

    xpath = \
                "./specie[contains(@CAS, '" + \
                cas_filter + "')]/" + \
                "formula_name_structure[contains(translate(" +\
                "formula_name_structure_1" + \
                ", '" + string.ascii_lowercase + "', '" + \
                string.ascii_uppercase + "'), '" + \
                name_filter + "')]/../@CAS/../" + \
                "phase[contains(translate(" + \
                "formula" + \
                ", '" + string.ascii_lowercase + "', '" + \
                string.ascii_uppercase + "'), '" + \
                formula_filter + "')]/../" + \
                "phase[contains(phase, '" + \
                phase_filter + "')]"


    for element in root.xpath(xpath):
        #print(element.getparent().get('CAS'))
        pass

    data = []
    for i in result.xpath('/*'):
        inner = {}
        for j in i.xpath('*'):
            inner[j.tag] = j.text
        data.append(inner)


    burcat_df = DataFrame(data)

    for column_name in [
            'a'+str(i)+'_low' for i in range(1,7+1) ] + [
            'a'+str(i)+'_high' for i in range(1,7+1)] + [
            'hf298_div_r', 'molecular_weight', 'range_1000_to_tmax',
            'range_tmin_to_1000'
            ]:
            burcat_df[column_name] = to_numeric(
                burcat_df[column_name].str.replace(' ', '')
                )

    ant_df = DataFrame(loadtxt(
            open(antoine_csv, 'rb'),
            delimiter='|',
            skiprows=9,
            dtype={
                'names': [
                    'ant_no', 'ant_formula', 'ant_name',
                    'cas_no', 'ant_a', 'ant_b',
                    'ant_c', 'ant_tmin', 'ant_tmax',
                    'ant_code'],
                'formats': [
                    int, object, object,
                    object, float, float,
                    float, float, float, object]},
            converters={
                0: helper_func1, 3: helper_func3,
                4: helper_func1, 5: helper_func1,
                6: helper_func1, 7: helper_func1
            }))

    poling_basic_i_df = DataFrame(loadtxt(
        open(poling_basic_i_csv, 'rb'),
        delimiter='|',
        skiprows=3,
        dtype={
            'names': [
                'poling_no', 'poling_formula', 'poling_name',
                'cas_no', 'poling_molwt', 'poling_tfp',
                'poling_tb', 'poling_tc', 'poling_pc',
                'poling_vc', 'poling_zc', 'poling_omega'],
            'formats': [
                int, object, object,
                object, float, float,
                float, float, float, float, float,
                float]},
        converters={
            5: helper_func2, 6: helper_func2,
            8: helper_func2, 9: helper_func2,
            10: helper_func2, 11: helper_func2
        }
    ))

    poling_basic_ii_df = DataFrame(loadtxt(
        open(poling_basic_ii_csv, 'rb'),
        delimiter='|',
        skiprows=3,
        usecols=(3, 4, 5, 6, 7, 8, 9, 10, 0),
        dtype={
            'names': [
                'cas_no',
                'poling_delhf0', 'poling_delgf0', 'poling_delhb',
                'poling_delhm', 'poling_v_liq', 'poling_t_liq',
                'poling_dipole', 'poling_no'],
            'formats': [
                object, float, float,
                float, float, float, float, float, int]},
        converters={
            4: helper_func2, 5: helper_func2,
            6: helper_func2, 7: helper_func2,
            8: helper_func2, 9: helper_func2,
            10: helper_func2
        }
    ))


    poling_cp_ig_l_df = DataFrame(loadtxt(
        open(poling_cp_l_ig_poly_csv, 'rb'),
        delimiter='|',
        skiprows=4,
        usecols=(3, 4, 5, 6, 7, 8, 9, 10, 11, 0),
        dtype={
            'names': [
                'cas_no',
                'poling_trange', 'poling_a0', 'poling_a1',
                'poling_a2', 'poling_a3', 'poling_a4',
                'poling_cpig', 'poling_cpliq', 'poling_no'],
            'formats': [
                object, object, float,
                float, float, float, float, float, float, int]},
        converters={
            5: helper_func2, 6: helper_func2,
            7: helper_func2, 8: helper_func2,
            9: helper_func2, 10: helper_func2,
            11: helper_func2
        }
    ))

    conv = dict([[i, lambda x: helper_func4(helper_func2(x)).decode('utf-8')]
                 for i in [0, 1, 2, 4, 5,
                           ]+[8, 9, 10, 11, 12, 13, 14, 15, 16]])
    conv[3] = lambda x: helper_func3(x)
    conv[6] = lambda x: helper_func4(helper_func2(x)).decode('utf-8')
    conv[7] = lambda x: helper_func4(helper_func2(x)).decode('utf-8')

    poling_pv_df = DataFrame(loadtxt(
        open(poling_pv_csv, 'rb'),
        delimiter='|',
        skiprows=3,
        dtype={
            'names': [
                'poling_no',
                'poling_formula',
                'poling_name',
                'cas_no',
                'poling_pv_eq',
                'A/A/Tc',
                'B/B/a',
                'C/C/b',
                'Tc/c',
                'to/d',
                'n/Pc',
                'E',
                'F',
                'poling_pv_pmin',
                'poling_pv_tmin',
                'poling_pv_pmax',
                'poling_pv_tmax'],
            'formats': [
                           int, object, object, object, int] + [float] * 12},
        converters=conv
    ))

    poling_pv_df_eq_3 = poling_pv_df[poling_pv_df['poling_pv_eq'] == 3][
        [x for x in poling_pv_df.keys() if x not in [
            'E', 'F', 'poling_name', 'poling_formula', 'poling_pv_pmin', 'poling_pv_tmin']]
    ].rename(columns={
        'A/A/Tc': 'wagn_tc', 'B/B/a': 'wagn_a', 'C/C/b': 'wagn_b',
        'Tc/c': 'wagn_c', 'to/d': 'wagn_d', 'n/Pc': 'wagn_pc',
        'poling_pv_pmin':'wagn_pmin', 'poling_pv_pmax':'wagn_pmax',
        'poling_pv_tmin':'wagn_tmin', 'poling_pv_tmax':'wagn_tmax'})
    poling_pv_df_eq_2 = poling_pv_df[poling_pv_df['poling_pv_eq'] == 2][
        [x for x in poling_pv_df.keys() if x not in [
            'poling_name', 'poling_formula']]
    ].rename(columns={
        'A/A/Tc': 'p_ant2_a', 'B/B/a': 'p_ant2_b',
        'C/C/b': 'p_ant2_c', 'Tc/c': 'p_ant2_tc', 'to/d': 'p_ant2_to',
        'n/Pc': 'p_ant2_n', 'E': 'p_ant2_e', 'F': 'p_ant2_f',
        'poling_pv_pmin':'p_ant2_pmin', 'poling_pv_pmax':'p_ant2_pmax',
        'poling_pv_tmin':'p_ant2_tmin', 'poling_pv_tmax':'p_ant2_tmax'})
    poling_pv_df_eq_1 = poling_pv_df[poling_pv_df['poling_pv_eq'] == 1][
        [x for x in poling_pv_df.keys() if x not in [
            'E', 'F', 'Tc/c', 'to/d', 'n/Pc', 'poling_name', 'poling_formula']]
    ].rename(columns={
        'A/A/Tc': 'p_ant1_a', 'B/B/a': 'p_ant1_b', 'C/C/b': 'p_ant1_c',
        'poling_pv_pmin':'p_ant1_pmin','poling_pv_pmax':'p_ant1_pmax',
        'poling_pv_tmin':'p_ant1_tmin','poling_pv_tmax':'p_ant1_tmax'})

    poling_pv_df = merge(merge(
        poling_pv_df_eq_1, poling_pv_df_eq_2,
        how='outer', on=['poling_no', 'cas_no'], suffixes=['_1', '_2']),
          poling_pv_df_eq_3, how='outer', on=['poling_no', 'cas_no'],
        suffixes=['', '_3'])

    poling_df = merge(merge(    
        poling_basic_i_df, poling_basic_ii_df,
        how='outer', on=['cas_no', 'poling_no']),
        poling_cp_ig_l_df, how='outer', on='cas_no')

    del poling_df['poling_no_y']
    poling_df = poling_df.rename(columns={'poling_no_x': 'poling_no'})
    poling_df = merge(poling_df, poling_pv_df, on=['cas_no', 'poling_no'], how='outer')

    del poling_df['poling_pv_eq']

    poling_burcat_df = merge(burcat_df, poling_df, on='cas_no', how='outer')

    poling_burcat_ant_df = merge(poling_burcat_df, ant_df, on='cas_no', how='outer')

    for j in range(4):
        poling_burcat_ant_df[f'poling_a{j+1}'] = poling_burcat_ant_df[f'poling_a{j+1}']*[1e-3,1e-5,1e-8,1e-11][j]

    df = poling_burcat_ant_df.copy()
    # estimate acentric factor where missing. Def. w=-log10(prsat)-1, at Tr=0.7
    tr = 0.7
    tau = 1-tr
    prsat_tr_0_7 = exp(
        1/tr*(df['wagn_a']*tau+df['wagn_b']*tau**1.5+df['wagn_c']*tau**2.5+df['wagn_d']*tau**5)
    )  # Wagner: ln(Pvp)=ln(Pc)+(Tc/T)*(a*tau+b*tau^1.5+c*tau^2.5+d*tau^5); tau=(1-T/Tc); (For H2O, last two terms are c*tau^3+d*tau^6)
    water_idx = df[df['poling_name'].fillna('').str.contains('water',case=False)].index
    prsat_tr_0_7[water_idx] = exp(
        1/tr*(df['wagn_a']*tau+df['wagn_b']*tau**1.5+df['wagn_c']*tau**3+df['wagn_d']*tau**6)[water_idx]
    )  # Wagner: ln(Pvp)=ln(Pc)+(Tc/T)*(a*tau+b*tau^1.5+c*tau^2.5+d*tau^5); tau=(1-T/Tc); (For H2O, last two terms are c*tau^3+d*tau^6)
    df['wagn_omega'] = -log10(prsat_tr_0_7)-1

    # test: w=df['wagn_omega']; (w[~df['wagn_a'].isna() & ~df['poling_omega'].isna()]-df.loc[~df['wagn_a'].isna() & ~df['poling_omega'].isna(),'poling_omega'])/w[~df['wagn_a'].isna() & ~df['poling_omega'].isna()]*100

    prsat_tr_0_7=10**(
            df['p_ant1_a']-df['p_ant1_b']/(tr*df['poling_tc']+df['p_ant1_c']-273.15)
    )/df['poling_pc']  # extended Antoine: log10(Pvp)=A-B/(T+C-273.15)+0.43429*x^n+E*x^8+F*x^12; x=(T-to-273.15)/Tc
    df['p_ant1_omega']=-log10(prsat_tr_0_7)-1

    # test: (df['p_ant1_omega'][~df['p_ant1_a'].isna()]-df.loc[~df['p_ant1_a'].isna(),'poling_omega'])/df['p_ant1_omega'][~df['p_ant1_a'].isna()]*100

    x = tr-df['p_ant2_to']/df['p_ant2_tc']-273.15/df['p_ant2_tc']
    prsat_tr_0_7 = 10**(
        df['p_ant2_a']-df['p_ant2_b']/(tr*df['p_ant2_tc']+df['p_ant2_c']-273.15)+0.43429*x**df['p_ant2_n']+df['p_ant2_e']*x**8+df['p_ant2_f']*x**12
    )/df['poling_pc'] # extended Antoine: log10(Pvp)=A-B/(T+C-273.15)+0.43429*x^n+E*x^8+F*x^12; x=(T-to-273.15)/Tc
    df['p_ant2_omega']=-log10(prsat_tr_0_7)-1

    # test: (df['p_ant2_omega'][~df['p_ant2_a'].isna()]-df.loc[~df['p_ant2_a'].isna(),'poling_omega'])/df['p_ant2_omega'][~df['p_ant2_a'].isna()]*100

    prsat_tr_0_7=10**(
            df['ant_a']-df['ant_b']/(tr*df['poling_tc']+df['ant_c']-273.15)
    )*1.01325/760/df['poling_pc']  # extended Antoine: log10(Pvp)=A-B/(T+C-273.15)+0.43429*x^n+E*x^8+F*x^12; x=(T-to-273.15)/Tc
    df['yaws_omega']=-log10(prsat_tr_0_7)-1

    # test: (df['yaws_omega'][~df['ant_a'].isna() & ~df['poling_omega'].isna()]-df.loc[~df['ant_a'].isna() & ~df['poling_omega'].isna(),'poling_omega'])/df['yaws_omega'][~df['ant_a'].isna() & ~df['poling_omega'].isna()]*100

    return poling_burcat_ant_df

@streamlit.cache_data
def filter_data(df, cas_filter='', name_filter='', formula_filter='', phase_filter=''):
    df_out = df.copy()
    df_out = df[df['cas_no'].str.contains(cas_filter, na=False, case=False) & 
        df['combined_formula'].str.contains(formula_filter, na=False, case=False) & 
        df['combined_name'].str.contains(name_filter, na=False, case=False) &
        df['phase'].str.contains(phase_filter, na=False, case=False)
        ]
    return df_out

def update_temp(df):
    t = temp_input

    mm_i = df['poling_molwt']/1000  # kg/mol
    tc_i = df['poling_tc']  # K
    pc_i = df['poling_pc']*1e5  # Pa
    omega_i = df['poling_omega']
    vc_i = df['poling_vc']*10**-6  # m^3/mol
    delhf0_poling = df['poling_delhf0']*1000  # J/mol
    delgf0_poling = df['poling_delgf0']*1000  # J/mol
    delsf0_poling = (delhf0_poling - delgf0_poling)/t_ref # J/mol/K
    a_low = df[['a'+str(i)+'_low' for i in range(1, 7+1)]]
    a_high = df[['a'+str(i)+'_high' for i in range(1, 7+1)]]

    if t>1000: # poly a_low is for 200 - 1000 K; a_high is for 1000 - 6000 K
        cp_r=concat([df[f'a{j+1}_high']*t**j for j in range(4+1)],axis=1).sum(axis=1)  # cp/R
        s_cp_r_dt = concat([
            1/(j+1)*df[f'a{j+1}_high']*t**(j+1)-1/(j+1)*df[f'a{j+1}_low']*t_ref**(j+1)
            for j in range(4+1)], axis=1).sum(axis=1) # int(Cp/R*dT,t_refK,T)
        s_cp_r_t_dt=df['a1_high']*log(t)+df['a7_high']+concat([
            1/(j+1)*df[f'a{j+1+1}_high']*t**(j+1)
            for j in range(2+1)],axis=1).sum(axis=1)  # int(Cp/(RT)*dT,0,T)
    else:
        cp_r = concat([df[f'a{j+1}_low']*t**j for j in range(4+1)],axis=1).sum(axis=1)  # cp/R
        s_cp_r_dt=concat([
            1/(j+1)*df[f'a{j+1}_low']*t**(j+1)-1/(j+1)*df[f'a{j+1}_low']*t_ref**(j+1)
            for j in range(4+1)],axis=1).sum(axis=1) # int(Cp/R*dT,t_refK,T)
        s_cp_r_t_dt=df['a1_low']*log(t)+df['a7_low']+concat([
            1/(j+1)*df[f'a{j+1+1}_low']*t**(j+1)
            for j in range(2+1)],axis=1).sum(axis=1)  # int(Cp/(RT)*dT,0,T)

    cp_ig=r*cp_r  # J/mol/K
    h_ig=delhf0_poling+r*s_cp_r_dt # J/mol
    s_ig=r*s_cp_r_t_dt # J/mol
    g_ig=h_ig-t*s_ig # J/mol

    df['cp_ig'] = cp_ig
    df['h_ig'] = h_ig
    df['s_ig'] = s_ig
    df['g_ig'] = g_ig

    return df

def plot(df):
    t = linspace(60, 220, 10)
    p = 1.01325  # bar
    phase_fraction = empty_like(t)
    v_l = empty_like(t)
    v_v = empty_like(t)
    z_i = df['z_i']
    # normalize z_i
    sum_z_i = sum(z_i)
    if sum_z_i <= 0:
        z_i = 1 / len(z_i) * ones(len(z_i))
        z_i = z_i.tolist()
    elif sum_z_i != 1.0:
        z_i = z_i / sum_z_i
        z_i = z_i.to_list()
    else:
        z_i = z_i.to_list()
    mm_i = (df['poling_molwt']/1000).tolist()  # kg/mol
    tc_i = df['poling_tc'].tolist()  # K
    pc_i = (df['poling_pc']).tolist()  # bar
    omega_i = df['poling_omega'].tolist()
    vc_i = (df['poling_vc']*10**-6).tolist()  # m^3/mol
    state = State(t[0], p, z_i, mm_i, tc_i, pc_i, omega_i, 'pr')

names_units_dtypes=array([
    ['cas_no','',str],
    ['phase','',str],
    ['formula','',str],
    ['formula_name_structure','',str],
    ['ant_name','',str],
    ['poling_no','',float],
    ['poling_formula','',str],
    ['poling_name','',str],
    ['poling_molwt','g/mol',float],
    ['poling_tfp','K',float],
    ['poling_tb','K',float],
    ['poling_tc','K',float],
    ['poling_pc','bar',float],
    ['poling_vc','cm3/mol',float],
    ['poling_zc','-',float],
    ['poling_omega','-',float],
    ['poling_delhf0','kJ/mol',float],
    ['poling_delgf0','kJ/mol',float],
    ['poling_delhb','kJ/mol',float],
    ['poling_delhm','kJ/mol',float],
    ['poling_v_liq','cm3/mol',float],
    ['poling_t_liq','K',float],
    ['poling_dipole','Debye',float],
    ['p_ant1_a','-',float],
    ['p_ant1_b','K',float],
    ['p_ant1_c','K',float],
    ['p_ant1_tmin','K',float],
    ['p_ant1_tmax','K',float],
    ['p_ant1_pmin','bar',float],
    ['p_ant1_pmax','bar',float],
    ['p_ant2_a','-',float],
    ['p_ant2_b','K',float],
    ['p_ant2_c','K',float],
    ['p_ant2_tc','K',float],
    ['p_ant2_to','K',float],
    ['p_ant2_n','-',float],
    ['p_ant2_e','-',float],
    ['p_ant2_f','-',float],
    ['p_ant2_tmin','K',float],
    ['p_ant2_tmax','K',float],
    ['p_ant2_pmin','bar',float],
    ['p_ant2_pmax','bar',float],
    ['wagn_a','-',float],
    ['wagn_b','-',float],
    ['wagn_c','-',float],
    ['wagn_d','-',float],
    ['wagn_tc','K',float],
    ['wagn_pc','bar',float],
    ['wagn_tmax','K',float],
    ['wagn_pmax','bar',float],
    ['range_tmin_to_1000','K',float],
    ['range_1000_to_tmax','K',float],
    ['molecular_weight','g/mol',float],
    ['hf298_div_r','K',float],
    ['a1_low','-',float],
    ['a2_low','K^-1',float],
    ['a3_low','K^-2',float],
    ['a4_low','K^-3',float],
    ['a5_low','K^-4',float],
    ['a6_low','K^-1',float],
    ['a7_low','-',float],
    ['a1_high','-',float],
    ['a2_high','K^-1',float],
    ['a3_high','K^-2',float],
    ['a4_high','K^-3',float],
    ['a5_high','K^-4',float],
    ['a6_high','K^-1',float],
    ['a7_high','-',float],
    ['reference','',str],
    ['source','',str],
    ['date','',str],
    ['ant_no','',float],
    ['ant_formula','',str],
    ['ant_a','',float],
    ['ant_b','',float],
    ['ant_c','',float],
    ['ant_tmin','',float],
    ['ant_tmax','°C',float],
    ['ant_code','°C',str]
])

ed_cols=[['z_i','h_ig','s_ig','g_ig','cp_ig'],
        ['-','J/mol','J/mol/K','J/mol','J/mol/K']]

elems=[
        ['G','1333-74-0','hydrogen','H2'],
        ['G','74-82-8','methane','CH4'],
        ['G','124-38-9','carbon dioxide','CO2'],
        ['G','630-08-0','carbon monoxide','CO'],
        ['G','7732-18-5','water','H2O'],
        ['G','7727-37-9','nitrogen','N2'],
        ['G','7782-44-7','oxygen','O2'],
        ['G','7440-37-1','argon','Ar'],
        ['G','7440-59-7','helium','He'],
        ['G','71-43-2','benzene','C6H6'],
        ['G','110-82-7','cyclohexane','C6H12'],
        ['G','110-54-3','n-hexane','C6H14'],
        ['G','67-56-1','methanol','CH3OH'],
        ['G','107-31-3','methyl formate','C2H4O2'],
        ['G','64-18-6','formic acid','CH2O2'],
        ['G','107-31-3','methyl formate','C2H4O2'],
        ['G','74-84-0','ethane','c2h6'],
        ['G','74-85-1','ethene','c2h4'],
        ['G','74-98-6','propane','c3h8'],
        ['G','115-07-1','propene','c3h6'],
        ['G','7783-06-4','','h2s'],
        ['G','7664-93-9','sulfuric acid','h2so4'],
        ['G','7446-11-9','','SO3'],
        ['G','7446-09-5','','SO2']
]
elems = [[x[j] for j in [1,2,3,0]] for x in elems if 'sulfuric acid' not in x]

df = load_data()[names_units_dtypes[:,0]]
df['combined_formula'] = df['formula_name_structure'].fillna('').astype(str)+df['formula'].fillna('').astype(str)+df['poling_formula'].fillna('').astype(str)+df['ant_formula'].fillna('').astype(str)
df['combined_name'] = df['formula_name_structure'].fillna('').astype(str)+df['ant_name'].fillna('').astype(str)+df['poling_name'].fillna('').astype(str)

df_filtered = filter_data(df, cas_filter=cas_filter, name_filter=name_filter, formula_filter=formula_filter, phase_filter=phase_filter)
if not 'Select' in df_filtered.keys():
    df_filtered.loc[:, 'Select']=False
    df_filtered=df_filtered[['Select']+[x for x in df_filtered.keys() if not x in ['Select', 'combined_name', 'combined_formula']]]

if col3.button('default') or not 'idx_sel' in session_state: 
    idx_sel = [int(filter_data(df,*x).iloc[0].name) for x in elems]
    session_state['idx_sel'] = idx_sel

df_with_selections = df_filtered.copy()
edited_df = data_editor(
        df_with_selections, hide_index=True, 
        column_config={'Select':column_config.CheckboxColumn(required=True)},
        disabled=[x for x in df.columns if x!= 'Select'],
        key='edited_df_key'
        )
selected_indices = edited_df.where(edited_df.Select).index.to_list() #list(where(edited_df.Select)[0])
selected_rows = df_filtered[edited_df.Select]

if col1.button('clear'):
    idx_sel = []
    session_state['idx_sel'] = []

if 'edited_df' in session_state:
    if len(edited_df) != len(session_state['edited_df']):
        # keep selections immediately after filtering
        idx_sel = session_state['idx_sel']
    else:
        edited_dict = session_state['edited_df_key'].get('edited_rows') # dict such as e.g. {'7':{'Select':True},'8':{'Select':False}}
        idx_deselected = [edited_df.iloc[x[0]].name for x in edited_dict.items() if not x[1]['Select']]
        #write('idx_deselected',idx_deselected)
        idx_sel = list(set(selected_rows.index.to_list()+session_state['idx_sel']))
        idx_sel = [x for x in idx_sel if not x in idx_deselected] # remove newly unselected


df_zi = df.loc[idx_sel].copy()
df_zi = df_zi[[x for x in df_zi.keys() if not x in ['combined_formula', 'combined_name']]]
for j in range(len(ed_cols[0])):
    col = ed_cols[0][j]
    if not col in df_zi.columns and col != 'z_i':
        df_zi[col] = None
df_zi = df_zi[['formula_name_structure']+[x for x in ed_cols[0] if x != 'z_i']+[x for x in df_zi.columns if not x in ed_cols[0]+['formula_name_structure']]] # order

df_zi_with_selections = df_zi.copy()
df_zi_with_selections.insert(0, 'z_i',0.0)
df_zi_with_selections = update_temp(df_zi_with_selections)
edited_df_zi = sidebar.data_editor(
        df_zi_with_selections, hide_index=True,
        column_config={'z_i':column_config.NumberColumn(required=True,min_value=0,max_value=1)},
        disabled=df_zi.columns,
        );

session_state['idx_sel'] = idx_sel
session_state['edited_df_zi'] = edited_df_zi
session_state['edited_df'] = edited_df
session_state['temp_input'] = temp_input

#write('idx_sel', idx_sel)

if True or col4.button('⟳'):
    plot(edited_df_zi)
