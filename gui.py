import streamlit
from streamlit import write, markdown, sidebar, text_input
from lxml import etree
from os.path import sep
from pandas import DataFrame, to_numeric, merge
from numpy import loadtxt
import string
from re import search
from os import linesep

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
def load_data():
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
                'poling_pvpmin',
                'poling_pv_tmin',
                'poling_pvpmax',
                'poling_pv_tmax'],
            'formats': [
                           int, object, object, object, int] + [float] * 12},
        converters=conv
    ))

    poling_pv_df_eq_3 = poling_pv_df[poling_pv_df['poling_pv_eq'] == 3][
        [x for x in poling_pv_df.keys() if x not in [
            'E', 'F', 'poling_name', 'poling_formula']]
    ].rename(columns={
        'A/A/Tc': 'poling_tc', 'B/B/a': 'wagn_a', 'C/C/b': 'wagn_b',
        'Tc/c': 'wagn_c', 'to/d': 'wagn_d', 'n/Pc': 'poling_pc'})
    poling_pv_df_eq_2 = poling_pv_df[poling_pv_df['poling_pv_eq'] == 2][
        [x for x in poling_pv_df.keys() if x not in [
            'A/A/Tc', 'B/B/a', 'C/C/b', 'poling_name', 'poling_formula']]
    ].rename(columns={
        'A/A/Tc': 'p_ant_a', 'B/B/a': 'p_ant_b',
        'C/C/b': 'p_ant_c', 'Tc/c': 'poling_tc', 'to/d': 'eant_to',
        'n/Pc': 'eant_n', 'E': 'eant_e', 'F': 'eant_f'})
    poling_pv_df_eq_1 = poling_pv_df[poling_pv_df['poling_pv_eq'] == 1][
        [x for x in poling_pv_df.keys() if x not in [
            'E', 'F', 'Tc/c', 'to/d', 'n/Pc', 'poling_name', 'poling_formula']]
    ].rename(columns={
        'A/A/Tc': 'p_ant_a', 'B/B/a': 'p_ant_b', 'C/C/b': 'p_ant_c'})

    poling_pv_df = merge(merge(
        poling_pv_df_eq_1, poling_pv_df_eq_2,
        how='outer', on=['poling_no', 'cas_no'], suffixes=['_1', '_2']),
          poling_pv_df_eq_3, how='outer', on=['poling_no', 'cas_no'],
        suffixes=['', '_3'])

    poling_pv_df.update(
        poling_pv_df.rename(
            columns={'poling_tc': 'poling_tc_4',
                     'poling_tc_3': 'poling_tc',
                     'poling_pc': 'poling_pc_4',
                     'poling_pc_3': 'poling_pc'}))
    poling_pv_df.update(
        poling_pv_df.rename(columns={
            'poling_tc': 'poling_tc_4',
            'poling_tc_3': 'poling_tc'}))

    poling_df = merge(merge(    
        poling_basic_i_df, poling_basic_ii_df,
        how='outer', on=['cas_no', 'poling_no']),
        poling_cp_ig_l_df, how='outer', on='cas_no')

    del poling_df['poling_no_y']
    poling_df = poling_df.rename(columns={'poling_no_x': 'poling_no'})
    poling_df = merge(poling_df, poling_pv_df, on=['cas_no', 'poling_no'], how='outer')

    poling_df = poling_df.rename(
        columns={'poling_pc_x': 'poling_pc', 'poling_tc_x': 'poling_tc',
                 'poling_pv_tmin_1': 'p_ant_tmin',
                 'poling_pv_tmax_1': 'p_ant_tmax',
                 'poling_pvpmin_1': 'p_ant_pvpmin',
                 'poling_pvpmax_1': 'p_ant_pvpmax',
                 'poling_pv_tmin_2': 'eant_tmin',
                 'poling_pv_tmax_2': 'eant_tmax',
                 'poling_pvpmin_2': 'eant_pvpmin',
                 'poling_pvpmax_2': 'eant_pvpmax',
                 'poling_pv_tmin': 'wagn_tmin',
                 'poling_pv_tmax': 'wagn_tmax',
                 'poling_pvpmin': 'wagn_pvpmin',
                 'poling_pvpmax': 'wagn_pvpmax'
                 })
    poling_df.update(poling_df.rename(
        columns={'poling_pc': 'poling_pc_x', 'poling_pc_y': 'poling_pc'}))

    del poling_df['poling_tc_y']
    del poling_df['poling_pc_y']
    del poling_df['poling_tc_3']
    del poling_df['poling_pv_eq_1']
    del poling_df['poling_pv_eq_2']
    del poling_df['poling_pv_eq']

    poling_burcat_df = merge(burcat_df, poling_df, on='cas_no', how='outer')

    poling_burcat_ant_df = merge(poling_burcat_df, ant_df, on='cas_no', how='outer')

    for j in range(4):
        poling_burcat_ant_df[f'poling_a{j+1}'] = poling_burcat_ant_df[f'poling_a{j+1}']*[1e-3,1e-5,1e-8,1e-11][j]

    return poling_burcat_ant_df

column_names = [
    'cas_no', 'phase', 'formula',
    'formula_name_structure', 'ant_name',
    'poling_no', 'poling_formula', 'poling_name',
    'poling_molwt', 'poling_tfp',
    'poling_tb', 'poling_tc', 'poling_pc',
    'poling_vc', 'poling_zc', 'poling_omega',
    'poling_delhf0', 'poling_delgf0', 'poling_delhb',
    'poling_delhm', 'poling_v_liq', 'poling_t_liq',
    'poling_dipole',
    'p_ant_a', 'p_ant_b', 'p_ant_c',
    'p_ant_tmin', 'p_ant_tmax',
    'p_ant_pvpmin', 'p_ant_pvpmax',
    'eant_to', 'eant_n',
    'eant_e', 'eant_f',
    'eant_tmin', 'eant_tmax',
    'eant_pvpmin', 'eant_pvpmax',
    'wagn_a', 'wagn_b',
    'wagn_c', 'wagn_d',
    'wagn_tmin', 'wagn_tmax',
    'wagn_pvpmin', 'wagn_pvpmax',
    'range_tmin_to_1000',
    'range_1000_to_tmax', 'molecular_weight',
    'hf298_div_r'] + [
    'a' + str(i) + '_low' for i in range(1, 7 + 1)] + [
    'a' + str(i) + '_high' for i in range(1, 7 + 1)
] + [
    'reference', 'source', 'date',
    'ant_no', 'ant_formula',
    'ant_a', 'ant_b',
    'ant_c', 'ant_tmin', 'ant_tmax',
    'ant_code'
]

column_dtypes = [
    str, str, str,
    str, str,
    float, str, str,
    float, float,
    float, float, float,
    float, float, float,
    float, float, float,
    float, float, float,
    float,
    float, float, float,
    float, float,
    float, float,
    float, float,
    float, float,
    float, float,
    float, float,
    float, float,
    float, float,
    float, float,
    float, float,
    float,
    float, float,
    float] + [
    float] * 7 + [float] * 7 + [
    str, str, str,
    float, str,
    float, float,
    float, float, float,
    str
]

column_units = [
    '', '', '',
    '', '',
    '', '', '',
    'g/mol', 'K',
    'K', 'K', 'bar',
    'cm3/mol', '-', '-',
    'kJ/mol', 'kJ/mol', 'kJ/mol',
    'kJ/mol', 'cm3/mol', 'K',
    'Debye',
    '-', 'K', 'K',
    'K', 'K', 
    'bar', 'bar',
    'K', '-', 
    '-', '-',
    'K', 'K', 
    'bar', 'bar',
    '-', '-', 
    '-', '-',
    'K', 'K', 
    'bar', 'bar',
    'K',
    'K', 'g/mol',
    'K',
    '-', 'K^-1', 'K^-2', 'K^-3', 'K^-4',
    'K^-1', '-',
    '-', 'K^-1', 'K^-2', 'K^-3', 'K^-4',
    'K^-1', '-',
    '', '', '',
    '', '', '',
    '', '', 
    '', '°C', '°C',
    ''
]

df = load_data()[column_names]
df = df[df['cas_no'].str.contains(cas_filter, na=False, case=False)]
df = df[df['formula'].str.contains(formula_filter, na=False, case=False)]
df = df[df['formula_name_structure'].str.contains(name_filter, na=False, case=False)]
df = df[df['phase'].str.contains(phase_filter, na=False, case=False)]

write(df)

write('filtered to', str(len(df)), 'records')
