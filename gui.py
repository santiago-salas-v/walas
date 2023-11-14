import streamlit
from streamlit import write, markdown, sidebar, text_input
from lxml import etree
from os.path import sep
from pandas import DataFrame, to_numeric
from numpy import loadtxt
import string
import re

data_path = sep.join(['data', 'BURCAT_THR.xml'])
template_path = sep.join(['data', 'xsl_stylesheet_burcat.xsl'])
antoine_csv = './data/The-Yaws-Handbook-of-Vapor-Pressure-Second-Edition-Antoine-coefficients.csv'
poling_basic_i_csv = './data/basic_constants_i_properties_of_gases_and_liquids.csv'
poling_basic_ii_csv = './data/basic_constants_ii_properties_of_gases_and_liquids.csv'
poling_cp_l_ig_poly_csv = './data/ig_l_heat_capacities_properties_of_gases_and_liquids.csv'
poling_pv_csv = './data/vapor_pressure_correlations_parameters_clean.csv'

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
    matches = re.search(b'(\.(\d+\.\d+))', x)
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

    return burcat_df

df = load_data()
df = df[df['cas_no'].str.contains(cas_filter, na=False, case=False)]
df = df[df['formula'].str.contains(formula_filter, na=False, case=False)]
df = df[df['formula_name_structure'].str.contains(name_filter, na=False, case=False)]
df = df[df['phase'].str.contains(phase_filter, na=False, case=False)]

write(df)

write('filtered to', str(len(df)), 'records')
