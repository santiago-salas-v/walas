import streamlit
from streamlit import write, markdown, sidebar, text_input
from lxml import etree
from os.path import sep
from pandas import DataFrame
import string

data_path = sep.join(['data', 'BURCAT_THR.xml'])
template_path = sep.join(['data', 'xsl_stylesheet_burcat.xsl'])

markdown('# thr')
sidebar.markdown('# thr')
sidebar.write('filter by:')
cas_filter = sidebar.text_input(label='cas', key='cas', placeholder='cas', help='type cas no.')
name_filter = sidebar.text_input(label='name', key='name', placeholder='name', help='type name')
formula_filter = sidebar.text_input(label='formula', key='formula', placeholder='formula', help='type formula')
phase_filter = sidebar.selectbox('phase',['','G','L','S','C'])

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
    return burcat_df

write(cas_filter)
df = load_data()
df = df[df['cas_no'].str.contains(cas_filter, na=False)]
df = df[df['formula'].str.contains(formula_filter, na=False)]
df = df[df['formula_name_structure'].str.contains(name_filter, na=False)]
df = df[df['phase'].str.contains(phase_filter, na=False)]

write(df)
