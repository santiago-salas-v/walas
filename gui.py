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
text_input('', key='name', placeholder='formula')



@streamlit.cache_data
def load_data():
    tree = etree.parse(data_path)
    root = tree.getroot()
    xsl = etree.parse(template_path)
    transformer = etree.XSLT(xsl)
    result = transformer(tree)

    cas_filter, name_filter, \
    formula_filter, phase_filter = \
    '', '', '', ''


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

df = load_data()
write(df)

