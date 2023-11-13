from streamlit import write
from lxml import etree
from os.path import sep
from pandas import DataFrame
import string

data = sep.join(['data', 'BURCAT_THR.xml'])
template = sep.join(['data', 'xsl_stylesheet_burcat.xsl'])

tree = etree.parse(data)
root = tree.getroot()
xsl = etree.parse(template)
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

write(burcat_df)
