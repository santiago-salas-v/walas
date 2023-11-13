from html.parser import HTMLParser
from urllib.request import urlretrieve, urlopen
from subprocess import check_call
from lxml import etree as et
import string
from pandas import DataFrame
from os.path import sep
from os import chdir, getcwd

# 1. burcat tc
# ref. https://gist.github.com/santiago-salas-v/6f408779c65a0267372c0ed66ff21fe4

folder = getcwd()
chdir('data')
outfile = 'BURCAT_THR.xml'
template = 'xsl_stylesheet_burcat.xsl'
thermodyn2xml = 'Thermodyn2XML_vanilla_7_11_05.py'
burcat_thr = 'BURCAT.THR'

url = 'http://garfield.chem.elte.hu/Burcat/BURCAT.THR'
patch = 'patch_BURCAT_THR.diff'

path, headers = urlretrieve(url, burcat_thr)
print(check_call(['git', 'apply', patch]))

url = 'http://garfield.chem.elte.hu/Burcat/dist/Thermodyn2XML_vanilla_7_11_05.py'
patch = 'patch_Thermodyn2XML_vanilla_7_11_05_py.diff'

path, headers = urlretrieve(url, thermodyn2xml)
print(check_call(['git', 'apply', patch]))

print('both patches applied successfully.')

check_call(['python', thermodyn2xml, burcat_thr])

print(f'file produced successfully: {outfile}')

print(et.parse(outfile))

tree = et.parse(outfile)
root = tree.getroot()
xsl = et.parse(template)
transformer = et.XSLT(xsl)
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

print(burcat_df)

# 2. unifac

chdir(folder)

ddbst_adress = 'http://www.ddbst.com/published-parameters-unifac.html'

class TableHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)

        self.table_open = False
        self.tr_open = False
        self.td_open = False
        self.tables = []

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.table_open = True
            self.tables += [[]]
        if tag == 'tr':
            self.tr_open = True
            self.tables[-1] += [[]]
        if tag == 'td':
            self.td_open = True
            print('Start :' + tag)
    def handle_endtag(self, tag):
        if tag == 'table':
            self.table_open = False
            print('End tag:' + tag)
        if tag == 'tr':
            self.tr_open = False
            print('End tag:' + tag)
        if tag == 'td':
            self.td_open = False
            print('End tag:' + tag)
    def handle_data(self, data):
        if self.td_open:
            self.tables[-1][-1] += [data]
            print('Data:'+data)


page_str = ''
for line in urlopen(ddbst_adress):
    page_str += line.decode('utf-8')

parser = TableHTMLParser()
parser.feed(page_str)

f = open('./data/unifac_list_of_interaction_parameters.csv', 'w')
f.write('sep=,'+'\n')
for line in parser.tables[0]:
    f.write(','.join(line)+'\n')
f.close()

f = open('./data/unifac_sub_groups_surfaces_and_volumes.csv', 'w')
f.write('sep=,'+'\n')
for line in parser.tables[1]:
    f.write(','.join(line)+'\n')
f.close()

f = open('./data/unifac_main_groups.csv', 'w')
f.write('sep=,'+'\n')
for line in parser.tables[2]:
    f.write(','.join(line)+'\n')
f.close()
