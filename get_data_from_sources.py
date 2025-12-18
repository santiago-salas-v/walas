from html.parser import HTMLParser
from urllib import request, parse
import json
from time import sleep
from subprocess import check_call, CalledProcessError
from lxml import etree as et
import string, re
from numpy import loadtxt
from pandas import DataFrame
from os.path import sep
from os import chdir, getcwd
from pathlib import Path

# 1. burcat tc
# ref. https://gist.github.com/santiago-salas-v/6f408779c65a0267372c0ed66ff21fe4

folder = getcwd()
chdir('data')
outfile = 'BURCAT_THR.xml'
template = 'xsl_stylesheet_burcat.xsl'
thermodyn2xml = 'Thermodyn2XML_vanilla_7_11_05.py'
burcat_thr = 'BURCAT.THR'
antoine_csv = Path('data/The-Yaws-Handbook-of-Vapor-Pressure-Second-Edition-Antoine-coefficients.csv')
antoine_csv_new = Path('/'.join([x if j==0 else antoine_csv.stem+'_more_cas_resolved.csv' for j,x in enumerate(antoine_csv.parts)]))
burcat_url = 'https://respecth.elte.hu/burcat/BURCAT.THR.txt'
patch_url = 'https://gist.githubusercontent.com/santiago-salas-v/6f408779c65a0267372c0ed66ff21fe4/raw/38f1621ea9dc03e265468f4d3ab5e8f5ce2515d4/Thermodyn2XML_vanilla_7_11_05.py' # original py script was removed (only exe remaining) from the original path in https://respecth.elte.hu/burcat/dist.zip
resolver_url = 'https://hcd.rtpnc.epa.gov/api/resolver/lookup?%s'
ddbst_adress = 'http://www.ddbst.com/published-parameters-unifac.html'

if not Path(burcat_thr).exists():
    url = burcat_url
    patch = 'patch_BURCAT_THR.diff'

    path, headers = urlretrieve(url, burcat_thr)
    print(f'file produced successfully: {burcat_thr}')
    try:
        print(check_call(['patch', '-p1', burcat_thr, patch])) # newer version with fixes already available
        print(f'patch applied successfully: {burcat_thr}.')
    except CalledProcessError as e:
        print(f'called process error: {burcat_thr}')
else:
    print(f'file already exists: {burcat_thr}')

if not Path(outfile).exists():
    url = patch_url
    patch = 'patch_Thermodyn2XML_vanilla_7_11_05_py.diff'

    path, headers = urlretrieve(url, thermodyn2xml)
    try:
        print(check_call(['patch', '-p1', thermodyn2xml,  patch]))
        print('patch applied successfully.')
    except CalledProcessError as e:
        print(f'called process error: {thermodyn2xml}')

    check_call(['python', thermodyn2xml, burcat_thr])

    print(f'file produced successfully: {outfile}')
else:
    print(f'file already exists: {outfile}')

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
for line in request.urlopen(ddbst_adress):
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

def helper_func1(x):
    # helper function for csv reading
    # replace dot in original file for minus in some cases ocr'd like .17.896
    str_with_dot_actually_minus = x
    matches = re.search('(\.(\d+\.\d+))', x)
    if matches:
        str_with_dot_actually_minus = '-' + matches.groups()[1]
    return str_with_dot_actually_minus.replace(' ', '')

def helper_func3(x):
    # helper function 3 for csv reading.
    # replace - for . in cas numbers (Antoine coeff)
    return x.replace('.', '-')

ant_df = DataFrame(loadtxt(
    open(antoine_csv, 'r'),
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

def get_cas(x):
    params=parse.urlencode({'query':x,'idType':'AnyId','fuzzy':'Not','mol':'false'})
    with request.urlopen(resolver_url % params) as f:
        response=json.loads(f.read().decode('utf-8'))
    sleep(1)
    if len(response)==0:
        return ''
    elif 'casrn' in response[0]['chemical'].keys():
        print(x,response[0]['chemical']['casrn'])
        return response[0]['chemical']['casrn']
    else:
        return ''

idx=(ant_df.cas_no=='---')|(ant_df.cas_no=='—') # missinc CAS
#idx=idx.index[idx][:15] # limit to first 15
ant_df.loc[idx,'cas_no']=ant_df.ant_name.loc[idx].apply(get_cas)

with open(antoine_csv,'r') as f:
    with open(antoine_csv_new,'w') as n:
        for j in range(9):
            t=f.readline()
            print(t)
            n.write(t)

ant_df.to_csv(antoine_csv_new,sep='|',index=False,header=False,mode='a')

