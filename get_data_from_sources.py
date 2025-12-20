from html.parser import HTMLParser
from urllib import request
from time import sleep
from subprocess import check_call, CalledProcessError
from lxml import etree as et
import string, re
from numpy import loadtxt,nan
from pandas import DataFrame, read_csv, merge, isna, concat
from os import chdir, getcwd
from pathlib import Path
from cirpy import resolve # conda install -c conda-forge cirpy
import camelot

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
wagn_2_csv=Path('data/table2_wagner_doi.org_10.1016_j.jct.2011.03.011.csv')
wagn_2_csv_new = Path('/'.join([antoine_csv.parts[0],wagn_2_csv.stem+'_cas.csv']))
wagn_1_csv=Path('data/table1_wagner_doi.org_10.1016_j.jct.2011.03.011.csv')
wagn_1_csv_new = Path('/'.join([antoine_csv.parts[0],wagn_1_csv.stem+'_cas.csv']))
wagn_m_csv=Path('data/tables_wagner_doi.org_10.1016_j.jct.2011.03.011.csv')
wagn_m_csv_new = Path('/'.join([antoine_csv.parts[0],wagn_m_csv.stem+'_cas.csv']))
burcat_url = 'https://respecth.elte.hu/burcat/BURCAT.THR.txt'
patch_url = 'https://gist.githubusercontent.com/santiago-salas-v/6f408779c65a0267372c0ed66ff21fe4/raw/38f1621ea9dc03e265468f4d3ab5e8f5ce2515d4/Thermodyn2XML_vanilla_7_11_05.py' # original py script was removed (only exe remaining) from the original path in https://respecth.elte.hu/burcat/dist.zip
resolver_url = 'https://cactus.nci.nih.gov/chemical/structure/%s/cas'
ddbst_adress = 'http://www.ddbst.com/published-parameters-unifac.html'
vdi_vp_tab_pdf=Path('data/vdi_heat_atlas_2nd_2010_3.1-table_3.pdf')
vdi_csv=Path('data/vdi_heat_atlas_2nd_2010_3.1-table_3.csv')

if not Path(burcat_thr).exists():
    url = burcat_url
    patch = 'patch_BURCAT_THR.diff'

    path, headers = request.urlretrieve(url, burcat_thr)
    print(f'file produced successfully: {burcat_thr}')
    try:
        print(check_call(['patch', '-p1', burcat_thr, patch])) # newer version with fixes already available
        print(f'patch applied successfully: {burcat_thr}.')
    except CalledProcessError:
        print(f'called process error: {burcat_thr}')
else:
    print(f'file already exists: {burcat_thr}')

if not Path(outfile).exists():
    url = patch_url
    patch = 'patch_Thermodyn2XML_vanilla_7_11_05_py.diff'

    path, headers = request.urlretrieve(url, thermodyn2xml)
    try:
        print(check_call(['patch', '-p1', thermodyn2xml,  patch]))
        print('patch applied successfully.')
    except CalledProcessError:
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
    matches = re.search(r'(\.(\d+\.\d+))', x)
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
    try:
        with request.urlopen(resolver_url % x) as f:
            response=f.read().decode('utf-8').replace('\n',' ')
        print(x,response)
    except request.HTTPError as e:
        response=''
        print(x,'cas not found:\t'+str(e))
    except request.URLError as e:
        response=''
        print(x,'url error:\t'+str(e))
    sleep(1)
    return response

def get_cas_cir(x):
    response=resolve(x,'cas',['name_by_cir','name_by_opsin'])
    if response is None:
        response=''
    elif type(response)==list:
        response=','.join(response)
    print(x,response)
    return response

idx=(ant_df.cas_no=='---')|(ant_df.cas_no=='—') # missinc CAS
idx=idx.index[idx][:3] # limit to first 15
ant_df.loc[idx,'cas_no']=ant_df.ant_name.loc[idx].apply(get_cas)

with open(antoine_csv,'r') as f:
    with open(antoine_csv_new,'w') as n:
        for j in range(9):
            t=f.readline()
            print(t)
            n.write(t)

ant_df.to_csv(antoine_csv_new,sep='|',index=False,header=False,mode='a')

def helper_func4(x):
    # helper function 4 for csv reading.
    # replace whitespace ' ' for '' in float numbers
    return x.replace(' ', '')

def helper_func5(x):
    # helper function 5 for csv reading.
    # replace '−' for '-' (otherwise float not possible)
    return nan if len(x.strip())==0 else float(x.replace('−','-'))

df_wgn2=read_csv(wagn_2_csv,skiprows=2,sep='\t',converters={1:helper_func4}|{x:helper_func5 for x in range(2,18)})
df_wgn2['cas_no']=df_wgn2.Substance.apply(get_cas_cir)
df_wgn2=df_wgn2.rename(columns={'Tc':'Tc/K','Pc':'Pc/kPa'})

df_wgn1=read_csv(wagn_1_csv,skiprows=2,sep='\t')
df_wgn1['cas_no']=df_wgn1.Substance.apply(get_cas_cir)
df_wgn1['Tr,min']=df_wgn1['Experimental Tr range'].apply(lambda x:float(x.split('-')[0]))
df_wgn1['Tr,max']=df_wgn1['Experimental Tr range'].apply(lambda x:float(x.split('-')[1]))
df_wgn1=df_wgn1.rename(columns={'Tc':'Tc/K','Pc':'Pc/kPa'})


df_wgn_m=merge(df_wgn1,df_wgn2,how='outer',on=['Substance'])
labels=[x for x in ['cas_no']+[y for y in 'ABCD']+['Tc/K','Pc/kPa','Tr,min','Tr,max']]
for label in [x for x in [y for y in 'ABCD']+['cas_no','Tc/K','Pc/kPa','Tr,min','Tr,max']]:
    df_wgn_m[label]=df_wgn_m[label+'_x']
    idx=isna(df_wgn_m[label])
    df_wgn_m.loc[idx,label]=df_wgn_m.loc[idx,label+'_y']

with open(wagn_1_csv,'r') as f:
    with open(wagn_1_csv_new,'w') as n:
        for j in range(3):
            t=f.readline()
            if j > 0:
                t=t.split('\n')[0]+'\t'+'cas_no'+'\n'
            print(t)
            n.write(t)

df_wgn1.to_csv(wagn_1_csv_new,sep='\t',index=False,header=False,mode='a')

with open(wagn_2_csv,'r') as f:
    with open(wagn_2_csv_new,'w') as n:
        for j in range(3):
            t=f.readline()
            if j > 0:
                t=t.split('\n')[0]+'\t'+'cas_no'+'\n'
            print(t)
            n.write(t)

df_wgn2.to_csv(wagn_2_csv_new,sep='\t',index=False,header=False,mode='a')

df_wgn_m[['Substance']+labels].to_csv(wagn_m_csv_new,sep='\t',index=False,header=True)

# 1. read tables with flavor stream generates some duplicates
# 2. character for - recognized as (cid:2) --> replace
# 3. drop 1st and second empty rows, fix 1,1,2,2-Tetrachlorodifluoroethane issue
# 3.1 add column names
# 4. some rows are just classification -> for numeric columns check which have only nan
# 5. convert numeric columns to float by first explicitly adding nan where empty text present
# 6. reset index to perform indexed operations, get index of fully nan-rows (classifications)
# 7. propagate classifications to rows below class names
# 8. remove duplicates and reset index, results in 275 components
# 9. add cas numbers, add cas numbers to refrigerants (additional RXX number)
tables=camelot.read_pdf(vdi_vp_tab_pdf,pages='all', flavor='stream')
df_vdi_vp=concat([tables[j].df.map(lambda x:x.replace('(cid:2)','-')) for j in range(tables.n)],ignore_index=True)
df_vdi_vp=df_vdi_vp.drop(index=[0,1]).rename(columns={j:x for j,x in enumerate(['Substance','Formula','5','10','50','100','250','500','1000','2000','5000','10000','A','B','C','D'])})
df_vdi_vp.loc[df_vdi_vp.Formula=='C2Cl4F2','Substance']='1,1,2,2-Tetrachlorodifluoroethane'
df_vdi_vp=df_vdi_vp.drop(df_vdi_vp.index[df_vdi_vp.Substance=='Tetrachlorodifluoroethane'])
numeric_keys=[x for x in df_vdi_vp.keys() if x not in ['Substance','Formula']]
df_vdi_vp=df_vdi_vp.drop(df_vdi_vp.index[(df_vdi_vp=='Vapor pressure in mbar').any(axis=1)|(df_vdi_vp.Substance=='Substance')|(df_vdi_vp.Substance.str.contains('Table'))|df_vdi_vp[numeric_keys].map(lambda x:re.search('[a-zA-Z]|,|\s',str(x).replace('nan','')) is not None).any(axis=1)]).reset_index(drop=True)
df_vdi_vp[df_vdi_vp[numeric_keys]=='']=nan
df_vdi_vp[numeric_keys]=df_vdi_vp[numeric_keys].map(float)
df_vdi_vp=df_vdi_vp.reset_index(drop=True)
idx=df_vdi_vp[isna(df_vdi_vp[numeric_keys]).all(axis=1)].index.to_list()+[df_vdi_vp.shape[0]-1]
for j in range(1,len(idx)):
    print(idx[j-1],idx[j],df_vdi_vp.Substance.loc[idx[j-1]])
    df_vdi_vp.loc[idx[j-1]:idx[j],'class']=df_vdi_vp.Substance.loc[idx[j-1]]
df_vdi_vp=df_vdi_vp.drop(df_vdi_vp.index[df_vdi_vp.Substance.duplicated()|isna(df_vdi_vp[numeric_keys]).all(axis=1)]).reset_index(drop=True)
idx=df_vdi_vp.Substance.str.findall('\(R[\d]+\w*\)').apply(len)>0 # rows with refrigerant names
df_vdi_vp['refrigerant']=df_vdi_vp.Substance.str.findall('\(R[\d]+\w*\)').apply(lambda x: x[0] if len(x)>0 else '')
df_vdi_vp.Substance=df_vdi_vp.Substance.str.replace('\s*\(R[\d]+\w*\)','',regex=True)
df_vdi_vp['cas_no']=df_vdi_vp.Substance.apply(get_cas_cir)

df_vdi_vp[['Substance','Formula','refrigerant','cas_no']+numeric_keys].to_csv(vdi_csv)
