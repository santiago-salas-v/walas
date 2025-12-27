from html.parser import HTMLParser
from urllib import request
from time import sleep
from subprocess import check_call, CalledProcessError
from lxml import etree as et
import string, re
from numpy import loadtxt,nan,log
from pandas import DataFrame, read_csv, merge, isna, concat
from pathlib import Path
from py2opsin import py2opsin # pip install py2opsin # fast get StdInChIKey
from pubchempy import PubChemHTTPError, get_synonyms # slow get synonims (CAS)
import camelot

# 1. burcat tc
# ref. https://gist.github.com/santiago-salas-v/6f408779c65a0267372c0ed66ff21fe4

burcat_xml_file = Path('data/BURCAT_THR.xml')
template = Path('data/xsl_stylesheet_burcat.xsl')
thermodyn2xml = Path('data/Thermodyn2XML_vanilla_7_11_05.py')
burcat_thr = Path('data/BURCAT.THR')
burcat_patch = Path('data/patch_BURCAT_THR.diff')
thermodyn2xml_patch = Path('data/patch_Thermodyn2XML_vanilla_7_11_05_py.diff')
antoine_csv = Path('data/The-Yaws-Handbook-of-Vapor-Pressure-Second-Edition-Antoine-coefficients.csv')
antoine_csv_new = Path('/'.join([x if j==0 else antoine_csv.stem+'_more_cas_resolved.csv' for j,x in enumerate(antoine_csv.parts)]))
wagn_2_csv=Path('data/table2_wagner_doi.org_10.1016_j.jct.2011.03.011.csv')
wagn_2_csv_new = Path('/'.join([antoine_csv.parts[0],wagn_2_csv.stem+'_cas.csv']))
wagn_1_csv=Path('data/table1_wagner_doi.org_10.1016_j.jct.2011.03.011.csv')
wagn_1_csv_new = Path('/'.join([antoine_csv.parts[0],wagn_1_csv.stem+'_cas.csv']))
wagn_m_csv=Path('data/tables_wagner_doi.org_10.1016_j.jct.2011.03.011.csv')
wagn_m_csv_new = Path('/'.join([antoine_csv.parts[0],wagn_m_csv.stem+'_cas.csv']))
burcat_url = 'https://respecth.elte.hu/burcat/BURCAT.THR.txt'
thermodyn2xml_url = 'https://gist.githubusercontent.com/santiago-salas-v/6f408779c65a0267372c0ed66ff21fe4/raw/38f1621ea9dc03e265468f4d3ab5e8f5ce2515d4/Thermodyn2XML_vanilla_7_11_05.py' # original py script was removed (only exe remaining) from the original path in https://respecth.elte.hu/burcat/dist.zip
ddbst_adress = 'http://www.ddbst.com/published-parameters-unifac.html'
vdi_vp_tab_pdf=Path('data/vdi_heat_atlas_2nd_2010_3.1-table_3.pdf')
vdi_csv=Path('data/vdi_heat_atlas_2nd_2010_3.1-table_3.csv')
poling_basic_i_csv = Path('data/basic_constants_i_properties_of_gases_and_liquids.csv')
poling_basic_ii_csv = Path('data/basic_constants_ii_properties_of_gases_and_liquids.csv')
poling_cp_l_ig_poly_csv = Path('data/ig_l_heat_capacities_properties_of_gases_and_liquids.csv')
poling_pv_csv = Path('data/vapor_pressure_correlations_parameters_clean.csv')
merged_df_csv = Path('data/th_data_df.csv')

if not burcat_thr.exists():
    url = burcat_url

    path, headers = request.urlretrieve(url, burcat_thr)
    print(f'file produced successfully: {burcat_thr}')
    try:
        print(check_call(['patch', '-p1', burcat_thr, burcat_patch])) # newer version with fixes already available
        print(f'patch applied successfully: {burcat_thr}.')
    except CalledProcessError:
        print(f'called process error: {burcat_thr}')
else:
    print(f'file already exists: {burcat_thr}')

if not burcat_xml_file.exists():
    url = thermodyn2xml_url

    path, headers = request.urlretrieve(url, thermodyn2xml)
    try:
        print(check_call(['patch', '-p1', thermodyn2xml,  thermodyn2xml_patch]))
        print('patch applied successfully.')
    except CalledProcessError:
        print(f'called process error: {thermodyn2xml}')

    check_call(['python', thermodyn2xml, burcat_thr])

    Path(burcat_thr.name.replace('.','_')+'.xml').rename(burcat_xml_file)
    print(f'file produced successfully: {burcat_xml_file}')
else:
    print(f'file already exists: {burcat_xml_file}')

print(et.parse(burcat_xml_file))

tree = et.parse(burcat_xml_file)
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

for column_name in [
    'a'+str(i)+'_low' for i in range(1,7+1) ] + [
    'a'+str(i)+'_high' for i in range(1,7+1)] + [
    'hf298_div_r', 'molecular_weight', 'range_1000_to_tmax',
    'range_tmin_to_1000'
    ]:
    burcat_df[column_name] =  burcat_df[column_name].str.replace(' ', '').apply(lambda x: float(x) if x not in ['N/A'] else nan)


print(burcat_df)

# 2. unifac

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

f = open('data/unifac_list_of_interaction_parameters.csv', 'w', encoding='utf-8')
f.write('sep=,'+'\n')
for line in parser.tables[0]:
    f.write(','.join(line)+'\n')
f.close()

f = open('data/unifac_sub_groups_surfaces_and_volumes.csv', 'w', encoding='utf-8')
f.write('sep=,'+'\n')
for line in parser.tables[1]:
    f.write(','.join(line)+'\n')
f.close()

f = open('data/unifac_main_groups.csv', 'w', encoding='utf-8')
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


def helper_func2(x):
    # helper function 2 for csv reading.
    # Return None when empty string
    if len(x) == 0:
        return 'nan'
    return x.replace(' ', '')


def helper_func3(x):
    # helper function 3 for csv reading.
    # replace - for . in cas numbers (Antoine coeff)
    return x.replace('.', '-')


def helper_func4(x):
    # helper function 4 for csv reading.
    # replace whitespace ' ' for '' in float numbers
    return x.replace(' ', '')


def helper_func5(x):
    # helper function 5 for csv reading.
    # replace '−' for '-' (otherwise float not possible)
    return nan if len(x.strip())==0 else float(x.replace('−','-'))


ant_df = DataFrame(loadtxt(
    open(antoine_csv, 'r', encoding='utf-8'),
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
    # try pubchem directly, 
    response=[] # name not resolved
    synonyms=get_synonyms(x,'name')
    for c in synonyms:
        response+=[y for y in c['Synonym'] if re.match(r'^\d{2,7}-\d{2}-\d{1}$',y.strip())]
    response=','.join(response)
    print(x,response)
    return response

idx=(ant_df.cas_no=='---')|(ant_df.cas_no=='—') # missinc CAS
idx=idx.index[idx][:3] # limit to first 15
ant_df.loc[idx,'cas_no']=ant_df.ant_name.loc[idx].apply(get_cas)

with open(antoine_csv,'r', encoding='utf-8') as f:
    with open(antoine_csv_new,'w', encoding='utf-8') as n:
        for j in range(9):
            t=f.readline()
            print(t)
            n.write(t)

ant_df.to_csv(antoine_csv_new,sep='|',index=False,header=False,mode='a')

poling_basic_i_df = DataFrame(loadtxt(
    open(poling_basic_i_csv, 'r', encoding='utf-8'),
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
    open(poling_basic_ii_csv, 'r', encoding='utf-8'),
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
    open(poling_cp_l_ig_poly_csv, 'r', encoding='utf-8'),
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

conv = dict([[i, lambda x: helper_func4(helper_func2(x))]
             for i in [0, 1, 2, 4, 5,
                       ]+[8, 9, 10, 11, 12, 13, 14, 15, 16]])
conv[3] = lambda x: helper_func3(x)
conv[6] = lambda x: helper_func4(helper_func2(x))
conv[7] = lambda x: helper_func4(helper_func2(x))

poling_pv_df = DataFrame(loadtxt(
    open(poling_pv_csv, 'r', encoding='utf-8'),
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
df = merge(poling_burcat_df, ant_df, on='cas_no', how='outer')

df['poling_a1'] = df['poling_a1'] * 1e-3
df['poling_a2'] = df['poling_a2'] * 1e-5
df['poling_a3'] = df['poling_a3'] * 1e-8
df['poling_a4'] = df['poling_a4'] * 1e-11

idx=isna(df['poling_omega'])
pc_bar=df.loc[idx,'poling_pc']
tbr=df.loc[idx,'poling_tb']/df.loc[idx,'poling_tc']
f0_tbr,f1_tbr,f2_tbr=(-5.97616*(1-tbr)+1.29874*(1-tbr)**1.5-0.60394*(1-tbr)**2.5-1.06841*(1-tbr)**5)/tbr,(-5.03365*(1-tbr)+1.11505*(1-tbr)**1.5-5.41217*(1-tbr)**2.5-7.46628*(1-tbr)**5)/tbr,(-0.64771*(1-tbr)+2.41539*(1-tbr)**1.5-4.26979*(1-tbr)**2.5+3.25259*(1-tbr)**5)/tbr # Ambrose-Walton 1989
df.loc[idx,'poling_omega']=-(log(pc_bar/1.01325)+f0_tbr)/f1_tbr

df.loc[isna(df.formula_name_structure),'formula_name_structure']=''

with open(merged_df_csv, 'w', encoding='utf-8') as buf:
    buf.write('sep=,\n')
df.to_csv(merged_df_csv, na_rep='NaN', mode='a')

wgn2_df=read_csv(wagn_2_csv,skiprows=2,sep='\t',converters={1:helper_func4}|{x:helper_func5 for x in range(2,18)})
# cas no. based on pubchem (1st) + opsin (backup). Ref. https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest#section=Input
# since unfortunately only cid's and inchikeys (not names) can be entered as full lists, would need
# to transform names into cids, which has the same cost as getting synonyms directly. Therefore:
# 1. call all synonyms using py2opsin to get StdInChIKey in batch. OPSIN is secure but falsifies diatomic molecules (hydrogen, bromine, etc. interpreted as monatomic), though sulfur is no longer interpreted as H2S. Keep this response as fallback in case the pubchem call fails (because of nomenclature)
# 2. call for each record the pubchem name method: slow but diatomic molecules interpreted correctly.
wgn2_df['stdinchikeys']=py2opsin(wgn2_df.Substance,'StdInChIKey')
idx=wgn2_df['stdinchikeys'].apply(len)==0
wgn2_df.loc[idx,'stdinchikeys']=wgn2_df.loc[0,'stdinchikeys'] # placeholder
wgn2_df['cas_no']=wgn2_df.Substance.apply(get_cas)
wgn2_df=wgn2_df.rename(columns={'Tc':'Tc/K','Pc':'Pc/kPa'})

wgn1_df=read_csv(wagn_1_csv,skiprows=2,sep='\t')
wgn1_df['cas_no']=wgn1_df.Substance.apply(get_cas)
wgn1_df['Tr,min']=wgn1_df['Experimental Tr range'].apply(lambda x:float(x.split('-')[0]))
wgn1_df['Tr,max']=wgn1_df['Experimental Tr range'].apply(lambda x:float(x.split('-')[1]))
wgn1_df=wgn1_df.rename(columns={'Tc':'Tc/K','Pc':'Pc/kPa'})


wgn_m_df=merge(wgn1_df,wgn2_df,how='outer',on=['Substance'])
labels=[x for x in ['cas_no']+[y for y in 'ABCD']+['Tc/K','Pc/kPa','Tr,min','Tr,max']]
for label in [x for x in [y for y in 'ABCD']+['cas_no','Tc/K','Pc/kPa','Tr,min','Tr,max']]:
    wgn_m_df[label]=wgn_m_df[label+'_x']
    idx=isna(wgn_m_df[label])
    wgn_m_df.loc[idx,label]=wgn_m_df.loc[idx,label+'_y']

with open(wagn_1_csv,'r', encoding='utf-8') as f:
    with open(wagn_1_csv_new,'w', encoding='utf-8') as n:
        for j in range(3):
            t=f.readline()
            if j > 0:
                t=t.split('\n')[0]+'\t'+'cas_no'+'\n'
            print(t)
            n.write(t)

wgn1_df.to_csv(wagn_1_csv_new,sep='\t',index=False,header=False,mode='a')

with open(wagn_2_csv,'r', encoding='utf-8') as f:
    with open(wagn_2_csv_new,'w', encoding='utf-8') as n:
        for j in range(3):
            t=f.readline()
            if j > 0:
                t=t.split('\n')[0]+'\t'+'cas_no'+'\n'
            print(t)
            n.write(t)

wgn2_df.to_csv(wagn_2_csv_new,sep='\t',index=False,header=False,mode='a')

wgn_m_df[['Substance']+labels].to_csv(wagn_m_csv_new,sep='\t',index=False,header=True)

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
tables=camelot.read_pdf(str(vdi_vp_tab_pdf),pages='all', flavor='stream')
vdi_vp_df=concat([tables[j].df.map(lambda x:x.replace('(cid:2)','-')) for j in range(tables.n)],ignore_index=True)
vdi_vp_df=vdi_vp_df.drop(index=[0,1]).rename(columns={j:x for j,x in enumerate(['Substance','Formula','5','10','50','100','250','500','1000','2000','5000','10000','A','B','C','D'])})
vdi_vp_df.loc[vdi_vp_df.Formula=='C2Cl4F2','Substance']='1,1,2,2-Tetrachlorodifluoroethane'
vdi_vp_df=vdi_vp_df.drop(vdi_vp_df.index[vdi_vp_df.Substance=='Tetrachlorodifluoroethane'])
numeric_keys=[x for x in vdi_vp_df.keys() if x not in ['Substance','Formula']]
vdi_vp_df=vdi_vp_df.drop(vdi_vp_df.index[(vdi_vp_df=='Vapor pressure in mbar').any(axis=1)|(vdi_vp_df.Substance=='Substance')|(vdi_vp_df.Substance.str.contains('Table'))|vdi_vp_df[numeric_keys].map(lambda x:re.search(r'[a-zA-Z]|,|\s',str(x).replace('nan','')) is not None).any(axis=1)]).reset_index(drop=True)
vdi_vp_df[vdi_vp_df[numeric_keys]=='']=nan
vdi_vp_df[numeric_keys]=vdi_vp_df[numeric_keys].map(float)
vdi_vp_df=vdi_vp_df.reset_index(drop=True)
idx=vdi_vp_df[isna(vdi_vp_df[numeric_keys]).all(axis=1)].index.to_list()+[vdi_vp_df.shape[0]-1]
for j in range(1,len(idx)):
    print(idx[j-1],idx[j],vdi_vp_df.Substance.loc[idx[j-1]])
    vdi_vp_df.loc[idx[j-1]:idx[j],'class']=vdi_vp_df.Substance.loc[idx[j-1]]
vdi_vp_df=vdi_vp_df.drop(vdi_vp_df.index[vdi_vp_df.Substance.duplicated()|isna(vdi_vp_df[numeric_keys]).all(axis=1)]).reset_index(drop=True)
idx=vdi_vp_df.Substance.str.findall(r'\(R[\d]+\w*\)').apply(len)>0 # rows with refrigerant names
vdi_vp_df['refrigerant']=vdi_vp_df.Substance.str.findall(r'\(R[\d]+\w*\)').apply(lambda x: x[0] if len(x)>0 else '')
vdi_vp_df.Substance=vdi_vp_df.Substance.str.replace(r'\s*\(R[\d]+\w*\)','',regex=True)
vdi_vp_df['cas_no']=vdi_vp_df.Substance.apply(get_cas)

vdi_vp_df[['Substance','Formula','refrigerant','cas_no']+numeric_keys].to_csv(vdi_csv)

