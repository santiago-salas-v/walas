from pandas import concat, read_csv
import string, re
from pathlib import Path
from py2opsin import py2opsin # pip install py2opsin # fast get StdInChIKey
from pubchempy import PubChemHTTPError, get_synonyms # slow get synonims (CAS)
import camelot

names = ['CO', 'CO2', 'H2', 'H2O', 'CH3OH', 'CH4', 'N2', 'C2H5OH', 'n-C3H7OH', 'CH3COOCH3']
lj_csv=Path('data/lennard_jones_params_properties_of_gases_and_liquids.csv')
merged_df_csv = 'data/th_data_df.csv'

tables=camelot.read_pdf("C:/Users/sala_sn/Downloads/The properties of gases and Liquids Poling Prausnitz O'Connell.pdf",pages='779-780', flavor='stream')

df=concat([tables[j].df for j in range(tables.n)]).drop_duplicates()
# lj_b0 in cm3/gmol, lj_sigma in Angstrom, lj_epsilon_kB in K
df=df[df[0].apply(lambda x: x not in ['','B.2'])].reset_index(drop=True).rename(columns={0:'Formula',1:'Substance',2:'lj_b0',3:'lj_sigma',4:'lj_epsilon_kB'}).map(lambda x:x.replace('§','').replace('ﬂu','flu').replace('ﬁde','fide'))
df=df.astype(dtype={'Formula':str,'Substance':str,'lj_b0':float,'lj_sigma':float,'lj_epsilon_kB':float})

def get_cas(x):
    # try pubchem directly, 
    response=[] # name not resolved
    synonyms=get_synonyms(x,'name') #[0] # acetic acid has 3 CIDS first is ok.
    if len(synonyms)==0: 
        # not directly found but attempt py2opsin (can manage e.g. "ethyl cyclohexane")
        inchikey=py2opsin(x,'StdInChIKey')
        if inchikey:
            synonyms=get_synonyms(inchikey,'inchikey') #[0] # acetic acid has 3 CIDS first is ok.
    for c in synonyms:
        if 'Synonym' in c.keys():
            response+=[y for y in c['Synonym'] if re.match(r'^\d{2,7}-\d{2}-\d{1}$',y.strip())]
    response=','.join(response)
    print(x,response)
    return response

df['cas_no']=df.Substance.apply(get_cas)

dtypes_dict = {
    'cas_no': object, 'phase': object, 'formula_name_structure': object,
    'reference': object, 'hf298': object, 'max_lst_sq_error': object,
    'formula': object, 'source': object, 'date': object,
    'range_tmin_to_1000': float, 'range_1000_to_tmax': float,
    'molecular_weight': float, 'hf298_div_r': float,
    'a1_low': float, 'a2_low': float, 'a3_low': float,
    'a4_low': float, 'a5_low': float, 'a6_low': float,
    'a7_low': float, 'a1_high': float, 'a2_high': float,
    'a3_high': float, 'a4_high': float, 'a5_high': float,
    'a6_high': float, 'a7_high': float, 'poling_no': float,
    'poling_formula': object, 'poling_name': object, 'poling_molwt': float,
    'poling_tfp': float, 'poling_tb': float, 'poling_tc': float,
    'poling_pc': float, 'poling_vc': float, 'poling_zc': float,
    'poling_omega': float, 'poling_delhf0': float, 'poling_delgf0': float,
    'poling_delhb': float, 'poling_delhm': float, 'poling_v_liq': float,
    'poling_t_liq': float, 'poling_dipole': float, 'poling_trange': object,
    'poling_a0': float, 'poling_a1': float, 'poling_a2': float,
    'poling_a3': float, 'poling_a4': float, 'poling_cpig': float,
    'poling_cpliq': float, 'p_ant_a': float, 'p_ant_b': float,
    'p_ant_c': float, 'p_ant_pvpmin': float, 'p_ant_tmin': float,
    'p_ant_pvpmax': float, 'p_ant_tmax': float, 'eant_to': float,
    'eant_n': float, 'eant_e': float, 'eant_f': float,
    'eant_pvpmin': float, 'eant_tmin': float, 'eant_pvpmax': float,
    'eant_tmax': float, 'wagn_a': float, 'wagn_b': float,
    'wagn_c': float, 'wagn_d': float, 'wagn_pvpmin': float,
    'wagn_tmin': float, 'wagn_pvpmax': float, 'wagn_tmax': float,
    'ant_no': float, 'ant_formula': object, 'ant_name': object,
    'ant_a': float, 'ant_b': float, 'ant_c': float,
    'ant_tmin': float, 'ant_tmax': float, 'ant_code': object,
    'lj_b0': float, 'lj_sigma': float, 'lj_epsilon_kB': float}

df_merged=read_csv(merged_df_csv, skiprows=1, sep=',', index_col=0, keep_default_na=False, na_values=['NaN'], dtype=dtypes_dict)

[[[j]+[y]+df_merged[df_merged.cas_no.str.contains(y)].formula_name_structure.to_list()+df_merged[df_merged.cas_no.str.contains(y)].wagn_a.to_list() for y in x.split(',') if len(y)>0] for j,x in enumerate(df.cas_no)]

for j,x in df.cas_no.items():
    for y in x.split(','):
        if len(y)>0 and y not in ['---','—','—','NaN']:
            for label in ['lj_b0','lj_sigma','lj_epsilon_kB']:
                df_merged.loc[df_merged.cas_no.str.contains(y),label]=df.loc[j,label]

with open(merged_df_csv, 'w', encoding='utf-8') as buf:
    buf.write('sep=,\n')
df_merged.to_csv(merged_df_csv, sep=',', na_rep='NaN', mode='a',encoding='utf-8')
