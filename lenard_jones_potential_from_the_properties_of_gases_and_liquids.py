import os

names = ['CO', 'CO2', 'H2', 'H2O', 'CH3OH', 'CH4', 'N2', 'C2H5OH', 'n-C3H7OH', 'CH3COOCH3']

file = open(os.path.abspath('../lennard_jones_potentials_properties_of_gases_liquids.txt'), 
    mode='r', encoding='utf8')
result_dict = dict()
for line in file:
    arr = line.split(' ')
    vals = arr[-3:]
    float_vals = [float(x.replace('\n','').replace('§','')) for x in vals]
    txt = arr[0]
    result_dict[txt]=float_vals
print('Available names:')
print('\n'.join(result_dict.keys()))

print('')
print('specified list:')
print(names)
print('')
print('b_0/(cm^3/g-mol)')
print(str([result_dict[name][0] for name in names]).replace('[','').replace(']',''))
print('sigma/(Angstrom)')
print(str([result_dict[name][1] for name in names]).replace('[','').replace(']',''))
print('(epsilon/k_B)/(K)')
print(str([result_dict[name][2] for name in names]).replace('[','').replace(']',''))