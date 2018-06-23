import os
import re

sel_component_numbers = [30, 31, 438, 440, 27, 26, 455, 66, 96, 61]
print('Selected components:')
print(sel_component_numbers)
print('')

test_exp_descriptors = '(^[0-9]{1,3})\|(.*)\|(.*)\|' + \
    '([0-9]{1,7}-[0-9]{1,7}-[0-9]{1,7})'
test_exp_basic_constants_i = test_exp_descriptors + \
    '[\||\ ](\-?\ ?[0-9]+\.?[0-9]+)?' * 8  # 8 numeric columns
test_exp_basic_constants_ii = test_exp_descriptors + \
    '[\||\ ](\-?\ ?[0-9]+\.?[0-9]+)?' * 7  # 7 numeric columns
test_exp_ig_heat_capacities = test_exp_descriptors + \
    '\|(—|[0-9]{1,4}-[0-9]{1,4})?'+ \
    '[\||\ ](\-?\ ?[0-9]+\.?[0-9]+)?' * 7  # 7 numeric columns

file_name_basic_constants_i = os.path.abspath(
    '../basic_constants_i_properties_of_gases_and_liquids.csv'
)
file_name_basic_constants_ii = os.path.abspath(
    '../basic_constants_ii_properties_of_gases_and_liquids.csv'
)
file_name_ig_heat_capacities = os.path.abspath(
    '../ig_l_heat_capacities_properties_of_gases_and_liquids.csv'
)

file = open(file_name_basic_constants_i, mode='r', encoding='utf-8-sig')
matches = []
k = 0
print('======================')
for line in file:
  k += 1
  # print header
  if k < 4:
    print(line)
  match = re.search(test_exp_basic_constants_i, line)
  if match is not None:
    matches.append(match.groups())

file.close()
print('Basic Constants I: ' + str(len(matches)) +
      ' Components in table')

no = [int(item[0]) for item in matches]
formula = [item[1] for item in matches]
name = [item[2] for item in matches]
cas_no = [item[3] for item in matches]
mol_wt = [float(item[4].replace(' ', ''))
          if item[4] is not None
          else 0.0
          for item in matches]  # g/mol
tfp = [float(item[5].replace(' ', ''))
       if item[5] is not None
       else 0.0
       for item in matches]  # K
tb = [float(item[6].replace(' ', ''))
      if item[6] is not None
      else 0.0
      for item in matches]  # K
tc = [float(item[7].replace(' ', ''))
      if item[7] is not None
      else 0.0
      for item in matches]  # K
pc = [float(item[8].replace(' ', ''))
      if item[8] is not None
      else 0.0
      for item in matches]  # bar
vc = [float(item[9].replace(' ', ''))
      if item[9] is not None
      else 0.0
      for item in matches]  # cm^3/mol
zc = [float(item[10].replace(' ', ''))
      if item[10] is not None
      else 0.0
      for item in matches]  # []
omega = [float(item[11].replace(' ', ''))
         if item[11] is not None
         else 0.0
         for item in matches]  # []

table_indexes_of_comp_nos = [
    no.index(comp_no) for comp_no in sel_component_numbers
]

print('')
print('======================')

props = ['no', 'formula', 'name', 'cas_no',
         'mol_wt', 'tfp', 'tb', 'tc',
         'pc', 'zc', 'omega']

print('Values in Table:')
for prop in props:
  print(prop)
  print(str([
      (globals()[prop])[comp_no]
      for comp_no in table_indexes_of_comp_nos
  ]).replace('[', '').replace(']', ''))
  print('')


file = open(file_name_basic_constants_ii, mode='r', encoding='utf-8-sig')
matches = []
k = 0
print('======================')
for line in file:
  k += 1
  # print header
  if k < 9:
    print(line)
  match = re.search(test_exp_basic_constants_ii, line)
  if match is not None:
    matches.append(match.groups())

file.close()
print('Basic Constants II: ' + str(len(matches)) +
      ' Components in table')


no = [int(item[0]) for item in matches]
formula = [item[1] for item in matches]
name = [item[2] for item in matches]
cas_no = [item[3] for item in matches]
delHf0 = [float(item[4].replace(' ', ''))
          if item[4] is not None
          else 0.0
          for item in matches]  # kJ/mol
delGf0 = [float(item[5].replace(' ', ''))
          if item[5] is not None
          else 0.0
          for item in matches]  # kJ/mol
delHb = [float(item[6].replace(' ', ''))
         if item[6] is not None
         else 0.0
         for item in matches]  # kJ/mol
delHm = [float(item[7].replace(' ', ''))
         if item[7] is not None
         else 0.0
         for item in matches]  # kJ/mol
v_liq = [float(item[8].replace(' ', ''))
         if item[8] is not None
         else 0.0
         for item in matches]  # cm^3/mol
t_liq = [float(item[9].replace(' ', ''))
         if item[9] is not None
         else 0.0
         for item in matches]  # K
dipole = [float(item[10].replace(' ', ''))
          if item[10] is not None
          else 0.0
          for item in matches]  # Debye


table_indexes_of_comp_nos = [
    no.index(comp_no) for comp_no in sel_component_numbers
]

print('======================')

props = ['no', 'formula', 'name', 'cas_no',
         'delHf0', 'delGf0', 'delHb', 'delHm',
         'v_liq', 't_liq', 'dipole']

print('Values in Table:')
for prop in props:
  print(prop)
  print(str([
      (globals()[prop])[comp_no]
      for comp_no in table_indexes_of_comp_nos
  ]).replace('[', '').replace(']', ''))
  print('')


file = open(file_name_ig_heat_capacities, mode='r', encoding='utf-8-sig')
matches = []
k = 0
print('======================')
for line in file:
  k += 1
  # print header
  if k < 5:
    print(line)
  match = re.search(test_exp_ig_heat_capacities, line)
  if match is not None:
    matches.append(match.groups())

file.close()
print('Ideal Gas and Liquid Heat Capacities: ' + str(len(matches)) +
      ' Components in table')


no = [int(item[0]) for item in matches]
formula = [item[1] for item in matches]
name = [item[2] for item in matches]
cas_no = [item[3] for item in matches]
trange = [item[4] for item in matches]
a0 = [float(item[5].replace(' ', ''))
          if item[5] is not None
          else 0.0
          for item in matches]
a1 = [1e-3*float(item[6].replace(' ', ''))
         if item[6] is not None
         else 0.0
         for item in matches]
a2 = [1e-5*float(item[7].replace(' ', ''))
         if item[7] is not None
         else 0.0
         for item in matches]
a3 = [1e-8*float(item[8].replace(' ', ''))
         if item[8] is not None
         else 0.0
         for item in matches]
a4 = [1e-11*float(item[9].replace(' ', ''))
         if item[9] is not None
         else 0.0
         for item in matches] 
cpig = [float(item[10].replace(' ', ''))
          if item[10] is not None
          else 0.0
          for item in matches]
cpliq = [float(item[11].replace(' ', ''))
          if item[11] is not None
          else 0.0
          for item in matches]  # J/mol/K

cpig_test = [8.3145 * (
        a0[i] + a1[i] * 298.15 + a2[i] * 298.15**2 +
        a3[i] * 298.15**3 + a4[i] * 298.15**4
) for i in range(len(matches))]# J/mol/K

table_indexes_of_comp_nos = [
    no.index(comp_no) for comp_no in sel_component_numbers
]

print('======================')

props = ['no', 'formula', 'name', 'cas_no',
         'trange', 'a0', 'a1', 'a2', 
         'a3', 'a4', 'cpig', 'cpliq',
         'cpig_test']

print('Values in Table:')
for prop in props:
  print(prop)
  print(str([
      (globals()[prop])[comp_no]
      for comp_no in table_indexes_of_comp_nos
  ]).replace('[', '').replace(']', ''))
  print('')