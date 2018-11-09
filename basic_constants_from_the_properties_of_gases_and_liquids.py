import os
import re
import locale
locale.setlocale(locale.LC_ALL, '')  # decimals according to locale

out_file_name = './logs/output_basic_constants.csv'
sep_char_for_csv = '|'

out_file = open(out_file_name, mode='w')
out_file_full_path = os.path.abspath(out_file_name)


def str_list_to_np_array_str(param):
    return 'np.array([' + param + '])'


#sel_component_numbers = [130, 460, 461, 463, 95]
sel_component_numbers = [461, 455, 460, 463, 465]
#sel_component_numbers = [130, 460, 31, 440, 455]
#sel_component_numbers = [66, 438, ]
out_file.write('sep=' + sep_char_for_csv + '\n')
out_file.write("""
# Source of data:
# Poling, Bruce E., John M. Prausnitz,
# and John P. O'connell.
# The properties of gases and liquids.
# Vol. 5. New York: Mcgraw-hill, 2001.
# Basic Constants I: 468 Components in table
""")
out_file.write('\n')

test_exp_descriptors = '(^[0-9]{1,3})\|(.*)\|(.*)\|' + \
    '([0-9]{1,7}-[0-9]{1,7}-[0-9]{1,7})'
test_exp_basic_constants_i = test_exp_descriptors + \
    '[\||\ ](\-?\ ?[0-9]+\.?[0-9]+)?' * 8  # 8 numeric columns
test_exp_basic_constants_ii = test_exp_descriptors + \
    '[\||\ ](\-?\ ?[0-9]+\.?[0-9]+)?' * 7  # 7 numeric columns
test_exp_ig_heat_capacities = test_exp_descriptors + \
    '\|(—|[0-9]{1,4}-[0-9]{1,4})?' + \
    '[\||\ ](\-?\ ?[0-9]+\.?[0-9]+)?' * 7  # 7 numeric columns

file_name_basic_constants_i = os.path.abspath(
    './data/basic_constants_i_properties_of_gases_and_liquids.csv'
)
file_name_basic_constants_ii = os.path.abspath(
    './data/basic_constants_ii_properties_of_gases_and_liquids.csv'
)
file_name_ig_heat_capacities = os.path.abspath(
    './data/ig_l_heat_capacities_properties_of_gases_and_liquids.csv'
)

file = open(file_name_basic_constants_i, mode='r', encoding='utf-8-sig')
matches = []
k = 0
out_file.write('# ======================')
out_file.write('\n')
for line in file:
    k += 1
    # print header
    if k < 4:
        out_file.write(line)
        out_file.write('\n')
    match = re.search(test_exp_basic_constants_i, line)
    if match is not None:
        matches.append(match.groups())

file.close()
out_file.write('# Basic Constants I: ' + str(len(matches)) +
               ' Components in table')
out_file.write('\n')

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

out_file.write('# ======================')
out_file.write('\n')

props = ['no', 'formula', 'name', 'cas_no',
         'mol_wt', 'tfp', 'tb', 'tc',
         'pc', 'zc', 'omega']

out_file.write('# Values in Table:')
out_file.write('\n')
for prop in props:
    is_numeric_prop = not isinstance((globals()[prop])[0], str)
    out_file.write(prop)
    if is_numeric_prop:
        for comp_no in table_indexes_of_comp_nos:
            out_file.write(' | ' +
                           locale.str((globals()[prop])[comp_no])
                           )
    else:
        for comp_no in table_indexes_of_comp_nos:
            out_file.write(' | ' +
                           (globals()[prop])[comp_no]
                           )
    out_file.write('\n')


file = open(file_name_basic_constants_ii, mode='r', encoding='utf-8-sig')
matches = []
k = 0
out_file.write('# ======================')
out_file.write('\n')
for line in file:
    k += 1
    # print header
    if k < 9:
        out_file.write(line)
        out_file.write('\n')
    match = re.search(test_exp_basic_constants_ii, line)
    if match is not None:
        matches.append(match.groups())

file.close()
out_file.write('# Basic Constants II: ' + str(len(matches)) +
               ' Components in table')
out_file.write('\n')

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

out_file.write('# ======================')
out_file.write('\n')

props = ['no', 'formula', 'name', 'cas_no',
         'delHf0', 'delGf0', 'delHb', 'delHm',
         'v_liq', 't_liq', 'dipole']

out_file.write('# Values in Table:')
out_file.write('\n')
for prop in props:
    is_numeric_prop = not isinstance((globals()[prop])[0], str)
    out_file.write(prop)
    if is_numeric_prop:
        for comp_no in table_indexes_of_comp_nos:
            out_file.write(' | ' +
                           locale.str((globals()[prop])[comp_no])
                           )
    else:
        for comp_no in table_indexes_of_comp_nos:
            out_file.write(' | ' +
                           (globals()[prop])[comp_no]
                           )
    out_file.write('\n')


file = open(file_name_ig_heat_capacities, mode='r', encoding='utf-8-sig')
matches = []
k = 0
out_file.write('# ======================')
out_file.write('\n')
for line in file:
    k += 1
    # print header
    if k < 5:
        out_file.write(line)
        out_file.write('\n')
    match = re.search(test_exp_ig_heat_capacities, line)
    if match is not None:
        matches.append(match.groups())

file.close()
out_file.write('# Ideal Gas and Liquid Heat Capacities: ' + str(len(matches)) +
               ' Components in table')
out_file.write('\n')


no = [int(item[0]) for item in matches]
formula = [item[1] for item in matches]
name = [item[2] for item in matches]
cas_no = [item[3] for item in matches]
trange = [item[4] for item in matches]
a0 = [float(item[5].replace(' ', ''))
      if item[5] is not None
      else 0.0
      for item in matches]
a1 = [1e-3 * float(item[6].replace(' ', ''))
      if item[6] is not None
      else 0.0
      for item in matches]
a2 = [1e-5 * float(item[7].replace(' ', ''))
      if item[7] is not None
      else 0.0
      for item in matches]
a3 = [1e-8 * float(item[8].replace(' ', ''))
      if item[8] is not None
      else 0.0
      for item in matches]
a4 = [1e-11 * float(item[9].replace(' ', ''))
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
) for i in range(len(matches))]  # J/mol/K

table_indexes_of_comp_nos = [
    no.index(comp_no) for comp_no in sel_component_numbers
]

out_file.write('# ======================')
out_file.write('\n')

props = ['no', 'formula', 'name', 'cas_no',
         'trange', 'a0', 'a1', 'a2',
         'a3', 'a4', 'cpig', 'cpliq',
         'cpig_test']

out_file.write('# Values in Table:')
out_file.write('\n')
for prop in props:
    is_numeric_prop = not isinstance((globals()[prop])[0], str)
    out_file.write(prop)
    if is_numeric_prop:
        for comp_no in table_indexes_of_comp_nos:
            out_file.write(' | ' +
                           locale.str((globals()[prop])[comp_no])
                           )
    else:
        for comp_no in table_indexes_of_comp_nos:
            out_file.write(' | ' +
                           (globals()[prop])[comp_no]
                           )
    out_file.write('\n')

out_file.close()

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')  # decimals to US
out_file = open(out_file_name, mode='r')
i = 0
for line in out_file:
    if i > 1:
        line_to_print = line.replace('\n', '')
        if len(line_to_print) >= 1  \
                and line_to_print[0] != "#"  \
                and line_to_print.count('|') > 1:
            parts = [x.strip().replace(',', '.')
                     for x in line_to_print.split('|')]
            prop = parts[0]
            line_to_print = prop + ' = '
            if prop not in [
                'No.',
                'no',
                'formula',
                'name',
                'cas_no',
                    'trange']:
                for i in range(1, len(parts)):
                    parts[i] = float(parts[i])
            line_to_print += str_list_to_np_array_str(str(
                parts[1:]).replace('[', '').replace(']', '')
            )
        print(line_to_print)
    i += 1
out_file.close()
print('')
print('saved to ' + out_file_full_path)
