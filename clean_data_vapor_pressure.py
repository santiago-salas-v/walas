f1 = open('./data/vapor_pressure_correlations_parameters.csv', 'rb')
f2 = open('./data/vapor_pressure_correlations_parameters_clean.csv', 'wb')
i = 0
line0 = ''
for line in f1:
    i += 1
    if i <= 3:
        f2.write(line)
        print(line)
    else:
        if len(line0) == 0:
            # init line0
            line0 = line.split(b'|')
        line1 = line.split(b'|')
        line2 = []
        if len(line1[0]) == 0:
            for j in range(4):
                line2 += [line0[j]]
        else:
            for j in range(4):
                line2 += [line1[j]]
            line0 = line1
        for j in range(4, len(line1)):
            if line1[j] == b'640.5.0\r\n':
                line2 += [b'640.50\r\n']
            elif b'*' in line1[j]:
                line2 += [line1[j].replace(b'*', b'')]
            else:
                line2 += [line1[j]]
        if len(line1[4]) > 0:
            f2.write(b'|'.join(line2))
            print(b'|'.join(line2))
f1.close()
f2.close()