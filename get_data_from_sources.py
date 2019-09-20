from html.parser import HTMLParser
import urllib.request

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
for line in urllib.request.urlopen(ddbst_adress):
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