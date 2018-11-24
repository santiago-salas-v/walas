import os, sys, re
import locale
from numpy import loadtxt
from functools import partial
import lxml.etree as ET
import csv, io
from PyQt5.QtWidgets import QTableView, QApplication, QWidget
from PyQt5.QtWidgets import QDesktopWidget, QGridLayout, QLineEdit
from PyQt5.QtWidgets import QComboBox, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt, QVariant, QEvent
from PyQt5.QtGui import QKeySequence, QGuiApplication, QFont
from PyQt5.QtCore import pyqtSignal as SIGNAL


locale.setlocale(locale.LC_ALL, '')
burcat_xml_file = './data/BURCAT_THR.xml'
yaws_csv_file = './data/The-Yaws-Handbook-of-Vapor-Pressure-Second-Edition-Antoine-coefficients_2.csv'

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Thermodynamic data from burcat'

        screen = QDesktopWidget().screenGeometry()
        width = 600
        height = 300

        x = screen.width()/2 - width/2
        y = screen.height()/2 - height/2

        self.width = width
        self.height = height
        self.left = x
        self.top = y
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.tableView1 = QTableView()
        self.tableWidget1 = QTableWidget()
        self.model = thTableModel()
        self.cas_filter = QLineEdit()
        self.name_filter = QLineEdit()
        self.formula_filter = QLineEdit()
        self.phase_filter = QComboBox()

        self.tableView1.setModel(self.model)
        self.tableWidget1.added_column_names = ['']
        self.tableWidget1.setColumnCount(self.tableView1.model().columnCount())
        self.tableWidget1.setHorizontalHeaderLabels(self.tableView1.model().column_names_burcat)
        self.phase_filter.addItems(['', 'G', 'L', 'S', 'C'])
        self.tableWidget1.setEnabled(False)
        self.tableWidget1.setSelectionBehavior(QTableWidget.SelectRows)
        self.tableWidget1.installEventFilter(self)

        # Add box layout, add table to box layout and add box layout to widget
        self.layout = QGridLayout()
        self.layout.addWidget(self.tableView1, 1, 1, 1, 4)
        self.layout.addWidget(self.cas_filter, 3, 1, 1, 1)
        self.layout.addWidget(self.name_filter, 3, 2, 1, 1)
        self.layout.addWidget(self.formula_filter, 3, 3, 1, 1)
        self.layout.addWidget(self.phase_filter, 3, 4, 1, 1)
        self.layout.addWidget(QLabel('find by: cas'), 2, 1, 1, 1)
        self.layout.addWidget(QLabel('name'), 2, 2, 1, 1)
        self.layout.addWidget(QLabel('formula'), 2, 3, 1, 1)
        self.layout.addWidget(QLabel('phase'), 2, 4, 1, 1)
        self.layout.addWidget(QLabel('selection'), 4, 1, 1, 4)
        self.layout.addWidget(self.tableWidget1, 5, 1, 1, 4)



        self.setLayout(self.layout)

        for item in [self.cas_filter, self.name_filter, self.formula_filter]:
            item.textChanged.connect(partial(
                self.model.apply_filter,
                self.cas_filter, self.name_filter, self.formula_filter, self.phase_filter))
        self.phase_filter.currentTextChanged.connect(partial(
                self.model.apply_filter,
                self.cas_filter, self.name_filter, self.formula_filter, self.phase_filter))
        self.tableView1.selectionModel().selectionChanged.connect(partial(
            self.add_selection_to_widget
        ))

        # Show widget
        self.show()

    def add_selection_to_widget(self):
        indexes = self.tableView1.selectedIndexes()
        index = indexes[0]
        column_of_cas = self.tableView1.model().column_names_burcat.index('cas')
        column_of_phase = self.tableView1.model().column_names_burcat.index('phase')
        phase = self.tableView1.model().index(index.row(), column_of_phase).data()
        cas = self.tableView1.model().index(index.row(), column_of_cas).data()
        already_in_table = False
        for item in self.tableWidget1.findItems(cas, Qt.MatchExactly):
            if phase == self.tableWidget1.item(item.row(), column_of_phase).text():
                already_in_table = True
        if not already_in_table:
            self.tableWidget1.setRowCount(self.tableWidget1.rowCount() + 1)
            for column_name in self.tableView1.model().column_names_burcat:
                column_no = self.tableView1.model().column_names_burcat.index(column_name)
                data = self.tableView1.model().index(index.row(), column_no).data()
                if type(data) != str:
                    item_to_add = QTableWidgetItem(locale.str(data))
                else:
                    item_to_add = QTableWidgetItem(data)
                item_to_add.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled| Qt.ItemIsEditable)
                self.tableWidget1.setItem(
                    self.tableWidget1.rowCount()-1, column_no,
                    item_to_add)
            column_no += 1
        if len(indexes) > 0 or self.tableWidget1.rowCount() > 0:
            self.tableWidget1.setEnabled(True)


    def copy_selection(self):
        col_names = self.tableView1.model().column_names_burcat
        selection = self.tableWidget1.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = range(len(col_names))
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount+1)]
            table[0] = col_names
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row+1][column] = index.data()
            stream = io.StringIO()
            csv.writer(stream, delimiter=';', quoting=csv.QUOTE_NONE).writerows(table)
            QApplication.clipboard().setText(stream.getvalue())

    def eventFilter(self, source, event):
        if (event.type() == QEvent.KeyPress and
                event.matches(QKeySequence.Copy)):
            self.copy_selection()
            return True
        if (event.type() == QEvent.KeyPress and
                event.matches(QKeySequence.Delete)):
            if len(self.tableWidget1.selectedIndexes()) > 1:
                self.delete_selected()
            return True
        return super(App, self).eventFilter(source, event)

    def delete_selected(self):
        while len(self.tableWidget1.selectedIndexes()) > 0:
            self.tableWidget1.removeRow(self.tableWidget1.selectedIndexes()[0].row())
        if self.tableWidget1.rowCount() == 0:
            self.tableWidget1.setEnabled(False)


def helper_func1(x):
    # helper function for csv reading
    str_with_dot_actually_minus = x
    matches = re.search(b'(\.(\d+\.\d+))', x)
    if matches:
        str_with_dot_actually_minus = b'-' + matches.groups()[1]
    return str_with_dot_actually_minus.replace(b' ', b'')


class thTableModel(QAbstractTableModel):

    def __init__(self):
        super(thTableModel, self).__init__()
        self.tree = ET.parse(burcat_xml_file)
        self.root = self.tree.getroot()
        self.cas_filter = ''
        self.name_filter = ''
        self.formula_filter = ''
        self.phase_filter = ''

        self.ant = loadtxt(
            open(yaws_csv_file, 'rb'),
            delimiter='|',
            skiprows=2,
            dtype={
                'names': [
                    'ant_no', 'ant_formula', 'ant_name',
                    'ant_cas', 'ant_a', 'ant_b',
                    'ant_c', 'ant_tmin', 'ant_tmax',
                    'ant_code'],
                'formats': [
                    int, object, object,
                    object, float, float,
                    float, float, float, object]},
            converters={
                0: helper_func1, 4: helper_func1,
                5: helper_func1, 6: helper_func1,
                7: helper_func1
            })
        #self.xpath = \
        #    ".//formula_name_structure[contains(formula_name_structure_1, '" + \
        #    self.name_filter + "')]/" + "../../specie[contains(@CAS, '')]/" +\
        #    "phase[contains(phase, '')]"
        self.xpath = \
            "./specie[contains(@CAS, '" + \
            self.cas_filter + "')]/" + \
            "formula_name_structure[contains(formula_name_structure_1, '" + \
            self.name_filter + "')]/../" + \
            "phase[contains(formula, '" + \
            self.formula_filter + "')]/../" + \
            "phase[contains(phase, '" + \
            self.phase_filter + "')]"
        self.results = self.root.xpath(self.xpath)
        self.column_names_burcat = [
            'cas', 'formula_name_structure_1',
            'phase', 'formula', 'molecular_weight',
            'hf298_div_r', 'range_tmin_to_1000',
            'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7',
            'range_1000_to_tmax',
            'a1_high', 'a2_high', 'a3_high', 'a4_high',
            'a5_high', 'a6_high', 'a7_high']
        self.column_names_ant = [
            'ant_no', 'ant_formula', 'ant_name',
            'ant_cas', 'ant_a', 'ant_b',
            'ant_c', 'ant_tmin', 'ant_tmax',
            'ant_code'
        ]

    def apply_filter(self, cas_filter, name_filter, formula_filter, phase_filter):
        self.name_filter = name_filter.text().upper()
        self.cas_filter = cas_filter.text().upper()
        self.formula_filter = formula_filter.text().upper()
        self.phase_filter = phase_filter.currentText()
        self.beginResetModel()
        self.xpath = \
            "./specie[contains(@CAS, '" + \
            self.cas_filter + "')]/" + \
            "formula_name_structure[contains(formula_name_structure_1, '" + \
            self.name_filter + "')]/../" + \
            "phase[contains(formula, '" + \
            self.formula_filter + "')]/../" + \
            "phase[contains(phase, '" + \
            self.phase_filter + "')]"
        self.results = self.root.xpath(self.xpath)
        self.endResetModel()


    def data(self, index, role=Qt.DisplayRole):
        comp_phase = self.results[index.row()]
        phase_parent = comp_phase.getparent()
        cas = phase_parent.get('CAS')
        formula_name_structure_1 = phase_parent.find(
            'formula_name_structure').find(
            'formula_name_structure_1').text
        phase = comp_phase.find('phase').text
        formula = comp_phase.find('formula').text
        range_tmin_to_1000 = float(
            comp_phase.find('temp_limit').get('low').replace(' ', ''))
        range_1000_to_tmax = float(
            comp_phase.find('temp_limit').get('high').replace(' ', ''))
        molecular_weight = float(
            comp_phase.find('molecular_weight').text.replace(' ', ''))
        hf298_div_r = float(comp_phase.find(
            'coefficients').find('hf298_div_r').text.replace(' ', ''))
        #for coef in ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']:
        for coef in comp_phase.find('coefficients').find(
                'range_Tmin_to_1000').getiterator('coef'):
            if coef.get('name') == 'a1':
                a1 = float(coef.text.replace(' ', ''))
            if coef.get('name') == 'a2':
                a2 = float(coef.text.replace(' ', ''))
            if coef.get('name') == 'a3':
                a3 = float(coef.text.replace(' ', ''))
            if coef.get('name') == 'a4':
                a4 = float(coef.text.replace(' ', ''))
            if coef.get('name') == 'a5':
                a5 = float(coef.text.replace(' ', ''))
            if coef.get('name') == 'a6':
                a6 = float(coef.text.replace(' ', ''))
            if coef.get('name') == 'a7':
                a7 = float(coef.text.replace(' ', ''))
        for coef in comp_phase.find('coefficients').find(
                'range_1000_to_Tmax').getiterator('coef'):
            if coef.get('name') == 'a1':
                a1_high = float(coef.text.replace(' ', ''))
            if coef.get('name') == 'a2':
                a2_high = float(coef.text.replace(' ', ''))
            if coef.get('name') == 'a3':
                a3_high = float(coef.text.replace(' ', ''))
            if coef.get('name') == 'a4':
                a4_high = float(coef.text.replace(' ', ''))
            if coef.get('name') == 'a5':
                a5_high = float(coef.text.replace(' ', ''))
            if coef.get('name') == 'a6':
                a6_high = float(coef.text.replace(' ', ''))
            if coef.get('name') == 'a7':
                a7_high = float(coef.text.replace(' ', ''))

        ant_record = self.ant[self.ant['ant_cas'] == cas]
        ant_no, ant_formula, ant_name, \
        ant_cas, ant_a, ant_b, ant_c, \
        ant_tmin, ant_tmax, ant_code = [None for i in range(len(self.column_names_ant))]
        if len(ant_record) == 1:
            ant_no, ant_formula, ant_name, \
            ant_cas,  ant_a, ant_b, ant_c, \
            ant_tmin, ant_tmax, ant_code = ant_record[0]
            ant_no = int(ant_no)
            ant_a = float(ant_a)
            ant_b = float(ant_b)
            ant_c = float(ant_c)
            ant_tmin = float(ant_tmin)
            ant_tmax = float(ant_tmax)

        if not index.isValid() or \
                index.row() < 0 or index.row() > len(self.results) or \
                index.column() < 0 or index.column() > self.columnCount():
            return QVariant()
        column = index.column()
        if role == Qt.DisplayRole:
            if column < len(self.column_names_burcat):
                return QVariant(locals()[self.column_names_burcat[column]])
            else:
                return QVariant(locals()[self.column_names_ant[column-len(self.column_names_burcat)]])

    def headerData(self, section: int, orientation: Qt.Orientation, role: int= Qt.DisplayRole):
        if role == Qt.TextAlignmentRole:
            if orientation == Qt.Horizontal:
                return QVariant(int(Qt.AlignLeft|Qt.AlignVCenter))
            return QVariant(int(Qt.AlignRight|Qt.AlignVCenter))
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            if section < len(self.column_names_burcat):
                return QVariant(self.column_names_burcat[section])
            else:
                return QVariant(self.column_names_ant[section-len(self.column_names_burcat)])
        if orientation == Qt.Vertical:
            return QVariant(section + 1)


    def rowCount(self, index=QModelIndex()):
        return len(self.results)

    def columnCount(self, index=QModelIndex()):
        # CAS, formula_name_structure_1, (phase)phase ,
        # (phase)formula, (phase)temp_limit low,
        # (phase)molecular_weight,
        # (phase)coefficients range_Tmin_to_1000 a1,
        # (phase)coefficients range_Tmin_to_1000 a2,
        # (phase)coefficients range_Tmin_to_1000 a3,
        # (phase)coefficients range_Tmin_to_1000 a4,
        # (phase)coefficients range_Tmin_to_1000 a5,
        # (phase)coefficients range_Tmin_to_1000 a6,
        # (phase)coefficients range_Tmin_to_1000 a7,
        # (phase)coefficients hf298_div_r
        return len(self.column_names_burcat) + len(self.column_names_ant)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    QApplication.setFont(QFont('Consolas', 10))
    ex = App()
    sys.exit(app.exec_())