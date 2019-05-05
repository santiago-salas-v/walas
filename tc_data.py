import os
import sys
import re
import string
import locale
from numpy import loadtxt, asscalar, argwhere, in1d, vectorize
from functools import partial
import lxml.etree as ET
import csv
import io
from pandas import DataFrame, merge
import ctypes  # Needed to set the app icon correctly
from PyQt5.QtWidgets import QTableView, QApplication, QWidget
from PyQt5.QtWidgets import QDesktopWidget, QGridLayout, QLineEdit, QPushButton
from PyQt5.QtWidgets import QComboBox, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt, QVariant, QEvent
from PyQt5.QtGui import QKeySequence, QGuiApplication, QFont, QIcon
from PyQt5.QtCore import pyqtSignal as SIGNAL


locale.setlocale(locale.LC_ALL, '')
burcat_xml_file = './data/BURCAT_THR.xml'
antoine_csv = './data/The-Yaws-Handbook-of-Vapor-Pressure-Second-Edition-Antoine-coefficients.csv'
poling_basic_i_csv = './data/basic_constants_i_properties_of_gases_and_liquids.csv'
poling_basic_ii_csv = './data/basic_constants_ii_properties_of_gases_and_liquids.csv'
poling_cp_l_ig_poly_csv = './data/ig_l_heat_capacities_properties_of_gases_and_liquids.csv'
template = "./data/xsl_stylesheet_burcat.xsl"


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Thermodynamic data from burcat'

        screen = QDesktopWidget().screenGeometry()
        width = 600
        height = 300

        x = screen.width() / 2 - width / 2
        y = screen.height() / 2 - height / 2

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
        self.delete_selected_button = QPushButton()
        self.copy_selected_button = QPushButton()

        self.tableView1.setModel(self.model)
        self.tableWidget1.added_column_names = ['']
        self.tableWidget1.setColumnCount(self.tableView1.model().columnCount())
        self.tableWidget1.setHorizontalHeaderLabels(
            self.tableView1.model().column_names
        )
        self.phase_filter.addItems(['', 'G', 'L', 'S', 'C'])
        self.tableWidget1.setEnabled(False)
        self.tableWidget1.setSelectionBehavior(QTableWidget.SelectRows)
        self.tableWidget1.installEventFilter(self)
        self.delete_selected_button.setText('delete selected')
        self.copy_selected_button.setText('copy selected')

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
        self.layout.addWidget(self.delete_selected_button, 6, 4, 1, 1)
        self.layout.addWidget(self.copy_selected_button, 6, 1, 1, 1)

        self.setLayout(self.layout)

        for item in [self.cas_filter, self.name_filter, self.formula_filter]:
            item.textChanged.connect(self.apply_filters)
        self.phase_filter.currentTextChanged.connect(self.apply_filters)
        self.tableView1.selectionModel().selectionChanged.connect(partial(
            self.add_selection_to_widget
        ))
        self.delete_selected_button.clicked.connect(partial(
            self.delete_selected
        ))
        self.copy_selected_button.clicked.connect(partial(
            self.copy_selection
        ))

        # Show widget
        self.show()

    def apply_filters(self):
        cas_filter = self.cas_filter.text().upper().strip().replace(' ', '')
        name_filter = self.name_filter.text().upper().strip()
        formula_filter = self.formula_filter.text().upper().strip()
        phase_filter = self.phase_filter.currentText()
        self.model.apply_filter(
            cas_filter,
            name_filter,
            formula_filter,
            phase_filter)

    def add_selection_to_widget(self):
        indexes = self.tableView1.selectedIndexes()
        index = indexes[0]
        column_of_cas = self.tableView1.model().column_names.index('cas')
        column_of_phase = self.tableView1.model().column_names.index('phase')
        phase = self.tableView1.model().index(index.row(), column_of_phase).data()
        cas = self.tableView1.model().index(index.row(), column_of_cas).data()
        already_in_table = False
        for item in self.tableWidget1.findItems(cas, Qt.MatchExactly):
            if phase == self.tableWidget1.item(
                    item.row(), column_of_phase).text():
                already_in_table = True
        if not already_in_table:
            column_names = self.tableView1.model().column_names
            self.tableWidget1.setRowCount(self.tableWidget1.rowCount() + 1)
            for i in range(len(column_names)):
                data = self.tableView1.model().index(index.row(), i).data()
                if isinstance(data, str) or data is None:
                    item_to_add = QTableWidgetItem(data)
                else:
                    item_to_add = QTableWidgetItem(locale.str(data))
                item_to_add.setFlags(
                    Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
                self.tableWidget1.setItem(
                    self.tableWidget1.rowCount() - 1, i,
                    item_to_add)
        if len(indexes) > 0 or self.tableWidget1.rowCount() > 0:
            self.tableWidget1.setEnabled(True)

    def copy_selection(self):
        col_names = self.tableView1.model().column_names
        selection = self.tableWidget1.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = range(len(col_names))
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount + 1)]
            table[0] = col_names
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row + 1][column] = index.data()
            stream = io.StringIO()
            csv.writer(
                stream,
                delimiter=';',
                quoting=csv.QUOTE_NONE).writerows(table)
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
        if len(self.tableWidget1.selectedIndexes()) > 0:
            current_row = self.tableWidget1.selectedIndexes()[0].row()
            while len(self.tableWidget1.selectedIndexes()) > 0:
                current_row = self.tableWidget1.selectedIndexes()[0].row()
                self.tableWidget1.removeRow(current_row)
            self.tableWidget1.selectRow(current_row)
            if self.tableWidget1.rowCount() == 0:
                self.tableWidget1.setEnabled(False)


def helper_func1(x):
    # helper function for csv reading
    # replace dot in original file for minus in some cases ocr'd like .17.896
    str_with_dot_actually_minus = x
    matches = re.search(b'(\.(\d+\.\d+))', x)
    if matches:
        str_with_dot_actually_minus = b'-' + matches.groups()[1]
    return str_with_dot_actually_minus.replace(b' ', b'')


def helper_func2(x):
    # helper function 2 for csv reading.
    # Return None when empty string
    if len(x) == 0:
        return 'nan'
    return x.replace(b' ', b'')


def helper_func3(x):
    # helper function 2 for csv reading.
    # replace - for . in cas numbers (Antoine coeff)
    return x.replace(b'.', b'-').decode('utf-8')


def indexes_containing_string(array_a, string_x):
    if len(array_a) < 1:
        return []
    return vectorize(lambda y: string_x.upper() in y.upper())(array_a)


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
            open(antoine_csv, 'rb'),
            delimiter='|',
            skiprows=9,
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
                0: helper_func1, 3: helper_func3,
                4: helper_func1, 5: helper_func1,
                6: helper_func1, 7: helper_func1
            })

        self.poling_basic_i = loadtxt(
            open(poling_basic_i_csv, 'rb'),
            delimiter='|',
            skiprows=3,
            dtype={
                'names': [
                    'poling_no', 'poling_formula', 'poling_name',
                    'poling_cas', 'poling_molwt', 'poling_tfp',
                    'poling_tb', 'poling_tc', 'poling_pc',
                    'poling_Vc', 'poling_Zc', 'poling_omega'],
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
        )

        self.poling_basic_ii = loadtxt(
            open(poling_basic_ii_csv, 'rb'),
            delimiter='|',
            skiprows=3,
            usecols=(3, 4, 5, 6, 7, 8, 9, 10),
            dtype={
                'names': [
                    'poling_cas',
                    'poling_delhf0', 'poling_delgf0', 'poling_delhb',
                    'poling_delhm', 'poling_v_liq', 'poling_t_liq',
                    'poling_dipole'],
                'formats': [
                    object, float, float,
                    float, float, float, float, float]},
            converters={
                4: helper_func2, 5: helper_func2,
                6: helper_func2, 7: helper_func2,
                8: helper_func2, 9: helper_func2,
                10: helper_func2
            }
        )

        self.poling_cp_ig_l = loadtxt(
            open(poling_cp_l_ig_poly_csv, 'rb'),
            delimiter='|',
            skiprows=4,
            usecols=(3, 4, 5, 6, 7, 8, 9, 10, 11),
            dtype={
                'names': [
                    'poling_cas',
                    'poling_trange', 'poling_a0', 'poling_a1',
                    'poling_a2', 'poling_a3', 'poling_a4',
                    'poling_cpig', 'poling_cpliq'],
                'formats': [
                    object, object, float,
                    float, float, float, float, float, float]},
            converters={
                5: helper_func2, 6: helper_func2,
                7: helper_func2, 8: helper_func2,
                9: helper_func2, 10: helper_func2,
                11: helper_func2
            }
        )
        self.poling_cp_ig_l['poling_a1'] = self.poling_cp_ig_l['poling_a1'] * 1e-3
        self.poling_cp_ig_l['poling_a2'] = self.poling_cp_ig_l['poling_a2'] * 1e-5
        self.poling_cp_ig_l['poling_a3'] = self.poling_cp_ig_l['poling_a3'] * 1e-8
        self.poling_cp_ig_l['poling_a4'] = self.poling_cp_ig_l['poling_a4'] * 1e-11

        self.column_names_burcat = [
            'cas', 'formula_name_structure_1',
            'phase', 'formula', 'molecular_weight',
            'hf298_div_r', 'range_tmin_to_1000',
            'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7',
            'range_1000_to_tmax',
            'a1_high', 'a2_high', 'a3_high', 'a4_high',
            'a5_high', 'a6_high', 'a7_high']
        self.column_names_ant = list(self.ant.dtype.names)
        self.column_names_poling_basic_i = list(
            self.poling_basic_i.dtype.names)
        self.column_names_poling_basic_ii = list(
            self.poling_basic_ii.dtype.names)
        self.column_names_poling_cp_ig_l = list(
            self.poling_cp_ig_l.dtype.names)
        self.column_names = \
            self.column_names_poling_basic_i + self.column_names_burcat + \
            self.column_names_poling_basic_ii + self.column_names_poling_cp_ig_l + \
            self.column_names_ant


        all_burcat_phases_cas_nos = self.root.xpath(
            "./specie/formula_name_structure/../phase/phase/../../@CAS")
        all_burcat_phases = self.root.xpath(
            "./specie/formula_name_structure/../phase/phase/../..")
        # for item in :
        #    if item in self.poling_basic_i['poling_cas']:
        #        self.poling_basic_burcat_intersects += [item]
        # for item in self.poling_basic_i
        self.apply_filter('', '', '', '')

    def apply_filter(
            self,
            cas_filter,
            name_filter,
            formula_filter,
            phase_filter):
        self.cas_filter = cas_filter
        self.name_filter = name_filter
        self.formula_filter = formula_filter
        self.phase_filter = phase_filter
        self.beginResetModel()
        # without filters, all phases:
        # self.root.xpath('./specie/formula_name_structure/../phase/../phase')
        # with filters:
        self.xpath = \
            "./specie[contains(@CAS, '" + \
            self.cas_filter + "')]/" + \
            "formula_name_structure[contains(translate(" +\
            "formula_name_structure_1" + \
            ", '" + string.ascii_lowercase + "', '" + string.ascii_uppercase + "'), '" + \
            self.name_filter + "')]/../@CAS/../" + \
            "phase[contains(translate(" + \
            "formula" + \
            ", '" + string.ascii_lowercase + "', '" + string.ascii_uppercase + "'), '" + \
            self.formula_filter + "')]/../" + \
            "phase[contains(phase, '" + \
            self.phase_filter + "')]"
        self.burcat_results = self.root.xpath(self.xpath)
        self.poling_basic_i_results = self.poling_basic_i.copy()
        self.poling_basic_ii_results = self.poling_basic_ii.copy()
        self.poling_cp_ig_l_results = self.poling_cp_ig_l.copy()
        self.ant_results = self.ant.copy()

        self.poling_basic_i_results = self.poling_basic_i_results[
            indexes_containing_string(self.poling_basic_i_results['poling_cas'], self.cas_filter)
        ]
        self.poling_basic_i_results = self.poling_basic_i_results[
            indexes_containing_string(self.poling_basic_i_results['poling_name'], self.name_filter)
        ]
        self.poling_basic_i_results = self.poling_basic_i_results[
            indexes_containing_string(self.poling_basic_i_results['poling_formula'], self.formula_filter)
        ]

        self.ant_results = self.ant_results[
            indexes_containing_string(self.ant_results['ant_cas'], self.cas_filter)
        ]
        self.ant_results = self.ant_results[
            indexes_containing_string(self.ant_results['ant_name'], self.name_filter)
        ]
        self.ant_results = self.ant_results[
            indexes_containing_string(self.ant_results['ant_formula'], self.formula_filter)
        ]


        self.burcat_cas_nos = [x.getparent().get('CAS')
                               for x in self.burcat_results]
        self.poling_cas_nos = self.poling_basic_i_results['poling_cas']
        self.poling_basic_burcat_intersects_p = []
        self.poling_basic_burcat_intersects_b = []
        self.poling_basic_burcat_complements = list(
            range(len(self.burcat_results)))
        self.poling_basic_i_rows_to_table = []
        self.poling_basic_i_ii_intersects = []
        self.poling_basic_i_ii_complements = list(
            range(len(self.poling_basic_i_results)))
        self.poling_basic_cp_ig_l_intersects = []
        self.poling_basic_cp_ig_l_complements = list(
            range(len(self.poling_basic_i_results)))
        self.poling_basic_ant_intersects_p = []
        self.poling_basic_ant_intersects_a = []
        self.poling_basic_ant_complements = list(
            range(len(self.ant_results)))
        self.burcat_ant_intersects_a = []
        self.burcat_ant_intersects_b = []
        self.poling_basic_burcat_ant_complements_ant_part = list(
            range(len(self.ant_results)))

        for k, casno in enumerate(self.poling_cas_nos):
            indexes = []
            for i, j in enumerate(self.burcat_cas_nos):
                if j == casno:
                    indexes += [i]
                    self.poling_basic_burcat_complements.pop(
                        self.poling_basic_burcat_complements.index(i))
            if len(indexes) == 0:
                self.poling_basic_i_rows_to_table += [k]
                self.poling_basic_burcat_intersects_p += [[]]
                self.poling_basic_burcat_intersects_b += [[]]
            for i in indexes:
                self.poling_basic_burcat_intersects_p += [k]
                self.poling_basic_burcat_intersects_b += [i]
                self.poling_basic_i_rows_to_table += [k]

            if casno in self.poling_basic_ii_results['poling_cas']:
                self.poling_basic_i_ii_intersects += [k]
                self.poling_basic_i_ii_complements.pop(
                    self.poling_basic_i_ii_complements.index(k)
                )
            else:
                self.poling_basic_i_ii_intersects += [[]]

            if casno in self.poling_cp_ig_l_results['poling_cas']:
                self.poling_basic_cp_ig_l_intersects += [k]
                self.poling_basic_cp_ig_l_complements.pop(
                    self.poling_basic_cp_ig_l_complements.index(k)
                )
            else:
                self.poling_basic_cp_ig_l_intersects += [[]]

            ant_index = argwhere(self.ant_results['ant_cas'] == casno)
            if len(ant_index) > 0:
                ant_index = ant_index[0].item()
                self.poling_basic_ant_intersects_p += [k]
                self.poling_basic_ant_intersects_a += [ant_index]
                self.poling_basic_ant_complements.pop(
                    self.poling_basic_ant_complements.index(ant_index)
                )
            else:
                self.poling_basic_ant_intersects_p += [[]]
                self.poling_basic_ant_intersects_a += [[]]

        poling_burcat_complement_casnos = [
            self.burcat_cas_nos[x] for x in self.poling_basic_burcat_complements]
        poling_basic_burcat_ant_complement_casnos = [
            self.ant_results['ant_cas'][i] for i in self.poling_basic_ant_complements]
        indexes_of_ant_complements_in_poling_burcat_complements = argwhere(
            in1d(poling_basic_burcat_ant_complement_casnos, poling_burcat_complement_casnos))
        self.poling_basic_burcat_ant_complements_ant_part = [
            x for x in self.poling_basic_ant_complements]
        self.poling_basic_burcat_ant_complements_burcat_part = [
            x for x in self.poling_basic_burcat_complements]
        for k, ant_index in enumerate(
                self.poling_basic_burcat_ant_complements_ant_part):
            if k in indexes_of_ant_complements_in_poling_burcat_complements:
                casno = poling_basic_burcat_ant_complement_casnos[k]
                burcat_index = self.poling_basic_burcat_complements[
                    poling_burcat_complement_casnos.index(casno)]
                self.burcat_ant_intersects_a += [
                    self.poling_basic_ant_complements[k]]
                self.burcat_ant_intersects_b += [burcat_index]
                self.poling_basic_burcat_ant_complements_ant_part.pop(
                    self.poling_basic_burcat_ant_complements_ant_part.index(ant_index))
                if burcat_index in self.poling_basic_burcat_ant_complements_burcat_part:
                    self.poling_basic_burcat_ant_complements_burcat_part.pop(
                        self.poling_basic_burcat_ant_complements_burcat_part.index(burcat_index))
            else:
                pass

        # row counts
        count_ant = len(
            [x for x in self.poling_basic_ant_intersects_a if x != []])
        count_ant += len([x for x in self.poling_basic_burcat_ant_complements_ant_part if x != []])
        count_ant += len([x for x in self.burcat_ant_intersects_a if x != []])

        count_poling_basic = len(self.poling_basic_i_rows_to_table)

        count_burcat = len(
            [x for x in self.poling_basic_burcat_intersects_b if x != []])
        count_burcat += len([x for x in self.burcat_ant_intersects_b if x != []])
        count_burcat += len(
            [x for x in self.poling_basic_burcat_ant_complements_burcat_part if x != []])

        self.endResetModel()

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or \
                index.row() < 0 or \
                index.column() < 0 or index.column() > self.columnCount():
            return QVariant()
        if role == Qt.DisplayRole:
            column = index.column()
            row = index.row()
            sec_1 = len(self.column_names_poling_basic_i)
            sec_2 = len(self.column_names_poling_basic_i) + \
                len(self.column_names_burcat)
            sec_3 = len(self.column_names_poling_basic_i) + len(self.column_names_burcat) + \
                len(self.column_names_poling_basic_ii)
            sec_4 = len(self.column_names_poling_basic_i) + len(self.column_names_burcat) + \
                len(self.column_names_poling_basic_ii) + len(self.column_names_poling_cp_ig_l)
            sec_5 = len(self.column_names_poling_basic_i) + len(self.column_names_burcat) + \
                len(self.column_names_poling_basic_ii) + len(self.column_names_poling_cp_ig_l) +\
                len(self.column_names_ant)

            if column < sec_1:
                if row < len(self.poling_basic_i_rows_to_table):
                    cas_no = self.poling_basic_i_results['poling_cas'][
                        self.poling_basic_i_rows_to_table[row]
                    ]
                    record_poling_basic_i = self.poling_basic_i_results[
                        self.poling_basic_i_results['poling_cas'] == cas_no]
                    column_name = self.column_names_poling_basic_i[column]
                    return QVariant(
                        record_poling_basic_i[column_name].item())
                else:
                    return QVariant(None)
            elif column < sec_2:
                row_burcat = None
                if row < len(self.poling_basic_i_rows_to_table):
                    row_intercept = self.poling_basic_burcat_intersects_p[row]
                    if row_intercept != []:
                        row_burcat = self.poling_basic_burcat_intersects_b[row]
                    elif row in self.poling_basic_burcat_complements:
                        pass
                elif row < len(self.poling_basic_i_rows_to_table) \
                        + len(self.burcat_ant_intersects_a):
                    row_burcat = self.burcat_ant_intersects_b[
                        row - len(self.poling_basic_i_rows_to_table)]
                elif row < len(self.poling_basic_i_rows_to_table) \
                        + len(self.poling_basic_burcat_complements):
                    row_burcat = self.poling_basic_burcat_complements[
                        row - len(self.poling_basic_i_rows_to_table) -
                        len(self.burcat_ant_intersects_a)]
                if row_burcat is not None and row_burcat != []:
                    record_burcat = self.burcat_results[row_burcat]
                    column_name = self.column_names_burcat[column - len(
                        self.column_names_poling_basic_i)]
                    data = self.extract_info_from_burcat_phase(
                        record_burcat, column_name)
                    return QVariant(data)
                else:
                    return QVariant(None)
            elif column < sec_3:
                row_poling_basic_ii = None
                if row < len(self.poling_basic_i_rows_to_table):
                    table_row = self.poling_basic_i_rows_to_table[row]
                    if table_row in self.poling_basic_i_ii_intersects:
                        row_poling_basic_ii = self.poling_basic_i_ii_intersects[table_row]
                    elif table_row in self.poling_basic_i_ii_complements:
                        pass
                if row_poling_basic_ii is not None:
                    cas_no = self.poling_basic_i_results['poling_cas'][
                        self.poling_basic_i_rows_to_table[row]
                    ]
                    record_poling_basic_i = self.poling_basic_i_results[
                        self.poling_basic_i_results['poling_cas'] == cas_no]
                    record_poling_basic_ii = self.poling_basic_ii_results[
                        self.poling_basic_ii_results['poling_cas'] == cas_no]
                    column_name = self.column_names_poling_basic_ii[column - sec_2]
                    return QVariant(
                        record_poling_basic_ii[column_name].item())
                else:
                    return QVariant(None)
            elif column < sec_4:
                row_poling_cp_ig_l = None
                if row < len(self.poling_basic_i_rows_to_table):
                    table_row = self.poling_basic_i_rows_to_table[row]
                    if table_row in self.poling_basic_cp_ig_l_intersects:
                        row_poling_cp_ig_l = self.poling_basic_cp_ig_l_intersects[table_row]
                    elif table_row in self.poling_basic_cp_ig_l_complements:
                        pass
                if row_poling_cp_ig_l is not None:
                    cas_no = self.poling_basic_i_results['poling_cas'][
                        self.poling_basic_i_rows_to_table[row]
                    ]
                    record_poling_basic_i = self.poling_basic_i_results[
                        self.poling_basic_i_results['poling_cas'] == cas_no]
                    record_poling_cp_ig_l = self.poling_cp_ig_l_results[
                        self.poling_cp_ig_l_results['poling_cas'] ==
                        cas_no]
                    column_name = self.column_names_poling_cp_ig_l[column - sec_3]
                    return QVariant(
                        record_poling_cp_ig_l[column_name].item())
                else:
                    return QVariant(None)
            elif column < sec_5:
                row_ant = None
                if row < len(self.poling_basic_i_rows_to_table):
                    table_row = self.poling_basic_i_rows_to_table[row]
                    if table_row in self.poling_basic_ant_intersects_p:
                        row_ant = self.poling_basic_ant_intersects_p[table_row]
                        cas_no = self.poling_basic_i_results['poling_cas'][
                            self.poling_basic_i_rows_to_table[row]
                        ]
                        record_poling_basic_i = self.poling_basic_i_results[
                            self.poling_basic_i_results['poling_cas'] == cas_no]
                        record_ant = self.ant_results[
                            self.ant_results['ant_cas'] ==
                            cas_no]
                        column_name = self.column_names_ant[column - sec_4]
                        if len(record_ant) > 1:
                            record_ant = record_ant[0]
                            # here no longer structured array
                            item_to_return = record_ant[column_name]
                            if not isinstance(item_to_return, str):
                                item_to_return = asscalar(item_to_return)
                            return QVariant(item_to_return)
                        else:
                            # take item from structured array
                            return QVariant(
                                record_ant[column_name].item())
                    elif table_row in self.poling_basic_ant_complements:
                        pass
                elif row < len(self.poling_basic_i_rows_to_table) \
                        + len(self.burcat_ant_intersects_a):
                    row_ant = self.burcat_ant_intersects_a[
                        row - len(self.poling_basic_i_rows_to_table)]
                    record_ant = self.ant_results[row_ant]
                    column_name = self.column_names_ant[column - sec_4]
                    item_to_return = record_ant[column_name]
                    if not isinstance(item_to_return, str):
                        item_to_return = asscalar(item_to_return)
                    return QVariant(item_to_return)
                elif row < len(self.poling_basic_i_rows_to_table) \
                        + len(self.poling_basic_burcat_complements):
                    return QVariant(None)
                elif row < len(self.poling_basic_burcat_ant_complements_ant_part) + \
                           len(self.poling_basic_i_rows_to_table) + \
                           len(self.poling_basic_burcat_complements):
                    row_ant = self.poling_basic_burcat_ant_complements_ant_part[
                        row - len(self.poling_basic_i_rows_to_table) -
                        len(self.poling_basic_burcat_complements)]
                    record_ant = self.ant_results[row_ant]
                    column_name = self.column_names_ant[column - sec_4]
                    item_to_return = record_ant[column_name]
                    if not isinstance(item_to_return, str):
                        item_to_return = asscalar(item_to_return)
                    return QVariant(item_to_return)
                else:
                    return QVariant(None)

    def extract_info_from_burcat_phase(self, phase_element, column_name):
        phase_parent = phase_element.getparent()
        if column_name == 'cas':
            return phase_parent.get('CAS')
        elif column_name == 'formula_name_structure_1':
            return phase_parent.find(
                'formula_name_structure').find(
                'formula_name_structure_1').text
        elif column_name in ['phase', 'formula']:
            return phase_element.find(column_name).text
        elif column_name == 'range_tmin_to_1000':
            return float(phase_element.find(
                'temp_limit').get('low').replace(' ', ''))
        elif column_name == 'range_1000_to_tmax':
            return float(phase_element.find(
                'temp_limit').get('high').replace(' ', ''))
        elif column_name in ['molecular_weight']:
            return float(phase_element.find(
                'molecular_weight').text.replace(' ', ''))
        elif column_name == 'hf298_div_r':
            return float(phase_element.find('coefficients').find(
                'hf298_div_r').text.replace(' ', ''))
        elif column_name in ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']:
            for coef in phase_element.find('coefficients').find(
                    'range_Tmin_to_1000').getiterator('coef'):
                if coef.get('name') == column_name:
                    return float(coef.text.replace(' ', ''))
        elif column_name in ['a1_high', 'a2_high', 'a3_high', 'a4_high', 'a5_high', 'a6_high', 'a7_high']:
            column_name = column_name.replace('_high', '')
            for coef in phase_element.find('coefficients').find(
                    'range_1000_to_Tmax').getiterator('coef'):
                if coef.get('name') == column_name:
                    return float(coef.text.replace(' ', ''))

    def headerData(self, section: int, orientation: Qt.Orientation, role: int= Qt.DisplayRole):
        if role == Qt.TextAlignmentRole:
            if orientation == Qt.Horizontal:
                return QVariant(int(Qt.AlignLeft | Qt.AlignVCenter))
            return QVariant(int(Qt.AlignRight | Qt.AlignVCenter))
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            if section < len(self.column_names_poling_basic_i):
                return QVariant(self.column_names_poling_basic_i[
                    section])
            elif section < len(self.column_names_poling_basic_i) + len(self.column_names_burcat):
                return QVariant(
                    self.column_names_burcat[
                        section - len(self.column_names_poling_basic_i)
                    ])
            elif section < len(self.column_names_poling_basic_ii) + len(self.column_names_poling_basic_i) + \
                    len(self.column_names_burcat):
                return QVariant(self.column_names_poling_basic_ii[section - len(
                    self.column_names_poling_basic_i) - len(self.column_names_burcat)])
            elif section < len(self.column_names_poling_cp_ig_l) + len(self.column_names_poling_basic_ii) + \
                    len(self.column_names_burcat) + len(self.column_names_poling_basic_i):
                return QVariant(
                    self.column_names_poling_cp_ig_l[
                        section - len(self.column_names_burcat) - len(self.column_names_poling_basic_ii) -
                        len(self.column_names_poling_basic_i)
                    ])
            elif section < len(self.column_names_ant) + len(self.column_names_poling_cp_ig_l) + \
                    len(self.column_names_poling_basic_i) + len(self.column_names_poling_basic_ii) + \
                    len(self.column_names_burcat):
                return QVariant(self.column_names_ant[section -
                                                      len(self.column_names_poling_cp_ig_l) -
                                                      len(self.column_names_poling_basic_ii) -
                                                      len(self.column_names_burcat) -
                                                      len(self.column_names_poling_basic_i)])
        if orientation == Qt.Vertical:
            return QVariant(section + 1)

    def rowCount(self, index=QModelIndex()):
        # return len(self.burcat_results) #+
        # len(self.column_names_poling_basic_i)
        return len(self.poling_basic_burcat_ant_complements_ant_part) + \
               len(self.poling_basic_i_rows_to_table) + \
               len(self.poling_basic_burcat_complements)

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
        return len(self.column_names)


xsl = ET.parse(template)
tree = ET.parse(burcat_xml_file)
root = tree.getroot()
transformer = ET.XSLT(xsl)
result = transformer(tree)
cas_filter, name_filter, formula_filter, phase_filter = \
    '', '', '', 'S'
xpath = \
            "./specie[contains(@CAS, '" + \
            cas_filter + "')]/" + \
            "formula_name_structure[contains(translate(" +\
            "formula_name_structure_1" + \
            ", '" + string.ascii_lowercase + "', '" + string.ascii_uppercase + "'), '" + \
            name_filter + "')]/../@CAS/../" + \
            "phase[contains(translate(" + \
            "formula" + \
            ", '" + string.ascii_lowercase + "', '" + string.ascii_uppercase + "'), '" + \
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
print(burcat_df[
    ['cas_no', 'phase', 'formula', 
     'formula_name_structure','reference', 
     'source', 'date', 'range_tmin_to_1000',
     'range_1000_to_tmax', 'molecular_weight',
     'hf298_div_r']+
    ['a'+str(i)+'_low' for i in range(1,7+1)]+
    ['a'+str(i)+'_high' for i in range(1,7+1)]
])

ant_df = DataFrame(loadtxt(
    open(antoine_csv, 'rb'),
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
print(ant_df)

poling_basic_i_df = DataFrame(loadtxt(
    open(poling_basic_i_csv, 'rb'),
    delimiter='|',
    skiprows=3,
    dtype={
        'names': [
            'poling_no', 'poling_formula', 'poling_name',
            'cas_no', 'poling_molwt', 'poling_tfp',
            'poling_tb', 'poling_tc', 'poling_pc',
            'poling_Vc', 'poling_Zc', 'poling_omega'],
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
    open(poling_basic_ii_csv, 'rb'),
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
    open(poling_cp_l_ig_poly_csv, 'rb'),
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

print(poling_cp_ig_l_df)

poling_df = merge(merge(    
    poling_basic_i_df, poling_basic_ii_df,
    how='outer', on='cas_no'), 
    poling_cp_ig_l_df, how='outer', on='cas_no')

print(poling_df)

#poling_ant_df = merge(poling_df, ant_df, on='cas_no', how='outer')

#print(poling_ant_df)

#poling_ant_burcat_df = merge(burcat_df, poling_ant_df, on='cas_no', how='outer')

#print(poling_ant_burcat_df)

poling_burcat_df = merge(burcat_df, poling_df, on='cas_no', how='outer')
poling_burcat_ant_df = merge(burcat_df, ant_df, on='cas_no', how='outer')
print(poling_burcat_ant_df)

print(poling_df[(poling_df['poling_no_x']==205) | (poling_df['poling_no_x']==206) | (poling_df['poling_no_x']==204)])

print(poling_cp_ig_l_df['poling_no'].dtype)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    QApplication.setFont(QFont('Consolas', 10))
    # following 2 lines for setting app icon correctly
    myappid = u'mycompany.myproduct.subproduct.version'  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    # ended lines for setting app icon correctly
    QApplication.setWindowIcon(QIcon('utils/icon_batch_32X32.png'))
    ex = App()
    sys.exit(app.exec_())
