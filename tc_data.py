import os
import sys
import re
import string
import locale
from numpy import loadtxt, isnan, empty_like, linspace, dtype, zeros
from functools import partial
import lxml.etree as et
import csv
import io
from pandas import DataFrame, merge, to_numeric
import ctypes  # Needed to set the app icon correctly
from PyQt5.QtWidgets import QTableView, QApplication, QWidget
from PyQt5.QtWidgets import QDesktopWidget, QGridLayout, QLineEdit, QPushButton
from PyQt5.QtWidgets import QComboBox, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt, QVariant, QEvent
from PyQt5.QtGui import QKeySequence, QGuiApplication, QFont, QIcon
from PyQt5.QtCore import pyqtSignal as SIGNAL
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


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
        self.title = 'Thermodynamic data'

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
        self.phases_vol_button = QPushButton()
        self.plot_window = PlotWindow()
        self.cp_button = QPushButton()
        self.psat_button = QPushButton()

        self.tableView1.setModel(self.model)
        self.tableWidget1.added_column_names = ['']
        self.tableWidget1.setColumnCount(self.tableView1.model().columnCount()+1)
        self.tableWidget1.setHorizontalHeaderLabels(
            ['z_i'] + self.tableView1.model().column_names
        )
        self.phase_filter.addItems(['', 'G', 'L', 'S', 'C'])
        self.tableWidget1.setEnabled(False)
        self.tableWidget1.setSelectionBehavior(QTableWidget.SelectRows)
        self.tableWidget1.installEventFilter(self)
        self.delete_selected_button.setText('delete selected')
        self.copy_selected_button.setText('copy selected')
        self.phases_vol_button.setText('ph, v')

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
        self.layout.addWidget(self.phases_vol_button, 6, 2, 1, 1)

        self.setLayout(self.layout)

        for item in [self.cas_filter, self.name_filter, self.formula_filter]:
            item.textChanged.connect(self.apply_filters)
        self.phase_filter.currentTextChanged.connect(self.apply_filters)
        self.tableView1.selectionModel().selectionChanged.connect(partial(
            self.add_selection_to_widget
        ))
        self.tableWidget1.cellChanged.connect(partial(
            self.update_props
            ))
        self.delete_selected_button.clicked.connect(partial(
            self.delete_selected
        ))
        self.copy_selected_button.clicked.connect(partial(
            self.copy_selection
        ))
        self.phases_vol_button.clicked.connect(partial(
            self.phases_vol
        ))
        #self.tableView1.horizontalHeader().setClickable(False)

        self.props_i = self.tableView1.model().df.iloc[[]]
        self.props_i['z_i'] = zeros(0)

        # init set for testing
        self.phase_filter.setCurrentIndex(self.phase_filter.findText('G'))
        self.cas_filter.setText('7727-37-9')
        self.name_filter.setText('nitrogen')
        self.formula_filter.setText('N2')
        self.tableView1.selectRow(0)

        self.phase_filter.setCurrentIndex(self.phase_filter.findText('G'))
        self.cas_filter.setText('74-82-8')
        self.name_filter.setText('methane')
        self.formula_filter.setText('ch4')
        self.tableView1.selectRow(0)

        self.phase_filter.setCurrentIndex(self.phase_filter.findText('G'))
        self.cas_filter.setText('74-84-0')
        self.name_filter.setText('ethane')
        self.formula_filter.setText('c2h6')
        self.tableView1.selectRow(0)

        self.phase_filter.setCurrentIndex(self.phase_filter.findText('G'))
        self.cas_filter.setText('74-98-6')
        self.name_filter.setText('propane')
        self.formula_filter.setText('c3h8')
        self.tableView1.selectRow(0)
        self.tableView1.selectRow(-1)

        self.phase_filter.setCurrentIndex(self.phase_filter.findText(''))
        self.cas_filter.setText('')
        self.name_filter.setText('')
        self.formula_filter.setText('')

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
        column_of_cas = self.tableView1.model().column_names.index('cas_no')
        column_of_phase = self.tableView1.model().column_names.index('phase')
        phase = self.tableView1.model().index(index.row(), column_of_phase).data()
        cas = self.tableView1.model().index(index.row(), column_of_cas).data()
        already_in_table = False
        for item in self.tableWidget1.findItems(cas, Qt.MatchExactly):
            if phase == self.tableWidget1.item(
                    item.row(), column_of_phase+1).text():
                already_in_table = True
        if not already_in_table:
            row_index = int(self.tableView1.model().headerData(index.row(), Qt.Vertical).value())
            column_names = self.tableView1.model().column_names
            header_to_add = QTableWidgetItem(str(row_index))

            # save df with new props
            z_i_orig = self.props_i['z_i']
            index_orig = self.props_i.index
            self.props_i = self.tableView1.model().df.loc[
                [int(self.tableWidget1.verticalHeaderItem(i).text())
                 for i in range(self.tableWidget1.rowCount())]+[row_index],
                 :
            ]
            self.props_i['z_i'] = zeros(len(self.props_i))
            self.props_i.loc[index_orig, 'z_i'] = z_i_orig
            self.props_i.loc[row_index, 'z_i'] = float(0)

            # add item to widget
            self.tableWidget1.setRowCount(self.tableWidget1.rowCount() + 1)
            self.tableWidget1.setVerticalHeaderItem(
                    self.tableWidget1.rowCount() - 1,
                    header_to_add)
            for i in range(len(column_names)):
                # columns in TableWidget shifted by one vs. Tableview due to first column z_i
                data = self.tableView1.model().index(index.row(), i).data()
                if isinstance(data, str) or data is None:
                    item_to_add = QTableWidgetItem(data)
                else:
                    item_to_add = QTableWidgetItem(locale.str(data))
                item_to_add.setFlags(
                    Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
                self.tableWidget1.setItem(
                    self.tableWidget1.rowCount() - 1, i+1,
                    item_to_add)

            # additional column with z_i
            item_to_add = QTableWidgetItem(locale.str(0))
            item_to_add.setFlags(
                    Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
            self.tableWidget1.setItem(
                    self.tableWidget1.rowCount() - 1, 0,
                    item_to_add)

        if len(indexes) > 0 or self.tableWidget1.rowCount() > 0:
            self.tableWidget1.setEnabled(True)

    def update_props(self, row, col):
        if col == 0:
            # change z_i
            df_index = int(self.tableWidget1.verticalHeaderItem(row).text())
            new_value = float(
                self.tableWidget1.item(row, col).text()
                )
            self.props_i.loc[df_index, 'z_i'] = new_value
        else:
            pass

    def copy_selection(self):
        col_names = self.tableView1.model().column_names
        col_units = self.tableView1.model().column_units
        selection = self.tableWidget1.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = range(len(col_names))
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount + 1)]
            table[0] = []
            for i in range(len(col_names)):
                text_to_add = col_names[i]
                if len(col_units[i]) > 0:
                    text_to_add += ' [' + col_units[i] + '] '
                table[0] += [text_to_add]
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

    def phases_vol(self):
        self.plot_window.show()
        t = linspace(60, 220, 30)
        p = 101325
        phase_fraction = empty_like(t)
        # normalize z_i
        sum_z_i = sum(self.props_i['z_i'])
        if sum_z_i > 0:
            self.props_i.loc[:, 'z_i'] = self.props_i['z_i'] / sum_z_i
        z_i = self.props_i['z_i'].tolist()
        mm_i = (self.props_i['poling_molwt']/1000).tolist() # kg/mol
        tc_i = self.props_i['poling_tc'].tolist() # K
        pc_i = (self.props_i['poling_pc']*10**5).tolist() # Pa
        omega_i = self.props_i['poling_omega'].tolist()
        vc_i = (self.props_i['poling_Vc']*10**-6).tolist() # m^3/mol

        for i in range(len(t)):
            phase_fraction[i] = pvt(p, t, z_i, pc_i, tc_i, omega_i, method='RKS')['v_f']

class PlotWindow(QWidget):
    def __init__(self, parent=None):
        super(PlotWindow, self).__init__(parent)
        self.dpi=50
        self.glayout_2 = QGridLayout()
        self.fig = plt.figure(dpi=self.dpi)
        self.ax = plt.subplot()
        self.p1 = FigureCanvas(self.fig)
        self.title = 'plot results'
        self.setLayout(self.glayout_2)
        self.glayout_2.addWidget(self.p1)
        self.setWindowTitle(self.title)
        self.p1.setSizePolicy( 
            QSizePolicy.Expanding, 
            QSizePolicy.Expanding
            )
        self.fig.tight_layout()


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


class thTableModel(QAbstractTableModel):

    def __init__(self):
        super(thTableModel, self).__init__()

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
            burcat_df[column_name] = to_numeric(
                burcat_df[column_name].str.replace(' ', '')
                )

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

        poling_df = merge(merge(    
            poling_basic_i_df, poling_basic_ii_df,
            how='outer', on='cas_no'), 
            poling_cp_ig_l_df, how='outer', on='cas_no')

        poling_burcat_df = merge(burcat_df, poling_df, on='cas_no', how='outer')
        self.df = merge(poling_burcat_df, ant_df, on='cas_no', how='outer')

        self.df['poling_a1'] = self.df['poling_a1'] * 1e-3
        self.df['poling_a2'] = self.df['poling_a2'] * 1e-5
        self.df['poling_a3'] = self.df['poling_a3'] * 1e-8
        self.df['poling_a4'] = self.df['poling_a4'] * 1e-11

        self.column_names = [
            'cas_no', 'phase', 'formula', 
            'formula_name_structure', 'ant_name',
            'poling_no', 'poling_formula', 'poling_name',
            'poling_molwt', 'poling_tfp',
            'poling_tb', 'poling_tc', 'poling_pc',
            'poling_Vc', 'poling_Zc', 'poling_omega',
            'poling_delhf0', 'poling_delgf0', 'poling_delhb',
            'poling_delhm', 'poling_v_liq', 'poling_t_liq',
            'poling_dipole',
            'range_tmin_to_1000',
            'range_1000_to_tmax', 'molecular_weight',
            'hf298_div_r'] + [
            'a'+str(i)+'_low' for i in range(1,7+1) ] + [
            'a'+str(i)+'_high' for i in range(1,7+1)
            ] + [
            'reference', 'source', 'date', 
            'ant_no', 'ant_formula', 'ant_name', 
            'ant_a', 'ant_b',
            'ant_c', 'ant_tmin', 'ant_tmax',
            'ant_code'
            ]

        self.column_dtypes = [
            str, str, str,
            str, str,
            float, str, str,
            float, float,
            float, float, float,
            float, float, float,
            float, float, float,
            float, float, float,
            float,
            float,
            float, float,
            float] + [
            float]*7 + [float]*7 + [
            str, str, str,
            float, str, str,
            float, float,
            float, float, float,
            str
        ]

        self.column_units = [
            '', '', '',
            '', '',
            '', '', '',
            'g/mol', 'K', 
            'K', 'K', 'bar', 
            'cm3/mol', '', '',
            'kJ/mol', 'kJ/mol', 'kJ/mol',
            'kJ/mol', 'cm3/mol', 'K',
            'Debye', 
            'K', 
            'K', 'g/mol', 
            '',
            '', 'K^-1', 'K^-2', 'K^-3', 'K^-4',
            'K^-1', '',
            '', 'K^-1', 'K^-2', 'K^-3', 'K^-4',
            'K^-1', '',
            '', '', '', 
            '', '', '', 
            '', '', '', 
            '°C', '°C', 
            ''
        ]

        self.dtypes = []
        for units in self.column_units:
            if len(units) > 0:
                # numeric value with units
                self.dtypes += [float]
            else:
                self.dtypes += [str]

        self.apply_filter(
            cas_filter, name_filter, 
            formula_filter, phase_filter
            )

    def apply_filter(
            self,
            cas_filter,
            name_filter,
            formula_filter,
            phase_filter):
        self.cas_filter = cas_filter.lower()
        self.name_filter = name_filter.lower()
        self.formula_filter = formula_filter.lower()
        self.phase_filter = phase_filter.lower()
        self.beginResetModel()
        
        filtered_data = self.df.copy()

        if len(self.cas_filter) > 0:
            filtered_data = filtered_data[
                (filtered_data.cas_no.str.lower().str.find(self.cas_filter)>=0)
            ]
        if len(self.formula_filter) > 0:
            filtered_data = filtered_data[
                    (filtered_data.formula.str.lower().str.find(self.formula_filter)>=0) |
                    (filtered_data.formula_name_structure.str.lower().str.find(self.formula_filter)>=0) |
                    (filtered_data.ant_formula.str.lower().str.find(self.formula_filter)>=0) |
                    (filtered_data.poling_formula.str.lower().str.find(self.formula_filter)>=0)
            ]

        if len(self.name_filter) > 0:
            filtered_data = filtered_data[
                (filtered_data.poling_name.str.lower().str.find(self.name_filter)>=0) | 
                (filtered_data.ant_name.str.lower().str.find(self.name_filter)>=0) | 
                (filtered_data.formula_name_structure.str.lower().str.find(self.name_filter)>=0)
            ]
        if len(self.phase_filter) > 0:
            filtered_data = filtered_data[
                (filtered_data.phase.str.lower().str.find(self.phase_filter)>=0)
            ]

        self.filtered_data = filtered_data

        self.endResetModel()

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or \
                index.row() < 0 or \
                index.column() < 0 or index.column() > self.columnCount():
            return QVariant()
        if role == Qt.DisplayRole:
            column = index.column()
            row = index.row()

            column_name = self.column_names[column]
            datum = self.filtered_data[column_name].values[row]

            if type(datum) == str:
                return QVariant(datum)
            elif datum is not None and not isnan(datum):
                return QVariant(locale.str(datum))
            else:
                return QVariant(None)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int= Qt.DisplayRole):
        if role == Qt.TextAlignmentRole:
            if orientation == Qt.Horizontal:
                return QVariant(int(Qt.AlignLeft | Qt.AlignVCenter))
            return QVariant(int(Qt.AlignRight | Qt.AlignVCenter))
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            return QVariant(
                self.column_names[section]+'\n'+
                self.column_units[section])
            
        if orientation == Qt.Vertical:
            return QVariant(str(self.filtered_data.index[section]))

    def rowCount(self, index=QModelIndex()):
        # return len(self.burcat_results) #+
        # len(self.column_names_poling_basic_i)
        return len(self.filtered_data)

    def columnCount(self, index=QModelIndex()):
        return len(self.column_names)


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
