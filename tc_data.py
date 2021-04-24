import csv
import ctypes  # Needed to set the app icon correctly
import io
import locale
import os
import re
import string
import sys
from functools import partial
from os.path import exists

import lxml.etree as et
import matplotlib
from PySide2.QtCore import QAbstractTableModel, QModelIndex, Qt, QEvent
from PySide2.QtGui import QKeySequence, QFont, QIcon, QDoubleValidator
from PySide2.QtWidgets import QComboBox, QLabel, QTableWidget, QTableWidgetItem
from PySide2.QtWidgets import QDesktopWidget, QGridLayout, QLineEdit, QPushButton
from PySide2.QtWidgets import QSizePolicy
from PySide2.QtWidgets import QTableView, QApplication, QWidget
from numpy import loadtxt, isnan, empty_like, linspace, zeros, ones, dtype, log
from pandas import DataFrame, merge, to_numeric, read_csv

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from z_l_v import State

locale.setlocale(locale.LC_ALL, '')
burcat_xml_file = './data/BURCAT_THR.xml'
antoine_csv = './data/The-Yaws-Handbook-of-Vapor-Pressure-Second-Edition-Antoine-coefficients.csv'
poling_basic_i_csv = './data/basic_constants_i_properties_of_gases_and_liquids.csv'
poling_basic_ii_csv = './data/basic_constants_ii_properties_of_gases_and_liquids.csv'
poling_cp_l_ig_poly_csv = './data/ig_l_heat_capacities_properties_of_gases_and_liquids.csv'
poling_pv_csv = './data/vapor_pressure_correlations_parameters_clean.csv'
template = "./data/xsl_stylesheet_burcat.xsl"
merged_df_csv = 'data/th_data_df.csv'
linesep_b = os.linesep.encode('utf-8')


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Thermodynamic data'

        qd = QDesktopWidget()
        screen = qd.screenGeometry(qd)
        width = 600
        height = 300

        x = screen.width() / 2 - width / 2
        y = screen.height() / 2 - height / 2

        self.width = width
        self.height = height
        self.left = x
        self.top = y
        self.ignore_events = False
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
        self.temp_text = QLineEdit()
        self.plot_window = PlotWindow()
        self.cp_button = QPushButton()
        self.psat_button = QPushButton()

        self.tableView1.setModel(self.model)
        self.tableWidget1.setColumnCount(self.tableView1.model().columnCount()+1+1+1+1+1)
        self.widget_col_names=['z_i','h_ig','s_ig','g_ig','cp_ig'] + self.tableView1.model().column_names
        self.widget_col_units=['-','J/mol','J/mol/K','J/mol','J/mol/K'] + self.tableView1.model().column_units
        self.widget_col_names_units=[self.widget_col_names[i]+'\n'+self.widget_col_units[i] for i in range(len(self.widget_col_names))]
        self.tableWidget1.setHorizontalHeaderLabels(self.widget_col_names_units)
        self.phase_filter.addItems(['', 'G', 'L', 'S', 'C'])
        self.tableWidget1.setEnabled(False)
        self.tableWidget1.setSelectionBehavior(QTableWidget.SelectRows)
        self.tableWidget1.installEventFilter(self)
        self.delete_selected_button.setText('delete selected')
        self.copy_selected_button.setText('copy selected')
        self.phases_vol_button.setText('ph, v')
        self.temp_text.setValidator(QDoubleValidator(0,6000,16))
        self.temp_text.setPlaceholderText('1000 K')

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
        self.layout.addWidget(self.temp_text, 6, 3, 1, 1)

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
        self.temp_text.editingFinished.connect(partial(
            self.update_temp
        ))
        #self.tableView1.horizontalHeader().setClickable(False)

        self.props_i = self.tableView1.model().df.iloc[[]]
        self.props_i['z_i'] = zeros(0)
        self.props_i['h_ig'] = zeros(0)
        self.props_i['s_ig'] = zeros(0)
        self.props_i['g_ig'] = zeros(0)
        self.props_i['cp_ig'] = zeros(0)

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

    def add_selection_to_widget(self, selected, deselected):
        # indexes = self.tableView1.selectedIndexes()
        indexes = selected.indexes()
        index = indexes[0]
        column_of_cas = self.tableView1.model().column_names.index('cas_no')
        column_of_phase = self.tableView1.model().column_names.index('phase')
        phase = self.tableView1.model().index(index.row(), column_of_phase).data()
        cas = self.tableView1.model().index(index.row(), column_of_cas).data()
        already_in_table = False
        for item in self.tableWidget1.findItems(cas, Qt.MatchExactly):
            if phase == self.tableWidget1.item(
                    item.row(), column_of_phase+1+1+1+1+1).text():
                already_in_table = True
        if not already_in_table:
            row_index = int(self.tableView1.model().headerData(index.row(), Qt.Vertical))
            column_names = self.tableView1.model().column_names
            header_to_add = QTableWidgetItem(str(row_index))

            # save df with new props
            z_i_orig = self.props_i['z_i']
            h_ig_orig = self.props_i['h_ig']
            s_ig_orig = self.props_i['s_ig']
            g_ig_orig = self.props_i['g_ig']
            cp_ig_orig = self.props_i['cp_ig']
            index_orig = self.props_i.index
            self.props_i = self.tableView1.model().df.loc[
                [int(self.tableWidget1.verticalHeaderItem(i).text())
                 for i in range(self.tableWidget1.rowCount())]+[
                    row_index], :
            ]
            self.props_i['z_i'] = zeros(len(self.props_i))
            self.props_i['h_ig'] = zeros(len(self.props_i))
            self.props_i['s_ig'] = zeros(len(self.props_i))
            self.props_i['g_ig'] = zeros(len(self.props_i))
            self.props_i['cp_ig'] = zeros(len(self.props_i))
            self.props_i.loc[index_orig, 'z_i'] = z_i_orig
            self.props_i.loc[index_orig, 'h_ig'] = h_ig_orig
            self.props_i.loc[index_orig, 's_ig'] = s_ig_orig
            self.props_i.loc[index_orig, 'g_ig'] = g_ig_orig
            self.props_i.loc[index_orig, 'cp_ig'] = cp_ig_orig
            
            self.props_i.loc[row_index, 'z_i'] = float(0)

            # add item to widget
            self.tableWidget1.setRowCount(self.tableWidget1.rowCount() + 1)
            self.tableWidget1.setVerticalHeaderItem(
                    self.tableWidget1.rowCount() - 1,
                    header_to_add)
            for i in range(len(column_names)):
                # columns in TableWidget shifted by 1+1+1+1+1 vs. Tableview due to first columns z_i, h_ig, s_ig, g_ig, cp_ig
                data = self.tableView1.model().index(index.row(), i).data()
                if isinstance(data, str) or data is None:
                    item_to_add = QTableWidgetItem(data)
                else:
                    item_to_add = QTableWidgetItem(locale.str(data))
                item_to_add.setFlags(
                    Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
                self.tableWidget1.setItem(
                    self.tableWidget1.rowCount() - 1, i+1+1+1+1+1,
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
                self.tableWidget1.item(row, col).text().replace(',', '.')
                )
            self.props_i.loc[df_index, 'z_i'] = new_value
        else:
            pass

    def update_temp(self):
        if self.temp_text.hasAcceptableInput():
            self.temp_text.setPlaceholderText(self.temp_text.text()+' K')
            t=float(self.temp_text.text().replace(',','.')) # K      
            self.temp_text.clear()
        else:
            t=1000 # K
            self.temp_text.clear()
        mm_i = (self.props_i['poling_molwt']/1000).tolist()  # kg/mol
        tc_i = self.props_i['poling_tc'].tolist()  # K
        pc_i = (self.props_i['poling_pc']*1e5).tolist()  # Pa
        omega_i = self.props_i['poling_omega'].tolist()
        vc_i = (self.props_i['poling_vc']*10**-6).tolist()  # m^3/mol
        delhf0_poling = (self.props_i['poling_delhf0']*1000).tolist()  # J/mol
        delgf0_poling = (self.props_i['poling_delgf0']*1000).tolist()  # J/mol
        delsf0_poling = [(delhf0_poling[i]-delgf0_poling[i])/298.15 for i in range(len(self.props_i))] # J/mol/K
        a_low = [self.props_i['a'+str(i)+'_low'].tolist() for i in range(1,7+1)]
        a_high = [self.props_i['a'+str(i)+'_high'].tolist() for i in range(1,7+1)]


        cp_r_low=[sum([a_low[j][i]*t**j for j in range(4+1)]) for i in range(len(self.props_i))] # cp/R
        cp_r_high=[sum([a_high[j][i]*t**j for j in range(4+1)]) for i in range(len(self.props_i))] # cp/R

        if t>1000: # poly a_low is for 200 - 1000 K; a_high is for 1000 - 6000 K
            a=a_high
            cp_ig=[8.3145*cp_r_high[i] for i in range(len(self.props_i))] # J/mol/K
        else:
            a=a_low
            cp_ig=[8.3145*cp_r_low[i] for i in range(len(self.props_i))] # J/mol/K
        
        s_cp_r_dt=[
        sum([1/(j+1)*a[j][i]*t**(j+1) for j in range(4+1)]) 
        -sum([1/(j+1)*a_low[j][i]*298.15**(j+1) for j in range(4+1)]) 
        for i in range(len(self.props_i))] # int(Cp/R*dT,298,15K,T)
        # int(Cp/R/T*dT,298.15K,T)
        s_cp_r_t_dt=[a[0][i]*log(t)+a[6][i]+
        sum([1/(j)*a[j][i]*t**(j) for j in range(1,3+1)])
        for i in range(len(self.props_i))] # int(Cp/(RT)*dT,0,T)

        h_ig=[delhf0_poling[i]+8.3145*s_cp_r_dt[i] for i in range(len(self.props_i))]
        s_ig=[8.3145*s_cp_r_t_dt[i] for i in range(len(self.props_i))]
        g_ig=[h_ig[i]-t*s_ig[i] for i in range(len(self.props_i))]

        for i in range(len(self.props_i)):
            for j,col in enumerate([h_ig,s_ig,g_ig,cp_ig]):
                #print(col)
                item_to_add = QTableWidgetItem(locale.str(col[i]))
                item_to_add.setFlags(
                        Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
                self.tableWidget1.setItem(
                        i, j+1, item_to_add)

    def copy_selection(self):
        selection = self.tableWidget1.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = range(len(self.widget_col_names))
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1 + 1
            table = [[''] * colcount for _ in range(rowcount + 1)]
            table[0] = []
            for i in range(len(self.widget_col_names)):
                text_to_add = self.widget_col_names[i]+'/('+self.widget_col_units[i]+')'
                table[0] += [text_to_add]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row + 1][column] = index.data().replace(chr(34),'') # ensure string can be read as csv by removing quotation mark (ascii character 34)
            table=table+[['T='+self.temp_text.placeholderText()]+['' for _ in range(colcount-1)]]
            stream = io.StringIO()
            csv.writer(
                stream,
                delimiter=';',
                quoting=csv.QUOTE_NONE).writerows(table)
            QApplication.clipboard().setText(stream.getvalue())

    def eventFilter(self, source, event):
        if self.ignore_events:
            pass
        elif (event.type() == QEvent.KeyPress and
                event.matches(QKeySequence.Copy)):
            self.copy_selection()
            return True
        elif (event.type() == QEvent.KeyPress and
                event.matches(QKeySequence.Delete)):
            if len(self.tableWidget1.selectedIndexes()) > 1:
                self.delete_selected()
            return True
        return super(App, self).eventFilter(source, event)

    def delete_selected(self):
        if len(self.tableWidget1.selectedIndexes()) > 0:
            while len(self.tableWidget1.selectedIndexes()) > 0:
                current_row = self.tableWidget1.selectedIndexes()[0].row()
                row_index = int(self.tableWidget1.verticalHeaderItem(current_row).text())
                self.tableWidget1.removeRow(current_row)
                self.props_i = self.props_i.drop([row_index])
            self.tableWidget1.selectRow(current_row)
            if self.tableWidget1.rowCount() == 0:
                self.tableWidget1.setEnabled(False)

    def phases_vol(self):
        self.plot_window.show()
        t = linspace(60, 220, 10)
        p = 1.01325  # bar
        phase_fraction = empty_like(t)
        v_l = empty_like(t)
        v_v = empty_like(t)
        z_i = self.props_i['z_i']
        # normalize z_i
        sum_z_i = sum(z_i)
        if sum_z_i <= 0:
            z_i = 1 / len(z_i) * ones(len(z_i))
            z_i = z_i.tolist()
        elif sum_z_i != 1.0:
            z_i = z_i / sum_z_i
            z_i = z_i.to_list()
        else:
            z_i = z_i.to_list()
        mm_i = (self.props_i['poling_molwt']/1000).tolist()  # kg/mol
        tc_i = self.props_i['poling_tc'].tolist()  # K
        pc_i = (self.props_i['poling_pc']).tolist()  # bar
        omega_i = self.props_i['poling_omega'].tolist()
        vc_i = (self.props_i['poling_vc']*10**-6).tolist()  # m^3/mol
        state = State(t[0], p, z_i, mm_i, tc_i, pc_i, omega_i, 'pr')

        for i in range(len(t)):
            state.set_t(t[i])
            phase_fraction[i] = state.v_f
            v_l[i] = state.v_l
            v_v[i] = state.v_v
        self.plot_window.ax[0].plot(t, phase_fraction)
        self.plot_window.ax[0].set_xlabel('T / K')
        self.plot_window.ax[0].set_xlabel('V / F')
        self.plot_window.ax[1].semilogy(t, v_l, label='v_l')
        self.plot_window.ax[1].semilogy(t, v_v, label='v_v')
        self.plot_window.ax[1].set_xlabel('T / K')
        self.plot_window.ax[1].set_xlabel(r'$\frac{V}{m^3 / mol}$')
        self.plot_window.ax[1].legend()
        self.plot_window.fig.tight_layout()


class PlotWindow(QWidget):
    def __init__(self, parent=None):
        super(PlotWindow, self).__init__(parent)
        self.dpi=50
        self.glayout_2 = QGridLayout()
        # self.fig = plt.figure(dpi=self.dpi)
        self.fig, self.ax = plt.subplots(1, 2)
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
    if len(x) == 0 or x == linesep_b:
        return b'nan'
    return x.replace(b' ', b'')


def helper_func3(x):
    # helper function 3 for csv reading.
    # replace - for . in cas numbers (Antoine coeff)
    return x.replace(b'.', b'-').decode('utf-8')


def helper_func4(x):
    # helper function 4 for csv reading.
    # replace whitespace ' ' for '' in float numbers
    return x.replace(b' ', b'')


class thTableModel(QAbstractTableModel):

    def __init__(self):
        super(thTableModel, self).__init__()

        cas_filter, name_filter, \
        formula_filter, phase_filter = \
            '', '', '', ''

        self.column_names = [
                                'cas_no', 'phase', 'formula',
                                'formula_name_structure', 'ant_name',
                                'poling_no', 'poling_formula', 'poling_name',
                                'poling_molwt', 'poling_tfp',
                                'poling_tb', 'poling_tc', 'poling_pc',
                                'poling_vc', 'poling_zc', 'poling_omega',
                                'poling_delhf0', 'poling_delgf0', 'poling_delhb',
                                'poling_delhm', 'poling_v_liq', 'poling_t_liq',
                                'poling_dipole',
                                'p_ant_a', 'p_ant_b', 'p_ant_c',
                                'p_ant_tmin', 'p_ant_tmax',
                                'p_ant_pvpmin', 'p_ant_pvpmax',
                                'eant_to', 'eant_n',
                                'eant_e', 'eant_f',
                                'eant_tmin', 'eant_tmax',
                                'eant_pvpmin', 'eant_pvpmax',
                                'wagn_a', 'wagn_b',
                                'wagn_c', 'wagn_d',
                                'wagn_tmin', 'wagn_tmax',
                                'wagn_pvpmin', 'wagn_pvpmax',
                                'range_tmin_to_1000',
                                'range_1000_to_tmax', 'molecular_weight',
                                'hf298_div_r'] + [
                                'a' + str(i) + '_low' for i in range(1, 7 + 1)] + [
                                'a' + str(i) + '_high' for i in range(1, 7 + 1)
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
                                 float, float, float,
                                 float, float,
                                 float, float,
                                 float, float,
                                 float, float,
                                 float, float,
                                 float, float,
                                 float, float,
                                 float, float,
                                 float, float,
                                 float, float,
                                 float,
                                 float, float,
                                 float] + [
                                 float] * 7 + [float] * 7 + [
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
            '', 'K', 'K',
            'K', 'K', 'bar', 'bar',
            'K', '', '', '',
            'K', 'K', 'bar', 'bar',
            '', '', '', '',
            'K', 'K', 'bar', 'bar',
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

        self.df = DataFrame()
        found_existing_df = exists(merged_df_csv)
        dtypes_dict = {
            'cas_no': dtype('O'), 'phase': dtype('O'), 'formula_name_structure': dtype('O'),
            'reference': dtype('O'), 'hf298': dtype('O'), 'max_lst_sq_error': dtype('O'),
            'formula': dtype('O'), 'source': dtype('O'), 'date': dtype('O'),
            'range_tmin_to_1000': dtype('float64'), 'range_1000_to_tmax': dtype('float64'),
            'molecular_weight': dtype('float64'), 'hf298_div_r': dtype('float64'),
            'a1_low': dtype('float64'), 'a2_low': dtype('float64'), 'a3_low': dtype('float64'),
            'a4_low': dtype('float64'), 'a5_low': dtype('float64'), 'a6_low': dtype('float64'),
            'a7_low': dtype('float64'), 'a1_high': dtype('float64'), 'a2_high': dtype('float64'),
            'a3_high': dtype('float64'), 'a4_high': dtype('float64'), 'a5_high': dtype('float64'),
            'a6_high': dtype('float64'), 'a7_high': dtype('float64'), 'poling_no': dtype('float64'),
            'poling_formula': dtype('O'), 'poling_name': dtype('O'), 'poling_molwt': dtype('float64'),
            'poling_tfp': dtype('float64'), 'poling_tb': dtype('float64'), 'poling_tc': dtype('float64'),
            'poling_pc': dtype('float64'), 'poling_vc': dtype('float64'), 'poling_zc': dtype('float64'),
            'poling_omega': dtype('float64'), 'poling_delhf0': dtype('float64'), 'poling_delgf0': dtype('float64'),
            'poling_delhb': dtype('float64'), 'poling_delhm': dtype('float64'), 'poling_v_liq': dtype('float64'),
            'poling_t_liq': dtype('float64'), 'poling_dipole': dtype('float64'), 'poling_trange': dtype('O'),
            'poling_a0': dtype('float64'), 'poling_a1': dtype('float64'), 'poling_a2': dtype('float64'),
            'poling_a3': dtype('float64'), 'poling_a4': dtype('float64'), 'poling_cpig': dtype('float64'),
            'poling_cpliq': dtype('float64'), 'p_ant_a': dtype('float64'), 'p_ant_b': dtype('float64'),
            'p_ant_c': dtype('float64'), 'p_ant_pvpmin': dtype('float64'), 'p_ant_tmin': dtype('float64'),
            'p_ant_pvpmax': dtype('float64'), 'p_ant_tmax': dtype('float64'), 'eant_to': dtype('float64'),
            'eant_n': dtype('float64'), 'eant_e': dtype('float64'), 'eant_f': dtype('float64'),
            'eant_pvpmin': dtype('float64'), 'eant_tmin': dtype('float64'), 'eant_pvpmax': dtype('float64'),
            'eant_tmax': dtype('float64'), 'wagn_a': dtype('float64'), 'wagn_b': dtype('float64'),
            'wagn_c': dtype('float64'), 'wagn_d': dtype('float64'), 'wagn_pvpmin': dtype('float64'),
            'wagn_tmin': dtype('float64'), 'wagn_pvpmax': dtype('float64'), 'wagn_tmax': dtype('float64'),
            'ant_no': dtype('float64'), 'ant_formula': dtype('O'), 'ant_name': dtype('O'),
            'ant_a': dtype('float64'), 'ant_b': dtype('float64'), 'ant_c': dtype('float64'),
            'ant_tmin': dtype('float64'), 'ant_tmax': dtype('float64'), 'ant_code': dtype('O')}

        if found_existing_df:
            self.df = read_csv(merged_df_csv, skiprows=1, sep=',', index_col=0,
                               keep_default_na=False, na_values=['NaN'], dtype=dtypes_dict)
            # test = all(a.fillna(0) == self.df.fillna(0))
        else:
            self.construct_df()
            buf = open(merged_df_csv, 'w')
            buf.write('sep=,\n')
            buf.close()
            self.df.to_csv(merged_df_csv, na_rep='NaN', mode='a')

        self.apply_filter(
            cas_filter, name_filter,
            formula_filter, phase_filter
        )


    def construct_df(self):
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

        conv = dict([[i, lambda x: helper_func4(helper_func2(x)).decode('utf-8')]
                     for i in [0, 1, 2, 4, 5,
                               ]+[8, 9, 10, 11, 12, 13, 14, 15, 16]])
        conv[3] = lambda x: helper_func3(x)
        conv[6] = lambda x: helper_func4(helper_func2(x)).decode('utf-8')
        conv[7] = lambda x: helper_func4(helper_func2(x)).decode('utf-8')

        poling_pv_df = DataFrame(loadtxt(
            open(poling_pv_csv, 'rb'),
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
        self.df = merge(poling_burcat_df, ant_df, on='cas_no', how='outer')

        self.df['poling_a1'] = self.df['poling_a1'] * 1e-3
        self.df['poling_a2'] = self.df['poling_a2'] * 1e-5
        self.df['poling_a3'] = self.df['poling_a3'] * 1e-8
        self.df['poling_a4'] = self.df['poling_a4'] * 1e-11

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
            return None
        if role == Qt.DisplayRole:
            column = index.column()
            row = index.row()

            column_name = self.column_names[column]
            datum = self.filtered_data[column_name].values[row]

            if type(datum) == str:
                return datum
            elif datum is not None and not isnan(datum):
                return locale.str(datum)
            else:
                return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int= Qt.DisplayRole):
        if role == Qt.TextAlignmentRole:
            if orientation == Qt.Horizontal:
                return int(Qt.AlignLeft | Qt.AlignVCenter)
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return \
                self.column_names[section]+'\n'+ \
                self.column_units[section]
            
        if orientation == Qt.Vertical:
            return str(self.filtered_data.index[section])

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

    # init set for testing
    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('1333-74-0')
    ex.name_filter.setText('hydrogen')
    ex.formula_filter.setText('H2')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('74-82-8')
    ex.name_filter.setText('methane')
    ex.formula_filter.setText('CH4')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('124-38-9')
    ex.name_filter.setText('carbon dioxide')
    ex.formula_filter.setText('CO2')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('630-08-0')
    ex.name_filter.setText('carbon monoxide')
    ex.formula_filter.setText('CO')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('7732-18-5')
    ex.name_filter.setText('water')
    ex.formula_filter.setText('H2O')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('7727-37-9')
    ex.name_filter.setText('nitrogen')
    ex.formula_filter.setText('N2')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('7782-44-7')
    ex.name_filter.setText('oxygen')
    ex.formula_filter.setText('O2')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('7440-37-1')
    ex.name_filter.setText('argon')
    ex.formula_filter.setText('Ar')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('7440-59-7')
    ex.name_filter.setText('helium')
    ex.formula_filter.setText('He')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('71-43-2')
    ex.name_filter.setText('benzene')
    ex.formula_filter.setText('C6H6')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('110-82-7')
    ex.name_filter.setText('cyclohexane')
    ex.formula_filter.setText('C6H12')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('110-54-3')
    ex.name_filter.setText('n-hexane')
    ex.formula_filter.setText('C6H14')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('67-56-1')
    ex.name_filter.setText('methanol')
    ex.formula_filter.setText('CH3OH')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('107-31-3')
    ex.name_filter.setText('methyl formate')
    ex.formula_filter.setText('C2H4O2')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('64-18-6')
    ex.name_filter.setText('formic acid')
    ex.formula_filter.setText('CH2O2')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('107-31-3')
    ex.name_filter.setText('methyl formate')
    ex.formula_filter.setText('C2H4O2')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('74-84-0')
    ex.name_filter.setText('ethane')
    ex.formula_filter.setText('c2h6')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('74-85-1')
    ex.name_filter.setText('ethene')
    ex.formula_filter.setText('c2h4')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('74-98-6')
    ex.name_filter.setText('propane')
    ex.formula_filter.setText('c3h8')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('115-07-1')
    ex.name_filter.setText('propene')
    ex.formula_filter.setText('c3h6')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('7783-06-4')
    ex.name_filter.setText('')
    ex.formula_filter.setText('h2s')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('7664-93-9')
    ex.name_filter.setText('sulfuric acid')
    ex.formula_filter.setText('h2so4')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('7446-11-9')
    ex.name_filter.setText('')
    ex.formula_filter.setText('SO3')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText('G'))
    ex.cas_filter.setText('7446-09-5')
    ex.name_filter.setText('')
    ex.formula_filter.setText('SO2')
    ex.tableView1.selectRow(0)

    ex.phase_filter.setCurrentIndex(ex.phase_filter.findText(''))
    ex.cas_filter.setText('')
    ex.name_filter.setText('')
    ex.formula_filter.setText('')

    ex.update_temp()

    ex.tableView1.selectRow(-1)

    sys.exit(app.exec_())
