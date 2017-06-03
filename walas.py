# coding=utf-8
import sys
import string
import numpy as np
import matplotlib.cm as colormaps
# noinspection PyUnresolvedReferences
import PySide
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock
from pyqtgraph.Qt import QtGui, QtCore
from scipy.integrate import ode, quad
from scipy.optimize import root

app = QtGui.QApplication([])
dpi_res = 250
window_size = QtGui.QDesktopWidget().screenGeometry()
app_width = window_size.width() * 2 / 5.0
app_height = window_size.height() * 2 / 5.0

# noinspection PyUnresolvedReferences
colormap_colors = colormaps.viridis.colors + \
    colormaps.magma.colors + \
    colormaps.inferno.colors

brilliant_colors_criteria = np.apply_along_axis(
    lambda x: np.sqrt(x.dot(x)), 1, np.array(colormap_colors)
) > 0.25

colormap_colors = [
    item for k, item in enumerate(colormap_colors)
    if brilliant_colors_criteria[k]
]

series_names = [item for item in string.ascii_uppercase]

symbols = ['t', 't1', 't2', 't3', 's', 'p', 'h', 'star', '+', 'd']


def random_color():
    return tuple(
        [
            255 * elem for elem in
            colormap_colors[np.random.randint(
                0, len(colormap_colors), 1
            ).item()]
        ]
    )


def random_symbol():
    return symbols[
        np.random.randint(
            0, len(symbols), 1
        ).item()]


class tab_1_model(QtCore.QAbstractTableModel):
    # To populate tableview with right model and column formats

    def __init__(self, data, column_names, column_formats, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data
        self._column_formats = column_formats
        width = data.shape[1]
        if len(column_names) == width:
            self._column_names = column_names
        else:
            self._column_names = [''] * width

    def rowCount(self, *args, **kwargs):
        return self._data.shape[0]

    def columnCount(self, *args, **kwargs):
        return self._data.shape[1]

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal \
                and role == QtCore.Qt.DisplayRole:
            return self._column_names[col]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                # Format of columns: '1.2f', '1.2e', etc.
                column_format = \
                    '{0:' + \
                    self._column_formats[index.column()] + \
                    '}'
                return column_format.format(
                    self._data[index.row(), index.column()]
                )


def stop_shared_timer(timer):
    timer.stop()
    if timer.connected:
        timer.timeout.disconnect()
        timer.connected = False


def plot_successful_integration_step(t_int, y_int, curves, y_t, t, t1, timer):
    pos = len(t)
    if pos >= y_t.shape[0]:
        tmp = y_t.shape[0]
        y_t.resize(
            [y_t.shape[0] * 2, y_t.shape[1]],
            refcheck=False
        )
        y_t[tmp:, :] = np.empty(
            [tmp, y_t.shape[1]]
        )
    if t_int > t[0]:
        t.append(t_int)
        y_t[pos] = y_int
        pos += 1  # Ensure plotting gets right vector length
    for j, curve in enumerate(curves):
        curve.setData(x=t, y=y_t[:pos, j])
    if t1 is None:
        return None
    elif t_int >= t1:
        timer.stop()
        return -1  # Stop integration


def solve_p2_04_01(b, b1, timer, plt,
                   non_linear=True):
    if non_linear:
        b1.setEnabled(False)
        b.setEnabled(True)
    else:
        b.setEnabled(False)
        b1.setEnabled(True)
    timer.stop()
    if timer.connected:
        # Qt objects can have several connected slots.
        # Not disconnecting them all causes previously
        # running processes to continue when restarted.
        timer.timeout.disconnect()
    plt.clear()
    # Legends are overlaid on a "scene" graphics item.
    # Remove them from there, not from the plot widget.
    plt.getPlotItem().legend.scene().removeItem(
        plt.getPlotItem().legend
    )
    # Add the legend before plotting, for it to pick up
    # all the curves names and properties.
    plt.addLegend()
    # plt.getPlotItem().legend.setScale(2.0)
    y0, t0 = [100, 0, 0, 0, 0], 0
    if non_linear:
        t1 = 0.1
    else:
        t1 = 25
    curves = [None] * len(y0)
    for j, item_j in enumerate(y0):
        pen_color = random_color()
        symbol_color = random_color()
        symbol = random_symbol()
        curves[j] = plt.plot(
            name=string.ascii_uppercase[j],
            pen=pen_color,
            symbolBrush=symbol_color,
            symbol=symbol,
            size=0.2
        )

    # noinspection PyUnusedLocal
    def g(t, y, linear=False):
        if linear:
            p = 1.0
        else:
            p = (400 - y[1] - 2 * y[2] - 3 * y[3] - 4 * y[4])
        return [
            -y[0] * p,
            (y[0] - 0.5 * y[1]) * p,
            (0.5 * y[1] - 0.3 * y[2]) * p,
            (0.3 * y[2] - 0.2 * y[3]) * p,
            0.2 * y[3] * p
        ]

    y_t = np.empty([20, len(y0)])
    time_series = [t0]
    y_t[0, :] = y0

    r = ode(
        lambda t, y: g(t, y, linear=False)
    )
    if non_linear:
        r.f = lambda t, y: g(t, y, linear=False)
    else:
        r.f = lambda t, y: g(t, y, linear=True)
    r.set_initial_value(y0, t0)
    r.set_integrator('dopri5', nsteps=1)

    # For updating, pass y_t and time_series by reference
    # y_t: Already a mutable object (numpy array)
    # time_series: Mutable object also
    r.set_solout(
        lambda t_l, y_l:
        plot_successful_integration_step(
            t_l,
            y_l,
            curves,
            y_t,
            time_series,
            t1,
            timer)
    )
    timer.timeout.connect(
        lambda: r.integrate(t1)
    )
    timer.connected = True
    timer.start(50)


def gui_docks_p2_04_01(d_area, timer):
    vlayout_1 = pg.LayoutWidget()
    vlayout_2 = pg.LayoutWidget()
    p_2 = pg.PlotWidget(name='Plot_2')
    p_1 = pg.PlotWidget(name='Plot_1')
    btn_1 = QtGui.QPushButton('Run Linear')
    btn_2 = QtGui.QPushButton('Run non-Linear')
    d1 = Dock('Non-Linear', size=(1, 1), closable=True)
    d2 = Dock('Linear', size=(1, 1), closable=True)
    d_area.addDock(d1, 'left')
    d_area.addDock(d2, 'right')

    vlayout_1.addWidget(p_1, row=0, col=0)
    vlayout_1.addWidget(btn_2, row=1, col=0)
    vlayout_2.addWidget(p_2, row=0, col=0)
    vlayout_2.addWidget(btn_1, row=1, col=0)
    d1.addWidget(vlayout_1)
    d2.addWidget(vlayout_2)
    # 3rd row, use 1 row and 2 columns of grid
    # layout.addWidget(btn_3, 2, 0, 1, 2)

    p_2.setYLink('Plot_1')
    # p.setYLink('Plot_2')
    p_1.setTitle('Non-linear')
    p_2.setTitle('Linear')
    p_1.setLabel('left', text='n_i, mol')
    p_1.setLabel('bottom', text='t, min')
    p_2.setLabel('left', text='n_i, mol')
    p_2.setLabel('bottom', text='t, min')
    p_1.addLegend()
    p_2.addLegend()
    p_1.setDownsampling(mode='peak')
    p_1.setClipToView(True)
    p_1.setLimits(xMin=0, yMin=0, yMax=100)
    p_2.setLimits(xMin=0, yMin=0, yMax=100)

    # noinspection PyUnresolvedReferences
    btn_1.clicked.connect(
        lambda: solve_p2_04_01(
            btn_1,
            btn_2,
            timer,
            p_2,
            non_linear=False
        ))
    # noinspection PyUnresolvedReferences
    btn_2.clicked.connect(
        lambda: solve_p2_04_01(
            btn_1,
            btn_2,
            timer,
            p_1,
            non_linear=True
        ))

    btn_2.setEnabled(False)

    solve_p2_04_01(
        btn_1,
        btn_2,
        timer,
        p_1,
        non_linear=True
    )


def gui_docks_p2_04_02(d_area, _):
    d1 = Dock('Table', size=(1, 1), closable=True)
    tab_1 = pg.TableWidget()
    d1.addWidget(tab_1)
    d_area.addDock(d1, 'right')

    c = np.arange(2.0, 0.0, -0.1)

    def solve_for_r(conc, start_r):
        return root(
            lambda r:
            -r + 1.5 * (conc - 0.8 * r) /
            (1 + 0.3 * np.sqrt(conc - 0.8 * r) + 0.1 * (conc - 0.8 * r)) ** 2.0,
            np.array([start_r])
        ).x

    r_calc = np.array(
        map(lambda conc: solve_for_r(conc, 0.1), c)
    ).flatten()
    simple_t = -np.gradient(c) * 1 / r_calc
    simple_t[0] = 0
    simple_t = np.cumsum(simple_t)
    quad_t = np.array(
        map(
            lambda c_min:
            quad(
                lambda conc:
                1 / solve_for_r(conc, 0.1),
                c_min, 2.0),
            c)
    )[:, 0]
    data = np.array([c, 1 / r_calc, simple_t, quad_t]).T
    tab_1.setData(data)
    tab_1.setHorizontalHeaderLabels(
        ['c', '1/r', 't_simple', 't_quadfunc']
    )
    tab_1.sortByColumn(2, QtCore.Qt.AscendingOrder)
    tab_1.horizontalHeader().setResizeMode(
        QtGui.QHeaderView.ResizeToContents
    )


def gui_docks_p4_03_01(d_area, timer):
    d1 = Dock('Y3, GLUCONIC ACID BY FERMENTATION',
              size=(1, 1), closable=True)
    p1 = pg.PlotWidget(name='Plot 1')
    d1.addWidget(p1)
    d_area.addDock(d1, 'right')
    timer.stop()
    if timer.connected:
        # Qt objects can have several connected slots.
        # Not disconnecting them all causes previously
        # running processes to continue when restarted.
        timer.timeout.disconnect()

    time_interval = [0, 10]

    # noinspection PyUnusedLocal
    def g(t, y):
        b1, b2, b3, b4, b5 = \
            0.949, 3.439, 18.72, 37.51, 1.169
        y1, y2, y3, y4 = y  # len(y) == 4
        return [
            b1 * y1 * (1 - y1 / b2),
            b3 * y1 * y4 / (b4 + y4) - 0.9802 * b5 * y2,
            b5 * y2,
            -1.011 * b3 * y1 * y4 / (b4 + y4)
        ]

    p1.setLabel('bottom', text='t, h')

    y0 = [0.5, 0, 0, 50.0]
    y_t = np.empty([20, len(y0)])
    time_series = [time_interval[0]]
    y_t[0, :] = y0

    r = ode(
        lambda t, y: g(t, y)
    )
    r.set_initial_value(y0, time_interval[0])
    r.set_integrator('dopri5', nsteps=1)
    # Add the legend before plotting, for it to pick up
    # all the curves names and properties.
    p1.setLimits(xMin=0, yMin=0, yMax=60)
    p1.addLegend()

    curves = [None] * len(y0)
    curve_names = ['y' + str(it) for it in range(len(y0))]
    for j, item_j in enumerate(y0):
        pen_color = random_color()
        symbol_color = random_color()
        symbol = random_symbol()
        curves[j] = p1.plot(
            name=curve_names[j],
            pen=pen_color,
            symbolBrush=symbol_color,
            symbol=symbol,
            size=0.2
        )

    # For updating, pass y_t and time_series by reference
    # y_t: Already a mutable object (numpy array)
    # time_series: Mutable object also
    r.set_solout(
        lambda t_l, y_l:
        plot_successful_integration_step(
            t_l,
            y_l,
            curves,
            y_t,
            time_series,
            time_interval[1],
            timer)
    )
    timer.timeout.connect(
        lambda: r.integrate(time_interval[1])
    )
    timer.connected = True
    timer.start(50)


def gui_docks_p4_03_04(d_area, timer):
    d1 = Dock('A<<==>>B<<==>>C', size=(1, 1), closable=True)
    p1 = pg.PlotWidget(name='Plot 1')
    d1.addWidget(p1)
    d_area.addDock(d1, 'right')
    timer.stop()
    if timer.connected:
        # Qt objects can have several connected slots.
        # Not disconnecting them all causes previously
        # running processes to continue when restarted.
        timer.timeout.disconnect()

    time_interval = [0, 25000]

    # noinspection PyUnusedLocal
    def g(t, y):
        k1, k2, kc1, kc2 = 0.001, 0.01, 0.8, 0.6
        ca, cb, cc = y  # len(y) == 3
        return [
            -k1 * (ca - cb / kc1),
            +k1 * (ca - cb / kc1) - k2 * (cb - cc / kc2),
            +k2 * (cb - cc / kc2)
        ]

    p1.setLabel('bottom', text='t, h')

    y0 = [1.0, 0, 0]
    y_t = np.empty([20, len(y0)])
    time_series = [time_interval[0]]
    y_t[0, :] = y0

    r = ode(
        lambda t, y: g(t, y)
    )
    r.set_initial_value(y0, time_interval[0])
    r.set_integrator('dopri5', nsteps=1)
    # Add the legend before plotting, for it to pick up
    # all the curves names and properties.
    p1.setLimits(xMin=0, yMin=0, yMax=1)
    p1.addLegend()

    curves = [None] * len(y0)
    for j, item_j in enumerate(y0):
        pen_color = random_color()
        symbol_color = random_color()
        symbol = random_symbol()
        curves[j] = p1.plot(
            name=string.ascii_uppercase[j],
            pen=pen_color,
            symbolBrush=symbol_color,
            symbol=symbol,
            size=0.2
        )

    # For updating, pass y_t and time_series by reference
    # y_t: Already a mutable object (numpy array)
    # time_series: Mutable object also
    r.set_solout(
        lambda t_l, y_l:
        plot_successful_integration_step(
            t_l,
            y_l,
            curves,
            y_t,
            time_series,
            time_interval[1],
            timer)
    )
    timer.timeout.connect(
        lambda: r.integrate(time_interval[1])
    )
    timer.connected = True
    timer.start(50)


def gui_docks_p4_03_06(d_area, timer):
    d1 = Dock('ADDITION POLYMERIZATION', size=(1, 1), closable=True)
    p1 = pg.PlotWidget(name='Plot 1')
    d1.addWidget(p1)
    d_area.addDock(d1, 'right')
    timer.stop()
    if timer.connected:
        # Qt objects can have several connected slots.
        # Not disconnecting them all causes previously
        # running processes to continue when restarted.
        timer.timeout.disconnect()

    time_interval = [0, 150]

    # noinspection PyUnusedLocal
    def g(t, y):
        k0, k1, k2, k3, k4, k5 = 0.01, 0.1, 0.1, 0.1, 0.1, 0.1
        cm, cp1, cp2, cp3, cp4, cp5 = y  # len(y) == 5
        # Note, in book dM/dt=+k0M - kM sum(P_n)
        # Monomer only reacts, it should be dM/dt=-k0M - kM sum(P_n)
        return [
            -k0 * cm - k1 * cm * cp1 - k2 * cm *
            cp2 - k3 * cm * cp3 - k4 * cm * cp4 -
            k5 * cm * cp5,
            +k0 * cm - k1 * cm * cp1,
            +k1 * cm * cp1 - k2 * cm * cp2,
            +k2 * cm * cp2 - k3 * cm * cp3,
            +k3 * cm * cp3 - k4 * cm * cp4,
            +k4 * cm * cp4 - k5 * cm * cp5,
        ]

    p1.setLabel('bottom', text='t')

    y0 = [1.0, 0, 0, 0, 0, 0]
    y_t = np.empty([20, len(y0)])
    time_series = [time_interval[0]]
    y_t[0, :] = y0

    r = ode(
        lambda t, y: g(t, y)
    )
    r.set_initial_value(y0, time_interval[0])
    r.set_integrator('dopri5', nsteps=1)
    # Add the legend before plotting, for it to pick up
    # all the curves names and properties.
    p1.setLimits(xMin=0, yMin=0, yMax=1)
    p1.addLegend()

    curves = [None] * len(y0)
    curve_names = ['M']
    for it in range(1, 5 + 1):
        curve_names.append('P' + str(it))
    for j, item_j in enumerate(y0):
        pen_color = random_color()
        symbol_color = random_color()
        symbol = random_symbol()
        curves[j] = p1.plot(
            name=curve_names[j],
            pen=pen_color,
            symbolBrush=symbol_color,
            symbol=symbol,
            size=0.2
        )

    # For updating, pass y_t and time_series by reference
    # y_t: Already a mutable object (numpy array)
    # time_series: Mutable object also
    r.set_solout(
        lambda t_l, y_l:
        plot_successful_integration_step(
            t_l,
            y_l,
            curves,
            y_t,
            time_series,
            time_interval[1],
            timer)
    )
    timer.timeout.connect(
        lambda: r.integrate(time_interval[1])
    )
    timer.connected = True
    timer.start(50)


def gui_docks_p4_04_41(d_area, _):
    d1 = Dock('CSTR WITH HEATED RECYCLE 2A ==>> 2B',
              size=(1, 1), closable=True)
    tab_1 = QtGui.QTableView()
    d1.addWidget(tab_1)
    d_area.addDock(d1, 'bottom')

    def k(temperature):
        temperature = float(temperature)
        # 1/2 , weil in den Stoffmengenbilanzen, die im Buch stehen
        # die Stöchiometrie der Reaktion  nicht berücksichtigt
        # wird.
        result = 1 / 2.0 * np.exp(
            24.1 - 6500 / temperature
        )  # 1/h (kgmol/L)^-1
        return result

    mw_00 = 500.0  # kg/h solvent
    na_00 = 20.0  # kgmol/h
    t_00 = 300.0  # K
    v_flow_00 = 600.0  # L/h
    mw_01 = 100.0  # kg/h solvent
    nab_01 = 4.0  # kgmol/h
    t_01 = 350.0  # K
    v_flow_01 = 120.0  # L/h
    r = 1 / 6.0  # ratio
    v_flow_0 = v_flow_00 + v_flow_01  # L/h
    v_r = 25000.0  # L

    # 2A<<==>>2B delta_Hr = -2000 cal/(gmolAconv)
    delta_hr = +2000 * 2  # cal/2gmolA, exot.
    cp_w_masse = 1.0  # kcal/(kg K)
    cp_s_mol = 40.0  # kcal/(kgmol K), any solutes

    # Adiabatische Wärmebilanz am Zulaufstrom
    # 0 =  (cp_w_masse*mw_00 + cp_s_mol*na_00)*(300K-t_0)
    #     +(cp_w_masse*mw_01 + cp_s_mol*na_01)*(350K-t_0)
    #   =  (cp_w_masse*mw_00 + cp_s_mol*na_00)*(300K-t_0)
    #     +1/6(cp_w_masse*mw_00 + cp_s_mol*na_00)*(350K-t_0)
    mw_0 = mw_00 + mw_01
    nab_0 = na_00 + nab_01
    t_0 = 5 / 6.0 * t_00 + 1 / 6.0 * t_01
    m_c_cp_0 = (cp_w_masse * mw_00 + cp_s_mol * na_00) + \
               (cp_w_masse * mw_01 + cp_s_mol * nab_01)

    # Wärmebilanz 1. Am Reaktorkessel
    # dT/dt = 0 = (mw_0*cp_w_masse+na_0*cp_s_mol)*(t_0-t_1) +
    #             (-delta_hr)*r*Vr
    # Wärmebilanz 2. Am Wärmetauscher
    # Q = (mw_1*cp_w_masse+na_0*cp_s_mol)
    # Stoffmengenbilanz 1. Am Reaktor
    # dCa/dt = 0 = +na0 + 1/6 na1 - na1 - 2*r*Vr
    # ==>>> r*Vr = 1/2(na0-5/6 na1)
    def null_werte(y, recycle):
        t1 = y[0]
        na1 = y[1]
        t_0 = (1 - recycle * 25 / 30.0) * t_00 + (recycle * 25 / 30.0) * t_01
        nab_01 = na_00 * recycle
        mw_01 = mw_00 * recycle
        nab_0 = na_00 + nab_01
        v_flow_0 = v_flow_00 * (1 + recycle)  # L/h
        m_c_cp_0 = (cp_w_masse * mw_00 + cp_s_mol * na_00) + \
                   (cp_w_masse * mw_01 + cp_s_mol * nab_01)
        # [0 (Energie)
        #  0 (Masse)]
        # 5/6 = 4/5*25/30 , ~effect of molar masses on recycle
        return [
            -t1 + t_0 +
            (-delta_hr * (1 / 2.0) *
             (na_00 - (1 - recycle * 25 / 30.0) * na1)) / (m_c_cp_0),
            na_00 - (1 - recycle * 25 / 30.0) * na1 -
            2.0 * v_r * k(t1) * (na1 / v_flow_0)**2
        ]

    # q = root(
    #     lambda q_load:
    #     + q_load
    #     + m0*cp*(temp0 - temp)
    #     + m0*cp*(temp_r - temp)*x
    #     + m0*cp*(temp - temp_r)
    #     + (-delta_hr)*x*na0 # =0
    # ).x # kcal/h

    soln_at_rec = []
    for recycle in [float(x) / 5.0 for x in range(0, 5 + 1, 1)]:
        sol = root(
            lambda y: null_werte(y, recycle),
            np.array([t_0, nab_0])
        )
        na_1 = sol.x[1]
        t_1 = sol.x[0]
        # ['R', 'na_1', 'T_1', 'k', 'x']
        soln_at_rec.append([
            recycle, na_1, t_1, k(t_1),
            1 - (1 - recycle * 25 / 30.0) * na_1 / na_00
        ])

    tab_1.setModel(tab_1_model(
        data=np.array(soln_at_rec),
        column_names=['R', 'na_1', 'T_1', 'k', 'x'],
        column_formats=['.2%', '1.2f', '1.2f', '1.2f', '1.3f']
    ))
    tab_1.horizontalHeader().setResizeMode(
        QtGui.QHeaderView.ResizeToContents
    )


def gui_docks_p4_04_53(d_area, _):
    d1 = Dock('PUMPAROUND SYSTEM A<<==>>B',
              size=(1, 1), closable=True)
    tab_1 = pg.TableWidget()
    d1.addWidget(tab_1)
    d_area.addDock(d1, 'bottom')

    def k(temperature):
        temperature = float(temperature)
        return np.exp(17.2 - 5800 / temperature)  # 1/h

    def ke(temperature):
        temperature = float(temperature)
        return np.exp(10000 / temperature - 24.7)  # [adim]
    # cb0 = 0. T at which X = 90%   = (na0 - na)/na0
    #                               = (nb - nb0)/na0 = nb/na0
    #                               = Kc na/na0 = Kc (1-90%)
    #          ==>>    Kc = nb / na = 90%/(1-90%)
    temp = root(
        lambda temp: ke(temp) - 0.9 / (1 - 0.9), 300.0
    ).x

    # A<<==>>B delta_Hr = 19870.0 cal/gmolA
    delta_hr = 19870.0  # cal/gmolA, exot.
    cp = 0.50  # cal/(gm K)
    k1 = k(temp)  # 1/h
    m0 = 10000  # kg/h
    na0 = 50  # kgmol/h
    vf0 = 5000  # L/h
    temp0 = 300.0  # K
    temp_r = 60.0  # K
    x = 0.80  # actual conversion

    q = root(
        lambda q_load:
        + q_load
        + m0 * cp * (temp0 - temp)
        + m0 * cp * (temp_r - temp) * x
        + m0 * cp * (temp - temp_r)
        + (-delta_hr) * x * na0,  # = 0
        10000
    ).x  # kcal/h

    tab_1.set_data

    tab_1.setHorizontalHeaderLabels(
        ['n_a', 'T', 'k', 'K_e', 'Integrand', 'S']
    )
    tab_1.horizontalHeader().setResizeMode(
        QtGui.QHeaderView.ResizeToContents
    )


def add_which_dock(text, d_area, timer):
    if text == 'P2.04.01':
        gui_docks_p2_04_01(d_area, timer)
    elif text == 'P2.04.02':
        gui_docks_p2_04_02(d_area, timer)
    elif text == 'P4.03.01':
        gui_docks_p4_03_01(d_area, timer)
    elif text == 'P4.03.04':
        gui_docks_p4_03_04(d_area, timer)
    elif text == 'P4.03.06':
        gui_docks_p4_03_06(d_area, timer)
    elif text == 'P4.04.41':
        gui_docks_p4_04_41(d_area, timer)
    elif text == 'P4.04.53':
        gui_docks_p4_04_53(d_area, timer)

wind = QtGui.QWidget()
area = DockArea()
tree = pg.TreeWidget()
btn_3 = QtGui.QPushButton('STOP')
vlayout_0 = QtGui.QVBoxLayout()
splitter = QtGui.QSplitter()
shared_timer = pg.QtCore.QTimer()

splitter.addWidget(tree)
splitter.addWidget(area)
splitter.setSizes([app_width * 1 / 3.0, app_width * 2 / 3.0])

vlayout_0.addWidget(splitter)
vlayout_0.addWidget(btn_3)

wind.setLayout(vlayout_0)
wind.setWindowTitle('Walas problems')
wind.resize(app_width, app_height)

covered_chapters = [2, 4]
chapter_texts = dict(zip(
    covered_chapters,
    [
        'REACTION RATES AND OPERATING MODES',
        'IDEAL REACTORS'
    ]
))
chapter_problems = dict(zip(
    covered_chapters,
    [
        zip(['P2.04.01', 'P2.04.02'],
            ['ALKYLATION OF ISOPROPYLBENZENE',
             'DIFFUSION AND SOLID CATALYSIS']),
        zip(['P4.03.01', 'P4.03.04', 'P4.03.06',
             'P4.04.41', 'P4.04.53'],
            ['GLUCONIC ACID BY FERMENTATION',
             'CONSECUTIVE REVERSIBLE REACTIONS',
             'ADDITION POLYMERIZATION',
             'PUMPAROUND SYSTEM'])
    ]
))
problem_counter = 0
tree.setColumnCount(len(covered_chapters))
for chapter in covered_chapters:
    locals()['ti' + str(chapter)] = QtGui.QTreeWidgetItem()
    tix = locals()['ti' + str(chapter)]
    tix.setText(
        0, 'CHAPTER ' + str(chapter) + '.')
    tix.setText(
        1, chapter_texts[chapter]
    )
    tree.addTopLevelItem(tix)
    for problem in chapter_problems[chapter]:
        problem_counter += 1
        name_item = 'ti' + str(chapter) + str(problem_counter) + str(1)
        name_label = 'lab_' + str(chapter) + str(problem_counter) + str(1)
        locals()[name_item] = QtGui.QTreeWidgetItem([problem[0]])
        locals()[name_label] = QtGui.QLabel(problem[1])
        tix.addChild(locals()[name_item])
        tree.setItemWidget(
            locals()[name_item],
            1,
            locals()[name_label])

tree.setDragEnabled(False)
tree.expandAll()
tree.resizeColumnToContents(0)
tree.resizeColumnToContents(1)
tree.setHeaderHidden(True)

# noinspection PyUnresolvedReferences
tree.itemClicked.connect(
    lambda it, column:
    add_which_dock(it.text(0), area, shared_timer)
)
# noinspection PyUnresolvedReferences
btn_3.clicked.connect(
    lambda: stop_shared_timer(shared_timer))
# Qt objects can have several connected slots.
# Not disconnecting them all causes previously
# running processes to continue when restarted.
# Keep track in this shared timer via added
# attribute.
shared_timer.connected = False

wind.show()

if __name__ == '__main__':
    if (sys.flags.interactive != 1) or \
            not hasattr(QtCore, 'PYQT_VERSION'):
        # noinspection PyArgumentList
        QtGui.QApplication.instance().exec_()
