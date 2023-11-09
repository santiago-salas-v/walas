# coding=utf-8
import sys
import string
import time
import numpy as np
import matplotlib.cm as colormaps
# noinspection PyUnresolvedReferences
# import PySide
import matplotlib
import matplotlib.pyplot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from scipy.integrate import ode, quad
from scipy.optimize import root, leastsq


app = QtWidgets.QApplication([])
dpi_res = 250
window_size = QtWidgets.QDesktopWidget().screenGeometry()
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

markers = list(matplotlib.markers.MarkerStyle.markers.keys())


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


def random_marker():
    return markers[
        np.random.randint(
            0, len(markers), 1
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
                # If format is a function, use function.
                # Else a number format
                format_of_index = self._column_formats[index.column()]
                if callable(format_of_index):
                    return format_of_index(
                        self._data[index.row(), index.column()]
                    )
                else:
                    # Format of columns: '1.2f', '1.2e', etc.
                    column_format = \
                        '{0:' + \
                         format_of_index + \
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


def plot_successful_integration_step_mpl(
        t_int, y_int, curves, y_t, t, t1, timer):
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
        curve.set_data(t, y_t[:pos, j])
        # Optimzation with artists:
        # curve.axes.draw_artist(curve)
    # for spine in curve.axes.spines.values():
    #     curve.axes.draw_artist(spine)
    #
    # Relim and autoscale view:
    curve.axes.relim()
    curve.axes.autoscale_view()
    #
    # Use either canvas.draw() or optimization draw_artist
    # for specific artists, and uptade:
    curve.axes.figure.canvas.draw()
    # Optimzation with artists:
    # curve.axes.draw_artist(curve.axes.patch)
    # curve.axes.draw_artist(curve.axes.xaxis)
    # curve.axes.draw_artist(curve.axes.yaxis)
    # curve.axes.figure.canvas.update()
    #
    # Flush events
    curve.axes.figure.canvas.flush_events()
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


def gui_docks_p2_04_01(d_area, timer, title=None):
    vlayout_1 = pg.LayoutWidget()
    vlayout_2 = pg.LayoutWidget()
    p_2 = pg.PlotWidget(name='Plot_2')
    p_1 = pg.PlotWidget(name='Plot_1')
    btn_1 = QtWidgets.QPushButton('Run Linear')
    btn_2 = QtWidgets.QPushButton('Run non-Linear')
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


def gui_docks_p2_04_02(d_area, _, title=None):
    d1 = Dock(title, size=(1, 1), closable=True)
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
        [solve_for_r(conc, 0.1) for conc in c]
    ).flatten()
    simple_t = -np.gradient(c) * 1 / r_calc
    simple_t[0] = 0
    simple_t = np.cumsum(simple_t)
    quad_t = np.array(
        [quad(
            lambda conc: 1 / solve_for_r(conc, 0.1),
            c_min, 
            2.0) for c_min in c
        ]
    )[:, 0]
    data = np.array([c, 1 / r_calc, simple_t, quad_t]).T
    tab_1.setData(data)
    tab_1.setHorizontalHeaderLabels(
        ['c', '1/r', 't_simple', 't_quadfunc']
    )
    tab_1.sortByColumn(2, QtCore.Qt.AscendingOrder)
    tab_1.horizontalHeader().setSectionResizeMode(
        QtWidgets.QHeaderView.ResizeToContents
    )


def gui_docks_p4_03_01(d_area, timer, title=None):
    d1 = Dock(title,
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


def gui_docks_p4_03_04(d_area, timer, title=None):
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


def gui_docks_p4_03_06(d_area, timer, title=None, use_mpl=False):
    d1 = Dock('ADDITION POLYMERIZATION', size=(1, 1), closable=True)
    vlayout = pg.LayoutWidget()
    b1 = QtWidgets.QPushButton()
    # Establish initial conditions before curve numbers
    time_interval = [0, 150]
    y0 = [1.0, 0, 0, 0, 0, 0]
    y_t = np.empty([20, len(y0)])
    time_series = [time_interval[0]]
    y_t[0, :] = y0
    curves = [None] * len(y0)
    curve_names = ['M']
    for it in range(1, 5 + 1):
        curve_names.append('P' + str(it))
    if use_mpl:
        matplotlib.pyplot.style.use('dark_background')
        fig, ax = matplotlib.pyplot.subplots()
        p1 = FigureCanvas(fig)
        ax.set_xlabel('t')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1)
        ax.legend()
        ax.autoscale(enable=True, axis='x')
        for j, item_j in enumerate(y0):
            pen_color = tuple(
                [item / 255.0 for item in random_color()]
            )
            marker_color = tuple(
                [item / 255.0 for item in random_color()]
            )
            marker = random_marker()
            curves[j], = ax.plot(
                [], [],
                color=pen_color,
                markeredgecolor=pen_color,
                marker=marker,
                markerfacecolor=marker_color,
                label=curve_names[j]
            )
        ax.legend(handles=curves, loc='best', fancybox=True).draggable(True)
        opt_function = plot_successful_integration_step_mpl
        ax.set_title('C vs. t (using mpl)')
        b1.setText('test pyqtgraph')
        b1.clicked.connect(
            lambda: gui_docks_p4_03_06(d_area, timer, title=None, use_mpl=False)
        )
    else:
        p1 = pg.PlotWidget(name='Plot 1')
        p1.setLabel('bottom', text='t')
        # Add the legend before plotting, for it to pick up
        # all the curves names and properties.
        p1.setLimits(xMin=0, yMin=0, yMax=1)
        p1.addLegend()
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
        opt_function = plot_successful_integration_step
        p1.setWindowTitle('C vs. t (using pyqtgraph)')
        b1.setText('test mpl')
        b1.clicked.connect(
            lambda: gui_docks_p4_03_06(d_area, timer, title=None, use_mpl=True)
        )
    vlayout.addWidget(p1, 0, 0)
    vlayout.addWidget(b1, 1, 0)
    d1.addWidget(vlayout)
    d_area.addDock(d1, 'right')
    timer.stop()
    if timer.connected:
        # Qt objects can have several connected slots.
        # Not disconnecting them all causes previously
        # running processes to continue when restarted.
        timer.timeout.disconnect()

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

    r = ode(
        lambda t, y: g(t, y)
    )
    r.set_initial_value(y0, time_interval[0])
    r.set_integrator('dopri5', nsteps=1)
    # For updating, pass y_t and time_series by reference
    # y_t: Already a mutable object (numpy array)
    # time_series: Mutable object also
    r.set_solout(
        lambda t_l, y_l:
        opt_function(
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


def gui_docks_p4_04_41(d_area, _, title=None):
    d1 = Dock('CSTR WITH HEATED RECYCLE 2A ==>> 2B',
              size=(1, 1), closable=True)
    tab_1 = QtWidgets.QTableView()
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
    tab_1.horizontalHeader().setSectionResizeMode(
        QtWidgets.QHeaderView.ResizeToContents
    )


def gui_docks_p4_04_53(d_area, _, title=None):
    d1 = Dock('PUMPAROUND SYSTEM A<<==>>B',
              size=(1, 1), closable=True)
    tab_1 = QtWidgets.QTableView()
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
        lambda temp: ke(temp) - 90.0 / (100.0 - 90.0), 300.0
    ).x.item()

    # A<<==>>B delta_Hr = 19870.0 cal/gmolA
    delta_hr = -19870.0  # cal/gmolA, exot.
    cp = 0.50  # cal/(gm K)
    k1 = k(temp)  # 1/h
    ketf = ke(temp)  # adim
    m0 = 10000  # kg/h
    na0 = 50  # kgmol/h
    vf0 = 5000  # L/h
    temp0 = 300.0  # K
    delta_t = 60.0  # K
    xa = 0.80  # actual conversion
    naf = (1 - xa) * na0  # kgmol/h , net reactor outlet
    # proportion due to stoich. nb0=0, nb/na = xa/(1-xa)
    nbf = xa / (1 - xa) * naf
    temp_f = temp - delta_t

    # Balance on complete system with reaction
    q = root(
        lambda q_load:
        + q_load
        + m0 * cp * temp0
        - m0 * cp * temp_f
        + (-delta_hr) * xa * na0,  # = 0
        10000
    ).x  # kcal/h

    # Balance on heat exchanger
    mt = q / (cp * (temp0 - temp_f))  # feed + recycle
    mr = mt - m0  # recycle
    nar = mr / m0 * naf  # recycled along with mass proportion
    # proportion due to stoich. nb0=0, nb/na = xa/(1-xa)
    nbr = xa / (1 - xa) * nar

    # Conc. in tank (Assune A and B are solutes with ~ high density)
    ca = na0 * (1 - xa) / vf0  # kgmol/L , conc. in CSTR
    # proportion due to stoich. nb0=0, nb/na = xa/(1-xa)
    cb = xa / (1 - xa) * ca
    # ra = r1 - r1' = k1ca - k1'cb = k1/vf0 * (na - 1/Ke * nb)
    ra = k1 * (ca - 1 / ketf * cb)  # kgmol / (L h)
    vr = root(
        lambda vr: - na0 + naf + ra * vr,  # = 0
        100.0
    )

    tab_variables = []
    na0st = na0 * (1 - xa)
    t0st = temp_f

    def opt_na(na):
        t = t0st + (-delta_hr) / (m0 * cp) * (na0st - na)
        k1 = k(t)
        ketf = ke(t)
        summation, err = quad(
            lambda na_var: 1 / (k1 * (na_var - (na0 - na_var) / ketf)),
            na, na0st
        )
        return summation, t, k1, ketf

    for na in [10, 9.5, 9.0, 8.67, 8.5]:
        # 0 = m0cp T0 - m0cp T + (-delta_Hr)*na0*xa
        # -delta_Hr (naf - na) = m0 cp (T - T0)
        summation, t, k1, ketf = opt_na(na)
        tab_variables.append(
            [na, t, k1, ketf, summation]
        )

    final_na = root(
        lambda na: 2500.0 / 5000.0 - opt_na(na)[0], na0st
    )
    summation, t, k1, ketf = opt_na(final_na.x)
    tab_variables.append(
        [final_na.x, t, k1, ketf, summation]
    )

    tab_1.setModel(tab_1_model(
        data=np.array(tab_variables),
        column_names=['n', 'T', 'k', 'K_e', 'S'],
        column_formats=['1.5g'] * 6
    ))
    tab_1.horizontalHeader().setSectionResizeMode(
        QtWidgets.QHeaderView.ResizeToContents
    )


def gui_docks_p3_02_58(d_area, _, title=None):
    d1 = Dock('IODINATION, FOURTH ORDER',
              size=(1, 1), closable=True)
    d2 = Dock('IODINATION, FOURTH ORDER',
              size=(1, 1), closable=True)
    tab_1 = QtWidgets.QTableView()
    matplotlib.pyplot.style.use('dark_background')
    fig, ax = matplotlib.pyplot.subplots()
    p1 = FigureCanvas(fig)
    ax.set_xlabel('t')
    ax.autoscale(enable=True, axis='x|y')

    cici_vs_t = np.array([
        [0, 0.1750],
        [1, 0.0815],
        [3, 0.0519],
        [5, 0.0409],
        [7.5, 0.0341],
        [10, 0.0294],
        [13, 0.0261],
        [15.5, 0.0239],
        [17, 0.0229],
        [20, 0.0210],
        [24, 0.0191]
    ], dtype=float)

    cici0 = cici_vs_t[0, 1]

    pen_color = tuple(
        [item / 255.0 for item in random_color()]
    )
    marker_color = tuple(
        [item / 255.0 for item in random_color()]
    )
    marker = random_marker()

    ax.plot(cici_vs_t[:, 0], cici_vs_t[:, 1],
            linestyle='None',
            color=pen_color,
            markeredgecolor=pen_color,
            marker=marker,
            markerfacecolor=marker_color,
            label='$C_{ICI} vs. t$'
            )

    def residuals(p, x, y):
        q, k_q = p
        t = x
        cici = y
        # linear form: 0 = 1 - k_q * t * (q-1) * cici^(q-1) -
        #                                (cici/cici0)^(q-1)
        return 1 - k_q * t * (q - 1) * cici ** (q - 1)  - \
            (cici / cici0)**(q - 1)

    plsq = leastsq(
        residuals,
        [float(2), float(58)],
        args=(cici_vs_t[:, 0], cici_vs_t[:, 1])
    )
    q_lsq, k_q_lsq = plsq[0]

    max_t = int(round(max(cici_vs_t[:, 0]), 0))
    x_plot_range = np.arange(0, max_t + 1, (max_t + 1 - 0) / 100.0)
    y_plot_range = (k_q_lsq * (q_lsq - 1) * x_plot_range +
                    1 / cici0**(q_lsq - 1))**(1 / (1 - q_lsq))
    k_q_str = '{0:1.4g}'.format(k_q_lsq)
    q_str = '{0:1.4g}'.format(q_lsq)
    pen_color = tuple(
        [item / 255.0 for item in random_color()]
    )
    marker_color = tuple(
        [item / 255.0 for item in random_color()]
    )
    marker = 'None'
    ax.plot(x_plot_range, y_plot_range,
            linestyle='-',
            color=pen_color,
            markeredgecolor=pen_color,
            marker=marker,
            markerfacecolor=marker_color,
            label='$' + k_q_str +
                  '=1/(t*(' + q_str + '-1))*' +
                  '(1/ C_{ICI}^{' + q_str + '-1}' +
                  '-1/ 0.175^{' + q_str + '-1})' + '$'
            )

    ax.legend()

    ki = np.empty([cici_vs_t.shape[0], 3 + 1], dtype=float)
    q = list(range(1, 4, 1))
    q.append(q_lsq)
    for row, k in enumerate(ki):
        t = cici_vs_t[row, 0]
        cici = cici_vs_t[row, 1]
        for col in range(len(q)):
            if q[col] == 1 and t != 0:
                k[col] = np.log(cici0 / cici)
            elif q[col] != 1 and t != 0:
                k[col] = \
                    1 / (t * (q[col] - 1)) * \
                    (1 / (cici ** (q[col] - 1)) - 1 /
                     (cici0 ** (q[col] - 1)))
            else:
                k[col] = np.nan

    tab_1_data = np.append(
        cici_vs_t,
        np.array(ki),
        1
    )

    tab_1.setModel(tab_1_model(
        data=tab_1_data,
        column_names=['t', 'C_{ICI}', 'k1', 'k2', 'k3',
                      'k' + '{0:1.3f}'.format(q_lsq)],
        column_formats=['g', '1.3f', '1.3f', '1.3f', '1.3f',
                        '1.3f']
    ))
    tab_1.horizontalHeader().setSectionResizeMode(
        QtWidgets.QHeaderView.ResizeToContents
    )

    d1.addWidget(tab_1)
    d2.addWidget(p1)
    d_area.addDock(d1, 'bottom')
    d_area.addDock(d2, 'right')


def gui_docks_a4_04_53(d_area, _, title=None):
    d1 = Dock(title,
              size=(1, 1), closable=True
              )
    fig, ax = matplotlib.pyplot.subplots()
    p1 = FigureCanvas(fig)
    ax.set_xlabel('t')
    ax.autoscale(enable=True, axis='x|y')

    emax = 0.97
    fmax = 0.95
    cazu = 1.0 # kmol/m3
    v10 = 0.01;v20 = 0.01 #m3
    v1min = 0.01;v2min = 0.01
    v1voll = 1.0;v2voll = 1.0
    v0 = 1e-03;v12=1e-03;v2=1e-03 #m3/s
    k2=100;k2r=100.0;k3=1.0;k4=1e-03

    y0 = np.empty([9, 1], dtype=float)

    y0[1] = 1.0e-2
    y0[0] = 2/3.0 * y0[1]
    y0[3] = 0.0
    y0[4] = 0.0
    y0[5] = 0.0
    y0[6] = 0.0
    y0[7] = 0.01
    y0[8] = 0.01
    y0[2] = y0[7]**3 * y0[0] / y0[1]**3

def gui_docks_ue3_1(d_area, _, title=None):
    d1 = Dock(title,
              size=(1, 1), closable=True
              )
    tab_1 = QtWidgets.QTableView()
    text1 = QtWidgets.QPlainTextEdit()
    vlayout1 = pg.LayoutWidget()

    x0so2 = 0.078
    x0n2 = 0.814
    x0o2 = 0.108
    x0so3 = 0.0

    nuso2 = -1
    nun2 = 0
    nuo2 = -1/2.0
    nuso3 = 1
    n0 = 1 + 1/2.0

    var0 = [x0so2, x0n2, x0o2, x0so3, n0, 0.0]

    def func_set(x):
        xso2 = x[0]
        xn2 = x[1]
        xo2 = x[2]
        xso3 = x[3]
        n = x[4]
        csi1 = x[5]

        return [
            - xso2 * n + x0so2 * n0 + nuso2 * csi1,
            - xn2 * n + x0n2 * n0 + nun2 * csi1,
            - xo2 * n + x0o2 * n0 + nuo2 * csi1,
            - xso3 * n + x0so3 * n0 + nuso3 * csi1,
            1 - 0.96 - (xso2 / x0so2) * (n / n0),
            -1 + xso2 + xn2 + xo2 + xso3
        ]

    soln = root(func_set, var0)

    text1.setPlainText(str(soln))

    xso2, xn2, xo2, xso3, n, csi1 = soln.x

    tab1_data = np.array(
        [
            [
                'xso2', 'xn2', 'xo2', 'xso3', 'n', 'csi1',
                'U', 'n0', 'n/n0', 'csi1/n0'
            ],
            np.append(var0, [0, n0, 0, 0]),
            np.append(soln.x,
                      [1 - xso2/x0so2 * n / n0, n0, n/n0, csi1/n0])
        ],
        dtype=object
    )

    for n0 in range(1,20):
        var0[-2] = n0
        new_root = root(func_set, var0)
        xso2, xn2, xo2, xso3, n, csi1 = new_root.x
        new_row = np.append(
            new_root.x,
            [1 - xso2/x0so2 * n / n0, n0, n/n0, csi1/n0])
        tab1_data = np.append(
            tab1_data,
            new_row.reshape([1,len(new_row)]), axis=0
        )
        text1.appendPlainText(str(new_root))

    tab_1.setModel(
        tab_1_model(
            data = tab1_data.T,
            column_names=['Var', 'Val(init)', 'Val(eq)'] +
                         ['Val(eq)'] * (len(tab1_data) - 3),
            column_formats= [str, 'g', 'g'] +
                            ['g'] * (len(tab1_data) - 3)
        )
    )

    tab_1.horizontalHeader().setSectionResizeMode(
        QtWidgets.QHeaderView.ResizeToContents
    )

    vlayout1.addWidget(tab_1, row=0, col=0)
    vlayout1.addWidget(text1, row=1, col=0)

    d1.addWidget(vlayout1)
    d_area.addDock(d1)


def add_which_dock(text, d_area, timer, title):
    if text == 'P2.04.01':
        gui_docks_p2_04_01(d_area, timer, title)
    elif text == 'P2.04.02':
        gui_docks_p2_04_02(d_area, timer, title)
    elif text == 'P3.02.58':
        gui_docks_p3_02_58(d_area, timer, title)
    elif text == 'P4.03.01':
        gui_docks_p4_03_01(d_area, timer, title)
    elif text == 'P4.03.04':
        gui_docks_p4_03_04(d_area, timer, title)
    elif text == 'P4.03.06':
        gui_docks_p4_03_06(d_area, timer, title, use_mpl=True)
    elif text == 'P4.04.41':
        gui_docks_p4_04_41(d_area, timer, title)
    elif text == 'P4.04.53':
        gui_docks_p4_04_53(d_area, timer, title)
    elif text == 'A4.1.2':
        gui_docks_a4_04_53(d_area, timer, title)
    elif text == u'Ü3.1':
        gui_docks_ue3_1(d_area, timer, title)

wind = QtWidgets.QWidget()
area = DockArea()
tree = pg.TreeWidget()
btn_3 = QtWidgets.QPushButton('STOP')
vlayout_0 = QtWidgets.QVBoxLayout()
splitter = QtWidgets.QSplitter()
shared_timer = pg.QtCore.QTimer()

splitter.addWidget(tree)
splitter.addWidget(area)
splitter.setSizes([int(app_width * 1 / 3.0), int(app_width * 2 / 3.0)])

vlayout_0.addWidget(splitter)
vlayout_0.addWidget(btn_3)

wind.setLayout(vlayout_0)
wind.setWindowTitle('Walas problems')
wind.resize(int(app_width), int(app_height))

chapter_layout = [
    [
        2,
        'REACTION RATES AND OPERATING MODES',
        [
            ['P2.04.01', 'ALKYLATION OF ISOPROPYLBENZENE'],
            ['P2.04.02', 'DIFFUSION AND SOLID CATALYSIS'],
        ]
    ],
    [
        3,
        'TREATMENT OF EXPERIMENTAL DATA',
        [
            ['P3.02.58', 'IODINATION. FOURTH ORDER']
        ]
    ],
    [
        4,
        'IDEAL REACTORS',
        [
            ['P4.03.01', 'GLUCONIC ACID BY FERMENTATION'],
            ['P4.03.04', 'CONSECUTIVE REVERSIBLE REACTIONS'],
            ['P4.03.06', 'ADDITION POLYMERIZATION'],
            ['P4.04.41', 'CSTR WITH HEATED RECYCLE'],
            ['P4.04.53', 'PUMPAROUND SYSTEM']
        ]
    ],
    [
        'ARNO 4.1',
        'ZWEISTUFIGER PROZESS',
        [
            ['A4.1.2', 'Zweistufiger Prozess'],
        ]
    ],
    [
        'HAGEN 3.1',
        'Umsatz',
        [
            [u'Ü3.1', u'Schwefelsäure Herstellung']
        ]
    ]
]

problem_counter = 0
tree.setColumnCount(len(chapter_layout))
for chapter in chapter_layout:
    chapter_code = chapter[0]
    chapter_text = chapter[1]
    chapter_problems = chapter[2]
    if not isinstance(chapter_code, str):
        chapter_title = 'CHAPTER ' + str(chapter_code) + '. '
    else:
        chapter_title = chapter_code
    locals()['ti' + str(chapter_code)] = QtWidgets.QTreeWidgetItem()
    tix = locals()['ti' + str(chapter_code)]
    tix.setText(
        0, chapter_title)
    tix.setText(
        1, chapter_text
    )
    tree.addTopLevelItem(tix)
    for problem in chapter_problems:
        problem_counter += 1
        name_item = 'ti' + str(chapter_code) + str(problem_counter) + str(1)
        name_label = 'lab_' + str(chapter_code) + str(problem_counter) + str(1)
        locals()[name_item] = QtWidgets.QTreeWidgetItem([problem[0]])
        locals()[name_label] = QtWidgets.QLabel(problem[1])
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
    add_which_dock(it.text(0), area, shared_timer,
                   tree.itemWidget(it, 1).text())
)
# noinspection PyUnresolvedReferences
btn_3.clicked.connect(
    lambda: stop_shared_timer(shared_timer))
# Qt objects can have several connected slots.
# Not disconnecting them all ca uses previously
# running processes to continue when restarted.
# Keep track in this shared timer via added
# attribute.
shared_timer.connected = False

wind.show()

if __name__ == '__main__':
    if (sys.flags.interactive != 1) or \
            not hasattr(QtCore, 'PYQT_VERSION'):
        # noinspection PyArgumentList
        QtWidgets.QApplication.instance().exec_()
