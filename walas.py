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
    d1 = Dock('Plot', size=(1, 1), closable=True)
    p1 = pg.PlotWidget(name='Y3, GLUCONIC ACID BY FERMENTATION')
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
    d1 = Dock('Plot', size=(1, 1), closable=True)
    p1 = pg.PlotWidget(name='A<<==>>B<<==>>C')
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
        ca, cb, cc = y  # len(y) == 4
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


def add_which_dock(text, d_area, timer):
    if text == 'P2.04.01':
        gui_docks_p2_04_01(d_area, timer)
    elif text == 'P2.04.02':
        gui_docks_p2_04_02(d_area, timer)
    elif text == 'P4.03.01':
        gui_docks_p4_03_01(d_area, timer)
    elif text == 'P4.03.04':
        gui_docks_p4_03_04(d_area, timer)

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
        zip(['P4.03.01', 'P4.03.04'],
            ['GLUCONIC ACID BY FERMENTATION',
             'CONSECUTIVE REVERSIBLE REACTIONS'])
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
