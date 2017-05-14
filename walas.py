import sys
import string
import numpy as np
import matplotlib.cm as colormaps
# noinspection PyUnresolvedReferences
import PySide
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock
from pyqtgraph.Qt import QtGui, QtCore
from scipy.integrate import ode

app = QtGui.QApplication([])
dpi_res = 250
window_size = QtGui.QDesktopWidget().screenGeometry()
app_width = window_size.width() * 2 / 5.0
app_height = window_size.height() * 2 / 5.0

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
                   non_linear=True, timer_connected=True):
    if non_linear:
        b1.setEnabled(False)
        b.setEnabled(True)
    else:
        b.setEnabled(False)
        b1.setEnabled(True)
    timer.stop()
    if timer_connected:
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
    timer.start(50)


def gui_docks_p2_04_01(wind, area, shared_timer):
    vlayout_1 = pg.LayoutWidget()
    vlayout_2 = pg.LayoutWidget()
    p_2 = pg.PlotWidget(name='Plot_2')
    p_1 = pg.PlotWidget(name='Plot_1')
    btn_1 = QtGui.QPushButton('Run Linear')
    btn_2 = QtGui.QPushButton('Run non-Linear')
    d1 = Dock('Non-Linear', size=(1, 1), closable=True)
    d2 = Dock('Linear', size=(1, 1), closable=True)
    area.addDock(d1, 'left')
    area.addDock(d2, 'right')

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
            shared_timer,
            p_2,
            non_linear=False
        ))
    # noinspection PyUnresolvedReferences
    btn_2.clicked.connect(
        lambda: solve_p2_04_01(
            btn_1,
            btn_2,
            shared_timer,
            p_1,
            non_linear=True
        ))
    # noinspection PyUnresolvedReferences
    btn_3.clicked.connect(lambda: shared_timer.stop())

    btn_2.setEnabled(False)

    solve_p2_04_01(
        btn_1,
        btn_2,
        shared_timer,
        p_1,
        non_linear=True,
        timer_connected=False
    )

def add_which_dock(text, wind, area, shared_timer):
    if text == 'P2.04.01':
        gui_docks_p2_04_01(wind, area, shared_timer)

wind = QtGui.QWidget()
area = DockArea()
tree = pg.TreeWidget()
btn_3 = QtGui.QPushButton('STOP')
vlayout_0 = QtGui.QVBoxLayout()
splitter = QtGui.QSplitter()
shared_timer = pg.QtCore.QTimer()

ti2 = QtGui.QTreeWidgetItem([
    'CHAPTER 2.'
])
lab_1 = QtGui.QLabel('REACTION RATES AND OPERATING MODES')
ti211 = QtGui.QTreeWidgetItem([
    'P2.04.01'
])
lab_2 = QtGui.QLabel('ALKYLATION OF ISOPROPYLBENZENE')
ti221 = QtGui.QTreeWidgetItem([
    'P2.04.02'
])
lab_3 = QtGui.QLabel('DIFFUSION AND SOLID CATALYSIS')

wind.setLayout(vlayout_0)
wind.setWindowTitle('Walas problems')
wind.resize(app_width, app_height)

tree.setColumnCount(2)
tree.addTopLevelItem(ti2)
ti2.addChild(ti211)
ti2.addChild(ti221)
tree.setItemWidget(ti2, 1, lab_1)
tree.setItemWidget(ti211, 1, lab_2)
tree.setItemWidget(ti221, 1, lab_3)
tree.setDragEnabled(False)
tree.expandAll()
tree.resizeColumnToContents(0)
tree.resizeColumnToContents(1)
tree.setHeaderHidden(True)

splitter.addWidget(tree)
splitter.addWidget(area)
splitter.setSizes([app_width * 1 / 3.0, app_width * 2 / 3.0])

vlayout_0.addWidget(splitter)
vlayout_0.addWidget(btn_3)

tree.itemClicked.connect(
    lambda item, column:
    add_which_dock(item.text(0), wind, area, shared_timer)
)

wind.show()

if __name__ == '__main__':
    if (sys.flags.interactive != 1) or \
            not hasattr(QtCore, 'PYQT_VERSION'):
        # noinspection PyArgumentList
        QtGui.QApplication.instance().exec_()
