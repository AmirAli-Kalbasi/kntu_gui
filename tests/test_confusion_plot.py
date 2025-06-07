import matplotlib
matplotlib.use('Agg')

from gui.main_window import plot_confusion
import numpy as np


def test_plot_confusion_returns_figure():
    cm = np.array([[2, 1], [0, 3]])
    fig = plot_confusion(cm, "TestModel")
    assert fig is not None
    assert fig.axes, "Figure should contain axes"
