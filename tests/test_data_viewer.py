import matplotlib
matplotlib.use('Agg')

import os
from PyQt5 import QtWidgets
import pandas as pd

from gui.data_viewer import DataViewer


def test_viewer_populates_list():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = DataViewer()
    df = pd.DataFrame({
        'sample_num': [0, 1],
        'Acc_X': [0, 1],
        'Acc_Y': [0, 1],
        'Acc_Z': [0, 1],
        'Gyro_X': [0, 1],
        'Gyro_Y': [0, 1],
        'Gyro_Z': [0, 1],
        'folder': ['f', 'f'],
        'source_file': ['file.mat', 'file.mat'],
        'label': ['normal', 'normal']
    })
    viewer.data = df
    viewer.populate_file_list()
    assert viewer.file_list.count() == 1
    text = viewer.file_list.item(0).text()
    assert 'file.mat' in text
    assert 'normal' in text
