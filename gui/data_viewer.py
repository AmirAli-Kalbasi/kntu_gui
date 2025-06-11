from PyQt5 import QtWidgets, QtCore
import pandas as pd
import matplotlib.pyplot as plt

from models import load_and_label_data, load_test_data
from features import SENSOR_COLS


class DataViewer(QtWidgets.QDialog):
    """Simple window to visualise sensor data from train/test folders."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Viewer")
        self.resize(800, 600)
        self.data = pd.DataFrame()
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Top controls
        controls = QtWidgets.QHBoxLayout()
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(["Train", "Test"])
        self.path_edit = QtWidgets.QLineEdit()
        select_btn = QtWidgets.QPushButton("Select Folder")
        select_btn.clicked.connect(self.select_folder)
        controls.addWidget(self.type_combo)
        controls.addWidget(self.path_edit)
        controls.addWidget(select_btn)
        layout.addLayout(controls)

        # Sensor check boxes
        sensor_layout = QtWidgets.QHBoxLayout()
        self.sensor_checks = {}
        for sensor in SENSOR_COLS:
            cb = QtWidgets.QCheckBox(sensor)
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_plot)
            sensor_layout.addWidget(cb)
            self.sensor_checks[sensor] = cb
        layout.addLayout(sensor_layout)

        # File list and plot area
        main = QtWidgets.QHBoxLayout()
        self.file_list = QtWidgets.QListWidget()
        self.file_list.currentItemChanged.connect(self.update_plot)
        main.addWidget(self.file_list, 1)

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

        self.figure = plt.Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        main.addWidget(self.canvas, 3)
        layout.addLayout(main)

    def select_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        self.path_edit.setText(folder)
        if self.type_combo.currentText() == "Train":
            self.data, _ = load_and_label_data(folder, verbose=False)
        else:
            self.data = load_test_data(folder)
        self.populate_file_list()

    def populate_file_list(self):
        self.file_list.clear()
        if self.data.empty:
            return
        if "folder" in self.data.columns:
            group = self.data.groupby(["folder", "source_file"])
        else:
            group = self.data.groupby(["source_file"])
        for keys, g in group:
            if isinstance(keys, tuple):
                folder, file = keys
            else:
                folder, file = "", keys
            label = g["label"].iloc[0] if "label" in g.columns else None
            text = f"{folder}/{file}" if folder else file
            if label is not None:
                text += f" - {label}"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, (folder, file))
            self.file_list.addItem(item)

    def update_plot(self, *args):
        item = self.file_list.currentItem()
        if item is None or self.data.empty:
            return
        folder, file = item.data(QtCore.Qt.UserRole)
        if folder:
            df = self.data[(self.data["folder"] == folder) & (self.data["source_file"] == file)]
        else:
            df = self.data[self.data["source_file"] == file]
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for sensor, cb in self.sensor_checks.items():
            if cb.isChecked() and sensor in df.columns:
                ax.plot(df["sample_num"], df[sensor], label=sensor)
        ax.set_xlabel("Sample")
        ax.legend()
        self.canvas.draw()
