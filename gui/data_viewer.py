from PyQt5 import QtWidgets, QtCore
import pandas as pd
import matplotlib.pyplot as plt

from models import load_and_label_data, load_test_data
from features import SENSOR_COLS


class DataViewer(QtWidgets.QDialog):
    """Simple window to visualise sensor data from train/test folders."""

    def __init__(self, parent=None, train_folder="", test_folder=""):
        super().__init__(parent)
        self.setWindowTitle("Data Viewer")
        self.resize(800, 600)
        self.data = pd.DataFrame()
        self.paths = {"Train": train_folder, "Test": test_folder}
        self._init_ui()
        self.type_changed(self.type_combo.currentText())

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Top controls
        controls = QtWidgets.QHBoxLayout()
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(["Train", "Test"])
        self.type_combo.currentTextChanged.connect(self.type_changed)
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

    def type_changed(self, text):
        """Update the displayed path and data when the type combo changes."""
        self.path_edit.setText(self.paths.get(text, ""))
        self.load_current_data()

    def load_current_data(self):
        """Load data for the current folder if a path is set."""
        folder = self.path_edit.text()
        if not folder:
            self.data = pd.DataFrame()
            self.populate_file_list()
            return
        if self.type_combo.currentText() == "Train":
            self.data, _ = load_and_label_data(folder, verbose=False)
        else:
            self.data = load_test_data(folder)
        self.populate_file_list()

    def select_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        current_type = self.type_combo.currentText()
        self.paths[current_type] = folder
        self.path_edit.setText(folder)
        self.load_current_data()

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
                if len(keys) == 2:
                    folder, file = keys
                elif len(keys) == 1:
                    folder, file = "", keys[0]
                else:
                    folder, file = keys[0], keys[-1]
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
