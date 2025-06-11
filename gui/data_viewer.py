from PyQt5 import QtWidgets, QtCore
import pandas as pd
import matplotlib.pyplot as plt

from models import load_and_label_data, load_test_data
from features import SENSOR_COLS, SENSOR_COLORS


class DataViewer(QtWidgets.QDialog):
    """Simple window to visualise sensor data from train/test folders."""

    def __init__(self, parent=None, train_folder="", test_folder=""):
        super().__init__(parent)
        self.setWindowTitle("Data Viewer")
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
        )
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

        # Sensor check boxes with colour indicators
        sensor_layout = QtWidgets.QHBoxLayout()
        self.sensor_checks = {}
        for sensor in SENSOR_COLS:
            cb = QtWidgets.QCheckBox(sensor)
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_plot)

            colour = SENSOR_COLORS.get(sensor, "#000000")
            indicator = QtWidgets.QLabel()
            indicator.setFixedSize(12, 12)
            indicator.setStyleSheet(
                f"background-color: {colour}; border: 1px solid #000;"
            )

            wrapper = QtWidgets.QWidget()
            w_layout = QtWidgets.QHBoxLayout(wrapper)
            w_layout.setContentsMargins(2, 2, 2, 2)
            w_layout.setSpacing(4)
            w_layout.addWidget(indicator)
            w_layout.addWidget(cb)

            sensor_layout.addWidget(wrapper)
            self.sensor_checks[sensor] = cb
        layout.addLayout(sensor_layout)

        # File list and plot area
        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Horizontal)

        self.file_list = QtWidgets.QListWidget()
        self.file_list.currentItemChanged.connect(self.update_plot)
        splitter.addWidget(self.file_list)

        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )

        self.figure = plt.Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.mpl_connect("resize_event", self._on_resize)

        right = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(right)
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.canvas)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)

        # Ensure the splitter expands while controls keep minimal height
        layout.setStretchFactor(splitter, 1)

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
                colour = SENSOR_COLORS.get(sensor, None)
                ax.plot(
                    df["sample_num"],
                    df[sensor],
                    label=sensor,
                    color=colour,
                )
        ax.set_xlabel("Sample")
        ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()

    def _on_resize(self, event):
        """Ensure axes adjust correctly when the window is resized."""
        self.figure.tight_layout()
        self.canvas.draw_idle()
