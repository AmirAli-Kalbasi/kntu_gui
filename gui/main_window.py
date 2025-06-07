from PyQt5 import QtWidgets, QtGui, QtCore

class MainWindow(QtWidgets.QMainWindow):
    """Main application window with placeholders for ML workflow."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("KNTU ML Trainer")
        self.resize(600, 400)
        self._init_ui()

    def _init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QVBoxLayout(central_widget)

        data_group = QtWidgets.QGroupBox("Data Folders")
        folder_layout = QtWidgets.QFormLayout()
        self.train_edit = QtWidgets.QLineEdit()
        self.test_edit = QtWidgets.QLineEdit()
        train_btn = QtWidgets.QPushButton("Train Folder…")
        test_btn = QtWidgets.QPushButton("Test Folder…")
        train_btn.clicked.connect(self.select_train_folder)
        test_btn.clicked.connect(self.select_test_folder)
        train_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        test_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        folder_layout.addRow(train_btn, self.train_edit)
        folder_layout.addRow(test_btn, self.test_edit)
        data_group.setLayout(folder_layout)
        main_layout.addWidget(data_group)

        model_group = QtWidgets.QGroupBox("Model")
        model_layout = QtWidgets.QVBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["CatBoost", "Decision Tree"])
        load_model_btn = QtWidgets.QPushButton("Load Pretrained Model…")
        load_model_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))
        load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(load_model_btn)
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # Train button
        train_model_btn = QtWidgets.QPushButton("Train")
        train_model_btn.clicked.connect(self.train_model)
        main_layout.addWidget(train_model_btn)

        # Placeholder for results
        self.results = QtWidgets.QTextEdit()
        self.results.setReadOnly(True)
        main_layout.addWidget(self.results)

    def select_train_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Train Folder")
        if folder:
            self.train_edit.setText(folder)

    def select_test_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Test Folder")
        if folder:
            self.test_edit.setText(folder)

    def load_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Model", filter="Model Files (*.bin *.pkl *.joblib)")
        if path:
            self.results.append(f"Loaded model: {path}")

    def train_model(self):
        # Placeholder for training logic
        self.results.append("Training not implemented yet.")
