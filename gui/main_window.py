from PyQt5 import QtWidgets

from models import load_and_label_data

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

        layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(layout)

        # Folder selection
        folder_layout = QtWidgets.QFormLayout()
        self.train_edit = QtWidgets.QLineEdit()
        self.test_edit = QtWidgets.QLineEdit()
        train_btn = QtWidgets.QPushButton("Select Train Folder")
        test_btn = QtWidgets.QPushButton("Select Test Folder")
        train_btn.clicked.connect(self.select_train_folder)
        test_btn.clicked.connect(self.select_test_folder)
        folder_layout.addRow(train_btn, self.train_edit)
        folder_layout.addRow(test_btn, self.test_edit)
        layout.addLayout(folder_layout)

        # Model selection
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["CatBoost", "Decision Tree"])
        layout.addWidget(self.model_combo)

        # Pretrained model load
        load_model_btn = QtWidgets.QPushButton("Load Pretrained Model")
        load_model_btn.clicked.connect(self.load_model)
        layout.addWidget(load_model_btn)

        # Train button
        train_model_btn = QtWidgets.QPushButton("Train")
        train_model_btn.clicked.connect(self.train_model)
        layout.addWidget(train_model_btn)

        # Placeholder for results
        self.results = QtWidgets.QTextEdit()
        self.results.setReadOnly(True)
        layout.addWidget(self.results)

    def select_train_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Train Folder"
        )
        if folder:
            data, counts = load_and_label_data(folder, verbose=False)
            self.train_data = data
            message = (
                f"{counts['normal']} normal files and "
                f"{counts['fault']} fault files detected."
            )
            QtWidgets.QMessageBox.information(
                self, "Training Data Loaded", message
            )
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
