from PyQt5 import QtWidgets, QtCore

from models import load_and_label_data, load_test_data
from features import extract_features

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


def _gpu_available() -> bool:
    """Return ``True`` if an NVIDIA GPU appears to be present."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return result.returncode == 0
    except Exception:
        return False


GPU_AVAILABLE = _gpu_available()
if GPU_AVAILABLE:
    logger.info("GPU detected, GPU-enabled models will run on GPU.")
else:
    logger.info("No GPU detected, running on CPU.")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

classifiers = {
    "DecisionTree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(random_state=42)),
    ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42)),
    ]),
    "AdaBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", AdaBoostClassifier(random_state=42)),
    ]),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(random_state=42, max_iter=500)),
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=42)),
    ]),
    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ]),
    "XGBoost": Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                tree_method="gpu_hist" if GPU_AVAILABLE else "auto",
            ),
        ),
    ]),
    "LightGBM": Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LGBMClassifier(
                random_state=42,
                device="gpu" if GPU_AVAILABLE else "cpu",
            ),
        ),
    ]),
    "CatBoost": Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            CatBoostClassifier(
                verbose=0,
                random_state=42,
                task_type="GPU" if GPU_AVAILABLE else "CPU",
            ),
        ),
    ]),
}

param_grids = {
    "DecisionTree": {
        "clf__max_depth": [3, 5, 10, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    },
    "RandomForest": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, 10, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__max_features": ["sqrt", "log2", None],
    },
    "AdaBoost": {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.01, 0.1, 1.0],
        "clf__algorithm": ["SAMME", "SAMME.R"],
    },
    "MLP": {
        "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": [0.0001, 0.001, 0.01],
        "clf__learning_rate_init": [0.001, 0.01],
    },
    "SVM": {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["linear", "rbf", "poly"],
        "clf__gamma": ["scale", "auto"],
        "clf__degree": [3, 5],
    },
    "GradientBoosting": {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__max_depth": [3, 5, 7],
        "clf__subsample": [0.8, 1.0],
    },
    "XGBoost": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    },
    "LightGBM": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__num_leaves": [31, 63],
        "clf__subsample": [0.8, 1.0],
    },
    "CatBoost": {
        "clf__iterations": [50, 100, 200],
        "clf__depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__l2_leaf_reg": [1, 3, 5],
    },
}


def plot_confusion(cm, model_name):
    """Return a matplotlib Figure showing the confusion matrix."""
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Normal", "Fault"]
    )
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    return disp.figure_


class MainWindow(QtWidgets.QMainWindow):
    """Main application window with placeholders for ML workflow."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("KNTU ML Trainer")
        self.resize(600, 400)
        self.dataset_counts = None
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
        self.model_combo.addItems(classifiers.keys())
        layout.addWidget(self.model_combo)

        # Feature selection
        feature_group = QtWidgets.QGroupBox("Feature Categories")
        feature_layout = QtWidgets.QVBoxLayout()
        self.time_cb = QtWidgets.QCheckBox("Time")
        self.freq_cb = QtWidgets.QCheckBox("Frequency")
        self.wave_cb = QtWidgets.QCheckBox("Time-Frequency")
        feature_layout.addWidget(self.time_cb)
        feature_layout.addWidget(self.freq_cb)
        feature_layout.addWidget(self.wave_cb)
        feature_group.setLayout(feature_layout)
        layout.addWidget(feature_group)

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

    def _display_figure(self, fig):
        """Show a matplotlib figure in a modal dialog with save option."""
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Confusion Matrix")
        layout = QtWidgets.QVBoxLayout(dialog)

        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, dialog)

        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        dialog.resize(600, 500)
        dialog.exec_()

    def _show_test_table(self):
        """Display a table listing test files and their assigned labels."""
        if not hasattr(self, "test_data") or self.test_data.empty:
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Test Data Labels")
        layout = QtWidgets.QVBoxLayout(dialog)

        table = QtWidgets.QTableWidget(dialog)
        file_labels = (
            self.test_data[["source_file", "label"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["File", "Label"])
        table.setRowCount(len(file_labels))

        for row, record in file_labels.iterrows():
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(record["source_file"]))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(record["label"]))

        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(table)

        dialog.resize(400, 300)
        dialog.exec_()

    def select_train_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Train Folder")
        if folder:
            self.train_edit.setText(folder)
            _, counts = load_and_label_data(folder, verbose=False)
            self.dataset_counts = counts
            self.results.append(
                f"Loaded {counts['normal']} normal files and {counts['fault']} fault files."
            )

    def select_test_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Test Folder")
        if folder:
            self.test_edit.setText(folder)
            data, counts = load_and_label_data(folder, verbose=False)
            self.test_data = data
            self.results.append(
                f"Loaded {counts['normal']} normal test files and {counts['fault']} fault test files."
            )
            self._show_test_table()

    def load_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Model", filter="Model Files (*.bin *.pkl *.joblib)")
        if path:
            self.results.append(f"Loaded model: {path}")

    def train_model(self):
        train_folder = self.train_edit.text()
        if not train_folder:
            self.results.append("Please select a train folder.")
            return

        if GPU_AVAILABLE:
            self.results.append("Running on GPU for supported models.")
        else:
            self.results.append("Running on CPU.")

        data, _ = load_and_label_data(train_folder, verbose=False)
        if data.empty:
            self.results.append("No training data found.")
            return

        feature_list = []
        if self.time_cb.isChecked():
            feature_list.append("time_domain")
        if self.freq_cb.isChecked():
            feature_list.append("freq_domain")
        if self.wave_cb.isChecked():
            feature_list.append("wavelet")

        if feature_list:
            feats = extract_features(data, feature_list)
            X = feats.drop(columns=["label", "source_file", "folder"], errors="ignore")
            y = feats["label"].map({"normal": 0, "fault": 1})
        else:
            X = data.drop(columns=["label", "source_file", "folder"], errors="ignore")
            y = data["label"].map({"normal": 0, "fault": 1})

        model_name = self.model_combo.currentText()
        pipeline = classifiers[model_name]
        params = param_grids[model_name]

        self.results.append(f"Running grid search for {model_name}...")

        progress = QtWidgets.QProgressDialog("Training...", None, 0, 0, self)
        progress.setWindowModality(QtCore.Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.show()
        QtWidgets.QApplication.processEvents()

        grid = GridSearchCV(pipeline, params, cv=3, n_jobs=-1)
        grid.fit(X, y)

        progress.close()

        self.results.append(f"Best score: {grid.best_score_:.3f}")
        self.results.append(f"Best params: {grid.best_params_}")

        preds = grid.predict(X)
        cm = confusion_matrix(y, preds)
        fig = plot_confusion(cm, model_name)
        if "agg" not in plt.get_backend().lower():
            plt.show()
        else:
            if QtWidgets.QApplication.instance() is not None:
                self._display_figure(fig)
            else:
                fig.savefig("confusion_matrix.png")
                self.results.append(
                    "Confusion matrix saved to confusion_matrix.png (non-interactive backend)."
                )

        self._predict_test_folder(grid.best_estimator_)

    def _predict_test_folder(self, model):
        """Predict labels for each subfolder in the selected test directory."""
        test_folder = self.test_edit.text()
        if not test_folder:
            return

        data = load_test_data(test_folder)
        if data.empty:
            self.results.append("No test data found.")
            return
        feature_list = []
        if self.time_cb.isChecked():
            feature_list.append("time_domain")
        if self.freq_cb.isChecked():
            feature_list.append("freq_domain")
        if self.wave_cb.isChecked():
            feature_list.append("wavelet")

        if feature_list:
            feats = extract_features(data, feature_list)
            X_test = feats.drop(columns=["label", "source_file", "folder"], errors="ignore")
            preds = model.predict(X_test)
            feats["pred"] = preds
            file_predictions = []
            for _, row in feats.iterrows():
                folder = row.get("folder", "")
                label = "fault" if row["pred"] == 1 else "normal"
                file_predictions.append((folder, row["source_file"], label))
            data_for_cm = feats
        else:
            drop_cols = ["folder", "source_file"]
            if "label" in data.columns:
                drop_cols.append("label")

            X_test = data.drop(columns=drop_cols)
            preds = model.predict(X_test)
            data["pred"] = preds

            file_predictions = []
            for (folder_name, file_name), group in data.groupby(["folder", "source_file"]):
                label = "fault" if group["pred"].mean() >= 0.5 else "normal"
                file_predictions.append((folder_name, file_name, label))
            data_for_cm = data

        for folder, file, label in file_predictions:
            logger.debug("Predicted %s for %s/%s", label, folder, file)
        labels = [lbl for _, _, lbl in file_predictions]
        logger.info(
            "Prediction summary - normal: %s, fault: %s",
            labels.count("normal"),
            labels.count("fault"),
        )

        html = [
            "<h3>Test File Predictions</h3>",
            "<table border='1' cellspacing='0' cellpadding='3'>",
            "<tr><th>Folder</th><th>File</th><th>Predicted Label</th></tr>",
        ]
        for folder, file, label in file_predictions:
            color = "red" if label == "fault" else "green"
            html.append(
                f"<tr><td>{folder}</td><td>{file}</td><td style='color:{color}'><b>{label}</b></td></tr>"
            )
        html.append("</table>")
        self.results.append("".join(html))

        # If the test data included labels, show a confusion matrix
        if "label" in data_for_cm.columns and data_for_cm["label"].notna().any():
            y_true = data_for_cm["label"].map({"normal": 0, "fault": 1})
            cm = confusion_matrix(y_true, data_for_cm["pred"])
            fig = plot_confusion(cm, "Test Data")
            if "agg" not in plt.get_backend().lower():
                plt.show()
            else:
                if QtWidgets.QApplication.instance() is not None:
                    self._display_figure(fig)
                else:
                    fig.savefig("test_confusion_matrix.png")
                    self.results.append(
                        "Test confusion matrix saved to test_confusion_matrix.png (non-interactive backend)."
                    )
