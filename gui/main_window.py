from PyQt5 import QtWidgets, QtCore

from models import load_and_label_data

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
        ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
    ]),
    "LightGBM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(random_state=42)),
    ]),
    "CatBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CatBoostClassifier(verbose=0, random_state=42)),
    ]),
}

param_grids = {
    "DecisionTree": {
        "clf__max_depth": [3, 5, 10, None],
        "clf__min_samples_split": [2, 5, 10],
    },
    "RandomForest": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, 10, None],
        "clf__min_samples_split": [2, 5, 10],
    },
    "AdaBoost": {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.01, 0.1, 1.0],
    },
    "MLP": {
        "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": [0.0001, 0.001, 0.01],
    },
    "SVM": {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", "auto"],
    },
    "GradientBoosting": {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__max_depth": [3, 5, 7],
    },
    "XGBoost": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.1, 0.2],
    },
    "LightGBM": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.1, 0.2],
    },
    "CatBoost": {
        "clf__iterations": [50, 100, 200],
        "clf__depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.1, 0.2],
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

    def load_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Model", filter="Model Files (*.bin *.pkl *.joblib)")
        if path:
            self.results.append(f"Loaded model: {path}")

    def train_model(self):
        train_folder = self.train_edit.text()
        if not train_folder:
            self.results.append("Please select a train folder.")
            return

        data, _ = load_and_label_data(train_folder, verbose=False)
        if data.empty:
            self.results.append("No training data found.")
            return

        X = data.drop(columns=["label", "source_file"])
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
            fig.savefig("confusion_matrix.png")
            self.results.append(
                "Confusion matrix saved to confusion_matrix.png (non-interactive backend)."
            )
