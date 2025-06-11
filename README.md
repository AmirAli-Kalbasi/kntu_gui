# kntu_gui

This project provides a simple PyQt based GUI skeleton for machine learning workflows.
The interface allows users to select training and test folders, choose a model type
from a list of common classifiers and load a pretrained model. Training logic now
includes a grid search to tune hyperparameters for the selected model. The search
space was recently expanded with more options for each estimator so you can
experiment with deeper trees, different SVM kernels and additional boosting
settings.

## Structure

```
├── gui/           # GUI widgets and windows
│   ├── __init__.py
│   └── main_window.py
├── models/        # Future model implementations
├── tests/         # Unit tests
├── run_gui.py     # Entry point to start the application
├── requirements.txt
└── .gitignore
```

## Loading Dataset

The `models` package includes a helper function `load_and_label_data` that
collects training data from folders named `Positive` and `Negative` (or the
alternative `fault`/`normal`) inside a given directory. Data in the fault
folders is labeled as *fault* while data in the normal folders is labeled as
*normal*. The function returns both the loaded ``pandas.DataFrame`` and a
dictionary with counts of normal and fault files.
When ``verbose`` is ``True`` these counts are logged using Python's
``logging`` module and displayed in the GUI after selecting the training folder.

After each training run the confusion matrix is displayed in a modal window.
The window includes the standard matplotlib toolbar allowing the image to be
saved directly from the interface when a graphical backend is available.  When
running without a display, the matrix is written to ``confusion_matrix.png``.

When a test folder is selected, ``.mat`` files can either be placed directly in
the directory or organised into subfolders.  If folders named
``normal``/``fault`` (or their ``Negative``/``Positive`` variants) are
present, the files are treated as labeled and a confusion matrix for the test
data is displayed.  Otherwise each folder is predicted and the assigned label is
listed in a small table.

## Running

Install dependencies and start the GUI:

```bash
pip install -r requirements.txt
python run_gui.py
```
