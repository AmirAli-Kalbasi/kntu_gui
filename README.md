# kntu_gui

This project provides a simple PyQt based GUI skeleton for machine learning workflows.
The interface allows users to select training and test folders, choose a model type
(CatBoost or Decision Tree for now) and load a pretrained model. Training logic and
metrics are intentionally left as placeholders so the team can extend them later.

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
collects training data from folders named `Positive` and `Negative` inside a
given directory. Data in `Positive` is labeled as *fault* while data in
`Negative` is labeled as *normal*. After loading, the function prints how many
files were found for each label.

## Running

Install dependencies and start the GUI:

```bash
pip install -r requirements.txt
python run_gui.py
```
