# kntu_gui

This project provides a simple PyQt based GUI skeleton for machine learning workflows.
The interface allows users to select training and test folders, choose a model type
(CatBoost or Decision Tree for now) and load a pretrained model. Training logic and
metrics are intentionally left as placeholders so the team can extend them later.
The GUI applies a dark theme for a more modern appearance.

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

## Running

Install dependencies and start the GUI:

```bash
pip install -r requirements.txt
python run_gui.py
```
