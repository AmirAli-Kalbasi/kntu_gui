from PyQt5 import QtWidgets, QtGui

from gui.main_window import MainWindow


def apply_dark_palette(app: QtWidgets.QApplication) -> None:
    """Apply a dark color palette for a more modern look."""
    app.setStyle("Fusion")
    dark = QtGui.QPalette()
    dark.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    dark.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
    dark.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
    dark.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    dark.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor("white"))
    dark.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("white"))
    dark.setColor(QtGui.QPalette.Text, QtGui.QColor("white"))
    dark.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    dark.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("white"))
    dark.setColor(QtGui.QPalette.BrightText, QtGui.QColor("red"))
    dark.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    dark.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    dark.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("black"))
    app.setPalette(dark)


def main():
    app = QtWidgets.QApplication([])
    apply_dark_palette(app)
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
