from PyQt5 import QtWidgets
import logging

from gui.main_window import MainWindow


def main():
    logging.basicConfig(level=logging.INFO)
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
