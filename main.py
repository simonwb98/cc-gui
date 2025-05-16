import sys

import pyqtgraph as pg
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog
from pyqtgraph.Qt import QtWidgets

from src.utils import open_int_file


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("form.ui", self)

        # Find the QAction for 'Open File'
        self.actionOpen_File.triggered.connect(self.load_image)

    def load_image(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image or Binary File",
            "",
            "Binary Files (*.int);;Images (*.png *.jpg *.bmp *.tif *.tiff);;All Files (*)",
            options=options,
        )
        if filename:
            image = (
                pg.imread(filename) if hasattr(pg, "imread") else self.imread(filename)
            )
            self.iv.setImage(image)

    def imread(self, filename):
        # Custom imread function to read .int files
        if filename.endswith(".int"):
            image = open_int_file(filename)
            return image.T
        else:
            raise ValueError("Unsupported file format. Only .int files are supported.")


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
