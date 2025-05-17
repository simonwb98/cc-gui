import sys
import numpy as np
from scipy.ndimage import gaussian_filter

import pyqtgraph as pg
from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import QFileDialog
from pyqtgraph.Qt import QtWidgets

from src.utils import open_int_file

class CrossCorrelationWorker(QtCore.QThread):
    resultReady = QtCore.pyqtSignal(np.ndarray) # Signal to emit the result

    def __init__(self, image, template):
        super().__init__()
        self.image = image
        self.template = template

    def loop(self):
        from scipy.signal import correlate2d
        result = correlate2d(self.image, self.template, mode='same')
        self.resultReady.emit(result)  # Emit the result

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("form.ui", self)
        self.setWindowTitle("A Cunty Cross-Correlator")
        self.roi = None

        self.actionOpen_File.triggered.connect(self.load_image)
        self.actionPerform_Cross_Correlation.triggered.connect(self.perform_cross_correlation)
        
        self.actionRectangularROI.triggered.connect(self.show_roi)
        self.actionEllipsoidalROI.triggered.connect(self.show_roi)
        self.gaussianSlider.valueChanged.connect(self.update_image)
        self.actionPlay_GDR_worker_s_songs.triggered.connect(self.play_gdr_songs)

        # need to set up scan widget and cross-correlation widget
        self.scanView = pg.ImageView()
        self.scanImageWidget.layout().addWidget(self.scanView) if self.scanImageWidget.layout() else self.scanImageWidget.setLayout(pg.QtWidgets.QVBoxLayout())
        self.scanImageWidget.layout().addWidget(self.scanView)

        self.crossCorrelationView = pg.ImageView()
        self.correlationWidget.layout().addWidget(self.crossCorrelationView) if self.correlationWidget.layout() else self.correlationWidget.setLayout(pg.QtWidgets.QVBoxLayout())
        self.correlationWidget.layout().addWidget(self.crossCorrelationView)
        self.crossCorrelationWorker = None

    @QtCore.pyqtSlot()
    def show_roi(self):
        # If ROI already exists, remove it
        if self.roi is not None:
            self.scanView.removeItem(self.roi)

        sender = self.sender()
        if sender == self.actionRectangularROI:
            self.roi = pg.RectROI([50, 50], [100, 100], pen='r')
        elif sender == self.actionEllipsoidalROI:
            self.roi = pg.EllipseROI([50, 50], [100, 100], pen='r')
        else:
            pass

        self.scanView.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.update_roi)

    @QtCore.pyqtSlot()
    def update_roi(self):
        roi_state = self.roi.getArraySlice(self.data, self.imageview.imageItem)
        roi_data = roi_state[0]
        print("ROI data shape:", roi_data.shape)

    @QtCore.pyqtSlot()
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
            # write up support for other file formats.
            image = (
                pg.imread(filename) if hasattr(pg, "imread") else self.imread(filename)
            )
            self.originalImage = image
            self.processedImage = image
            self.scanView.setImage(self.processedImage)

    def imread(self, filename):
        if filename.endswith(".int"):
            image = open_int_file(filename)
            return image.T
        else:
            raise ValueError("Unsupported file format. Only .int files are supported.")
        
    @QtCore.pyqtSlot()
    def update_image(self):
        # Update the displayed image based on the Gaussian slider value
        if self.scanView.image is not None:
            gaussian_value = self.gaussianSlider.value()
            # apply Gaussian filter
            self.processed_image = gaussian_filter(self.originalImage, sigma=gaussian_value)
            # space for other processing

            self.scanView.setImage(self.processed_image)
        else:
            print("Whoa there, buddy! No image loaded to update.")
    
        
    @QtCore.pyqtSlot()
    def play_gdr_songs(self):
        from src.utils import play_gdr_song
        play_gdr_song()
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
