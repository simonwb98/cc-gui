import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import imageio

import pyqtgraph as pg
from PyQt5 import uic, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog
from pyqtgraph.Qt import QtWidgets

from src.utils import open_int_file, ndarray_to_qimage


class CrossCorrelationWorker(QtCore.QObject):
    resultReady = QtCore.pyqtSignal(np.ndarray)

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def do_correlation(self, image, template):
        from scipy.signal import correlate2d
        try:
            template -= template.mean()
            result = correlate2d(image, template, mode='same', boundary="symm")
        except Exception as e:
            print("correlate2d failed:", e)
            result = np.zeros_like(image)
        self.resultReady.emit(result)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("form.ui", self)
        self.setWindowTitle("A Cunty Cross-Correlator")
        self.roi = None
        self.template = None

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

        # set up template image widget
        self.templateView = self.templateLabel

        self.crossCorrelationView = pg.ImageView()
        self.correlationWidget.layout().addWidget(self.crossCorrelationView) if self.correlationWidget.layout() else self.correlationWidget.setLayout(pg.QtWidgets.QVBoxLayout())
        self.correlationWidget.layout().addWidget(self.crossCorrelationView)

        # Worker thread 
        self.correlationThread = QtCore.QThread()
        self.crossCorrelationWorker = CrossCorrelationWorker()
        self.crossCorrelationWorker.moveToThread(self.correlationThread)
        self.crossCorrelationWorker.resultReady.connect(self.display_cross_correlation)
        self.correlationThread.start()


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
        self.roi.sigRegionChanged.connect(self.update_template_preview)
        self.roi.sigRegionChangeFinished.connect(self.update_cross_correlation)

    @QtCore.pyqtSlot()
    def update_template_preview(self):
        roi_data = self.roi.getArrayRegion(self.processedImage, self.scanView.imageItem)
        if hasattr(roi_data, 'filled'):
            roi_data = roi_data.filled(0)
        # Update templateLabel only
        qimg = ndarray_to_qimage(roi_data)
        widget_width = self.templateLabel.width()
        widget_height = self.templateLabel.height()
        pixmap = QtGui.QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(
            widget_width,
            widget_height,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.templateLabel.setPixmap(scaled_pixmap)
        self.template = roi_data  # Optionally keep current template for later

    @QtCore.pyqtSlot()
    def update_cross_correlation(self):
        # Get the most recent template if you haven't already
        template = self.template
        if template is None:
            # Or, re-extract as in the preview function if you prefer
            template = self.roi.getArrayRegion(self.processedImage, self.scanView.imageItem)
            if hasattr(template, 'filled'):
                template = template.filled(0)

        # Start worker job as before...
        image = self.processedImage.copy()
        template = template.copy()
        QtCore.QMetaObject.invokeMethod(
            self.crossCorrelationWorker,
            "do_correlation",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(np.ndarray, image),
            QtCore.Q_ARG(np.ndarray, template)
        )

    @QtCore.pyqtSlot()
    def load_image(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Image or Binary File", "", 
            "Binary Files (*.int);;Images (*.png *.jpg *.bmp *.tif *.tiff);;All Files (*)", options=options)
        if filename:
            if filename.endswith(".int"):
                image = open_int_file(filename).T
            else:
                image = imageio.imread(filename)

            self.originalImage = image
            self.processedImage = image
            self.scanView.setImage(self.processedImage)
        else:
            print("Illegal file format.")
        
    @QtCore.pyqtSlot()
    def update_image(self):
        # Update the displayed image based on the Gaussian slider value
        if self.scanView.image is not None:
            gaussian_value = self.gaussianSlider.value()
            # apply Gaussian filter
            self.processedImage = gaussian_filter(self.originalImage, sigma=gaussian_value)
            # space for other processing

            self.scanView.setImage(self.processedImage)
        else:
            print("Whoa there, buddy! No image loaded to update.")

    @QtCore.pyqtSlot()
    def perform_cross_correlation(self):
        if self.roi is None:
            print("No ROI selected!")
            return

        self.template = self.roi.getArrayRegion(self.processedImage, self.scanView.imageItem)


    @QtCore.pyqtSlot(np.ndarray)
    def display_cross_correlation(self, cc_result):
        # This is called in the main (GUI) thread automatically
        self.crossCorrelationView.setImage(cc_result)

    def closeEvent(self, event):
        self.correlationThread.quit()
        self.correlationThread.wait()
        event.accept()
    
        
    @QtCore.pyqtSlot()
    def play_gdr_songs(self):
        from src.utils import play_gdr_song
        play_gdr_song()
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
