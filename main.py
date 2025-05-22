import sys
import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import correlate2d
import imageio

import pyqtgraph as pg
from PyQt5 import uic, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog
from pyqtgraph.Qt import QtWidgets

from src.utils import *


class CrossCorrelationWorker(QtCore.QObject):
    resultReady = QtCore.pyqtSignal(np.ndarray)
    rotatedresultReady = QtCore.pyqtSignal(np.ndarray, object)

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def do_correlation(self, image, template):
        try:
            template -= template.mean()
            result = correlate2d(image, template, mode='same', boundary="symm")
        except Exception as e:
            print("correlate2d failed:", e)
            result = np.zeros_like(image)
        self.resultReady.emit(result)

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, int)
    def do_rotated_correlation(self, image, template, angle):
        print(f"do_rotated_correlation called with angle: {angle}")
        try:
            template -= template.mean()

            angles = np.arange(0, 360, angle) 
            result_list = []
            rotated_images = []

            for angle in angles:
                rotated_img = rotate(image, angle, reshape=True)  # to avoid cropping
                rotated_images.append(rotated_img)
                result = correlate2d(rotated_img, template, mode='same', boundary="symm")
                result_list.append(result)
            # Stack results along a new axis
            stacked_result = np.stack(result_list) 
            
        except Exception as e:
            print("correlate2d failed:", e)
            stacked_result = np.zeros((len(angles), *image.shape))
            rotated_images = [np.zeros_like(image) for _ in angles]

        self.rotatedresultReady.emit(stacked_result, rotated_images)
        

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("form.ui", self)
        self.setWindowTitle("A Cunty Cross-Correlator")

        # Initialize variables
        self.roi = None
        self.template = None
        self.num_matches = self.matchesSlider.value()
        self.match_rects = []
        self.use_rotations = self.rotationCheckBox.isChecked()
        self.rotation_step = self.anglesSlider.value()

        # Connect UI actions/signals
        self.actionOpen_File.triggered.connect(self.load_image)
        self.actionPerform_Cross_Correlation.triggered.connect(self.perform_cross_correlation)

        self.actionRectangularROI.triggered.connect(self.show_roi)
        self.actionEllipsoidalROI.triggered.connect(self.show_roi)

        self.gaussianSlider.valueChanged.connect(self.update_image)
        self.matchesSlider.valueChanged.connect(self.on_match_count_changed)
        self.rotationCheckBox.toggled.connect(self.on_rotation_checkbox_toggled)
        self.anglesSlider.valueChanged.connect(self.on_angle_slider_changed)
        self.actionPlay_GDR_worker_s_songs.triggered.connect(self.play_gdr_songs)

        # Setup widgets
        self.scanView = pg.ImageView()
        if self.scanImageWidget.layout():
            self.scanImageWidget.layout().addWidget(self.scanView)
        else:
            self.scanImageWidget.setLayout(pg.QtWidgets.QVBoxLayout())
            self.scanImageWidget.layout().addWidget(self.scanView)

        self.templateView = self.templateLabel

        self.crossCorrelationView = pg.ImageView()
        if self.correlationWidget.layout():
            self.correlationWidget.layout().addWidget(self.crossCorrelationView)
        else:
            self.correlationWidget.setLayout(pg.QtWidgets.QVBoxLayout())
            self.correlationWidget.layout().addWidget(self.crossCorrelationView)

        # Setup worker thread
        self.correlationThread = QtCore.QThread()
        self.crossCorrelationWorker = CrossCorrelationWorker()
        self.crossCorrelationWorker.moveToThread(self.correlationThread)
        self.crossCorrelationWorker.resultReady.connect(self.display_cross_correlation)
        self.crossCorrelationWorker.rotatedresultReady.connect(self.display_rot_cross_correlation)
        self.correlationThread.start()

    # ----------- ROI related methods -----------
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

        self.scanView.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.update_template_preview)
        self.roi.sigRegionChangeFinished.connect(self.update_cross_correlation_based_on_mode)

    @QtCore.pyqtSlot()
    def update_template_preview(self):
        roi_data = self.roi.getArrayRegion(self.processedImage, self.scanView.imageItem)
        if hasattr(roi_data, 'filled'):
            roi_data = roi_data.filled(0)

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
        self.template = roi_data  # keep current template for later

    # ----------- Image loading and processing -----------
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
        if self.scanView.image is not None:
            gaussian_value = self.gaussianSlider.value()
            self.processedImage = gaussian_filter(self.originalImage, sigma=gaussian_value)
            self.scanView.setImage(self.processedImage)
        else:
            print("No image loaded to update.")

    # ----------- Cross-correlation trigger methods -----------

    @QtCore.pyqtSlot()
    def perform_cross_correlation(self):
        if self.roi is None:
            print("No ROI selected!")
            return

        self.template = self.roi.getArrayRegion(self.processedImage, self.scanView.imageItem)

    @QtCore.pyqtSlot()
    def update_cross_correlation(self):
        template = self.template
        if template is None:
            template = self.roi.getArrayRegion(self.processedImage, self.scanView.imageItem)
            if hasattr(template, 'filled'):
                template = template.filled(0)

        image = self.processedImage.copy()
        template = template.copy()

        QtCore.QMetaObject.invokeMethod(
            self.crossCorrelationWorker,
            "do_correlation",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(np.ndarray, image),
            QtCore.Q_ARG(np.ndarray, template)
        )

    def update_rotated_cross_correlation(self):
        image = self.processedImage.copy()
        template = self.template.copy()

        delta_angle = int(self.rotation_step)
        if delta_angle <= 0:
            delta_angle = 5

        print(f"Invoking do_rotated_correlation with delta_angle={delta_angle}")

        QtCore.QMetaObject.invokeMethod(
            self.crossCorrelationWorker,
            "do_rotated_correlation",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(np.ndarray, image),
            QtCore.Q_ARG(np.ndarray, template),
            QtCore.Q_ARG(int, delta_angle)
        )

    # ----------- Display methods -----------

    @QtCore.pyqtSlot(np.ndarray)
    def display_cross_correlation(self, cc_result):
        self.last_cc_result = cc_result  # cache

        self.crossCorrelationView.setImage(self.processedImage, autoLevels=True)

        if hasattr(self, "match_rects"):
            for rect in self.match_rects:
                self.crossCorrelationView.removeItem(rect)
        else:
            self.match_rects = []

        w = self.roi.size().x()
        h = self.roi.size().y()

        # Find top candidate matches
        num_candidates = self.num_matches * 20
        flat_indices = np.argpartition(cc_result.ravel(), -num_candidates)[-num_candidates:]
        coords = list(zip(*np.unravel_index(flat_indices, cc_result.shape)))

        coords.sort(key=lambda c: cc_result[c[0], c[1]], reverse=True)
        filtered_coords = non_max_suppression(coords, px_threshold=20)[:self.num_matches]

        for x, y in filtered_coords:
            x_top_left = x - w / 2
            y_top_left = y - h / 2

            rect = pg.RectROI(
                [x_top_left, y_top_left],
                [w, h],
                pen='g',
                movable=False,
                rotatable=False,
                resizable=False
            )
            self.crossCorrelationView.addItem(rect)
            self.match_rects.append(rect)

    @QtCore.pyqtSlot(np.ndarray, object)
    def display_rot_cross_correlation(self, stacked_cc, rotated_images):
        self.last_stacked_cc = stacked_cc
        self.last_rotated_images = rotated_images

        self.crossCorrelationView.setImage(self.processedImage, autoLevels=True)

        if hasattr(self, "match_rects"):
            for rect in self.match_rects:
                self.crossCorrelationView.removeItem(rect)
        else:
            self.match_rects = []

        num_angles, H_rot, W_rot = stacked_cc.shape
        angles = np.arange(0, 360, self.rotation_step)
        w = self.roi.size().x()
        h = self.roi.size().y()

        num_candidates = self.num_matches * 10
        flat_idx = np.argpartition(stacked_cc.ravel(), -num_candidates)[-num_candidates:]
        angle_idxs, y_idxs, x_idxs = np.unravel_index(flat_idx, stacked_cc.shape)

        scores = stacked_cc[angle_idxs, y_idxs, x_idxs]
        sorted_idx = np.argsort(scores)[::-1]

        coords = list(zip(angle_idxs[sorted_idx], y_idxs[sorted_idx], x_idxs[sorted_idx]))
        scores = scores[sorted_idx]

        filtered_coords = coords[:self.num_matches]

        H_orig, W_orig = self.processedImage.shape
        cx_orig, cy_orig = W_orig / 2, H_orig / 2

        for angle_idx, y_rot, x_rot in filtered_coords:
            angle = angles[angle_idx]
            rotated_shape = rotated_images[angle_idx].shape
            cx_rot, cy_rot = rotated_shape[1] / 2, rotated_shape[0] / 2

            x_shifted = x_rot - cx_rot
            y_shifted = y_rot - cy_rot

            angle_rad = -np.deg2rad(angle)
            x_orig_shifted = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
            y_orig_shifted = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)
            x_orig = x_orig_shifted + cx_orig
            y_orig = y_orig_shifted + cy_orig

            x_top_left = x_orig - w / 2
            y_top_left = y_orig - h / 2

            rect = pg.RectROI(
                [x_top_left, y_top_left],
                [w, h],
                pen='g',
                movable=False,
                rotatable=False,
                resizable=False,
                angle=angle
            )
            self.crossCorrelationView.addItem(rect)
            self.match_rects.append(rect)

    # ----------- Slots for UI events -----------

    @QtCore.pyqtSlot(int)
    def on_match_count_changed(self, value):
        self.num_matches = value
        if self.use_rotations:
            if hasattr(self, "last_stacked_cc"):
                self.display_rot_cross_correlation(self.last_stacked_cc, getattr(self, "last_rotated_images", None))
            else:
                self.update_cross_correlation_based_on_mode()
        else:
            if hasattr(self, "last_cc_result"):
                self.display_cross_correlation(self.last_cc_result)
            else:
                self.update_cross_correlation_based_on_mode()

    @QtCore.pyqtSlot(bool)
    def on_rotation_checkbox_toggled(self, state):
        print(f"Rotation checkbox toggled: {state}")
        self.use_rotations = state
        self.update_cross_correlation_based_on_mode()

    @QtCore.pyqtSlot(int)
    def on_angle_slider_changed(self, value):
        value = max(1, value)
        self.rotation_step = value
        if self.use_rotations:
            self.update_cross_correlation_based_on_mode()

    @QtCore.pyqtSlot()
    def update_cross_correlation_based_on_mode(self):
        if self.roi is None or self.template is None:
            return
        if self.use_rotations:
            print("Using rotated cross-correlation")
            self.update_rotated_cross_correlation()
        else:
            self.update_cross_correlation()

    # ----------- Cleanup -----------

    def closeEvent(self, event):
        self.correlationThread.quit()
        self.correlationThread.wait()
        event.accept()

    # ----------- Fun Add-Ons -----------

    @QtCore.pyqtSlot()
    def play_gdr_songs(self):
        from src.utils import play_gdr_song
        play_gdr_song()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())