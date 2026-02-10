from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QStackedWidget, QLineEdit, QPushButton, QLabel, QSpinBox, QGridLayout, QSizePolicy, QProgressBar, QDoubleSpinBox, QScrollArea
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
import cv2
import sys
import numpy as np
import os
import pickle
import time
from CVModel import CVModel


TK_BG = "#f0f0f0"
TK_BTN = "#d9d9d9"
TK_BORDER = "#a3a3a3"


def apply_tk_layout(layout):
    layout.setContentsMargins(40, 25, 40, 25)
    layout.setSpacing(12)


def make_tk_header(text: str) -> QLabel:
    header = QLabel(text)
    header.setAlignment(Qt.AlignHCenter)
    header.setObjectName("tkHeader")
    return header

cvm = None
train_screen = None
t_screen = None
u_screen = None
tfv_screen = None

class HomeScreen(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        layout = QVBoxLayout()
        apply_tk_layout(layout)
        layout.addWidget(make_tk_header("Computer Vision Playground"))
        layout.addWidget(QPushButton("Load", clicked=self.go_to_screen1))
        layout.addWidget(QPushButton("Create New", clicked=self.go_to_screen2))
        self.setLayout(layout)

    def go_to_screen1(self):
        self.stack.setCurrentIndex(1)
    
    def go_to_screen2(self):
        self.stack.setCurrentIndex(2) 

class Placeholder(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        layout = QVBoxLayout()
        apply_tk_layout(layout)
        self.setLayout(layout)

class Load(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.setWindowTitle("Load CV Model")
        layout = QVBoxLayout()
        apply_tk_layout(layout)
        layout.addWidget(make_tk_header("Load CV Model"))
        layout.addWidget(QPushButton("Home", clicked=self.go_to_screen1))
        self.setLayout(layout)

        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("Model To Load (no .pkl)")
        self.line_edit.setFixedWidth(240)
        layout.addWidget(self.line_edit)

        button = QPushButton("Load And Train")
        button.clicked.connect(self.load_train)
        layout.addWidget(button)

        button1 = QPushButton("Load And Use")
        button1.clicked.connect(self.load_use)
        layout.addWidget(button1)
        layout.addStretch()

    def load_train(self):
        global cvm
        try:
            with open(f"{self.line_edit.text()}.pkl", "rb") as f:
                cvm = pickle.load(f)
            if t_screen is not None:
                t_screen.refresh_model()
            self.stack.setCurrentIndex(4) 
        except(FileNotFoundError):
            print("file not found")

    def load_use(self):
        global cvm
        try:
            with open(f"{self.line_edit.text()}.pkl", "rb") as f:
                cvm = pickle.load(f)
            if u_screen is not None:
                u_screen.refresh_model()
            self.stack.setCurrentIndex(3) 
        except(FileNotFoundError):
            print("file not found")

    def go_to_screen1(self):
        self.stack.setCurrentIndex(0) 

class CreateNewScreen(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.setWindowTitle("Create New CV Model")
        layout = QVBoxLayout()
        apply_tk_layout(layout)
        layout.addWidget(make_tk_header("Create New CV Model"))
        layout.addWidget(QPushButton("Home", clicked=self.go_to_screen1))
        self.setLayout(layout)

        xy_layout = QGridLayout()
        xy_layout.setHorizontalSpacing(16)
        xy_layout.setVerticalSpacing(10)

        self.size_spin_box = QSpinBox()
        self.size_spin_box.setMinimum(0)
        self.size_spin_box.setMaximum(100)
        self.size_spin_box.setSingleStep(1)
        self.size_spin_box.setFixedWidth(140)
        xy_layout.addWidget(QLabel("Size x Size:"))
        xy_layout.addWidget(self.size_spin_box)

        self.kernel_spin_box = QSpinBox()
        self.kernel_spin_box.setMinimum(1)
        self.kernel_spin_box.setMaximum(7)
        self.kernel_spin_box.setSingleStep(2)
        self.kernel_spin_box.setFixedWidth(140)
        xy_layout.addWidget(QLabel("Kernel Size:"))
        xy_layout.addWidget(self.kernel_spin_box)
        kernel_hint = QLabel("* Kernel size must be an odd number")
        xy_layout.addWidget(kernel_hint, 2, 0, 1, 2)

        self.layer_spin_box = QSpinBox()
        self.layer_spin_box.setMinimum(1)
        self.layer_spin_box.setMaximum(50)
        self.layer_spin_box.setSingleStep(1)
        self.layer_spin_box.setFixedWidth(140)
        xy_layout.addWidget(QLabel("Layers:"))
        xy_layout.addWidget(self.layer_spin_box)

        self.kpl_spin_box = QSpinBox()
        self.kpl_spin_box.setMinimum(1)
        self.kpl_spin_box.setMaximum(50)
        self.kpl_spin_box.setSingleStep(1)
        self.kpl_spin_box.setFixedWidth(140)
        xy_layout.addWidget(QLabel("Kernels/Layer:"))
        xy_layout.addWidget(self.kpl_spin_box)

        self.outp_spin_box = QSpinBox()
        self.outp_spin_box.setMinimum(1)
        self.outp_spin_box.setMaximum(50)
        self.outp_spin_box.setSingleStep(1)
        self.outp_spin_box.setFixedWidth(140)
        xy_layout.addWidget(QLabel("outputs:"))
        xy_layout.addWidget(self.outp_spin_box)

        button = QPushButton("Submit")
        button.clicked.connect(self.createCV)
        layout.addWidget(button)

        self.result_label = QLabel("")

        layout.addLayout(xy_layout)
        layout.addWidget(self.result_label)
        layout.addStretch()
        
        self.setLayout(layout)

    def createCV(self):
        global cvm, train_screen
        input_size = self.size_spin_box.value()
        cvm = CVModel(
            (input_size, input_size),
            self.outp_spin_box.value(),
            self.layer_spin_box.value(),
            self.kpl_spin_box.value(),
            3,
            self.kernel_spin_box.value()
        )
        if train_screen is not None:
            train_screen.refresh_model()
        self.stack.setCurrentIndex(5)

    def go_to_screen1(self):
        self.stack.setCurrentIndex(0)

class TrainScreen(QWidget):
    def __init__(self, stack):
        global cvm
        self.training = False
        self.last_loss = None
        super().__init__()
        self.stack = stack
        self.setWindowTitle("Train Model")
        layout = QVBoxLayout()
        apply_tk_layout(layout)
        layout.addWidget(make_tk_header("Live Training"))
        layout.addWidget(QPushButton("Home", clicked=self.go_to_screen1))
        layout.addWidget(QPushButton("Start/Stop Training", clicked=self.train))
        self.setLayout(layout)

        xy_layout = QGridLayout()
        xy_layout.setHorizontalSpacing(14)
        xy_layout.setVerticalSpacing(8)
        self.outp_spin_box = QSpinBox()
        self.outp_spin_box.setMinimum(1)
        self.outp_spin_box.setMaximum(cvm.outputs if cvm!=None else 1)
        self.outp_spin_box.setFixedWidth(120)
        xy_layout.addWidget(QLabel("Correct Output:"))
        xy_layout.addWidget(self.outp_spin_box)

        layout.addLayout(xy_layout)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.camera_label)

        self.training_label = QLabel("Training: OFF")
        layout.addWidget(self.training_label)

        self.error_label = QLabel("Loss: -")
        layout.addWidget(self.error_label)

        self.last_output = QLabel("Last Output: -")
        layout.addWidget(self.last_output)

        self.learning_rate_in = QSpinBox()
        self.learning_rate_in.setMinimum(0)
        self.learning_rate_in.setMaximum(10000)
        self.learning_rate_in.setFixedWidth(140)
        layout.addWidget(QLabel("Learning Rate (/10000):"))
        layout.addWidget(self.learning_rate_in)
        layout.addStretch()

        self.learning_rate = self.learning_rate_in.value()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        if self.cap.isOpened():
            self.timer.start(30)
        else:
            self.camera_label.setText("Unable to open camera")

    def update_frame(self):
        if not self.cap.isOpened():
            return

        ret, frame_bgr = self.cap.read()
        if not ret:
            self.camera_label.setText("Failed to read frame")
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.training:
            global cvm
            self.learning_rate = self.learning_rate_in.value()
            if cvm is not None:
                tensor = self.frame_to_tensor(frame_rgb, cvm.inputSize[0])
                output = cvm.forwardPass(tensor)
                expected = [1.0 if i + 1 == self.outp_spin_box.value() else 0.0 for i in range(cvm.outputs)]
                loss = cvm.backpropigate(expected, self.learning_rate/10000)
                if loss is not None:
                    self.last_loss = loss
                    self.error_label.setText(f"Loss: {loss:.4f}")
                    self.last_output.setText(f"Last Output: {output}")

        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.camera_label.setPixmap(scaled_pixmap)
    
    def train(self):
        global cvm
        if cvm is None:
            self.training = False
            self.training_label.setText("Training: OFF (no model)")
            return

        self.training = not self.training
        status = "ON" if self.training else "OFF"
        self.training_label.setText(f"Training: {status}")

    def go_to_screen1(self):
        self.stack.setCurrentIndex(0)
    

    def frame_to_tensor(self, frame_rgb, size: int):
        size = max(1, size)
        resized = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return chw

    def refresh_model(self):
        global cvm
        if cvm is None:
            return

        self.outp_spin_box.setMaximum(max(1, cvm.outputs))
        self.outp_spin_box.setValue(1)
        self.training_label.setText("Training: OFF")
        self.error_label.setText("Loss: -")
        self.training = False

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)

class UseScreen(QWidget):
    def __init__(self, stack):
        global cvm
        self.using = False
        self.last_loss = None
        super().__init__()
        self.stack = stack
        self.setWindowTitle("Use Model")
        layout = QVBoxLayout()
        apply_tk_layout(layout)
        layout.addWidget(make_tk_header("Use Model"))
        layout.addWidget(QPushButton("Home", clicked=self.go_to_screen1))
        layout.addWidget(QPushButton("Start/Stop", clicked=self.use))
        self.setLayout(layout)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.camera_label)

        self.using_label = QLabel("Using: OFF")
        layout.addWidget(self.using_label)

        self.last_output = QLabel("Last Output: -")
        layout.addWidget(self.last_output)

        layout.addWidget(make_tk_header("CNN Feature Maps"))

        self.feature_scroll = QScrollArea()
        self.feature_scroll.setWidgetResizable(True)
        self.feature_container = QWidget()
        self.feature_grid = QGridLayout()
        self.feature_grid.setContentsMargins(0, 0, 0, 0)
        self.feature_grid.setSpacing(6)
        self.feature_container.setLayout(self.feature_grid)
        self.feature_scroll.setWidget(self.feature_container)
        layout.addWidget(self.feature_scroll)

        self.feature_labels: list[QLabel] = []
        self.feature_buffers: list[np.ndarray | None] = []
        self.feature_columns = 5
        layout.addStretch()

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        if self.cap.isOpened():
            self.timer.start(30)
        else:
            self.camera_label.setText("Unable to open camera")

    def update_frame(self):
        if not self.cap.isOpened():
            return

        ret, frame_bgr = self.cap.read()
        if not ret:
            self.camera_label.setText("Failed to read frame")
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.using:
            global cvm
            if cvm is not None:
                tensor = self.frame_to_tensor(frame_rgb, cvm.inputSize[0])
                output = cvm.forwardPass(tensor)
                if output is not None:
                    self.last_output.setText(f"Last Output: {output}")
                feature_maps = cvm.get_last_feature_maps()
                if feature_maps is not None:
                    self.update_feature_maps_display(feature_maps)

        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.camera_label.setPixmap(scaled_pixmap)
    
    def use(self):
        global cvm
        if cvm is None:
            self.using = False
            self.using_label.setText("Using: OFF (no model)")
            return

        self.using = not self.using
        status = "ON" if self.using else "OFF"
        self.using_label.setText(f"Using: {status}")

    def go_to_screen1(self):
        self.stack.setCurrentIndex(0)
    

    def frame_to_tensor(self, frame_rgb, size: int):
        size = max(1, size)
        resized = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return chw

    def refresh_model(self):
        global cvm
        if cvm is None:
            return

        self.using_label.setText("Using: OFF")
        self.using = False
        self.clear_feature_grid()

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)

    def clear_feature_grid(self):
        for label in self.feature_labels:
            self.feature_grid.removeWidget(label)
            label.deleteLater()
        self.feature_labels = []
        self.feature_buffers = []

    def update_feature_maps_display(self, feature_maps: np.ndarray):
        if feature_maps.ndim != 3:
            self.clear_feature_grid()
            return

        channels = feature_maps.shape[0]
        if channels <= 0:
            self.clear_feature_grid()
            return

        if len(self.feature_labels) != channels:
            self.clear_feature_grid()
            for idx in range(channels):
                label = QLabel()
                label.setFixedSize(96, 96)
                label.setAlignment(Qt.AlignCenter)
                row = idx // self.feature_columns
                col = idx % self.feature_columns
                self.feature_grid.addWidget(label, row, col)
                self.feature_labels.append(label)
                self.feature_buffers.append(None)

        for idx in range(channels):
            fmap = feature_maps[idx]
            min_val = float(np.min(fmap))
            max_val = float(np.max(fmap))
            if np.isclose(min_val, max_val):
                normalized = np.zeros_like(fmap, dtype=np.float32)
            else:
                normalized = (fmap - min_val) / (max_val - min_val)

            buffer = (normalized * 255.0).clip(0, 255).astype(np.uint8)
            self.feature_buffers[idx] = buffer
            height, width = buffer.shape
            bytes_per_line = width
            qimage = QImage(buffer.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage).scaled(
                self.feature_labels[idx].width(),
                self.feature_labels[idx].height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.feature_labels[idx].setPixmap(pixmap)

class MakeTrainingInfoScreen(QWidget):
    def __init__(self, stack):
        global cvm
        self.training = False
        self.last_loss = None
        self.video_writer = None
        self.record_fps = 30
        self.record_size = None
        super().__init__()
        self.stack = stack
        self.setWindowTitle("Train Model")
        layout = QVBoxLayout()
        apply_tk_layout(layout)
        layout.addWidget(make_tk_header("Collect Training Clips"))
        layout.addWidget(QPushButton("Home", clicked=self.go_to_screen1))
        layout.addWidget(QPushButton("Start/Stop Collecting", clicked=self.train))
        layout.addWidget(QPushButton("Finished Collecting", clicked=self.done))
        self.setLayout(layout)

        selector_row = QGridLayout()
        selector_row.setHorizontalSpacing(12)
        selector_row.setVerticalSpacing(6)
        selector_row.addWidget(QLabel("Correct Output:"), 0, 0)
        self.outp_spin_box = QSpinBox()
        self.outp_spin_box.setMinimum(1)
        self.outp_spin_box.setMaximum(cvm.outputs if cvm is not None else 1)
        self.outp_spin_box.setFixedWidth(140)
        selector_row.addWidget(self.outp_spin_box, 0, 1)

        selector_row.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(5)
        self.learning_rate_spin.setMinimum(0.0001)
        self.learning_rate_spin.setMaximum(1.0)
        self.learning_rate_spin.setSingleStep(0.0005)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setFixedWidth(140)
        selector_row.addWidget(self.learning_rate_spin, 1, 1)

        selector_row.addWidget(QLabel("Epochs:"), 2, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimum(1)
        self.epochs_spin.setMaximum(1000)
        self.epochs_spin.setValue(20)
        self.epochs_spin.setFixedWidth(140)
        selector_row.addWidget(self.epochs_spin, 2, 1)

        layout.addLayout(selector_row)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.camera_label)

        self.training_label = QLabel("Collecting Data: OFF")
        layout.addWidget(self.training_label)
        layout.addStretch()

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        if self.cap.isOpened():
            self.timer.start(30)
        else:
            self.camera_label.setText("Unable to open camera")

    def update_frame(self):
        if not self.cap.isOpened():
            return

        ret, frame_bgr = self.cap.read()
        if not ret:
            self.camera_label.setText("Failed to read frame")
            return

        if self.record_size is None:
            h, w, _ = frame_bgr.shape
            self.record_size = (w, h)

        if self.training and self.video_writer is not None:
            self.video_writer.write(frame_bgr)
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)
    
    def train(self):
        if self.record_size is None:
            self.training = False
            self.training_label.setText("Collecting Data: OFF (no camera frame yet)")
            return
        global cvm
        if cvm is None:
            self.training = False
            self.training_label.setText("Collecting Data: OFF (no model)")
            return

        self.training = not self.training

        if self.training:
            # start
            out_idx = self.outp_spin_box.value()
            os.makedirs("saved_videos", exist_ok=True)
            filename = f"saved_videos/correct_in_{out_idx}.mov"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # works for .mov and .mp4
            self.video_writer = cv2.VideoWriter(
                filename,
                fourcc,
                self.record_fps,
                self.record_size
            )

            self.training_label.setText("Collecting Data: ON")
        else:
            # stop recording 
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

            self.training_label.setText("Collecting Data: OFF")
    def go_to_screen1(self):
        self.stack.setCurrentIndex(0)
    

    def frame_to_tensor(self, frame_rgb, size: int):
        size = max(1, size)
        resized = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return chw

    def refresh_model(self):
        global cvm
        if cvm is None:
            return

        self.outp_spin_box.setMaximum(max(1, cvm.outputs))
        self.outp_spin_box.setValue(1)
        self.training_label.setText("Collecting Data: OFF")
        self.training = False

    def done(self):
        global tfv_screen
        if tfv_screen is not None:
            tfv_screen.set_training_params(
                learning_rate=self.learning_rate_spin.value(),
                epochs=self.epochs_spin.value()
            )
        self.stack.setCurrentIndex(6)

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.cap.isOpened():
            self.cap.release()

        super().closeEvent(event)

class TrainFromVideos(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.setWindowTitle("Train From Videos")
        layout = QVBoxLayout()
        apply_tk_layout(layout)
        layout.addWidget(make_tk_header("Train From Saved Clips"))
        layout.addWidget(QPushButton("Home", clicked=self.go_to_screen1))

        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setAlignment(Qt.AlignCenter)

        self.error = QLabel("Error: -")
        layout.addWidget(self.error)

        batch_row = QGridLayout()
        batch_row.setHorizontalSpacing(10)
        batch_row.addWidget(QLabel("Batch Size:"), 0, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setMinimum(1)
        self.batch_spin.setMaximum(256)
        self.batch_spin.setValue(8)
        self.batch_spin.setFixedWidth(120)
        self.batch_spin.valueChanged.connect(self.update_batch_size)
        batch_row.addWidget(self.batch_spin, 0, 1)
        layout.addLayout(batch_row)

        self.eta_label = QLabel("Estimated time remaining: --")
        layout.addWidget(self.eta_label)

        layout.addWidget(self.progress)

        self.continue_btn = QPushButton("Continue", clicked=self.save_and_done)
        self.continue_btn.setEnabled(False)
        layout.addWidget(self.continue_btn)

        self.setLayout(layout)

        self.epochs = 20
        self.learningRate = 0.001
        self.worker = None
        self._stopping_worker = False
    
    def set_training_params(self, learning_rate: float, epochs: int):
        self.learningRate = max(1e-6, float(learning_rate))
        self.epochs = max(1, int(epochs))

    def update_batch_size(self, value: int):
        if self.worker is not None:
            self.worker.set_batch_size(value)

    def go_to_screen1(self):
        self.stop_training(wait=True)
        self.stack.setCurrentIndex(0)

    def save_and_done(self):
        global cvm
        with open("vision_model.pkl", "wb") as f:
            pickle.dump(cvm, f)

        self.go_to_screen1()

    def start_training(self):
        global cvm
        if cvm == None:
            return
        if self.worker is not None:
            return

        video_paths = [
            f"saved_videos/correct_in_{i+1}.mov" for i in range(cvm.outputs)
        ]

        self.worker = VideoTrainingWorker(
            video_paths=video_paths,
            cvm=cvm,
            learning_rate=self.learningRate,
            epochs=self.epochs,
            batch_size=self.batch_spin.value()
        )

        self.progress.setValue(0)
        self.continue_btn.setEnabled(False)
        self.update_eta(-1)

        self.worker.progress.connect(self.progress.setValue)
        self.worker.error.connect(lambda val: self.error.setText(f"Error: {val:.4f}"))
        self.worker.eta.connect(self.update_eta)
        self.worker.finished.connect(self.training_done)

        self.worker.start()

    def stop_training(self, wait: bool = False):
        if self.worker:
            self._stopping_worker = True
            worker = self.worker
            worker.stop()
            if wait:
                worker.wait()
            self.worker = None
            self._stopping_worker = False

    def showEvent(self, event):
        super().showEvent(event)
        if cvm is not None:
            self.start_training()

    def hideEvent(self, event):
        self.stop_training(wait=True)
        super().hideEvent(event)

    def training_done(self):
        if self._stopping_worker:
            self.progress.setValue(0)
            self.update_eta(-1)
            self.continue_btn.setEnabled(False)
        else:
            self.progress.setValue(100)
            self.continue_btn.setEnabled(True)
            self.update_eta(0)
        self.worker = None

    def update_eta(self, seconds):
        if seconds is None or seconds < 0:
            text = "--"
        else:
            total_seconds = max(0, int(round(seconds)))
            if total_seconds >= 3600:
                hours, rem = divmod(total_seconds, 3600)
                minutes, secs = divmod(rem, 60)
                text = f"{hours}h {minutes}m {secs}s"
            elif total_seconds >= 60:
                minutes, secs = divmod(total_seconds, 60)
                text = f"{minutes}m {secs}s"
            else:
                text = f"{total_seconds}s"

        self.eta_label.setText(f"Estimated time remaining: {text}")

class VideoTrainingWorker(QThread):
    progress = Signal(int)
    finished = Signal()
    error = Signal(float)
    eta = Signal(float)

    def __init__(self, video_paths, cvm, learning_rate, epochs, batch_size):
        super().__init__()
        self.video_paths = video_paths
        self.cvm = cvm
        self.learning_rate = learning_rate
        self._running = True
        self.epochs = epochs
        self.batch_size = max(1, batch_size)

    def stop(self):
        self._running = False

    def set_batch_size(self, value: int):
        self.batch_size = max(1, int(value))

    def run(self):
        if not self.video_paths or self.epochs <= 0:
            self.progress.emit(100)
            self.eta.emit(0.0)
            self.finished.emit()
            return

        frame_counts = []
        for path in self.video_paths:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            else:
                frame_counts.append(0)
            cap.release()

        total_frames_per_epoch = sum(frame_counts)
        if total_frames_per_epoch <= 0:
            self.progress.emit(100)
            self.eta.emit(0.0)
            self.finished.emit()
            return

        total_work = total_frames_per_epoch * self.epochs
        processed_global = 0
        start_time = time.time()
        batch_inputs = []
        batch_expected = []

        for _ in range(self.epochs):
            if not self._running:
                break

            for path in self.video_paths:
                if not self._running:
                    break

                cap = cv2.VideoCapture(path)
                while cap.isOpened():
                    if not self._running:
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor = self.frame_to_tensor(frame_rgb, self.cvm.inputSize[0])
                    expected = self.get_expected_output(path)

                    batch_inputs.append(tensor)
                    batch_expected.append(expected)

                    if len(batch_inputs) >= self.batch_size:
                        processed_global = self._train_batch(
                            batch_inputs,
                            batch_expected,
                            processed_global,
                            total_work,
                            start_time
                        )

                cap.release()

        if self._running and batch_inputs:
            processed_global = self._train_batch(
                batch_inputs,
                batch_expected,
                processed_global,
                total_work,
                start_time
            )
        elif not self._running:
            batch_inputs.clear()
            batch_expected.clear()

        if processed_global >= total_work:
            self.progress.emit(100)
            self.eta.emit(0.0)
        else:
            self.eta.emit(-1.0)

        self.finished.emit()

    def frame_to_tensor(self, frame_rgb, size):
        frame = cv2.resize(frame_rgb, (size, size))
        frame = frame.astype("float32") / 255.0
        return frame.transpose(2, 0, 1)

    def get_expected_output(self, path):
        idx = int(os.path.basename(path).split("_")[-1].split(".")[0])
        return [1.0 if i + 1 == idx else 0.0 for i in range(self.cvm.outputs)]

    def _train_batch(self, batch_inputs, batch_expected, processed_global, total_work, start_time):
        if not batch_inputs:
            return processed_global

        batch_len = len(batch_inputs)
        avg_loss = self.cvm.train_on_batch(batch_inputs, batch_expected, self.learning_rate)
        if avg_loss is not None:
            self.error.emit(avg_loss)

        processed_global += batch_len
        fraction = processed_global / total_work if total_work else 1
        percent = int(fraction * 100)
        self.progress.emit(min(100, percent))

        if processed_global > 0 and total_work:
            elapsed = max(1e-6, time.time() - start_time)
            rate = elapsed / processed_global
            remaining = max(0, total_work - processed_global)
            eta_seconds = rate * remaining
            self.eta.emit(eta_seconds)

        batch_inputs.clear()
        batch_expected.clear()
        return processed_global

app = QApplication(sys.argv)
app.setStyleSheet(
    f"""
    QWidget {{
        background-color: {TK_BG};
        font-family: 'Segoe UI';
        color: #000;
        font-size: 14px;
    }}
    #tkHeader {{
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 10px;
    }}
    QPushButton {{
        background-color: {TK_BTN};
        border: 2px solid {TK_BORDER};
        padding: 4px 12px;
        min-width: 140px;
    }}
    QPushButton:pressed {{
        background-color: #c0c0c0;
    }}
    QLineEdit {{
        background-color: #ffffff;
        border: 2px solid {TK_BORDER};
        padding: 4px;
    }}
    QSpinBox {{
        background-color: #ffffff;
        border: 2px solid {TK_BORDER};
        padding: 4px;
        padding-right: 26px;
    }}
    QSpinBox::up-button {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid {TK_BORDER};
        border-bottom: 1px solid {TK_BORDER};
        background: {TK_BTN};
        padding: 0px;
    }}
    QSpinBox::down-button {{
        subcontrol-origin: padding;
        subcontrol-position: bottom right;
        width: 20px;
        border-left: 1px solid {TK_BORDER};
        border-top: 1px solid {TK_BORDER};
        background: {TK_BTN};
        padding: 0px;
    }}
    QSpinBox::up-button:pressed,
    QSpinBox::down-button:pressed {{
        background: #c0c0c0;
    }}
    QSpinBox::up-button:disabled,
    QSpinBox::down-button:disabled {{
        background: #e4e4e4;
    }}
    QSpinBox::up-arrow {{
        image: url(:/qt-project.org/styles/commonstyle/images/arrowup.png);
        width: 8px;
        height: 6px;
    }}
    QSpinBox::down-arrow {{
        image: url(:/qt-project.org/styles/commonstyle/images/arrowdown.png);
        width: 8px;
        height: 6px;
    }}
    QSpinBox::up-arrow:disabled,
    QSpinBox::down-arrow:disabled {{
        image: none;
    }}
    QProgressBar {{
        border: 2px solid {TK_BORDER};
        background: #ffffff;
        height: 22px;
    }}
    QProgressBar::chunk {{
        background-color: #4f81bd;
        margin: 1px;
    }}
    QLabel {{
        font-size: 14px;
    }}
    """
)
stack = QStackedWidget()

screen1 = HomeScreen(stack)
screen2 = Load(stack)
screen3 = CreateNewScreen(stack)
screen4 = UseScreen(stack)
screen5 = TrainScreen(stack)
screen6 = MakeTrainingInfoScreen(stack)
screen7 = TrainFromVideos(stack)
train_screen = screen6
t_screen = screen5
u_screen = screen4
tfv_screen = screen7

stack.addWidget(screen1)
stack.addWidget(screen2)
stack.addWidget(screen3) 
stack.addWidget(screen4) 
stack.addWidget(screen5)
stack.addWidget(screen6)  
stack.addWidget(screen7)

  # Start with Screen 1
stack.setCurrentIndex(0)
stack.setFixedSize(1000, 800)
stack.show()

sys.exit(app.exec())
