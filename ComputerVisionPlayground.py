from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QStackedWidget, QLineEdit, QPushButton, QLabel, QSpinBox, QGridLayout, QSizePolicy
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
import cv2
import sys
import numpy as np
from CVModel import CVModel

cvm = None
train_screen = None

class HomeScreen(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        layout = QVBoxLayout()
        layout.addWidget(QPushButton("Load And Use", clicked=self.go_to_screen1))
        layout.addWidget(QPushButton("Create New", clicked=self.go_to_screen2))
        layout.addWidget(QPushButton("Load And Train", clicked=self.go_to_screen3))
        self.setLayout(layout)

    def go_to_screen1(self):
        self.stack.setCurrentIndex(1)
    
    def go_to_screen2(self):
        self.stack.setCurrentIndex(2) 
    
    def go_to_screen3(self):
        self.stack.setCurrentIndex(3) 

class LoadAndUseScreen(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.setWindowTitle("Load And Use CV Model")
        layout = QVBoxLayout()
        layout.addWidget(QPushButton("Home", clicked=self.go_to_screen1))
        self.setLayout(layout)

    def go_to_screen1(self):
        self.stack.setCurrentIndex(0) 

class CreateNewScreen(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.setWindowTitle("Create New CV Model")
        layout = QVBoxLayout()
        layout.addWidget(QPushButton("Home", clicked=self.go_to_screen1))
        self.setLayout(layout)

        xy_layout = QGridLayout()

        self.size_spin_box = QSpinBox()
        self.size_spin_box.setMinimum(0)
        self.size_spin_box.setMaximum(100)
        xy_layout.addWidget(QLabel("S x S:"))
        xy_layout.addWidget(self.size_spin_box)

        self.kernel_spin_box = QSpinBox()
        self.kernel_spin_box.setMinimum(1)
        self.kernel_spin_box.setMaximum(7)
        self.kernel_spin_box.setSingleStep(2)
        xy_layout.addWidget(QLabel("Kernel Size:"))
        xy_layout.addWidget(self.kernel_spin_box)

        self.layer_spin_box = QSpinBox()
        self.layer_spin_box.setMinimum(1)
        self.layer_spin_box.setMaximum(50)
        xy_layout.addWidget(QLabel("Layers:"))
        xy_layout.addWidget(self.layer_spin_box)

        self.kpl_spin_box = QSpinBox()
        self.kpl_spin_box.setMinimum(1)
        self.kpl_spin_box.setMaximum(50)
        xy_layout.addWidget(QLabel("Kernels/Layer:"))
        xy_layout.addWidget(self.kpl_spin_box)

        self.outp_spin_box = QSpinBox()
        self.outp_spin_box.setMinimum(1)
        self.outp_spin_box.setMaximum(50)
        xy_layout.addWidget(QLabel("outputs:"))
        xy_layout.addWidget(self.outp_spin_box)

        button = QPushButton("Submit")
        button.clicked.connect(self.createCV)
        layout.addWidget(button)

        self.result_label = QLabel("")

        layout.addLayout(xy_layout)
        layout.addWidget(self.result_label)
        
        self.setLayout(layout)

    def createCV(self):
        global cvm, train_screen
        cvm = CVModel([self.layer_spin_box.value(), self.layer_spin_box.value()], self.outp_spin_box.value(), self.layer_spin_box.value(), self.kpl_spin_box.value(), 3, self.kernel_spin_box.value())
        if train_screen is not None:
            train_screen.refresh_model()
        self.stack.setCurrentIndex(4)

    def go_to_screen1(self):
        self.stack.setCurrentIndex(0)

#inputSize, outputSize, kernelSize, poolerSize, layers, kernelsPerLayer, inputChannels=1

class LoadAndTrainScreen(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.setWindowTitle("Load And Train CV Model")
        layout = QVBoxLayout()
        layout.addWidget(QPushButton("Home", clicked=self.go_to_screen1))
        self.setLayout(layout)

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
        layout.addWidget(QPushButton("Home", clicked=self.go_to_screen1))
        layout.addWidget(QPushButton("Start/Stop Training", clicked=self.train))
        self.setLayout(layout)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        layout.addWidget(self.camera_label)

        xy_layout = QGridLayout()
        self.outp_spin_box = QSpinBox()
        self.outp_spin_box.setMinimum(1)
        self.outp_spin_box.setMaximum(cvm.outputs if cvm!=None else 1)
        xy_layout.addWidget(QLabel("Correct Output:"))
        xy_layout.addWidget(self.outp_spin_box)

        layout.addLayout(xy_layout)

        self.training_label = QLabel("Training: OFF")
        layout.addWidget(self.training_label)

        self.error_label = QLabel("Loss: -")
        layout.addWidget(self.error_label)

        self.learning_rate = 1e-3
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
            if cvm is not None:
                tensor = self.frame_to_tensor(frame_rgb, cvm.inputSize[0])
                cvm.forwardPass(tensor)
                expected = [1.0 if i + 1 == self.outp_spin_box.value() else 0.0 for i in range(cvm.outputs)]
                loss = cvm.backpropigate(expected, self.learning_rate)
                if loss is not None:
                    self.last_loss = loss
                    self.error_label.setText(f"Loss: {loss:.4f}")

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
        self.camera_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
    
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

app = QApplication(sys.argv)
stack = QStackedWidget()

screen1 = HomeScreen(stack)
screen2 = LoadAndUseScreen(stack)
screen3 = CreateNewScreen(stack)
screen4 = LoadAndTrainScreen(stack)
screen5 = TrainScreen(stack)
train_screen = screen5

stack.addWidget(screen1)
stack.addWidget(screen2)
stack.addWidget(screen3) 
stack.addWidget(screen4) 
stack.addWidget(screen5) 

  # Start with Screen 1
stack.setCurrentIndex(0)
stack.setFixedSize(1000, 800)
stack.show()

sys.exit(app.exec())
