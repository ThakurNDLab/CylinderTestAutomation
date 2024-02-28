import sys
import os
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QSlider, QLabel, QFileDialog
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QScreen
import cv2
from moviepy.editor import VideoFileClip

class VideoPlayer(QWidget):
	def __init__(self, video_path, save_path):
		super().__init__()
		self.video_path = video_path
		self.save_path = save_path
		self.initUI()
		self.cap = cv2.VideoCapture(video_path)
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.updateFrame)
		self.isPlaying = False

		self.start_frame = 0
		self.end_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		self.slider.setMaximum(self.end_frame)
		self.slider.valueChanged.connect(self.sliderMoved)

		# Set the window size to 80% of the screen size
		self.setWindowSize()

	def setWindowSize(self):
		screen = QApplication.primaryScreen().geometry()
		self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))

	def initUI(self):
		self.setWindowTitle("Simple Video Trimmer")

		self.layout = QVBoxLayout()
		self.setLayout(self.layout)

		self.videoLabel = QLabel(self)
		self.layout.addWidget(self.videoLabel)

		self.playButton = QPushButton("Play/Pause", self)
		self.playButton.clicked.connect(self.playPauseVideo)
		self.layout.addWidget(self.playButton)

		self.slider = QSlider(Qt.Horizontal, self)
		self.layout.addWidget(self.slider)

		self.startButton = QPushButton("Set Start", self)
		self.startButton.clicked.connect(self.setStart)
		self.layout.addWidget(self.startButton)

		self.endButton = QPushButton("Set End", self)
		self.endButton.clicked.connect(self.setEnd)
		self.layout.addWidget(self.endButton)

		self.trimButton = QPushButton("Trim and Save", self)
		self.trimButton.clicked.connect(self.trimVideo)
		self.layout.addWidget(self.trimButton)

	def playPauseVideo(self):
		if self.isPlaying:
			self.timer.stop()
			self.isPlaying = False
		else:
			self.timer.start(30)
			self.isPlaying = True

	def updateFrame(self):
		ret, frame = self.cap.read()
		if ret:
			self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
			self.slider.setValue(self.current_frame)

			# Resize frame to fit in the window
			frame_width = int(self.width() * 0.8)
			frame_height = int(self.height() * 0.8)
			dim = (frame_width, frame_height)
			frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
			
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
			self.videoLabel.setPixmap(QPixmap.fromImage(image))

	def sliderMoved(self, position):
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)

	def setStart(self):
		self.start_frame = self.slider.value()
		self.startButton.setStyleSheet("background-color: lightgreen")

	def setEnd(self):
		self.end_frame = self.slider.value()
		self.endButton.setStyleSheet("background-color: lightgreen")

	def trimVideo(self):
		self.cap.release()
		with VideoFileClip(self.video_path) as video:
			start_time = self.start_frame / video.fps
			end_time = self.end_frame / video.fps
			trimmed_clip = video.subclip(start_time, end_time)
			trimmed_clip.write_videofile(self.save_path, codec="libx264")
			print(f"Video trimmed and saved to {self.save_path}")

def edit_window(video_path):
	save_path = os.path.join(os.path.dirname(video_path), "edited", os.path.basename(video_path))
	app = QApplication.instance()
	if not app:  # If not, create a new instance
		app = QApplication(sys.argv)
	player = VideoPlayer(video_path, save_path)
	player.resize(640, 480)
	player.show()
	return app.exec()
