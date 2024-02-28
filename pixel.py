import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
							   QLineEdit, QLabel, QFileDialog, QGraphicsView,
							   QGraphicsScene)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen
from PySide6.QtCore import Qt, QPoint, Signal


class ImageWithLine(QGraphicsView):
	def __init__(self):
		super().__init__()
		self.scene = QGraphicsScene(self)
		self.setScene(self.scene)
		self.start = None
		self.end = None
		self.line = None
		self.image_item = None

	def set_image(self, image_path):
		pixmap = QPixmap(image_path)
		self.image_item = self.scene.addPixmap(pixmap)
		self.setSceneRect(self.image_item.boundingRect())

	def mousePressEvent(self, event):
		if event.button() == Qt.LeftButton:
			self.start = self.mapToScene(event.pos())
			self.line = self.scene.addLine(self.start.x(), self.start.y(), self.start.x(), self.start.y(), QPen(Qt.green, 2))

	def mouseMoveEvent(self, event):
		if self.start and self.line:
			end = self.mapToScene(event.pos())
			self.line.setLine(self.start.x(), self.start.y(), end.x(), end.y())

	def mouseReleaseEvent(self, event):
		if event.button() == Qt.LeftButton and self.start:
			self.end = self.mapToScene(event.pos())
			self.line.setLine(self.start.x(), self.start.y(), self.end.x(), self.end.y())
			self.start = None
			self.end = None


class DistanceMeasurementPlugin(QWidget):

	conversion_factor_calculated = Signal(float)

	def __init__(self, image_path=None, callback=None):
		super().__init__()
		self.callback = callback
		self.setLayout(QVBoxLayout())

		# Image view with line drawing capability
		self.imageView = ImageWithLine()
		self.layout().addWidget(self.imageView)

		if image_path:
			self.imageView.set_image(image_path)

		# UI Components
		self.distance_label = QLabel('Enter known distance:')
		self.distance_input = QLineEdit()
		self.calculate_button = QPushButton('Calculate Conversion Factor')
		self.result_label = QLabel('Conversion Factor: Not calculated yet')

		# Add widgets to layout
		self.layout().addWidget(self.distance_label)
		self.layout().addWidget(self.distance_input)
		self.layout().addWidget(self.calculate_button)
		self.layout().addWidget(self.result_label)

		# Connect events
		self.calculate_button.clicked.connect(self.calculate_conversion_factor)

	def calculate_conversion_factor(self):
		if not self.imageView.line:
			self.result_label.setText('Please draw a line.')
			return

		try:
			known_distance = float(self.distance_input.text())
		except ValueError:
			self.result_label.setText('Please enter a valid distance.')
			return

		# Calculate pixel length of the line
		line = self.imageView.line.line()
		pixel_length = np.linalg.norm(np.array([line.x1(), line.y1()]) - np.array([line.x2(), line.y2()]))

		# Calculate conversion factor
		conversion_factor = known_distance / pixel_length
		self.result_label.setText(f'Conversion Factor: {conversion_factor}')

		if self.callback:
			self.callback(conversion_factor)

		self.conversion_factor_calculated.emit(conversion_factor)

def pixel_window(image_path):
	app = QApplication.instance() or QApplication([])
	if not app:  # Create a new instance if it doesn't exist
		app = QApplication(sys.argv)

	widget = DistanceMeasurementPlugin(image_path)
	conversion_factor = None

	def set_conversion_factor(value):
		nonlocal conversion_factor
		conversion_factor = value
		app.quit()  # Exit the app when the value is set

	widget.conversion_factor_calculated.connect(set_conversion_factor)
	widget.resize(800, 600)
	widget.show()

	app.exec()  # This will block until app.quit() is called

	return conversion_factor