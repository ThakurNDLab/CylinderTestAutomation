import os
import glob
import pandas as pd
import imageio
import re
import napari
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QApplication
from PySide6.QtCore import QObject

# Function to sort filenames numerically
def numerical_sort_key(filename):
	"""Extracts numbers from the filename and returns them for sorting."""
	numbers = re.findall(r'\d+', filename)
	return int(numbers[0]) if numbers else 0

class TouchAnnotationPlugin(QWidget):

	def __init__(self, napari_viewer):
		super().__init__()
		self.viewer = napari_viewer
		self.setLayout(QVBoxLayout())

		# Initialize state
		self.current_image_index = 0
		self.touch_status = []
		self.loaded_image_range = (0, 0)
		self.image_chunk_size = 25
		self.loaded_images = {}
		self.currently_loading = False

		# UI Components
		self.image_label = QLabel('No Image Loaded')
		self.left_button = QPushButton('Left')
		self.right_button = QPushButton('Right')
		self.save_button = QPushButton('Save Annotations')
		self.prev_button = QPushButton('Previous')
		self.next_button = QPushButton('Next')

		# Add widgets to layout
		self.layout().addWidget(self.image_label)
		self.layout().addWidget(self.left_button)
		self.layout().addWidget(self.right_button)
		self.layout().addWidget(self.save_button)
		self.layout().addWidget(self.prev_button)
		self.layout().addWidget(self.next_button)

		# Connect events
		self.left_button.clicked.connect(self.annotate_left)
		self.right_button.clicked.connect(self.annotate_right)
		self.save_button.clicked.connect(self.save_annotations)
		self.prev_button.clicked.connect(self.go_to_previous_image)
		self.next_button.clicked.connect(self.go_to_next_image)

	def load_images(self, image_paths, save_path):
		# Sort the image paths and load only metadata initially
		self.sorted_image_paths = sorted(image_paths)
		self.csv_path = save_path
		self.load_touch_status()
		self.load_image(self.sorted_image_paths[0])  # Load the first image

	def load_touch_status(self):
		# Load or initialize touch status from CSV
		if os.path.exists(self.csv_path):
			self.touch_status = pd.read_csv(self.csv_path).to_dict('records')
		else:
			self.touch_status = [{'left': False, 'right': False} for _ in self.sorted_image_paths]
			self.save_annotations()  # Save initial annotations

	def load_image(self, path):
	    if path in self.loaded_images:
	        return
	    self.unload_distant_images()
	    image = imageio.imread(path)
	    layer = self.viewer.add_image(image, name=os.path.basename(path))
	    self.loaded_images[path] = layer

	def unload_distant_images(self):
	    current_index = self.sorted_image_paths.index(self.sorted_image_paths[self.current_image_index])
	    unload_indices = []

	    for path, layer in self.loaded_images.items():
	        index = self.sorted_image_paths.index(path)
	        if abs(index - current_index) > self.image_chunk_size // 2:
	            unload_indices.append(path)

	    for index in unload_indices:
	        self.viewer.layers.remove(self.loaded_images[index])
	        del self.loaded_images[index]


	def update_image(self, index):
	    self.current_image_index = index
	    path = self.sorted_image_paths[index]

	    # Load the image if it's not already loaded
	    if path not in self.loaded_images:
	        self.load_image(path)

	    # Hide all layers and then show only the current one
	    for layer_path, layer in self.loaded_images.items():
	        layer.visible = False
	    self.loaded_images[path].visible = True

	    # Update UI components
	    self.image_label.setText(os.path.basename(path))
	    self.update_buttons()


	def synchronize_layers(self):
		# Ensure all layers are in sync with the loaded_images
		for path, layer in self.loaded_images.items():
			if layer.name not in self.viewer.layers:
				self.viewer.add_layer(layer)

	def go_to_previous_image(self):
		if self.current_image_index > 0:
			self.update_image(self.current_image_index - 1)
			self.synchronize_layers()

	def go_to_next_image(self):
		if self.current_image_index < len(self.sorted_image_paths) - 1:
			self.update_image(self.current_image_index + 1)
			self.synchronize_layers()

	def onImagesLoaded(self):
		self.update_image(self.current_image_index)


	def annotate_left(self):
		self.toggle_annotation('left')

	def annotate_right(self):
		self.toggle_annotation('right')

	def toggle_annotation(self, side):
		index = self.current_image_index
		self.touch_status[index][side] = not self.touch_status[index][side]
		self.update_buttons()

	def update_buttons(self):
		index = self.current_image_index
		self.left_button.setStyleSheet('background-color: {}'.format('green' if self.touch_status[index]['left'] else 'red'))
		self.right_button.setStyleSheet('background-color: {}'.format('green' if self.touch_status[index]['right'] else 'red'))

	def save_annotations(self):
		try:
			df = pd.DataFrame(self.touch_status).replace({True: 1, False: 0})
			df.to_csv(self.csv_path, index=False)
			napari.utils.notifications.show_info("Annotations updated successfully.")
		except Exception as e:
			napari.utils.notifications.show_error(f'An error occurred: {e}')

	def closeEvent(self, event):
		self.save_annotations()  # Ensure that changes are saved
		super().closeEvent(event)


def cl_anno(image_dir):
	save_dir = os.path.join(image_dir, "touch.csv")
	image_paths = [os.path.abspath(file) for file in glob.glob(os.path.join(image_dir, '*.png'))]

	app = QApplication.instance() or QApplication([])
	viewer = napari.Viewer()
	touch_annotation_plugin = TouchAnnotationPlugin(viewer)
	viewer.window.add_dock_widget(touch_annotation_plugin)
	touch_annotation_plugin.load_images(image_paths, save_dir)
	app.exec_()

	viewer.close()
	del viewer
	del app
