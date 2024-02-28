import os
import glob
import pandas as pd
import imageio
import re
import napari
import numpy
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QApplication, QListWidget
from PySide6.QtCore import QObject
from scripts.predict import count

class PredictionReviewPlugin(QWidget):

	def __init__(self, napari_viewer):
		super().__init__()
		self.viewer = napari_viewer
		self.setLayout(QVBoxLayout())
		self.annotations_df = None
		self.current_image_index = 0
		self.loaded_images = {}
		self.image_chunk_size = 25
		self.limit = 0
		self.current_chunk_start = 0
		self.current_chunk_end = 0

		# UI Components
		self.image_list = QLabel('Total Frames:'+ str(self.limit))
		self.image_label = QLabel('No Image Loaded')
		self.left_button = QPushButton('Left')
		self.right_button = QPushButton('Right')
		self.update_button = QPushButton('Update Annotations')
		self.save_exit_button = QPushButton('Save and Exit')
		self.next_button = QPushButton('Next Frame')
		self.prev_button = QPushButton('Previous Frame')
		self.junction_button = QPushButton('Go to Junction Frame')

		# Add widgets to layout
		self.layout().addWidget(self.image_list)
		self.layout().addWidget(self.image_label)
		self.layout().addWidget(self.left_button)
		self.layout().addWidget(self.right_button)
		self.layout().addWidget(self.update_button)
		self.layout().addWidget(self.save_exit_button)
		self.layout().addWidget(self.next_button)
		self.layout().addWidget(self.prev_button)
		self.layout().addWidget(self.junction_button)

		# Connect events
		self.update_button.clicked.connect(self.update_annotations)
		self.save_exit_button.clicked.connect(self.save_and_exit)
		self.next_button.clicked.connect(lambda: self.navigate_frames('next'))
		self.prev_button.clicked.connect(lambda: self.navigate_frames('prev'))
		self.junction_button.clicked.connect(lambda: self.navigate_frames('junction'))

		# Connect annotation buttons
		self.left_button.clicked.connect(self.annotate_left)
		self.right_button.clicked.connect(self.annotate_right)

		self.csv_path = None
		self.image_dir = None

	def set_csv_path(self, csv_path):
		self.csv_path = csv_path
		if self.csv_path and self.image_dir:
			self.load_annotations()
			self.load_images()

	def set_image_dir(self, image_dir):
		self.image_dir = image_dir
		if self.csv_path and self.image_dir:
			self.load_annotations()
			self.load_images()

	def load_annotations(self):
		self.annotations_df = pd.read_csv(self.csv_path)
		l,r,_,_,_ = count((self.annotations_df).to_numpy())
		self.limit = (pd.concat([l.max(), r.max()]).max())+1
		self.image_list.setText('Total Frames:'+ str(self.limit))


	def load_images(self):
		image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.png')), key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
		self.sorted_image_paths = image_paths[6:]
		self.load_image_chunk(0)

	def load_image_chunk(self, start_index):
		self.current_chunk_start = start_index
		end_index = min(start_index + self.image_chunk_size, len(self.sorted_image_paths))
		self.current_chunk_end = end_index
		to_unload = [name for name in self.loaded_images if int(re.findall(r'\d+', name)[0]) - 6 < start_index or int(re.findall(r'\d+', name)[0]) - 6 >= end_index]
		
		for name in to_unload:
			self.viewer.layers.remove(self.loaded_images[name])
			del self.loaded_images[name]

		for i in range(start_index, end_index):
			path = self.sorted_image_paths[i]
			image_name = os.path.basename(path)
			if image_name not in self.loaded_images:
				image = imageio.imread(path)
				layer = self.viewer.add_image(image, name=image_name)
				self.loaded_images[image_name] = layer

	def annotate_left(self):
		# Toggle the annotation status for the left touch
		index = self.current_image_index
		self.annotations_df.at[index, 'left'] = not self.annotations_df.at[index, 'left']
		self.update_buttons()

	def annotate_right(self):
		# Toggle the annotation status for the right touch
		index = self.current_image_index
		self.annotations_df.at[index, 'right'] = not self.annotations_df.at[index, 'right']
		self.update_buttons()

	def navigate_frames(self, direction):
		new_index = self.current_image_index
		if direction == 'next':
			new_index += 1
		elif direction == 'prev':
			new_index -= 1
		elif direction == 'junction':
			new_index = self.find_next_junction_frame(new_index)

		# Check if new index is outside the range of loaded images
		if (new_index < 0 or new_index >= len(self.sorted_image_paths) or \
		   new_index < self.current_chunk_start or new_index >= self.current_chunk_end) and new_index<=self.limit:
			new_chunk_start = max(0, new_index - self.image_chunk_size // 2)
			self.load_image_chunk(new_chunk_start)

		self.current_image_index = new_index
		self.update_image(self.current_image_index)

	def find_next_junction_frame(self, current_index):
		for i in range(current_index + 1, len(self.sorted_image_paths)):
				# Check for change in touch status between consecutive frames
				if i > 0 and (self.annotations_df.iloc[i][0] != self.annotations_df.iloc[i-1][0] or
							  self.annotations_df.iloc[i][1] != self.annotations_df.iloc[i-1][1]):
					return i

	def update_image(self, index):
		self.current_image_index = index
		image_name = os.path.basename(self.sorted_image_paths[index])
		for layer_path, layer in self.loaded_images.items():
			layer.visible = False
		self.loaded_images[image_name].visible = True
		# Update UI components
		self.image_label.setText(image_name)
		self.update_buttons()

	def update_annotations(self):
		# This assumes that the user edits are directly reflected in self.annotations_df
		self.annotations_df = self.annotations_df.replace({True: 1, False: 0})
		self.annotations_df.to_csv(self.csv_path, index=False)
		l,r,_,_,_ = count((self.annotations_df).to_numpy())
		self.limit = (pd.concat([l.max(), r.max()]).max())+1
		self.image_list.setText('Total Frames:'+ str(self.limit))

	def update_buttons(self):
		index = self.current_image_index
		# Update button colors based on annotation status
		left_status = self.annotations_df.at[index, 'left']
		right_status = self.annotations_df.at[index, 'right']
		self.left_button.setStyleSheet('background-color: {}'.format('green' if left_status else 'red'))
		self.right_button.setStyleSheet('background-color: {}'.format('green' if right_status else 'red'))

	def save_and_exit(self):
		self.update_annotations()  # Save any changes
		self.viewer.window.close()  # Close Napari viewer

	def closeEvent(self, event):
		# Handle the close event for the widget
		self.save_and_exit()  # Ensure that changes are saved and resources are released
		super().closeEvent(event)


def review_window(image_dir, csv_path):

	app = QApplication.instance() or QApplication([])
	viewer = None

	try:
		viewer = napari.Viewer()
		prediction_review_plugin = PredictionReviewPlugin(viewer)
		viewer.window.add_dock_widget(prediction_review_plugin)
		prediction_review_plugin.set_csv_path(csv_path)
		prediction_review_plugin.set_image_dir(image_dir)

		app.exec_()

	finally:
		if viewer is not None:
			viewer.close()
			del viewer