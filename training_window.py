import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import os
import subprocess
from cl_anno import cl_anno
import glob
from ruamel.yaml import YAML
import shutil
import pandas as pd
import random
import sys
from scripts.train import *
import glob
import shutil
import re
import threading
from sklearn.cluster import MiniBatchKMeans
import cv2

root = tk.Tk()
root.withdraw()  # Hide the main window

processes = []

# Global settings
font_settings = ("Helvetica", 12, "bold")
bg_color = "#f0f0f0"
button_color = "#4a7abc"
button_font = ("Helvetica", 12)
# Style configuration
style = ttk.Style(root)
style.configure("W.TButton", font=button_font, background=button_color)
button_style = "W.TButton"

def numerical_sort_key(filename):
	"""Extracts numbers from the filename and returns them for sorting."""
	numbers = re.findall(r'\d+', filename)
	return int(numbers[0]) if numbers else 0

def get_base_path():
	if getattr(sys, 'frozen', False):
		# If the application is run as a bundle (e.g., packaged with PyInstaller),
		# the sys._MEIPASS attribute contains the path to the bundle folder.
		return sys._MEIPASS
	else:
		# If it's not packaged, return the directory of this script file
		return os.path.dirname(os.path.abspath(__file__))

def get_env_vars(env_name):
	activate_script = os.path.join(env_name, 'bin', 'activate')
	command = f"source {activate_script} && env"
	proc = subprocess.Popen(['/bin/bash', '-c', command], stdout=subprocess.PIPE)
	env_vars = dict(line.split("=", 1) for line in proc.stdout.decode().splitlines())
	return env_vars

def run_dlc_command(command):
	env_vars = get_env_vars("dlc_env")
	# Update LD_LIBRARY_PATH to include the Conda environment's lib directory
	env_vars['LD_LIBRARY_PATH'] = os.path.join(env_vars['CONDA_PREFIX'], 'lib') + ':' + env_vars.get('LD_LIBRARY_PATH', '')
	python_path = os.path.join(env_vars['CONDA_PREFIX'], 'bin', 'python')
	full_command = f"{python_path} -c 'import deeplabcut; {command}'"
	process = subprocess.run(['/bin/bash', '-c', full_command], check=True, env=env_vars)
	return process

def run_command_in_env(env_name, command):
	activate_script = os.path.join(get_base_path(), env_name, 'bin', 'activate')
	try:
		full_command = f"source {activate_script} && {command}"
		process = subprocess.run(['/bin/bash', '-c', full_command], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		output = process.stdout.decode().strip()
		print("Command Succesfull", f"{command}")
		# Handle the output as required
		return output
	except Exception as e:
		print("Unexpected Error", f"An unexpected error occurred, while running '{command}': '{e}")

def run_napari():
	# Activates a virtual environment and runs a specified command within that environment
	activate_script = os.path.join(get_base_path(), "dlc_env", 'bin', 'activate')
	full_command = f"source {activate_script} && napari"
	try:
		process = subprocess.run(['/bin/bash', '-c', full_command], check=True)
	except:
		print("Error", f"Error running command '{command}' in environment {env_name}")
		messagebox.showerror("Error", f"Error running command '{command}' in environment '{env_name}''")
		return None


def get_video_duration(video_path):
	try:
		command = f"ffprobe -v error -show_entries format=duration {video_path}"

		result = run_command_in_env("cyl_env", command)
		return float(result)
	except Exception as e:
		print(f"Error getting duration of video '{video_path}'': '{e}'")
		return 0

def trim_videos_in_folder(folder_path):
	all_video_paths = glob.glob(os.path.join(folder_path, '*.avi'))
	video_paths = [vp for vp in all_video_paths if not vp.endswith('_trimmed.avi')]

	if not video_paths:
		print("Could not find video file. Have you copied the videos to the videos folder?")
		messagebox.showerror("Error", "No untrimmed Video files found in the specified folder.", parent=root)
		return []

	trimmed_video_paths = []
	for video_path in video_paths:
		base, ext = os.path.splitext(video_path)
		output_file = f"{base}_trimmed{ext}"

		if os.path.exists(output_file):
			trimmed_video_paths.append(output_file)
			continue

		video_duration = get_video_duration(video_path)
		if video_duration < 180:  # Less than 3 minutes
			print("Video duration too short:" f"{video_path}")
			messagebox.showerror(f"Video '{video_path}' is too short for trimming.")
			continue

		start_time = '00:00:30'  # 30 seconds
		end_time = '00:02:30'  # 2 minutes and 30 seconds
		if video_duration < 150:  # Less than 2 minutes and 30 seconds
			end_time = str(datetime.timedelta(seconds=int(video_duration - 30)))

		ffmpeg_command = f"ffmpeg -i {video_path} -ss {start_time} -to {end_time} -c:v libx264 -c:a aac {output_fil}"
		try:
			run_command_in_env("cyl_env", ffmpeg_command)
			trimmed_video_paths.append(output_file)
		except Exception as e:
			print("Trimming Error", f"Failed to trim video '{video_path}': '{e}'")
			messagebox.showerror("Trimming Error", f"Failed to trim video '{video_path}': '{e}'", parent=root)

	return trimmed_video_paths


def update_config(original_config_path, new_config_path, project_name, new_video_paths):
	if not os.path.exists(original_config_path):
		print("Error", f"Original configuration file not found: '{original_config_path}'")
		messagebox.showerror("Error", f"Original configuration file not found: '{original_config_path}'", parent=root)
		return False

	yaml = YAML()
	yaml.preserve_quotes = True
	
	try:
		# Read the original configuration file
		with open(original_config_path, 'r') as file:
			config_data = yaml.load(file)

		with open(new_config_path, 'r') as file:
			new_config_data = yaml.load(file)

		# Update the necessary fields
		config_data['Task'] = project_name
		config_data['scorer'] = 'admin_user'  # Assuming you want to set this to a fixed value
		config_data['date'] = new_config_data['date']
		config_data['project_path'] = new_config_path
		config_data['video_sets'] = new_config_data['video_sets']

		# Write the updated configuration to the new file
		with open(new_config_path, 'w') as file:
			yaml.dump(config_data, file)

		print("Success", f"Configuration updated and saved to '{new_config_path}'")
		messagebox.showinfo("Success", f"Configuration updated and saved to '{new_config_path}'", parent=root)
		return True

	except Exception as e:
		print("Error", f"Error reading or writing YAML configuration: '{e}'")
		messagebox.showerror("Error", f"Error reading or writing YAML configuration: '{e}'", parent=root)
		return False


def find_file(directory, filename):
	for root, dirs, files in os.walk(directory):
		if filename in files:
			return os.path.join(root, filename)
	return None


def replace_init_weights(file_path, new_value):
	if not os.path.exists(file_path):
		print("File Not Found", f"File not found: '{file_path}'")
		messagebox.showerror("File Not Found", f"File not found: '{file_path}'", parent=root)
		return False

	yaml = YAML()
	yaml.preserve_quotes = True

	try:
		with open(file_path, 'r') as file:
			pose_config = yaml.load(file)

			pose_config['init_weights'] = os.path.join(os.path.dirname(file_path), os.path.basename(new_value))

			with open(file_path, 'w') as file:
				yaml.dump(pose_config, file)

			
			src1 = os.path.join(os.path.splitext(new_value)[0] + ".data-00000-of-00001")
			srci = os.path.join(os.path.splitext(new_value)[0] + ".index")
			srcm = os.path.join(os.path.splitext(new_value)[0] + ".meta")

			dst1 = os.path.join(os.path.dirname(file_path), os.path.splitext(os.path.basename(new_value))[0] + ".data-00000-of-00001")
			dsti = os.path.join(os.path.dirname(file_path), os.path.splitext(os.path.basename(new_value))[0] + ".index")
			dstm = os.path.join(os.path.dirname(file_path), os.path.splitext(os.path.basename(new_value))[0] + ".meta")

			shutil.copyfile(src1, dst1)
			shutil.copyfile(srci, dsti)
			shutil.copyfile(srcm, dstm)

			print("Success", "Initialization weights updated successfully.")
			messagebox.showinfo("Success", "Initialization weights updated successfully.", parent=root)
			return True

	except IOError as e:
		print("IO Error", f"File IO error: '{e}', when initializing from fixed point")
		messagebox.showerror("IO Error", f"File IO error: '{e}'", parent=root)
		return False
	except Exception as e:
		print("Unexpected Error", f"An unexpected error occurred: '{e}', when initializing from fixed point")
		messagebox.showerror("Unexpected Error", f"An unexpected error occurred: '{e}'", parent=root)
		return False



def extract_frames(video_path, destination_directory):
	if not os.path.exists(video_path):
		print("Video File not found.")
		messagebox.showerror("Video File Not Found", f"Video file not found: '{video_path}'", parent=root)
		return False

	video_name = os.path.splitext(os.path.basename(video_path))[0]
	frame_directory = os.path.join(destination_directory, video_name)
	os.makedirs(frame_directory, exist_ok=True)

	ffmpeg_command = f"ffmpeg -i {video_path} -f image2 {os.path.join(frame_directory, '%04d.png')}"

	try:
		run_command_in_env("cyl_env", ffmpeg_command)
		print("Frame Extraction Successfull:", f"{video_path}")
		return True
	except Exception as e:
		print("Unexpected Error", f"An unexpected error ocurred: '{e.stderr.decode()}', for '{video_path}'")
		messagebox.showerror("Unexpected Error", f"An unexpected error occurred: '{e}'", parent=root)
		return False


def select_random_frames(frame_folder, num_frames=20):
	if not os.path.exists(frame_folder):
		print("Folder Not Found", f"Frame folder not found: '{frame_folder}'")
		messagebox.showerror("Folder Not Found", f"Frame folder not found: '{frame_folder}'", parent=root)
		return []

	frames = [f for f in os.listdir(frame_folder) if f.endswith('.png')]
	if not frames:
		print("Random Selection Unsuccessful", f"No frame images found in the specified folder.")
		messagebox.showerror("No Frames Found", f"No frame images found in the specified folder.", parent=root)
		return []

	# Sort the frames to ensure they are in order
	frames = sorted(frames, key=numerical_sort_key)

	# Exclude the first 40 and last 40 frames
	frames = frames[40:-40]

	# Check if there are enough frames left to select from
	if len(frames) < num_frames:
		print("Not Enough Frames", f"Not enough frames to select {num_frames} frames with the specified constraints.")
		messagebox.showerror("Not Enough Frames", f"Not enough frames to select {num_frames} frames with the specified constraints.", parent=root)
		return []

	selected_frames = []
	while len(selected_frames) < num_frames:
		# Randomly select a frame
		frame = random.choice(frames)

		# Check if it's sufficiently far from previously selected frames
		if all(abs(frames.index(frame) - frames.index(prev_frame)) >= 80 for prev_frame in selected_frames):
			selected_frames.append(frame)

	return selected_frames


def Kmeans_frames(frame_folder, num_frames=10):
	frame_paths = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.png')]
	frame_paths = sorted(frame_paths, key=numerical_sort_key)[40:-40]  # Exclude first and last 40 frames

	kmeans = MiniBatchKMeans(n_clusters=num_frames, tol=1e-3, batch_size=30, max_iter=75)
	
	for i in range(0, len(frame_paths), 30):
		batch_paths = frame_paths[i:i+30]
		batch_images = [cv2.resize(cv2.imread(p), (128, 128), interpolation=cv2.INTER_NEAREST) for p in batch_paths]
		batch_data = np.array(batch_images, dtype=np.float32).reshape(len(batch_images), -1)
		kmeans.partial_fit(batch_data)

	frames_selected = []
	for cluster_id in range(num_frames):
		cluster_ids = np.where(kmeans.labels_ == cluster_id)[0]
		if len(cluster_ids) > 0:
			selected_index = cluster_ids[np.random.randint(len(cluster_ids))]
			frames_selected.append(frame_paths[selected_index])

	return frames_selected

def process_video_frames(model_dir):
	classifier_dir = os.path.join(model_dir, "classifier")
	if not os.path.exists(classifier_dir):
		print(f"Classifier directory not found: '{classifier_dir}'", f"Unable to Generate Training Dataset")
		messagebox.showerror(f"Classifier directory not found: '{classifier_dir}'")
		return None

	training_folder = os.path.join(classifier_dir, "training")
	os.makedirs(training_folder, exist_ok=True)
	anno = os.listdir(training_folder)
	if anno:
		letters = [ann[0] for ann in anno]
		folder_letter = chr(ord(sorted(letters)[-1]) + 1)
	else:
		folder_letter = 'a'

	set_number = 1

	for folder in sorted(os.listdir(classifier_dir)):
		folder_path = os.path.join(classifier_dir, folder)
		if os.path.isdir(folder_path) and folder != "training" and folder != "model" and folder != "graphs":
			video_name = folder
			csv_file = os.path.join(classifier_dir, video_name)
			csv_paths = glob.glob(os.path.join(csv_file + "DLC" + "*.csv"))
			if not csv_paths:
				print(f"No CSV file found for video: '{video_name}'", f"Unable to generate images for training")
				messagebox.showerror(f"No CSV file found for video: '{video_name}'")
				continue
			csv_path = csv_paths[0]

			try:
				pose_data = pd.read_csv(csv_path, skiprows=2)
				selected_frames = Kmeans_frames(folder_path)

				for frame in sorted(selected_frames, key=numerical_sort_key):
					frame_number = int(os.path.splitext(frame)[0])
					
					# Calculate the range of frames to extract
					start_frame = max(1, frame_number - 40)
					end_frame = min(len(pose_data), frame_number + 39)

					# Create subfolder for each set
					set_folder_name = f"{folder_letter}{set_number}"
					set_folder_path = os.path.join(training_folder, set_folder_name)
					os.makedirs(set_folder_path, exist_ok=True)

					# Extracting pose data for the specified frames
					subset_pose_data = pose_data.iloc[start_frame-1:end_frame]
					subset_pose_data.reset_index(drop=True, inplace=True)

					# Save the subset pose data in the subfolder
					subset_pose_data.to_csv(os.path.join(set_folder_path, 'pose_data.csv'), index=False)

					# Save frames in the subfolder
					for idx, sub_frame_number in enumerate(range(start_frame, end_frame + 1), start_frame):
						src_frame_path = os.path.join(folder_path, f"{sub_frame_number:04d}.png")
						dest_frame_path = os.path.join(set_folder_path, f"{idx:04d}.png")  # Use original frame number for naming
						shutil.copy(src_frame_path, dest_frame_path)

					set_number += 1

			except Exception as e:
				print(f"Error processing CSV file '{csv_file}': '{e}'", f"During selection of training images.")
				messagebox.showerror(f"Error processing CSV file '{csv_file}': '{e}'")

			folder_letter = chr(ord(folder_letter) + 1) if set_number > 3 else folder_letter
			set_number = 1 if set_number > 3 else set_number

	print("Successfully created Classifier training pose data")
	messagebox.showinfo("Success", 'Pose data processed and saved.', parent=root)


def check_extracted_frames(model_dir, video_paths, required_count=29):
	for video_path in video_paths:
		video_name = os.path.splitext(os.path.basename(video_path))[0]
		pattern = os.path.join(model_dir, "dlc", "**", "labeled-data", video_name)
		frame_folders = glob.glob(pattern, recursive=True)
		
		if not frame_folders:  # Check if the list is empty
			return False  # No folder found for this video

		# Check each matching folder
		for folder in frame_folders:
			frame_files = glob.glob(os.path.join(folder, '*.png'))
			if len(frame_files) < required_count:
				return False  # Not enough frames in one of the folders
			
	return True  # All videos have enough frames

def select_subdirectory_for_annotation(parent_window, model_dir):
	frame_directory = os.path.join(model_dir, "classifier", "training")
	subdirs = sorted([d for d in os.listdir(frame_directory) if os.path.isdir(os.path.join(frame_directory, d))])

	if not subdirs:
		messagebox.showinfo("No Subdirectories Found", "There are no images to annotate.", parent=parent_window)
		return

	# Create a new selection window
	selection_win = tk.Toplevel(parent_window)
	selection_win.title("Select a Subdirectory")

	# Styling
	modern_font = ("Arial", 12)
	selection_win.configure(bg=bg_color)

	# Label
	tk.Label(selection_win, text="Select a subdirectory to annotate:", bg=bg_color, font=modern_font).pack(padx=10, pady=(10, 5))

	# Listbox to display subdirectories
	lb = tk.Listbox(selection_win, font=modern_font, bg="#ffffff", selectbackground="#4a7abc", selectmode=tk.SINGLE)
	for subdir in subdirs:
		lb.insert(tk.END, subdir)
	lb.pack(padx=10, pady=5)

	def on_subdir_select():
		if lb.curselection():
			selected_subdir = lb.get(lb.curselection())
			subdir_path = os.path.join(frame_directory, selected_subdir)
			cl_anno(subdir_path)
			selection_win.destroy()
		else:
			messagebox.showinfo("Selection Required", "Please select a subdirectory to proceed.", parent=selection_win)

	# Button to confirm selection
	select_button = ttk.Button(selection_win, text="Annotate Selected", style=button_style, command=on_subdir_select)
	select_button.pack(pady=(5, 10))

	selection_win.transient(parent_window)  # Set to be on top of the main window
	selection_win.grab_set()  # Ensure all interactions are with this window


def training_window(window, model_dir, name):
	training_win = tk.Toplevel(root)
	training_win.title("Training Window")

	status_label = tk.Label(training_win, text='Ready', font=font_settings, bg=bg_color)
	status_label.grid(row=0, column=0, columnspan=2, sticky='w')

	tk.Label(training_win, text="DLC and Classifier Training", font=("Helvetica", 16), bg=bg_color).grid(row=1, column=0, columnspan=2, sticky='w')

	# Buttons with command bindings
	buttons = {
		"Create DLC Project": lambda: handle_event("Create DLC Project", model_dir, name, status_label, training_win),
		"Annotate Frames - DLC": lambda: handle_event("Annotate Frames - DLC", model_dir, name, status_label, training_win),
		"Create DLC Dataset": lambda: handle_event("Create DLC Dataset", model_dir, name, status_label, training_win),
		"Train DLC": lambda: handle_event("Train DLC", model_dir, name, status_label, training_win),
		"Generate Classifier Frame": lambda: handle_event("Generate Classifier Frames", model_dir, name, status_label, training_win),
		"Annotate Frames - Classifier": lambda: handle_event("Annotate Frames - Classifier", model_dir, name, status_label, training_win),
		"Train Classifier": lambda: handle_event("Train Classifier", model_dir, name, status_label, training_win)
	}

	for idx, (text, command) in enumerate(buttons.items()):
		ttk.Button(training_win, text=text, style=button_style, command=command).grid(row=idx + 2, column=0, padx=10, pady=5, sticky='ew')
	
	def on_close():
		training_win.quit()
		training_win.destroy()
		window.deiconify()

	training_win.protocol("WM_DELETE_WINDOW", lambda: on_close())

	training_win.mainloop()



def handle_event(event, model_dir, name, status_label, training_win):
	# Disable all buttons
	for widget in training_win.winfo_children():
		if isinstance(widget, ttk.Button):
			widget.configure(state='disabled')
	
	status_label.configure(text='Status: Busy')

	if event == "Create DLC Project":
		folder_path = os.path.join(model_dir, "videos")
		if os.listdir(folder_path):
			try:
				trimmed_videos = trim_videos_in_folder(folder_path)
				print("Video trimming complete. Proceeding to DLC config creation")
				messagebox.showinfo("Video trimming complete. Proceeding to DLC config creation")
			except Exception as e:
				print(f'Error during video trimming: {e}')
				messagebox.showerror(f'Error during video trimming: {e}')

			if not find_file(os.path.join(model_dir, "dlc"), "config.yaml"):
				dlc_dir = os.path.join(model_dir, "dlc")
				trimmed_videos_str = ', '.join(f"\"{video}\"" for video in trimmed_videos)

				dlc_command = (
					f"deeplabcut.create_new_project(\"{name}\", \"admin_user\", [{trimmed_videos_str}], "
					f"working_directory=\"{dlc_dir}\", copy_videos=False, multianimal=False)"
				)

				try:
					run_dlc_command('dlc_env', dlc_command)
				except Exception as e:
					print("Error Creating DLC Project")
					messagebox.showerror(f'Error creating DLC project: {e}')

			config_path = find_file(os.path.join(model_dir, "dlc"), 'config.yaml')
			source_config = find_file(os.path.join(get_base_path(), "dlc"), "config.yaml")
			if config_path and source_config:
				value = update_config(source_config, config_path, name, trimmed_videos)
				if value:
					print("DLC project created!")
					messagebox.showinfo("DLC project created!")
				else:
					print("Error Creating Config File.")
					messagebox.showerror("Error creating config file")
			else:
				print("Error finding DLC Config File.")
				messagebox.showerror("Error finding DLC config files.")
		else:
			print('No Videos in the directory. Please add training videos before continuing.')
			messagebox.showerror('No Videos in the directory. Please add training videos before continuing.')


	elif event == "Annotate Frames - DLC":
		config_path = find_file(os.path.join(model_dir, "dlc"), 'config.yaml')
		if not config_path:
			print("DLC config file missing. Please create DLC project first.")
			messagebox.showerror("DLC config file missing. Please create DLC project first.")

		video_paths = [vp for vp in os.listdir(os.path.join(model_dir, "videos")) if vp.endswith('_trimmed.avi')]
		
		if not check_extracted_frames(model_dir, video_paths):
			extract = f"deeplabcut.extract_frames(\"{config_path}\", mode=\"automatic\", algo=\"kmeans\", userfeedback=False, crop=False)"
			result = run_dlc_command(extract)
			if result:
				print("Frame extraction successfull")
				messagebox.showinfo("Frame extraction successfull")
			else:
				print("Unable to extract frames")
				messagebox.showerror("Unable to extract frames")
		else:
			print("Frames already extracted. Proceeding to annotation.")
			messagebox.showinfo("Frames already extracted. Proceeding to annotation.")

		# Proceed with annotation
		label = f"napari"
		run_napari()


	elif event == "Create DLC Dataset":
		config_path = find_file(os.path.join(model_dir, "dlc"), 'config.yaml')
		if not config_path:
			print("DLC config file missing. Please create DLC project first.")
			messagebox.showerror("DLC config file missing. Please create DLC project first.")

		try:
			command = f"deeplabcut.create_training_dataset(\"{config_path}\", net_type=\"resnet_152\", augmenter_type=\"imgaug\")"
			run_dlc_command(command)
			print("DLC training dataset created successfully.")
			messagebox.showinfo("DLC training dataset created successfully.")
		except Exception as e:
			print(f'Error creating DLC dataset: {e}')
			messagebox.showerror(f'Error creating DLC dataset: {e}')


	elif event == "Train DLC":
		config_path = find_file(os.path.join(model_dir, "dlc"), 'config.yaml')
		if not config_path:
			print("DLC config file missing. Please create DLC project and dataset first.")
			messagebox.showerror("DLC config file missing. Please create DLC project and dataset first.")

		else:
			snap = find_file(os.path.join(get_base_path(), "dlc"), "snapshot-200000.data-00000-of-00001")
			pose = find_file(os.path.join(model_dir, "dlc"), 'pose_cfg.yaml')
			if not (snap and pose):
				print("Pose config and/or snapshot file(s) missing")
				messagebox.showerror("Required files for training not found.")
			else:
				if replace_init_weights(pose, snap):
					try:
						messagebox.showinfo("DLC training initiated.")
						command = f"deeplabcut.train_network(\"{config_path}\", shuffle=1, trainingsetindex=0, gputouse=0, max_snapshots_to_keep=5, autotune=False, displayiters=100, saveiters=15000, maxiters=75000, allow_growth=True)"
						run_dlc_command(command)
						print((f'DLC training complete'))
						messagebox.showinfo(f'DLC training complete')
					except Exception as e:
						print("Error initiating DLC training")
						messagebox.showerror(f'Error initiating DLC training: {e}')
				else:
					print("Failed to update initialization weights")
					messagebox.showerror("Failed to update initial weights.")
			

	elif event == "Generate Classifier Frames":
		config_path = find_file(os.path.join(model_dir, "dlc"), 'config.yaml')
		videos = glob.glob(os.path.join(model_dir, "videos", '*_trimmed.avi'))
		if not config_path or not videos:
			print("DLC config file and/or trimmed video(s) missing")
			messagebox.showerror("DLC config file or trimmed videos are missing.")

		dest = os.path.join(model_dir, "classifier")
		try:
			for video in videos:
				name = os.path.splitext(os.path.basename(video))[0]
				csv_file = os.path.join(dest, name)
				csv_path = glob.glob(os.path.join(csv_file + "DLC" + "*.csv"))
				if not csv_path:
					command = f"deeplabcut.analyze_videos(\"{config_path}\", [\"{video}\"], videotype=\"avi\", shuffle=1, trainingsetindex=0, gputouse=0, save_as_csv=True, destfolder=\"{dest}\")"
					run_dlc_command(command)
					print("Pose file generated:" f'{video}')
				if not os.path.isdir(os.path.join(dest, os.path.splitext(os.path.basename(video))[0])):
					extract_frames(video, dest)
					print("Frames extracted:" f'{video}')

			process_video_frames(model_dir)
			print("Classifier frames generated and processed successfully.")
			messagebox.showinfo("Classifier frames generated and processed successfully.")
		except Exception as e:
			print(f"Error generating classifier frames: '{e}'")
			messagebox.showerror(f"Error generating classifier frames: '{e}'")

	elif event == "Annotate Frames - Classifier":
		select_subdirectory_for_annotation(training_win, model_dir)

	elif event == "Train Classifier":
		snap = find_file(os.path.join(get_base_path(), "classifier"), "snapshot.pt")
		if not snap:
			print("Snapshot file for classifier training not found::", snap)
			messagebox.showerror("Snapshot file for classifier training not found.")
		else:
			pose_csvs = glob.glob(os.path.join(model_dir, "classifier/training/**/pose_data.csv"))
			touch_csvs = glob.glob(os.path.join(model_dir, "classifier/training/**/touch.csv"))
			if not pose_csvs:
				print("Training pose estimation files not found")
				messagebox.showerror("Training data CSV file not found.")
			else:
				try:
					print("Initiating Classifier Training.")
					train(model_dir, snap, sorted(pose_csvs), sorted(touch_csvs))
					print("Classifier training complete")
					messagebox.showinfo("Classifier training complete.")
				except Exception as e:
					print(f"Error initiating classifier training: '{e}'")
					messagebox.showerror(f"Error initiating classifier training: '{e}'")

	for widget in training_win.winfo_children():
		if isinstance(widget, ttk.Button):
			widget.configure(state='normal')
	
	status_label.configure(text='Status: Ready')