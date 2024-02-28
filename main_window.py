import os
import pandas as pd
import subprocess
import torch
import numpy as np
import pickle
import json
import cv2
from sklearn.preprocessing import RobustScaler
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import glob
from pixel import pixel_window
from video_editor import edit_window
from pred_review import review_window
from scripts.predict import *
from scripts.utils import *
from scripts.models import *
from PIL import Image, ImageTk
import re

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
	proc = subprocess.run_dlc(['/bin/bash', '-c', command], stdout=subprocess.PIPE)
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


def export_project_summary(project_dir, video_statuses):
	video_dfs = []  # List to hold dataframes for each video

	for video in get_video_files(project_dir):
		xlsx_path = os.path.join(project_dir, video.replace(".avi", ".xlsx"))
		if not os.path.exists(xlsx_path):
			print("Error: File Not Found ::", xlsx_path)
			continue
		try:
			df = pd.read_excel(xlsx_path)
			relevant_data = df.loc[0, 'Total_Left_Touches':'Total_Touches']
			video_data = pd.DataFrame([relevant_data.values], 
									  columns=['Left Count', 'Left Frequency', 'Left Duration', 'Left Drag', 
											   'Right Count', 'Right Frequency', 'Right Duration', 'Right Drag', 'Total Count'])
			video_data.insert(0, 'Video', video)
			video_dfs.append(video_data)
		except Exception as e:
			print('Error: Error reading file ::', e, xlsx_path)
			messagebox.showerror(f'Error reading file {xlsx_path}: {e}')
			break

	if video_dfs:
		try:
			summary_df = pd.concat(video_dfs, ignore_index=True)
			summary_path = os.path.join(project_dir, f"{os.path.basename(project_dir)}-summary.xlsx")
			summary_df.to_excel(summary_path, index=False)
			print("Project Summary Exported")
			messagebox.showinfo('Project summary has been exported.')
			return summary_path
		except Exception as e:
			print("Error Creating Project Summary::", e)
			messagebox.showerror(f'Error creating project summary: {e}')
			return None
	else:
		print("No data to create a summary.")
		messagebox.showinfo('No data to create a summary.')
		return None


def open_video_editor(video, project_dir):
	video_path = os.path.join(project_dir, "videos", video)
	if not os.path.exists(video_path):
		print("Video file not found::", video_path)
		messagebox.showerror(f'Video file not found: {video_path}')
		return False  
	try:
		edit_window(video_path)
		return True
	except Exception as e:
		print("An error ocurred while opening the video editor::", e)
		messagebox.showerror(f'An error occurred while opening the video editor: {e}')
		return False


def run_dlc_analysis(project_dir, video, config_path):
	if not os.path.exists(config_path):
		print("Configuration file not found::", config_path)
		messagebox.showerror(f'Configuration file not found: {config_path}')
		return False

	dlc_dir = os.path.join(project_dir, 'dlc')
	if not os.path.exists(dlc_dir):
		os.makedirs(dlc_dir)

	video_path = os.path.join(project_dir, 'videos', video)
	if not os.path.exists(video_path):
		print("Video file not found::", video_path)
		messagebox.showerror(f'Video file not found: {video_path}')
		return False
	if os.path.exists(os.path.join(project_dir, 'videos', 'edited', video)):
		video_path = os.path.join(project_dir, 'videos', 'edited', video)
	
	print("Starting pose estimation:", video)
	command = f"deeplabcut.analyze_videos(\"{config_path}\", [\"{video_path}\"], videotype=\"avi\", shuffle=1, trainingsetindex=0, gputouse=0, save_as_csv=True, destfolder=\"{dlc_dir}\")"
	run_dlc_command(command)
	return True

def run_model(dlc_path, output_path, config_path):

	edge_index = torch.from_numpy(np.array([[0,1,2,3,4,5,6,7,8],
					[9,2,9,4,9,6,9,8,9]
					], dtype=np.int_)).cuda()

	# Check for the existence of required files
	if not os.path.exists(dlc_path):
		print("DLC File Not Found::", dlc_path)
		messagebox.showerror(f'DLC file not found: {dlc_path}')
		return False

	model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(config_path))), "classifier", "model.pt")
	if not os.path.exists(model_path):
		print("Model File Not Found::", model_path)
		messagebox.showerror(f'Model file not found: {model_path}')
		return False

	scaler_path = os.path.join(os.path.dirname(model_path), "scaler.pkl")
	if not os.path.exists(scaler_path):
		print("Scaler File Not Found::", scaler_path)
		messagebox.showerror(f'Scaler file not found: {scaler_path}')
		return False

	try:
		scaler = RobustScaler()
		with open(scaler_path, 'rb') as file:
			scaler = pickle.load(file)

		coord_data = pd.read_csv(dlc_path, skiprows=2)
		X = preprocess_data(coord_data)
		X = np.asarray(X)
		X = reshape_array(X, 7)
		X = torch.from_numpy(X)
		vel = compute_velocities(X)
		X = np.concatenate([X, vel], axis=2)
		ang = compute_angles_for_sequence(X)
		ang = ang[..., np.newaxis]
		ang = np.repeat(ang, 10, axis=3)
		X = np.concatenate([X, ang], axis=2)
		X = transform_scaler(X, scaler)
		X = torch.from_numpy(X)
		lstm_dim = (X.shape[2])*10

		model = BehaviorModel(lstm_dim=lstm_dim, hidden_dim=256, num_labels=2)
		model.float().cuda()
		model.load_state_dict(torch.load(model_path))
		model.eval()
		thresh_file = os.path.join(os.path.dirname(model_path), "model.thresh")
		if os.path.exists(thresh_file):
			with open(thresh_file, 'r') as file:
				thresh = float(file.read())
		else:
			thresh = 0.5
		# Run the model prediction
		y_pred = model(X.float().cuda(), edge_index)
		y_pred = np.where(y_pred.cpu().data.numpy() >= thresh, 1, 0)
		y_pred = smoothen_data(y_pred)
		# Save the predictions
		df = pd.DataFrame(y_pred, columns=['left', 'right'])
		df.to_csv(output_path, index=False)
		print("Model Analysis Successful.")
		return True
	except Exception as e:
		print("Error occurred during model analysis:", e)
		messagebox.showerror(f'An error occurred during model analysis: {e}')
		return False


def run_classifier_analysis(project_dir, video, config_path):
	touch_dir = os.path.join(project_dir, 'touch')
	if not os.path.exists(touch_dir):
		os.makedirs(touch_dir)

	video_name = os.path.splitext(video)[0]
	dlc_path = os.path.abspath(glob.glob(os.path.join(project_dir, "dlc", video_name + "DLC" + "*.csv"))[0])
	output_path = os.path.join(touch_dir, video.replace(".avi", ".csv"))

	if not os.path.exists(dlc_path):
		print("DLC output file not found for::", video_name, "::", dlc_path)
		messagebox.showerror(f'DLC output file not found for {video_name}: {dlc_path}')
		return False

	try:
		success = run_model(dlc_path, output_path, config_path)
		if not success:
			print("Error occurred during analysis for::", video_name)
			messagebox.showerror(f'Error occurred during analysis for {video_name}.')
			return False

		print("Touch analysis completed for::", video_name)
		return True
	except Exception as e:
		print("An error occurred during analysis for::", video_name)
		messagebox.showerror(f'An error occurred during analysis for {video_name}: {e}')
		return False


def review(project_dir, video):
	video_path = os.path.join(project_dir, "videos", video)
	if not os.path.exists(video_path):
		print("Videofile not found::", video_path)
		messagebox.showerror(f'Video file not found: {video_path}')
		return False

	csv_path = os.path.join(project_dir, "touch", video.replace(".avi", ".csv"))
	if not os.path.exists(csv_path):
		print("Analysis CSV file not found::", csv_path)
		messagebox.showerror(f'Analysis CSV file not found: {csv_path}')
		return False

	frame_directory = os.path.join(project_dir, "videos", os.path.splitext(video)[0])
	os.makedirs(frame_directory, exist_ok=True)
	if  os.path.exists(os.path.join(project_dir, "videos", "edited", video)):
		video_path = os.path.join(project_dir, "videos", "edited", video)
	if not len(os.listdir(frame_directory))>0:
		try:
			ffmpeg_command = f"ffmpeg -i {video_path} -f image2 {os.path.join(frame_directory, '%04d.png')}"
			run_command_in_env("cyl_env", ffmpeg_command)
		except:
			print("Error extracting frames from video:", video)
	try:
		review_window(frame_directory, csv_path)
		return True
	except Exception as e:
		print("An error occurred during review::", e)
		messagebox.showerror(f'An error occurred during review: {e}')
		return False


def run_data_analysis(project_dir, video):
	factor_file = os.path.join(project_dir, "factors.json")
	video_path = os.path.join(project_dir, "videos", video)

	if not os.path.exists(factor_file):
		print("Factor file missing:", factor_file)
		messagebox.showerror('Error', f'Factor file not found: {factor_file}')
		return False

	factors = load_conversion_factors(factor_file)
	if factors is None:
		print("Error loading conversion factors.")
		return False

	factor = factors.get(video, 1)
	print(factor)  # Default to 1 if the video is not in the factors

	csv_path = os.path.join(project_dir, "touch", video.replace(".avi", ".csv"))
	posecsv = glob.glob(os.path.join(project_dir, 'dlc', os.path.splitext(os.path.basename(video))[0] + "DLC" + "*.csv"))[0]
	output_path = os.path.join(project_dir, video.replace(".avi", ".xlsx"))
	if not os.path.exists(csv_path):
		print("Analysis CSV file not found for:", video)
		return False

	try:
		ffmpeg_command = f"ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate {video_path}"
		fpsfrac = run_command_in_env("cyl_env", ffmpeg_command)
		frame_rate_match = re.search(r'avg_frame_rate=(\d+)/(\d+)', fpsfrac)
		if frame_rate_match:
			fps_num, fps_den = map(int, frame_rate_match.groups())
			fps = fps_num / fps_den
		print(fps)
		analyse(csv_path, posecsv, output_path, factor, fps)  # Assuming analyse function is defined elsewhere
		print("Analysis completed for:", video)
		return True

	except Exception as e:
		print("An error occurred during analysis for:", video, "::", e)
		return False

# global_image_references = {}

# def get_status_image(status, img_id):
#     global global_image_references

#     img_path = get_base_path()
#     image_file = "tick.bmp" if status else "cross.bmp"
#     full_path = os.path.join(img_path, image_file)
#     image = Image.open(full_path)
#     photo = ImageTk.PhotoImage(image)

#     # Store the reference in the global dictionary
#     global_image_references[img_id] = photo

#     return photo


def get_video_files(project_dir):
	video_dir = os.path.join(project_dir, 'videos')
	if not os.path.exists(video_dir):
		print("Video directory not found::", video_dir)
		messagebox.showerror(f'Video directory not found: {video_dir}')
		return []

	video_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]
	if not video_files:
		print("No videofile found in the directory")
		messagebox.showerror('No video files found in the directory.')

	return video_files


def get_video_status(project_dir):
	video_files = get_video_files(project_dir)
	if not video_files:
		return []

	statuses = []
	for video in video_files:
		edit_status = os.path.exists(os.path.join(project_dir, 'videos', 'edited', video))
		dlc_status = len(glob.glob(os.path.join(project_dir, 'dlc', os.path.splitext(os.path.basename(video))[0] + "DLC" + "*.csv")))>0
		predict_status = os.path.exists(os.path.join(project_dir, 'touch', video.replace(".avi", ".csv")))
		analyse_status = os.path.exists(os.path.join(project_dir, video.replace(".avi", ".xlsx")))
		statuses.append((video, edit_status, dlc_status, predict_status, analyse_status))
	return statuses


def extract_middle_frame(video_path):
	if not os.path.exists(video_path):
		print("Videofile Missing::", video_path)
		messagebox.showerror(f'Video file not found: {video_path}')
		return None

	try:
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			print("Unable to open Videofile::", video_path)
			messagebox.showerror(f'Unable to open video file: {video_path}')
			return None

		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
		ret, frame = cap.read()

		if not ret:
			print("Unable to read frames from the videofile")
			messagebox.showerror('Unable to read the frame from the video.')
			return None

		frame_path = video_path.replace('.avi', '_middle_frame.png')
		cv2.imwrite(frame_path, frame)
		cap.release()

		return frame_path if ret else None

	except Exception as e:
		print("An error occurred while extracting frame for calculating pixel conversion factor::", e)
		messagebox.showerror(f'An error occurred while extracting the middle frame: {e}')
		return None


def toggle_buttons_state(frame, state):
	for widget in frame.winfo_children():
		if isinstance(widget, ttk.Button):
			widget.config(state='normal' if state else 'disabled')


def create_overview_tab(root, project_dir):
	overview_tab = ttk.Frame(root)
	video_statuses = get_video_status(project_dir)

	headers = ["File Name", "Edit", "DLC", "Predict", "Analyse"]
	for i, header in enumerate(headers):
		header_label = ttk.Label(overview_tab, text=header)
		header_label.grid(row=0, column=i, padx=5, pady=5)

	for row, (video, edit, dlc, predict, analyse) in enumerate(video_statuses, start=1):
		ttk.Label(overview_tab, text=video).grid(row=row, column=0, padx=5, pady=5)
		for col, status in enumerate([edit, dlc, predict, analyse], start=1):
			status_text = "DONE" if status else "PENDING"
			status_color = "#00FF00" if status else "#FF0000"  # Green for DONE, Red for PENDING
			status_label = ttk.Label(overview_tab, text=status_text, background=status_color)
			status_label.grid(row=row, column=col, padx=5, pady=5)

	def on_export_button_click():
		export_project_summary(project_dir, video_statuses)

	export_button = ttk.Button(overview_tab, text='Export Project Summary', command=on_export_button_click)
	export_button.grid(row=len(video_statuses) + 1, column=len(headers) - 1, padx=5, pady=5)

	return overview_tab

def create_edit_tab(root, project_dir):
	edit_tab = ttk.Frame(root)
	video_files = get_video_files(project_dir)

	headers = ["File Name", "Status", "Actions"]
	for i, header in enumerate(headers):
		header_label = ttk.Label(edit_tab, text=header)
		header_label.grid(row=0, column=i, padx=5, pady=5)

	for row, video in enumerate(video_files, start=1):
		ttk.Label(edit_tab, text=video).grid(row=row, column=0, padx=5, pady=5)

		# Status of editing
		edit_status = check_edit_status(video, project_dir)
		status_text = "DONE" if edit_status else "PENDING"
		status_color = "#00FF00" if edit_status else "#FF0000"  # Green for DONE, Red for PENDING
		status_label = ttk.Label(edit_tab, text=status_text, background=status_color)
		status_label.grid(row=row, column=1, padx=5, pady=5)

		# Action button for editing
		action_button = ttk.Button(edit_tab, text='Edit', command=lambda v=video: edit_video(edit_tab, v, project_dir))
		action_button.grid(row=row, column=2, padx=5, pady=5)

	return edit_tab

def check_edit_status(video, project_dir):
	# Assuming edited videos are stored in a subdirectory named 'edited'
	edited_videos_dir = os.path.join(project_dir, 'videos', 'edited')
	edited_video_path = os.path.join(edited_videos_dir, video)

	return os.path.exists(edited_video_path)

def edit_video(edit_frame, video, project_dir):
		# Update status and disable buttons
		toggle_buttons_state(edit_frame, False)

		# Call the function to edit the video
		open_video_editor(video, project_dir)

		# Update status and re-enable buttons
		toggle_buttons_state(edit_frame, True)

def toggle_video_selection(video):
	if video in selected_videos_dlc:
		selected_videos_dlc[video] = not selected_videos_dlc[video]
	else:
		selected_videos_dlc[video] = True
	print(f"Toggled {video}: {selected_videos_dlc[video]}")

def create_dlc_tab(root, project_dir, config_path):
	global selected_videos_dlc
	selected_videos_dlc = {}

	dlc_tab = ttk.Frame(root)

	video_files = get_video_files(project_dir)
	selected_videos_dlc = {}  # Dictionary to store selected videos for DLC

	headers = ["Select", "File Name", "DLC Status", "Run DLC"]
	for i, header in enumerate(headers):
		header_label = ttk.Label(dlc_tab, text=header)
		header_label.grid(row=0, column=i, padx=5, pady=5)

	for row, video in enumerate(video_files, start=1):
		chk = ttk.Checkbutton(dlc_tab, text=video, command=lambda v=video: toggle_video_selection(v))
		chk.grid(row=row, column=0, padx=5, pady=5)
		selected_videos_dlc[video] = False

		ttk.Label(dlc_tab, text=video).grid(row=row, column=1, padx=5, pady=5)

		dlc_status = check_dlc_status(video, project_dir)
		status_text = "DONE" if dlc_status else "PENDING"
		status_color = "#00FF00" if dlc_status else "#FF0000"
		status_label = ttk.Label(dlc_tab, text=status_text, background=status_color)
		status_label.grid(row=row, column=2, padx=5, pady=5)

	def run_selected_dlcs():
		print("Selected videos for DLC:", selected_videos_dlc)
		for video, state in selected_videos_dlc.items():
			if state == 1:
				run_dlc(dlc_tab, video, project_dir, config_path)

	run_dlc_button = ttk.Button(dlc_tab, text='Run Selected DLCs', command=run_selected_dlcs)
	run_dlc_button.grid(row=len(video_files) + 1, column=3, padx=5, pady=5)

	return dlc_tab

def run_dlc(dlc_frame, video, project_dir, config_path):
	toggle_buttons_state(dlc_frame, False)

	run_dlc_analysis(project_dir, video, config_path)

	toggle_buttons_state(dlc_frame, True)

def check_dlc_status(video, project_dir):
	dlc_output_dir = os.path.join(project_dir, 'dlc')
	dlc_output_pattern = os.path.join(dlc_output_dir, os.path.splitext(os.path.basename(video))[0] + "DLC" + "*.csv")

	return len(glob.glob(dlc_output_pattern)) > 0

def toggle_video_selection_predict(video):
	if video in selected_videos_predict:
		selected_videos_predict[video] = not selected_videos_predict[video]
	else:
		selected_videos_predict[video] = True
	print(f"Toggled {video}: {selected_videos_predict[video]}")  # Debug print

def create_predict_tab(root, project_dir, config_path):
	global selected_videos_predict
	selected_videos_predict = {}

	predict_tab = ttk.Frame(root)

	video_files = get_video_files(project_dir)
	selected_videos_predict = {}  # Dictionary to store selected videos for prediction

	headers = ["Select", "File Name", "Prediction Status", "Run Prediction"]
	for i, header in enumerate(headers):
		header_label = ttk.Label(predict_tab, text=header)
		header_label.grid(row=0, column=i, padx=5, pady=5)

	for row, video in enumerate(video_files, start=1):
		chk = ttk.Checkbutton(predict_tab, text=video, command=lambda v=video: toggle_video_selection_predict(v))
		chk.grid(row=row, column=0, padx=5, pady=5)
		selected_videos_predict[video] = False 
		
		ttk.Label(predict_tab, text=video).grid(row=row, column=1, padx=5, pady=5)

		prediction_status = check_prediction_status(video, project_dir)
		status_text = "DONE" if prediction_status else "PENDING"
		status_color = "#00FF00" if prediction_status else "#FF0000"
		status_label = ttk.Label(predict_tab, text=status_text, background=status_color)
		status_label.grid(row=row, column=2, padx=5, pady=5)

	def run_selected_predictions():
		print("Selected videos for Prediction:", selected_videos_predict)
		for video, selected in selected_videos_predict.items():
			if selected:
				run_prediction(predict_tab, video, project_dir, config_path)

	run_prediction_button = ttk.Button(predict_tab, text='Run Selected Predictions', command=run_selected_predictions)
	run_prediction_button.grid(row=len(video_files) + 1, column=3, padx=5, pady=5)

	return predict_tab

def run_prediction(predict_frame, video, project_dir, config_path):
	toggle_buttons_state(predict_frame, False)

	run_classifier_analysis(project_dir, video, config_path)

	toggle_buttons_state(predict_frame, True)

def check_prediction_status(video, project_dir):
	classifier_output_dir = os.path.join(project_dir, 'touch')
	classifier_output_pattern = os.path.join(classifier_output_dir, os.path.splitext(os.path.basename(video))[0] + ".csv")

	return os.path.exists(classifier_output_pattern)


def create_review_tab(root, project_dir):
	review_tab = ttk.Frame(root)

	video_files = get_video_files(project_dir)  # Assuming this returns full file paths

	headers = ["File Name", "Review"]
	for i, header in enumerate(headers):
		header_label = ttk.Label(review_tab, text=header)
		header_label.grid(row=0, column=i, padx=5, pady=5)

	for row, video_path in enumerate(video_files, start=1):
		video_name = os.path.basename(video_path)
		ttk.Label(review_tab, text=video_name).grid(row=row, column=0, padx=5, pady=5)

		review_button = ttk.Button(review_tab, text='Review', command=lambda v=video_path: review_video(project_dir, review_tab, v))
		review_button.grid(row=row, column=1, padx=5, pady=5)

	return review_tab

def review_video(project_dir, review_frame, video):
	# Update status and disable buttons
	toggle_buttons_state(review_frame, False)

	# Call the function to review the video
	review(project_dir, video)

	# Update status and re-enable buttons
	toggle_buttons_state(review_frame, True)

def toggle_video_selection_analyse(video):
	selected_videos_analyse[video] = not selected_videos_analyse.get(video, False)
	print(f"Toggled {video}: {selected_videos_analyse[video]}")  # Debug print

def create_analyse_tab(root, project_dir):
	global selected_videos_analyse
	analyse_frame = ttk.Frame(root)

	video_files = get_video_files(project_dir)
	factor_file = os.path.join(project_dir, "factors.json")
	factors = load_conversion_factors(factor_file)
	factor_labels = {}
	selected_videos_analyse = {}

	headers = ["Select", "File Name", "Analysis Status", "Load Conversion Factor", "Pixel Conversion Factor"]
	for i, header in enumerate(headers):
		header_label = ttk.Label(analyse_frame, text=header)
		header_label.grid(row=0, column=i, padx=5, pady=5)

	for row, video in enumerate(video_files, start=1):
		chk = ttk.Checkbutton(analyse_frame, text=video, command=lambda v=video: toggle_video_selection_analyse(v))
		chk.grid(row=row, column=0, padx=5, pady=5)
		selected_videos_analyse[video] = False

		ttk.Label(analyse_frame, text=video).grid(row=row, column=1, padx=5, pady=5)

		# Assuming check_analysis_status is a function that returns True if analysis is done
		analysis_status = check_analysis_status(video, project_dir)  # Define or modify this function as needed
		status_text = "DONE" if analysis_status else "PENDING"
		status_color = "#00FF00" if analysis_status else "#FF0000"  # Green for DONE, Red for PENDING
		status_label = ttk.Label(analyse_frame, text=status_text, background=status_color)
		status_label.grid(row=row, column=2, padx=5, pady=5)

		extract_button = ttk.Button(analyse_frame, text='Load Conversion Factor',
									command=lambda v=video: extract(v, project_dir, factors))
		extract_button.grid(row=row, column=3, padx=5, pady=5)

		factor_label = ttk.Label(analyse_frame, text=factors.get(video, ""))
		factor_label.grid(row=row, column=4, padx=5, pady=5)
		factor_labels[video] = factor_label

	# Button to Run Analysis on Selected Videos
	def run_selected_analyses():
		print("Selected videos for Analysis:", selected_videos_analyse)
		for video, selected in selected_videos_analyse.items():
			if selected:
				analysis(analyse_frame, project_dir, video)

	analyse_button = ttk.Button(analyse_frame, text='Run Analysis on Selected', command=run_selected_analyses)
	analyse_button.grid(row=len(video_files) + 2, column=len(headers) - 1, padx=5, pady=5)

	return analyse_frame


def extract(video, project_dir, factors):
	video_path = os.path.join(project_dir, "videos", video)
	frame_path = extract_middle_frame(video_path)
	factor = pixel_window(frame_path)
	if video in factors:
		factors[video]['text'] = factor
	else:
		factors[video] = {'text': factor}
	save_conversion_factors(project_dir, factors)

def save_conversion_factors(project_dir, factors):
	factor_file = os.path.join(project_dir, "factors.json")
	with open(factor_file, 'w') as file:
		json.dump({k: v['text'] for k, v in factors.items()}, file)

def load_conversion_factors(factor_file):
	try:
		with open(factor_file, 'r') as file:
			return json.load(file)
	except FileNotFoundError:
		# Create an empty JSON file if it doesn't exist
		with open(factor_file, 'w') as file:
			json.dump({}, file)
		return json.load(file)

def analysis(analyse_frame, project_dir, video):
	# Update status and disable buttons
	toggle_buttons_state(analyse_frame, False)

	# Call the function to review the video
	run_data_analysis(project_dir, video)

	# Update status and re-enable buttons
	toggle_buttons_state(analyse_frame, True)

def check_analysis_status(video, project_dir):
	analyse_path = os.path.join(project_dir, video.replace('.avi', '.xlsx'))

	return os.path.exists(analyse_path)

def main_window(window, project_dir, config_path):
	root = tk.Tk()
	root.title("ACT Analysis")

	# Calculate and set window size
	screen_width = root.winfo_screenwidth()
	screen_height = root.winfo_screenheight()
	window_width = int(screen_width * 0.8)
	window_height = int(screen_height * 0.8)
	root.geometry(f"{window_width}x{window_height}+{int(screen_width*0.1)}+{int(screen_height*0.1)}")

	font_settings = ("Helvetica", 12, "bold")
	bg_color = "#f0f0f0"
	button_color = "#4a7abc"
	button_font = ("Helvetica", 12)
	# Style configuration
	style = ttk.Style(root)
	style.configure("W.TButton", font=button_font, background=button_color)
	button_style = "W.TButton"

	def on_close():
		root.destroy()
		window.deiconify()

	root.protocol("WM_DELETE_WINDOW", lambda: on_close())

	tab_control = ttk.Notebook(root)

	overview_tab = create_overview_tab(root, project_dir)
	tab_control.add(overview_tab, text='Overview')

	edit_tab = create_edit_tab(root, project_dir)
	tab_control.add(edit_tab, text='Edit')

	dlc_tab = create_dlc_tab(root, project_dir, config_path)
	tab_control.add(dlc_tab, text='DLC')

	predict_tab = create_predict_tab(root, project_dir, config_path)
	tab_control.add(predict_tab, text='Predict')

	review_tab = create_review_tab(root, project_dir)
	tab_control.add(review_tab, text='Review')

	analyse_tab = create_analyse_tab(root, project_dir)
	tab_control.add(analyse_tab, text='Analyse')

	tab_control.pack(expand=1, fill="both")
	root.mainloop()