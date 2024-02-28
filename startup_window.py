import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
from training_window import training_window
from main_window import main_window


font_settings = ("Helvetica", 12, "bold")
bg_color = "#f0f0f0"
button_color = "#4a7abc"
button_font = ("Helvetica", 12)
button_style = "W.TButton"


def find_file(directory, filename):
	for root, dirs, files in os.walk(directory):
		if filename in files:
			return os.path.join(root, filename)
	return None

def open_model(startup_window):
	folder = filedialog.askdirectory(title='Please select the model directory')
	if folder:
		pose = find_file(folder, 'pose_cfg.yaml')
		model = find_file(folder, 'model.pt')
		if pose and model:
			config_path = find_file(os.path.join(folder, "dlc"), 'config.yaml')
			if config_path:
				startup_window.withdraw()
				open_project_window(folder, config_path)
			else:
				messagebox.showerror('Error', 'Configuration file not found in the selected directory.')
				startup_window.deiconify()
		else:
			messagebox.showerror('Error', 'Required files not found in the selected directory.')
			startup_window.deiconify()
	else:
		messagebox.showinfo('Info', 'No folder selected. Please select a model directory.')
		startup_window.deiconify()
		
def create_project_window():
	project_win = tk.Toplevel()
	project_win.title("Create Project")
	project_win.configure(bg=bg_color)

	# Calculate and set window size
	screen_width = project_win.winfo_screenwidth()
	screen_height = project_win.winfo_screenheight()
	window_width = int(screen_width * 0.8)
	window_height = int(screen_height * 0.8)
	project_win.geometry(f"{window_width}x{window_height}+{int(screen_width*0.1)}+{int(screen_height*0.1)}")

	# Status Label
	status_label = tk.Label(project_win, text="Ready", font=font_settings, bg=bg_color)
	status_label.pack(pady=10)

	entries = {}

	# Project Directory Field

	tk.Label(project_win, text='Project Directory', font=font_settings, bg=bg_color).pack(pady=5)
	directory_frame = tk.Frame(project_win, bg=bg_color)
	directory_entry = tk.Entry(directory_frame, font=font_settings)
	directory_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
	browse_btn = ttk.Button(directory_frame, text="Browse", style=button_style,
							command=lambda: directory_entry.insert(0, filedialog.askdirectory()))
	browse_btn.pack(side=tk.LEFT)
	directory_frame.pack(fill=tk.X, pady=5)
	entries['Project Directory'] = directory_entry

	# Input Fields
	fields = ['Project Name', "Experimenter", 'Project Description']
	for field in fields:
		tk.Label(project_win, text=field, font=font_settings, bg=bg_color).pack(pady=5)
		entry = tk.Entry(project_win, font=font_settings)
		entry.pack(pady=5)
		entries[field] = entry


	# Control Buttons
	back_btn = ttk.Button(project_win, text="Back", style=button_style, command=project_win.destroy)
	back_btn.pack(side=tk.LEFT, padx=10, pady=10)

	create_btn = ttk.Button(project_win, text="Create", style=button_style, 
							command=lambda: create_project(entries))
	create_btn.pack(side=tk.LEFT, padx=10, pady=10)

def create_project(entries):
	project_dir = os.path.join(entries['Project Directory'].get(), entries['Project Name'].get())
	if not entries['Project Directory'].get() or not entries['Project Name'].get():
		messagebox.showerror('Error', 'Please specify both the directory and project name.')
		return
	if os.path.exists(project_dir):
		messagebox.showerror('Error', 'Project already exists. Please choose a different name or directory.')
		return
	try:
		# Creating directories
		for subdir in ['videos', 'dlc', 'videos/edited', 'touch']:
			os.makedirs(os.path.join(project_dir, subdir), exist_ok=True)
		
		# Creating info.txt file
		info_path = os.path.join(project_dir, 'info.txt')
		with open(info_path, 'w') as f:
			f.write(f"Project Name: {entries['Project Name'].get()}\n")
			f.write(f"Experimenter: {entries['Experimenter'].get()}\n")
			f.write(f"Project Description: {entries['Project Description'].get()}\n")

		messagebox.showinfo('Success', 'Project created successfully.')
		entries['Project Directory'].delete(0, tk.END)
		entries['Project Name'].delete(0, tk.END)
		entries['Experimenter'].delete(0, tk.END)
		entries['Project Description'].delete(0, tk.END)

	except Exception as e:
		messagebox.showerror('Error', f'An error occurred while creating the project: {e}')


def open_project_action(project_window, config_path):
	project_dir = filedialog.askdirectory(title='Please select the project directory')
	if project_dir and os.path.exists(os.path.join(project_dir, 'info.txt')):
		project_window.withdraw()
		main_window(project_window, project_dir, config_path)
	else:
		messagebox.showinfo('Info', 'Project files not found. Please select a valid project directory.')


def open_project_window(model_dir, config_path):
	project_window = tk.Toplevel()
	project_window.title("Project Window")
	project_window.configure(bg=bg_color)

	# Calculate and set window size
	screen_width = project_window.winfo_screenwidth()
	screen_height = project_window.winfo_screenheight()
	window_width = int(screen_width * 0.4)
	window_height = int(screen_height * 0.4)
	project_window.geometry(f"{window_width}x{window_height}+{int(screen_width*0.1)}+{int(screen_height*0.1)}")

	# Status Label
	status_label = tk.Label(project_window, text="Ready", font=font_settings, bg=bg_color)
	status_label.pack(pady=10)

	# Open Project Button
	open_project_btn = ttk.Button(project_window, text="Open Project", style=button_style, 
								  command=lambda: open_project_action(project_window, config_path))
	open_project_btn.pack(pady=10)

	# Create Project Button
	create_project_btn = ttk.Button(project_window, text="Create Project", style=button_style, 
									command=create_project_window)  # Function to be defined
	create_project_btn.pack(pady=10)

	# Back Button
	back_btn = ttk.Button(project_window, text="Back", style=button_style, 
						  command=lambda: close_window(project_window))
	back_btn.pack(pady=10)

	def close_project_window():
		project_window.destroy()
		startup_window.deiconify()

	project_window.protocol("WM_DELETE_WINDOW", lambda: close_project_window())


def train_model(startup_window):
	train_model_win = tk.Toplevel()
	train_model_win.title("Train Model")
	train_model_win.configure(bg=bg_color)

	# Calculate and set window size
	screen_width = train_model_win.winfo_screenwidth()
	screen_height = train_model_win.winfo_screenheight()
	window_width = int(screen_width * 0.4)
	window_height = int(screen_height * 0.4)
	train_model_win.geometry(f"{window_width}x{window_height}+{int(screen_width*0.1)}+{int(screen_height*0.1)}")

	# Status Label
	status_label = tk.Label(train_model_win, text="Ready", font=font_settings, bg=bg_color)
	status_label.pack(pady=10)

	# Source Directory Input
	src_dir_label = tk.Label(train_model_win, text="Source Directory:", font=font_settings, bg=bg_color)
	src_dir_label.pack(pady=10)
	src_dir_entry = tk.Entry(train_model_win, font=font_settings)
	src_dir_entry.pack(pady=10)
	src_dir_browse = ttk.Button(train_model_win, text="Browse", style=button_style,
							command=lambda: src_dir_entry.insert(0, filedialog.askdirectory()))
	src_dir_browse.pack(pady=10)

	# Model Name Input
	model_name_label = tk.Label(train_model_win, text="Model Name:", font=font_settings, bg=bg_color)
	model_name_label.pack(pady=10)
	model_name_entry = tk.Entry(train_model_win, font=font_settings)
	model_name_entry.pack(pady=10)

	def on_close():
		train_model_win.destroy()
		startup_window.deiconify()

	# Control Buttons
	back_btn = ttk.Button(train_model_win, text="Back", style=button_style, command=lambda: on_close())
	back_btn.pack(side=tk.LEFT, padx=10, pady=10)

	train_model_win.protocol("WM_DELETE_WINDOW", lambda: on_close())

	create_dirs_btn = ttk.Button(train_model_win, text="Create Directories", style=button_style, 
								 command=lambda: create_directories(src_dir_entry.get(), model_name_entry.get()))
	create_dirs_btn.pack(side=tk.LEFT, padx=10, pady=10)

	next_btn = ttk.Button(train_model_win, text="Next", style=button_style, 
						  command=lambda: proceed_to_next(train_model_win, src_dir_entry.get(), model_name_entry.get()))
	next_btn.pack(side=tk.LEFT, padx=10, pady=10)

def create_directories(source_dir, model_name):
	model_dir = os.path.join(source_dir, model_name)
	os.makedirs(model_dir, exist_ok=True)
	os.makedirs(os.path.join(model_dir, "dlc"), exist_ok=True)
	os.makedirs(os.path.join(model_dir, "videos"), exist_ok=True)
	os.makedirs(os.path.join(model_dir, "classifier"), exist_ok=True)

def proceed_to_next(window, source_dir, model_name):
	model_dir = os.path.join(source_dir, model_name)
	window.withdraw()
	training_window(window, model_dir, model_name)

def create_startup_window():
	window = tk.Tk()
	window.title("ACT Analysis")

	# Calculate and set window size
	screen_width = window.winfo_screenwidth()
	screen_height = window.winfo_screenheight()
	window_width = int(screen_width * 0.4)
	window_height = int(screen_height * 0.4)
	window.geometry(f"{window_width}x{window_height}+{int(screen_width*0.1)}+{int(screen_height*0.1)}")

	# Style configuration
	style = ttk.Style(window)
	style.configure("W.TButton", font=button_font, background=button_color)

	window.configure(bg=bg_color)

	def on_close_startup(window):
		window.quit()
		window.destroy()

	# Bind window close event to close_app function
	window.protocol("WM_DELETE_WINDOW", lambda: on_close_startup(window))

	# Welcome Text
	welcome_label = tk.Label(window, text="Welcome! Let's do some science.", font=font_settings, bg=bg_color)
	welcome_label.pack(pady=20)

	# Open Model Button
	open_model_btn = ttk.Button(window, text="Open Model", style=button_style, 
								command=lambda: open_model_action(window))
	open_model_btn.pack(pady=10)

	# Train Model Button
	train_model_btn = ttk.Button(window, text="Train Model", style=button_style, 
								 command=lambda: train_model_action(window))
	train_model_btn.pack(pady=10)

	return window

def open_model_action(window):
	window.withdraw()
	open_model(window)

def train_model_action(window):
	window.withdraw()
	train_model(window)

startup_window = create_startup_window()
startup_window.mainloop()