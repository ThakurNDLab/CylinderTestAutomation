import subprocess
import os
import sys
import tarfile

def find_file(directory, filename):
	try:
		for root, dirs, files in os.walk(directory):
			if filename in files:
				return os.path.join(root, filename)
	except Exception as e:
		print(f"Environment File: {filename}  Not Found")
		return None

def get_base_path():
	if getattr(sys, 'frozen', False):
		# If the application is run as a bundle (e.g., packaged with PyInstaller),
		# the sys._MEIPASS attribute contains the path to the bundle folder.
		return sys._MEIPASS
	else:
		# If it's not packaged, return the directory of this script file
		return os.path.dirname(os.path.abspath(__file__))


def extract_tar(tar_path, target_dir):
	# Extracts a tar.gz file using tarfile module (cross-platform)
	try:
		with tarfile.open(tar_path, "r:gz") as tar:
			tar.extractall(path=target_dir)
	except (tarfile.TarError, FileNotFoundError) as e:
		print(f"Error extracting tar file {tar_path}: {e}")
		sys.exit(1)

def set_executable_permission(file_path):
	try:
		os.chmod(file_path, os.stat(file_path).st_mode | 0o111)
	except OSError as e:
		print(f"Error setting executable permission for {file_path}: {e}")
		sys.exit(1)

def run_script(env_name, script_name):

	activate_script = os.path.join(get_base_path(), env_name, 'bin', 'activate')
	set_executable_permission(activate_script)

	python_exec = os.path.join(get_base_path(), env_name, 'bin', 'python')
	
	if not os.path.exists(python_exec):
		print(f"Python executable not found in {env_name}.")
		sys.exit(1)

	command = f'/bin/bash -c {activate_script} && {python_exec} {script_name}'
	try:
		subprocess.run(command, shell=True, check=True)
	except subprocess.CalledProcessError as e:
		print(f"Error running script {script_name} in environment {env_name}: {e}")
		sys.exit(1)

def main():
	script_dir = get_base_path()

	cyl_env_path = os.path.join(script_dir, "cyl_env")
	dlc_env_path = os.path.join(script_dir, "dlc_env")

	# Check and potentially extract 'cyl_env'
	if not os.path.isdir(cyl_env_path):
		cyl_archive = find_file(script_dir, 'cyl.tar.gz')
		if cyl_archive:
			extract_tar(cyl_archive, cyl_env_path)

	# Check and potentially extract 'dlc_env'
	if not os.path.isdir(dlc_env_path):
		dlc_archive = find_file(script_dir, 'dlc.tar.gz')
		if dlc_archive:
			extract_tar(dlc_archive, dlc_env_path)

	# Check if both environments exist
	if os.path.isdir(cyl_env_path) and os.path.isdir(dlc_env_path):
		# Run script in the 'cyl_env' environment
		script = os.path.join(script_dir, 'startup_window.py')
		run_script('cyl_env', script)
	else:
		# Show error and exit if either environment is not present
		print("Error: Both 'cyl_env' and 'dlc_env' need to be present to run the script.")
		sys.exit(1)

if __name__ == '__main__':
	main()