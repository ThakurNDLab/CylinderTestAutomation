#!/bin/bash

set -e

x=$(pwd)
num_mts_files=0
num_mp4_files=0

home_dir="/home/ndlab_admin"
config_path="/home/ndlab_admin/Desktop/Jyotirmay/cylinder test/Cylinder Test Trial 3-Jyotiray Srivastava-2023-01-23/config.yaml"

run_deeplabcut() {
		local mp4_file=$1
		conda run -n dlc python -c "import deeplabcut; deeplabcut.analyze_videos('$config_path', ['$mp4_file'], save_as_csv=True, gputouse=0)" || { echo "DeepLabCut analysis failed for $mp4_file"; exit 1; }
}

if find "$x" -type f -name "*.MTS" | grep -q '.'; then
	num_mts_files=$(find "$x" -type f -name "*.MTS" | wc -l)

	if find "$x" -type f -name "*.mp4" | grep -q '.'; then
		num_mp4_files=$(find "$x" -type f -name "*.mp4" | wc -l)
	else
		first=true
		for MTS_file in "$x"/*.MTS; do
			mp4_file="$x/$(basename "$MTS_file" .MTS).mp4"
			if [ ! -e "$mp4_file" ]; then
				ffmpeg -i "$MTS_file" -r 29.97 -c:v libx265 -preset veryfast -crf 0 "$mp4_file"

				# Extract frame from the first converted MP4 file
				if [ "$first" = true ]; then
					duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$mp4_file")
					middle_time=$(bc <<< "$duration / 2")
					ffmpeg -ss "$middle_time" -i "$mp4_file" -frames:v 1 "${mp4_file%.mp4}_middle_frame.jpg"
					first=false
				fi
			fi
		done
	fi

	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	read -p "Enter the pixel conversion value: " user_input

	for mp4_file in "$x"/*.mp4; do
		csv_file="${mp4_file%.mp4}DLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv"
		if [ ! -e "$csv_file" ]; then
				run_deeplabcut "$mp4_file"
		fi
	done

	if [ ! -e "$x/csv_file_list.txt" ]; then
			find "$x" -type f -name "*DLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv" > "$x/csv_file_list.txt"
	fi


	if [ ! -e "$x/summary.xlsx" ]; then
		csv_file_list="$x/csv_file_list.txt"
		conda run -n cylindertest python predict.py "$csv_file_list" "$user_input"
	fi

	if [ -e "$x/summary.xlsx" ]; then 
		echo "Full Analysis Complete!"
		echo "Please find the summary.xlsx file containing results in the video directory."
	else
		echo "An error occurred. Please try rerunning the script."
	fi


else
	find "$x" -type f -name "*.mp4" | grep -q '.'

	if find "$x" -maxdepth 1 -type f -name "*.jpg" | grep -q '.'; then
		echo "JPG files found. Continuing with the script."
	else
		echo "No JPG files found. Extracting frame from a random MP4 file."
		random_mp4=$(find "$x" -maxdepth 1 -type f -name "*.mp4" | shuf -n 1)
		echo "x"
		if [ -z "$random_mp4" ]; then
			echo "No MP4 files found. Exiting."
			exit 1
		fi
		duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$random_mp4")
		middle_time=$(bc <<< "$duration / 2")
		ffmpeg -ss "$middle_time" -i "$random_mp4" -frames:v 1 "${random_mp4%.mp4}_middle_frame.jpg"
	fi
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	echo "."
	read -p "Enter the pixel conversion value: " user_input

	for mp4_file in "$x"/*.mp4; do
		csv_file="${mp4_file%.mp4}DLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv"
		if [ ! -e "$csv_file" ]; then
				run_deeplabcut "$mp4_file"
		fi
	done

	if [ ! -e "$x/csv_file_list.txt" ]; then
			find "$x" -type f -name "*DLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv" > "$x/csv_file_list.txt"
	fi

	if [ ! -e "$x/summary.xlsx" ]; then
		csv_file_list="$x/csv_file_list.txt"
		conda run -n cylindertest python predict.py "$csv_file_list" "$user_input"
	fi

	if [ -e "$x/summary.xlsx" ]; then 
		echo "Full Analysis Complete!"
		echo "Please find the summary.xlsx file containing results in the video directory."
	else
		echo "An error occurred. Please try rerunning the script."
	fi
fi