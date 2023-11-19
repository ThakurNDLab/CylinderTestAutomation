from scipy.stats import mode
import math
import torch.nn.functional as F
from utils import *
from models import GCN, LSTMVAE, FFN
import numpy as np

def predict_cylinder(filename, project_test_directory, fps, timesteps, edge_index, pixel, num_nodes, use_gpu):
	X = preprocess_data(filename)
	X = np.asarray(X)
	X = reshape_array(X, timesteps)
	X = torch.from_numpy(X)
	vel = compute_velocities(X)
	X = np.concatenate([X, vel], axis=2)
	ang = compute_angles_for_sequence(X)
	ang = ang[..., np.newaxis]
	ang = np.repeat(ang, 10, axis=3)
	X = np.concatenate([X, ang], axis=2)

	lstm_dim = (X.shape[2])*num_nodes
	model = BehaviorModel(lstm_dim=lstm_dim, hidden_dim=256, num_labels=num_actions)
	if use_gpu == True:
		datasheet, left_touches, right_touches, touch_count, touch = cylinder_touch_detection(X.cuda(), model.cuda(), num_nodes, edge_index.cuda(), timesteps)
	else:
		datasheet, left_touches, right_touches, touch_count, touch = cylinder_touch_detection(X, model, num_nodes, edge_index, timesteps)
	print('Left Touch Count is:', left_touches)
	print('Right Touch Count is:', right_touches)
	print('Total Touch Count is:', touch_count)
	add_rows = pd.DataFrame(np.zeros((int(timesteps-1), len(touch.columns))), columns=touch.columns)
	touch = pd.concat([add_rows, touch], ignore_index=True)
	y_path = os.path.join(project_test_directory, os.path.splittext(os.path.basename(filename))[0], '_touches.csv')
	touch.to_csv(y_path, index=False)
	left_touch_freq = left_touches / ((datasheet['Left_Touch_Start'].max())+(timesteps-1))/fps
	right_touch_freq = right_touches / ((datasheet['Right_Touch_Start'].max())+(timesteps-1))/fps

	if len(datasheet['Left_Touch_Start']) != len(datasheet['Left_Touch_Ends']):
		datasheet = datasheet[:-1, :, :, :]
	if len(datasheet['Right_Touch_Start']) != len(datasheet['Right_Touch_Ends']):
		datasheet = datasheet[:, :, :-1, :]
	avg_left_len = ((datasheet['Left_Touch_Ends']-datasheet['Left_Touch_Start']).mean())/fps
	avg_right_len = ((datasheet['Right_Touch_Ends']-datasheet['Right_Touch_Start']).mean())/fps

	left_drags = []
	for i in range(len(datasheet['Left_Touch_Start'])):
		left_start_value = datasheet['Left_Touch_Start'].iloc[i-1]
		left_end_value = datasheet['Left_Touch_Ends'].iloc[i-1]
		dist=0
		if left_start_value >= 0 and left_start_value < len(X_gcn):
			for i in range(left_end_value - left_start_value):
				left_x1 = X.at[((left_start_value+i-1)+int(timesteps-1)),-1,0,1]
				left_y1 = X.at[((left_start_value+i-1)+int(timesteps-1)),-1, 1,1]
				left_x2 = X.at[((left_start_value+i)+int(timesteps-1)),-1,0,1]
				left_y2 = X.at[((left_start_value+i)+int(timesteps-1)),-1,1,1]
				dist += math.sqrt(pow((left_y2 - left_y1),2) + pow((left_x2 - left_x1),2))
		left_drags.append[dist]

	right_drags = []
	for i in range(len(datasheet['Right_Touch_Start'])):
		right_start_value = datasheet['Right_Touch_Start'].iloc[i-1]
		right_end_value = datasheet['Right_Touch_Ends'].iloc[i-1]
		dist=0
		if right_start_value >= 0 and right_start_value < len(X_gcn):
			for i in range(right_end_value - right_start_value):
				right_x1 = X.at[((right_start_value+i-1)+int(timesteps-1)),-1,0,3]
				right_y1 = X.at[((right_start_value+i-1)+int(timesteps-1)),-1,1,3]
				right_x2 = X.at[((right_start_value+i)+int(timesteps-1)),-1,0,3]
				right_y2 = X.at[((right_start_value+i)+int(timesteps-1)),-1,1,3]
				dist += math.sqrt(pow((right_y2 - right_y1),2) + pow((right_x2 - right_x1),2))
		right_drags.append[dist]

	left_drag_avg = (np.mean(left_drags) * pixel)
	right_drag_avg = (np.mean(right_drags) * pixel)

	filename_excel = file_path.replace(".csv", "_analysed.xlsx")
	datasheet.to_excel(filename_excel)

	return left_touches, right_touches, left_touch_freq, right_touch_freq, avg_left_len, avg_right_len, left_drag_avg, right_drag_avg


num_actions = 2
num_nodes = 10
timesteps = 7

use_gpu=True

edge_index = np.array([[0,1,2,3,4,5,6,7,8],
					[9,2,9,4,9,6,9,8,9]
					], dtype=np.int_)

txt_file_path = input("")
filelist = []
try:
	with open(txt_file_path, 'r') as file:
		for line in file:
			# Remove leading and trailing whitespace and add the element to the list
			element = line.strip()
			filelist.append(element)

pixel = float(input("Input pixel to milimeter conversion factor"))

summary_df = pd.DataFrame(columns=['File', 'Total_Left_Touches', 'Left_Touch_Freq', 'Left_Touch_Avg_Length', 'Left_Touch_Drag_Distance', 'Total_Right_Touches', 'Right_Touch_Freq', 'Right_Touch_Avg_Length', 'Left_Touch_Drag_Distance', 'Total_Touches'])
for filepath in filelist:
	with torch.no_grad():
		project_test_directory, filename = os.path.split(filepath)
		left_touches, right_touches, left_touch_freq, right_touch_freq, avg_left_len, avg_right_len, left_drag_avg, right_drag_avg = predict_cylinder(filename=filename, project_test_directory=project_test_directory, fps=29.97, timesteps=timesteps, edge_index=edge_index, pixel=pixel, num_nodes=num_nodes, use_gpu=use_gpu)
		summary_df = summary_df.append({'File': file, 
											'Total_Left_Touches': left_touches,
											'Left_Touch_Freq': left_touch_freq,
											'Left_Touch_Avg_Length': avg_left_len,
											'Left_Touch_Drag_Distance': left_drag_avg,
											'Total_Right_Touches': right_touches,
											'Right_Touch_Freq': right_touch_freq,
											'Right_Touch_Avg_Length': avg_right_len,
											'Right_Touch_Drag_Distance': right_drag_avg,
											'Total_Touches': touch_count,
											}, ignore_index=True)

summary_df.to_excel('summary.xlsx')
