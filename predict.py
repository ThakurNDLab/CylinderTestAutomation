from scipy.stats import mode
import math
import torch.nn.functional as F
from utils import *
from models import *
import numpy as np
import sys
import os
from scipy.signal import medfilt

def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

random_seed = 42
set_seed(random_seed)

def reset_model_state(model):
	if hasattr(model, 'reset_hidden_state'):
		model.reset_hidden_state()


def cylinder_touch_detection(X, model, num_nodes, edge_index):
	model.eval()
	with torch.no_grad():
		preds = model(X, edge_index)
		reset_model_state(model)
	preds = preds.cpu().data.numpy()
	y = np.where(preds >= 0.5, 1, 0)
	y = smoothen_data(y)

	starts_left = []
	ends_left = []
	starts_right = []
	ends_right = []

	in_left_touch, in_right_touch = False, False

	for i, (left_touch, right_touch) in enumerate(y):

		if left_touch == 1 and not in_left_touch:
			in_left_touch = True
			starts_left.append(i)
		elif left_touch == 0 and in_left_touch:
			in_left_touch = False
			ends_left.append(i-1)

		if right_touch == 1 and not in_right_touch:
			in_right_touch = True
			starts_right.append(i)
		elif right_touch == 0 and in_right_touch:
			in_right_touch = False
			ends_right.append(i-1)

		if len(starts_left) + len(starts_right) >= 30:
			if not in_left_touch and not in_right_touch:
				break

	if in_left_touch:
		ends_left.append(len(y) - 1)
	if in_right_touch:
		ends_right.append(len(y) - 1)

	left_df = pd.DataFrame({'Left_Touch_Start': starts_left, 'Left_Touch_Ends': ends_left})
	right_df = pd.DataFrame({'Right_Touch_Start': starts_right, 'Right_Touch_Ends': ends_right})

	return left_df, right_df, len(starts_left), len(starts_right), (len(starts_left) + len(starts_right)), pd.DataFrame(y)

def calculate_drags(X, datasheet, touch_start_column, touch_end_column, X_coord_index, Y_coord_index):
	drags = []
	for start, end in zip(datasheet[touch_start_column], datasheet[touch_end_column]):
		start, end = int(start), int(end)
		dist = 0
		if 0 <= start < len(X):
			for j in range(end - start):
				index1 = start + j + timesteps - 1
				index2 = start + j + 1 + timesteps - 1
				if index1 < len(X) and index2 < len(X):
					x1, y1 = X[index1, -1, 0, X_coord_index], X[index1, -1, 1, Y_coord_index]
					x2, y2 = X[index2, -1, 0, X_coord_index], X[index2, -1, 1, Y_coord_index]
					dist += math.sqrt(pow((y2 - y1), 2) + pow((x2 - x1), 2))
				else:
					break
		drags.append(dist)
	return drags

def predict_cylinder(filename, project_test_directory, fps, timesteps, pixel, num_nodes, use_gpu):
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

	edge_index = np.array([[0,1,2,3,4,5,6,7,8],
					[9,2,9,4,9,6,9,8,9]
					], dtype=np.int_)

	X = torch.from_numpy(X)
	edge_index = torch.from_numpy(edge_index)

	lstm_dim = (X.shape[2])*num_nodes
	model = BehaviorModel(lstm_dim=lstm_dim, hidden_dim=256, num_labels=num_actions)
	model.load_state_dict(torch.load('model.pt'))

	if use_gpu == True:
		left_data, right_data, left_touches, right_touches, touch_count, touch = cylinder_touch_detection(X.float().cuda(), model.float().cuda(), num_nodes, edge_index.cuda())
	else:
		left_data, right_data, left_touches, right_touches, touch_count, touch = cylinder_touch_detection(X.float(), model.float(), num_nodes, edge_index)
	print('Left Touch Count is:', left_touches)
	print('Right Touch Count is:', right_touches)
	print('Total Touch Count is:', touch_count)
	suffix_to_remove = 'DLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000'
	filename = filename.replace(suffix_to_remove, '')
	filename = os.path.splitext(filename)[0]
	y_path = os.path.join(path, filename + '.csv')
	touch.to_csv(y_path, index=False)
	left_touch_freq = left_touches / (((left_data['Left_Touch_Start'].max())+(timesteps))/fps)
	right_touch_freq = right_touches / (((right_data['Right_Touch_Start'].max())+(timesteps))/fps)

	filename_excel = y_path.replace(".csv", "_analysed.xlsx")
	datasheet = pd.concat([left_data, right_data], axis=1)
	datasheet.to_excel(filename_excel)

	avg_left_len = ((left_data['Left_Touch_Ends'] - left_data['Left_Touch_Start']).mean()) / fps
	avg_right_len = ((right_data['Right_Touch_Ends'] - right_data['Right_Touch_Start']).mean()) / fps

	left_drags = calculate_drags(X, left_data, 'Left_Touch_Start', 'Left_Touch_Ends', 1, 1)
	right_drags = calculate_drags(X, right_data, 'Right_Touch_Start', 'Right_Touch_Ends', 3, 3)
	left_drag_avg = (np.mean(left_drags) / pixel)
	right_drag_avg = (np.mean(right_drags) / pixel)

	return touch_count, left_touches, right_touches, left_touch_freq, right_touch_freq, avg_left_len, avg_right_len, left_drag_avg, right_drag_avg


num_actions = 2
num_nodes = 10
timesteps = 7

use_gpu=True

txt_file_path = sys.argv[1]
filelist = []
with open(txt_file_path, 'r') as file:
	for line in file:
		element = line.strip()
		path, name = os.path.split(element)
		filelist.append(name)

	pixel = float(sys.argv[2])

	summary_df = pd.DataFrame(columns=['File', 'Total_Left_Touches', 'Left_Touch_Freq', 'Left_Touch_Avg_Length', 'Left_Touch_Drag_Distance', 'Total_Right_Touches', 'Right_Touch_Freq', 'Right_Touch_Avg_Length', 'Right_Touch_Drag_Distance', 'Total_Touches'])
	for file in filelist:
		with torch.no_grad():
			touch_count, left_touches, right_touches, left_touch_freq, right_touch_freq, avg_left_len, avg_right_len, left_drag_avg, right_drag_avg = predict_cylinder(filename=file, project_test_directory=path, fps=29.97, timesteps=timesteps, pixel=pixel, num_nodes=num_nodes, use_gpu=use_gpu)
			df = pd.DataFrame({'File': [file], 
								'Total_Left_Touches': left_touches,
								'Left_Touch_Freq': left_touch_freq,
								'Left_Touch_Avg_Length': avg_left_len,
								'Left_Touch_Drag_Distance': left_drag_avg,
								'Total_Right_Touches': right_touches,
								'Right_Touch_Freq': right_touch_freq,
								'Right_Touch_Avg_Length': avg_right_len,
								'Right_Touch_Drag_Distance': right_drag_avg,
								'Total_Touches': touch_count,
								})
			summary_df = pd.concat([summary_df.reset_index(drop=True), df.reset_index(drop=True)], axis=0, ignore_index=True)
	summary_df.to_excel('summary.xlsx')