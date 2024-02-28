from scipy.stats import mode
import math
import torch.nn.functional as F
from scripts.utils import *
from scripts.models import *
import numpy as np
import sys
import os
from scipy.signal import medfilt


def reset_model_state(model):
	if hasattr(model, 'reset_hidden_state'):
		model.reset_hidden_state()

def count(y):
	starts_left = []
	ends_left = []
	starts_right = []
	ends_right = []

	in_left_touch, in_right_touch = False, False

	for i in range(len(y)):
		left_touch = y[i,0]
		right_touch = y[i,1]

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

	left_df = pd.DataFrame({'Left_Touch_Start': starts_left, 'Left_Touch_Ends': ends_left}) +6
	right_df = pd.DataFrame({'Right_Touch_Start': starts_right, 'Right_Touch_Ends': ends_right}) +6

	return left_df, right_df, len(starts_left), len(starts_right), (len(starts_left) + len(starts_right))

def calculate_drags(touchCoordinates, touchEvents, startColumn, endColumn, coordIndex):
    drags = []
    for start, end in zip(touchEvents[startColumn], touchEvents[endColumn]):
        start, end = int(start), int(end)
        if 0 <= start < len(touchCoordinates) and 0 <= end <= len(touchCoordinates):
            segment = touchCoordinates.iloc[start:end, coordIndex:coordIndex+2]
            distances = np.sqrt(np.sum(np.diff(segment, axis=0)**2, axis=1))
            total_distance = np.sum(distances)
            drags.append(total_distance)
        else:
            continue
    return drags

def analyse(touchcsv, posecsv, out, factor, fps):
	y = pd.read_csv(touchcsv)
	X = pd.read_csv(posecsv, skiprows=2)
	X = X.apply(lambda col: medfilt(col, kernel_size=5))
	left_df, right_df, left_touches, right_touches, total_touches = count(y.to_numpy())
	left_touch_freq = left_touches / (((left_df['Left_Touch_Start'].max()))/fps)
	right_touch_freq = right_touches / (((right_df['Right_Touch_Start'].max()))/fps)

	avg_left_len = ((left_df['Left_Touch_Ends'] - left_df['Left_Touch_Start']).mean()) / fps
	avg_right_len = ((right_df['Right_Touch_Ends'] - right_df['Right_Touch_Start']).mean()) / fps
	left_drags = calculate_drags(X, left_df, 'Left_Touch_Start', 'Left_Touch_Ends', 3)
	right_drags = calculate_drags(X, right_df, 'Right_Touch_Start', 'Right_Touch_Ends', 9)
	left_drag_avg = (np.mean(left_drags) * factor)
	right_drag_avg = (np.mean(right_drags) * factor)

	df = pd.DataFrame({
		    'Total_Left_Touches': [left_touches],
		    'Left_Touch_Freq': [left_touch_freq],
		    'Left_Touch_Avg_Length': [avg_left_len],
		    'Left_Touch_Drag_Distance': [left_drag_avg],
		    'Total_Right_Touches': [right_touches],
		    'Right_Touch_Freq': [right_touch_freq],
		    'Right_Touch_Avg_Length': [avg_right_len],
		    'Right_Touch_Drag_Distance': [right_drag_avg],
		    'Total_Touches': [total_touches],
		}, index=[0])
	
	datasheet = pd.concat([left_df, right_df, df], axis=1)
	datasheet.to_excel(out)