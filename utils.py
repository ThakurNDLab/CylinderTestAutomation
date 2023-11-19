import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import mode
from scipy.signal import medfilt
import numpy as np


def compute_velocities(data):
    velocities = np.diff(data, axis=1)
    euclidean_velocities = np.linalg.norm(velocities, axis=2)
    euclidean_velocities = euclidean_velocities[:, :, np.newaxis, :]
    zero_pad = np.zeros((data.shape[0], 1, 1, data.shape[3]))
    euclidean_velocities = np.concatenate([zero_pad, euclidean_velocities], axis=1)
    return euclidean_velocities

def compute_angle(A, O, B):
    a = np.array(A) - np.array(O)
    b = np.array(B) - np.array(O)
    dot_product = np.einsum('ijk,ijk->ij', a, b)
    norm_a = np.linalg.norm(a, axis=2)
    norm_b = np.linalg.norm(b, axis=2)
    epsilon = 1e-8
    cos_angle = dot_product / (norm_a * norm_b + epsilon)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    return angle

def compute_angles_for_sequence(data):
	angles = []
	for marker in [1, 3, 5, 7]:
		angles.append(compute_angle(data[:, :, :2, marker], data[:, :, :2, (marker+1)], data[:, :, :2, 9]))
	for marker in [1, 3, 5, 7]:
		angles.append(compute_angle(data[:, :, :2, marker], data[:, :, :2, (marker+1)], data[:, :, :2, 0]))
	for marker in [2, 4, 6, 8]:
		angles.append(compute_angle(data[:, :, :2, marker], data[:, :, :2, 9], data[:, :, :2, 0]))
	out = np.stack(angles, axis=-1)
	return out

def preprocess_data(file_path):
	coord_data = pd.read_csv(file_path, skiprows=2)
	coord_data = coord_data.drop(coord_data.columns[0], axis=1)
	cylcenter_x = coord_data.iloc[:,30].mean()
	cylcenter_y = coord_data.iloc[:,31].mean()
	coord_data = coord_data.drop(coord_data.columns[[30,31,32]], axis=1)
	coord_data.iloc[:,0:28:3] -= cylcenter_x
	coord_data.iloc[:,1:29:3] -= cylcenter_y
	coord_data = coord_data.iloc[:, [i for i in range(len(coord_data.columns)) if (i + 1) % 3 != 0]]
	coord_data = coord_data.apply(lambda col: medfilt(col, kernel_size=5))
	coord_data = pd.DataFrame(coord_data)
	return coord_data

def reshape_array(input_array, timesteps=11):
	timepoints, features = input_array.shape
	new_shape = (timepoints - timesteps + 1, timesteps, features)
	reshaped_array = np.zeros(new_shape)
	for i in range(timepoints - timesteps + 1):
		reshaped_array[i] = input_array[i:i + timesteps]
	reshaped_array = np.reshape(reshaped_array, (-1,timesteps,2,int(features/2)))
	return reshaped_array

def augment_data(x, noise_factor=0.05):
	noise = np.random.randn(*x.shape) * noise_factor
	x_augmented = x + noise
	return x_augmented

def majority_voting_filter(predictions, window_size):
	filtered_predictions = []
	for i in range(len(predictions)):
		start_index = max(0, i - window_size)
		end_index = min(len(predictions), i + window_size + 1)
		window_predictions = predictions[start_index:end_index]
		mode_prediction = mode(window_predictions)[0][0]
		filtered_predictions.append(mode_prediction)
	return np.array(filtered_predictions)

def cylinder_touch_detection(X, model, num_nodes, edge_index, timesteps):
	edge_index = torch.from_numpy(edge_index)
	preds = model(X, edge_index)
	preds = preds.cpu().data.numpy()
	y = np.where(preds>=0.5, 1, 0)
	y = majority_voting_filter(y, 5)
	y = pd.DataFrame(y)
	starts_left = []
	ends_left = []
	starts_right = []
	ends_right = []
	left=0
	right=0
	lc=0
	rc=0
	tc=0
	for i in range(len(y)):
		if y.iloc[i,0]==1 and y.iloc[i,1]==1:
			if left==1 and right==1:
				continue
			elif left==1 and right==0: 
				right=1
				rc+=1
				starts_right.append(i+int((timesteps-1)/2))
			elif left==0 and right==1:
				left=1
				lc+=1
				starts_left.append(i+int((timesteps-1)/2))
			else:
				left=1
				right=1
				lc+=1
				starts_left.append(i+int((timesteps-1)/2))
				rc+=1
				starts_right.append(i+int((timesteps-1)/2))
				
		elif y.iloc[i,0]==0 and y.iloc[i,1]==1:
			if left==1 and right==1:
				left=0
				ends_left.append(i+int((timesteps-1)/2))
			elif left==1 and right==0: 
				left=0
				ends_left.append(i+int((timesteps-1)/2))
				right=1
				rc+=1
				starts_right.append(i+int((timesteps-1)/2))
			elif left==0 and right==1:
				continue
			else:
				right=1
				rc+=1
				starts_right.append(i+int((timesteps-1)/2))
				
		elif y.iloc[i,0]==1 and y.iloc[i,1]==0:
			if left==1 and right==1:
				right=0
				ends_right.append(i+int((timesteps-1)/2))
			elif left==1 and right==0: 
				continue
			elif left==0 and right==1:
				left=1
				lc+=1
				starts_left.append(i+int((timesteps-1)/2))
				right=0
				ends_right.append(i+int((timesteps-1)/2))
			else:
				left=1
				lc+=1
				starts_left.append(i+int((timesteps-1)/2))
		
		else:
			if left==1 and right==1:
				right=0
				ends_right.append(i+int((timesteps-1)/2))
				left=0
				ends_left.append(i+int((timesteps-1)/2))
			elif left==1 and right==0: 
				left=0
				ends_left.append(i+int((timesteps-1)/2))
			elif left==0 and right==1:
				right=0
				ends_right.append(i+int((timesteps-1)/2))
			else:
				continue
		
		tc = rc + lc
	
		if tc>=30:
			break
		else:
			continue
			
	# Create series objects from the arrays
	starts_left = pd.DataFrame(starts_left, columns=['Left_Touch_Start'])
	ends_left = pd.DataFrame(ends_left, columns=['Left_Touch_Ends'])
	starts_right = pd.DataFrame(starts_right, columns=['Right_Touch_Start'])
	ends_right = pd.DataFrame(ends_right, columns=['Right_Touch_Ends'])
	
	datasheet = pd.concat([starts_left, ends_left, starts_right, ends_right], axis=1)
	return datasheet, lc, rc, tc, y