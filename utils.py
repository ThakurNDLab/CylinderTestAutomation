import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

def process_action_column(data, threshold):
    change_points = np.where(np.diff(data) != 0)[0] + 1
    start_point = 0
    for end_point in change_points:
        if (end_point - start_point) <= threshold:
            data[start_point:end_point] = data[start_point - 1]
        start_point = end_point
    if (len(data) - start_point) <= threshold:
        data[start_point:] = data[start_point - 1]
    return data

def process_actions(data, threshold=3):
	action_1 = process_action_column(data[:, 0], threshold)
	action_2 = process_action_column(data[:, 1], threshold)
	return np.column_stack((action_1, action_2))