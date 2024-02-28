import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline
import random
import numpy as np
from sklearn.preprocessing import RobustScaler


def compute_velocities(data):
	speed = np.diff(data, axis=1)
	euclidean_speed = np.linalg.norm(speed, axis=2)
	euclidean_speed = euclidean_speed[:, :, np.newaxis, :]
	zero_pad = np.zeros((data.shape[0], 1, 1, data.shape[3]))
	euclidean_speed = np.concatenate([zero_pad, euclidean_speed], axis=1)
	return euclidean_speed

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

def preprocess_data(coord_data):
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

def reshape_array(input_array, timesteps=7):
	timepoints, features = input_array.shape
	new_shape = (timepoints - timesteps + 1, timesteps, features)
	reshaped_array = np.zeros(new_shape)
	for i in range(timepoints - timesteps + 1):
		reshaped_array[i] = input_array[i:i + timesteps]
	reshaped_array = np.reshape(reshaped_array, (-1,timesteps,2,int(features/2)))
	return reshaped_array

def prep_training(coord_data, touch):
	x = preprocess_data(coord_data)
	y = touch.iloc[:,:2]
	x = x.fillna(0)
	y = y.fillna(0)
	x.reset_index(level=None, drop=True, inplace=True)
	y.reset_index(level=None, drop=True, inplace=True)
	x = np.asarray(x)
	y = np.asarray(y)
	y = y[6:, :]
	x = reshape_array(x, 7)
	vel = compute_velocities(x)
	x = np.concatenate([x, vel], axis=2)
	ang = compute_angles_for_sequence(x)
	ang = ang[..., np.newaxis]
	ang = np.repeat(ang, 10, axis=3)
	x = np.concatenate([x, ang], axis=2)

	return x, y

def time_warp(x, intensity=0.8):
	"""
	Apply time warping to a 4D data array.
	"""
	batch_size, timesteps, features, channels = x.shape
	warped = np.zeros_like(x)

	min_length = timesteps//1.5
	intensity = max(intensity, min_length/timesteps)
	new_length = max(int(intensity*timesteps), min_length)
	# Generate new time axis based on intensity
	t = np.linspace(0, 1, timesteps)
	t_new = np.linspace(0, 1, new_length)

	# Apply warping for each series in the dataset
	for i in range(batch_size):
		for j in range(features):
			for k in range(channels):
				# Extract the series
				series = x[i, :, j, k]
				cs = CubicSpline(t, series)
				warped_series = cs(t_new)
				# Assign the warped series back, handling dimensionality mismatch
				resampled_series = np.interp(t, t_new, warped_series)
				warped[i, :, j, k] = resampled_series

	return warped

def jitter(x, noise_factor=0.02):
	"""
	Apply jittering to a data array.
	"""
	noise = np.random.normal(0, noise_factor, x.shape)
	return x + noise

def random_erasing(x, p=0.35, scale=(0.015, 0.05), num_regions=2, erase_val_range=(0, 1)):
    """
    Apply random erasing to a data array with multiple regions and variable erase values.

    """
    if np.random.rand() > p:
        return x
    t, v, c = x.shape[1], x.shape[2], x.shape[3]

    if t == 0 or v == 0 or c ==0:
        return x  # Avoid zero dimension

    for _ in range(num_regions):
        # Ensure valid range for erasing
        lower_t, upper_t = int(t * scale[0]), max(int(t * scale[1]), int(t * scale[0]) + 1)
        lower_v, upper_v = int(v * scale[0]), max(int(v * scale[1]), int(v * scale[0]) + 1)
        lower_c, upper_c = int(c * scale[0]), max(int(c * scale[1]), int(c * scale[0]) + 1)

        erasing_t = np.random.randint(lower_t, upper_t)
        erasing_v = np.random.randint(lower_v, upper_v)
        erasing_c = np.random.randint(lower_c, upper_c)

        t1 = np.random.randint(0, max(t - erasing_t, 1))
        v1 = np.random.randint(0, max(v - erasing_v, 1))
        c1 = np.random.randint(0, max(c - erasing_c, 1))

        # Generate random values for erasing within the specified range
        erasing_values = np.random.uniform(erase_val_range[0], erase_val_range[1], 
                                           (erasing_t, erasing_v, erasing_c))

        x[:, t1:t1+erasing_t, v1:v1+erasing_v, c1:c1+erasing_c] = erasing_values

    return x


def augment_data(x, apply_time_warp=True, apply_jitter=True, apply_random_erasing=True):
	"""
	Augment data
	"""
	# Apply time warping
	if apply_time_warp:
		x_augmented = time_warp(x)

	# Apply jittering
	if apply_jitter:
		x_augmented = jitter(x_augmented)

	# Apply random erasing
	if apply_random_erasing:
		x_augmented = random_erasing(x_augmented)

	return x_augmented


def smoothen_data(arr, min_action_length=12, gap_tolerance=5):
	num_columns = arr.shape[1]
	smoothened_arr = np.zeros_like(arr)

	for column_index in range(num_columns):
		col = arr[:, column_index]
		n = len(col)
		smoothened_col = np.zeros_like(col)

		i = 0
		while i < n:
			if col[i] == 1:
				start = i
				while i < n and col[i] == 1:
					i += 1
				end = i
				gap_count = 0
				while i < n and gap_count <= gap_tolerance and col[i] == 0:
					gap_count += 1
					i += 1
					if i < n and col[i] == 1:
						end = i
				if end - start + gap_count >= min_action_length:
					smoothened_col[start:end] = 1
			i += 1
		smoothened_arr[:, column_index] = smoothened_col
	return smoothened_arr


def find_zero_stretches(B, stretch_length=200):
	stretches = []
	count = 0
	for i in range(len(B)):
		if B[i,0] == 0 and B[i,1] == 0:
			count += 1
			if count >= stretch_length:
				start_index = i - count + 1
		elif B[i,0] == 1 or B[i,1] == 1:
			if count >= stretch_length:
				stretches.append((start_index, i-1))
				count = 0

	return stretches


def fit_scaler(array_4d):
	# Reshape 4D array to 2D
	n_samples, x, y, z = array_4d.shape
	reshaped_array = array_4d.reshape(n_samples, -1)

	# Fit the RobustScaler
	scaler = RobustScaler()
	scaler.fit(reshaped_array)

	return scaler

def transform_scaler(array_4d, scaler):
	original_shape = array_4d.shape
	# Reshape 4D array to 2D
	reshaped_array = array_4d.reshape(original_shape[0], -1)
	# Transform the data
	transformed_array = scaler.transform(reshaped_array)
	# Reshape back to 4D
	transformed_array_4d = transformed_array.reshape(original_shape)
	return transformed_array_4d
