import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F

from scipy.signal import savgol_filter

from utils import *
from models import *

use_gpu = True

num_actions = 2
num_nodes = 10
timesteps = 7

num_epochs = 500
batch_size = 256

parser = argparse.ArgumentParser(description='GC-LSTM')
args = parser.parse_args()


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == '__main__':

	print('Data Preprocessing...')
	print('')

	X1 = preprocess_data('00001trimDLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv').iloc[:5000]
	X2 = preprocess_data('00002trimDLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv').iloc[:5000]
	X4 = preprocess_data('00004trimDLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv').iloc[:5000]
	X5 = preprocess_data('00005trimDLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv').iloc[:5000]
	X6 = preprocess_data('00006trimDLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv').iloc[:5000]
	X7 = preprocess_data('00007trimDLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv').iloc[:5000]
	X8 = preprocess_data('00008trimDLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv')
	X10 = preprocess_data('00010trimDLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv').iloc[:5000]
	X13 = preprocess_data('00013trimDLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv').iloc[:5000]
	X14 = preprocess_data('00014trimDLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv').iloc[:5000]
	X15 = preprocess_data('00015trimDLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv').iloc[:5000]
	X16 = preprocess_data('00016trimDLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv').iloc[:5000]
	X17 = preprocess_data('00017trimDLC_resnet152_Cylinder Test Trial 3Jan23shuffle1_200000.csv').iloc[:5000]


	X = pd.concat([X1, X2, X4, X5, X6, X7, X8, X10, X13, X14, X15, X16, X17], axis=0)
	Y = pd.concat([pd.read_csv('event_annotation001.csv'),
			   pd.read_csv('event_annotation002.csv'),
			   pd.read_csv('event_annotation004.csv'),
			   pd.read_csv('event_annotation005.csv'),
			   pd.read_csv('event_annotation006.csv'),
			   pd.read_csv('event_annotation007.csv'),
			   pd.read_csv('event_annotation008.csv'),
			   pd.read_csv('event_annotation010.csv'),
			   pd.read_csv('event_annotation013.csv'),
			   pd.read_csv('event_annotation014.csv'),
			   pd.read_csv('event_annotation015.csv'),
			   pd.read_csv('event_annotation016.csv'),
			   pd.read_csv('event_annotation017.csv')
			   ], axis=0)
	Y = Y.iloc[:,:2]
	
	X = X.fillna(0)
	Y = Y.fillna(0)

	zeros_count = 0
	to_delete = []
	X.reset_index(level=None, drop=True, inplace=True)
	Y.reset_index(level=None, drop=True, inplace=True)

	for i, row in Y.iterrows():
		if row.iloc[0] == 0 and row.iloc[1] == 0:
			zeros_count += 1
			if zeros_count == 150:
				to_delete.extend(list(range(i - 149, i + 1)))
				zeros_count = 0
		else:
			zeros_count = 0

	X = X.drop(to_delete, axis=0)
	Y = Y.drop(to_delete, axis=0)

	X.reset_index(level=None, drop=True, inplace=True)
	Y.reset_index(level=None, drop=True, inplace=True)

	X = np.asarray(X)
	Y = np.asarray(Y)

	train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.8, random_state=seed)
	scaler = RobustScaler()
	train_x = scaler.fit_transform(train_x)
	test_x = scaler.transform(test_x)

	train_y = train_y[int(timesteps-1):, :]
	test_y = test_y[int(timesteps-1):, :]

	train_x = reshape_array(train_x, timesteps)
	test_x = reshape_array(test_x, timesteps)

	train_velocities = compute_velocities(train_x)
	test_velocities = compute_velocities(test_x)
	train_x = np.concatenate([train_x, train_velocities], axis=2)
	test_x = np.concatenate([test_x, test_velocities], axis=2)

	train_angles = compute_angles_for_sequence(train_x)
	train_angles = train_angles[..., np.newaxis]
	train_angles = np.repeat(train_angles, 10, axis=3)
	test_angles = compute_angles_for_sequence(test_x)
	test_angles = test_angles[..., np.newaxis]
	test_angles = np.repeat(test_angles, 10, axis=3)
	train_x = np.concatenate([train_x, train_angles], axis=2)
	test_x = np.concatenate([test_x, test_angles], axis=2)

	edge_index = np.array([[0,1,2,3,4,5,6,7,8],
					[9,2,9,4,9,6,9,8,9]
					], dtype=np.int_)

	edge_index = torch.from_numpy(edge_index)
	
	train_x =torch.from_numpy(train_x)
	train_y =torch.from_numpy(train_y)
	test_x =torch.from_numpy(test_x)
	test_y =torch.from_numpy(test_y)

	train = torch.utils.data.TensorDataset(train_x, train_y)
	train_loader = DataLoader(train, batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
	test = torch.utils.data.TensorDataset(test_x, test_y)
	test_loader = DataLoader(test, batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))

	print('Starting Training...')
	print('')

	lstm_dim = (train_x.shape[2])*num_nodes
	model = BehaviorModel(lstm_dim=lstm_dim, hidden_dim=256, num_labels=num_actions)
	model = model.float().cuda()

	log_interval=10

	optimizer= torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.95, 0.999), weight_decay=1e-4, amsgrad=True)
	scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.001, mode='triangular2', gamma=0.9, cycle_momentum=False)

	loss_criterion = nn.BCELoss()

	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	writer = SummaryWriter('runs/model_{}'.format(timestamp))

	epoch_number = 0
	best_vloss = 1000000

	train_losses = []
	test_losses = []
	train_f1_scores = []
	test_f1_scores = []
	train_auc_scores = []
	test_auc_scores = []

	for epoch in range(num_epochs):
		print('EPOCH {} / {}:'.format((epoch_number + 1), num_epochs))
		# Train
		model.train()
		
		batch = 0
		running_loss = 0
		last_loss = 0
		running_vloss =0
		last_vloss =0
		for  x, y in train_loader:
			optimizer.zero_grad()
			inputs = augment_data(x.numpy(), noise_factor=0.05)
			inputs = torch.Tensor(inputs)
			outputs = model(inputs.float().cuda(), edge_index.cuda())
			loss = loss_criterion(outputs, y.float().cuda())
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()

			running_loss += loss.item()

			labels = y.data.numpy()

			outputs = outputs.cpu().data.numpy()

			tauc_score = average_precision_score(labels, outputs, average='weighted')

			outputs = np.where(outputs>=0.5, 1, 0)

			tf1_score = f1_score(labels, outputs, average='weighted', zero_division=0)
			
			batch +=1

			if batch % log_interval == log_interval-1:
				last_loss = running_loss / log_interval
				print('  batch {}  training loss: {} and training F1_score: {}'.format(batch + 1, last_loss, tf1_score))
				tb_x = i * len(train_loader) + i + 1
				writer.add_scalar('Loss/train', last_loss, tb_x)
				running_loss = 0

		train_losses.append(last_loss)
		train_f1_scores.append(tf1_score)
		train_auc_scores.append(tauc_score)

		scheduler.step()

		batch = 0
		# Test
		model.eval()
		with torch.no_grad():
			for  x, y in test_loader:

				voutputs = model(x.float().cuda(), edge_index.cuda())
				vloss = loss_criterion(voutputs, y.float().cuda())

				running_vloss += vloss.item()

				vlabels = y.data.numpy()

				voutputs = voutputs.cpu().data.numpy()

				vauc_score = average_precision_score(vlabels, voutputs, average='weighted')

				voutputs = np.where(voutputs>=0.5, 1, 0)

				vf1_score = f1_score(vlabels, voutputs, average='weighted', zero_division=0)

				batch += 1

				if batch % log_interval == log_interval-1:
					last_vloss = running_vloss / log_interval # loss per batch
					print('  batch {}  test loss: {} and test F1_score: {}'.format(batch + 1, last_vloss, vf1_score))
					vb_x = i * len(train_loader) + i + 1
					writer.add_scalar('Loss/train', last_vloss, vb_x)
					running_loss = 0

			test_losses.append(last_vloss)
			test_f1_scores.append(vf1_score)
			test_auc_scores.append(vauc_score)

		writer.add_scalars('Training vs. Test Loss',
					{ 'Training' : last_loss, 'Testing' : last_vloss },
					epoch_number + 1)
		writer.flush()

		# Track best performance, and save the model's state
		if last_vloss < best_vloss:
			best_vloss = last_vloss
			path = 'model/model{}_{}'.format(timestamp, (epoch_number+1))
			torch.save(model.state_dict(), path)
			
		epoch_number += 1

		print('')

	final_path = 'model.pt'
	model.load_state_dict(torch.load(path))
	
	model.eval()
	with torch.no_grad():

		cmoutputs = model(test_x.float().cuda(), edge_index.cuda())

		cmoutputs = cmoutputs.cpu().data.numpy()
		cmoutputs = np.where(cmoutputs>=0.5, 1, 0)

		cm = multilabel_confusion_matrix(test_y.data.numpy(), cmoutputs)
		f, axes = plt.subplots(1, 2)
		f.tight_layout(pad=5.0)
		axes = axes.ravel()
		for i in range(2):
			disp = ConfusionMatrixDisplay(confusion_matrix=cm[i,:])
			disp.plot(ax=axes[i], values_format='.4g')
				
	disp.figure_.savefig('conf_mat.tif',dpi=600)

	f.clear()
	plt.close(f)

	plt.figure(figsize=(10, 5))
	plt.plot(range(num_epochs), train_losses, label='Train')
	plt.plot(range(num_epochs), test_losses, label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('Loss')
	plt.savefig('loss_plot.tif',dpi=600)

	plt.figure(figsize=(10, 5))
	plt.plot(range(num_epochs), train_f1_scores, label='Train')
	plt.plot(range(num_epochs), test_f1_scores, label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('F1 Score')
	plt.legend()
	plt.title('F1 Score')
	plt.savefig('f1_plot.tif',dpi=600)

	plt.figure(figsize=(10, 5))
	plt.plot(range(num_epochs), train_auc_scores, label='Train')
	plt.plot(range(num_epochs), test_auc_scores, label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('AUC Score')
	plt.legend()
	plt.title('AUC Score')
	plt.savefig('auc_plot.tif',dpi=600)


	plt.clf()
	
	plt.figure(figsize=(10, 5))
	plt.plot(range(num_epochs), savgol_filter(train_losses, window_length=5, polyorder=2), label='Train')
	plt.plot(range(num_epochs), savgol_filter(test_losses, window_length=5, polyorder=2), label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('Loss')
	plt.savefig('loss_plot_smooth.tif',dpi=600)

	plt.figure(figsize=(10, 5))
	plt.plot(range(num_epochs), savgol_filter(train_f1_scores, window_length=5, polyorder=2), label='Train')
	plt.plot(range(num_epochs), savgol_filter(test_f1_scores, window_length=5, polyorder=2), label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('F1 Score')
	plt.legend()
	plt.title('F1 Score')
	plt.savefig('f1_plot_smooth.tif',dpi=600)

	plt.figure(figsize=(10, 5))
	plt.plot(range(num_epochs), savgol_filter(train_auc_scores, window_length=5, polyorder=2), label='Train')
	plt.plot(range(num_epochs), savgol_filter(test_auc_scores, window_length=5, polyorder=2), label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('AUC Score')
	plt.legend()
	plt.title('AUC Score')
	plt.savefig('auc_plot_smooth.tif',dpi=600)


	plt.clf()

	torch.save(model.state_dict(), final_path)
	
	print('Training complete...')
	print('')

