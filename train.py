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
from datetime import datetime
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
from scipy.signal import savgol_filter
from scripts.utils import *
from scripts.models import *
import pickle

use_gpu = True

num_actions = 2
num_nodes = 10

num_epochs = 200
batch_size = 512

def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

random_seed = 42
set_seed(random_seed)

def train(model_dir, snap, X, Y):

	print('Data Preprocessing...')
	print('')

	xall = np.empty((0, 7, 15, 10))
	yall = np.empty((0, 2))

	for i, x in enumerate(X):
		x,y = prep_training(pd.read_csv(x), pd.read_csv(Y[i-1]))
		xall = np.concatenate([xall, x])
		yall = np.concatenate([yall, y])

	print('Preprocessing Complete!')
	print('')
	print('Train/Test split and scaling...')
	print('')

	train_x, test_x, train_y, test_y = train_test_split(xall, yall, train_size=0.8, random_state=random_seed)
	scaler = fit_scaler(train_x)
	train_x = transform_scaler(train_x, scaler)
	test_x = transform_scaler(test_x, scaler)
	scaler_path = os.path.join(model_dir, "classifier", "scaler.pkl")
	with open(scaler_path, 'wb') as file:
		pickle.dump(scaler, file)

	trainstretches = find_zero_stretches(train_y)
	teststretches = find_zero_stretches(test_y)

	id_trX = []
	id_trY = []
	id_teX = []
	id_teY = []

	for start, end in trainstretches:
		id_trY.extend(range(start, end))
		id_trX.extend(range(start, end))

	train_x = np.delete(train_x, id_trX, axis=0)
	train_y = np.delete(train_y, id_trY, axis=0)

	for start, end in teststretches:
		id_teY.extend(range(start, end))
		id_teX.extend(range(start, end))

	test_x = np.delete(test_x, id_teX, axis=0)
	test_y = np.delete(test_y, id_teY, axis=0)


	edge_index = np.array([[0,1,2,3,4,5,6,7,8],
					[9,2,9,4,9,6,9,8,9]
					], dtype=np.int_)

	edge_index = torch.from_numpy(edge_index)
	
	train_x =torch.from_numpy(train_x)
	train_y =torch.from_numpy(train_y)
	test_x =torch.from_numpy(test_x)
	test_y =torch.from_numpy(test_y)

	train = torch.utils.data.TensorDataset(train_x, train_y)
	train_loader = DataLoader(train, batch_size, shuffle=True, generator=torch.Generator().manual_seed(random_seed))
	test = torch.utils.data.TensorDataset(test_x, test_y)
	test_loader = DataLoader(test, batch_size, shuffle=True, generator=torch.Generator().manual_seed(random_seed))

	print('Starting Training...')
	print('')

	lstm_dim = (train_x.shape[2])*num_nodes
	model = BehaviorModel(lstm_dim=lstm_dim, hidden_dim=256, num_labels=num_actions)
	model.load_state_dict(torch.load(snap))
	# model.freeze_layers()
	model = model.float().cuda()

	log_interval=1

	optimizer= torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.95, 0.999), weight_decay=1e-4, amsgrad=True)
	scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.001, mode='triangular2', gamma=0.9, cycle_momentum=False)

	loss_criterion = nn.BCELoss()

	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

	epoch_number = 0
	best_vloss = 1000000
	best_f1_score = 0
	best_auc_score = 0
	patience = 200
	no_improvement_counter = 0

	train_losses = []
	test_losses = []
	train_f1_scores = []
	test_f1_scores = []
	train_auc_scores = []
	test_auc_scores = []

	thresh_epoch = []

	for epoch in range(num_epochs):
		print('EPOCH {} / {}:'.format((epoch_number + 1), num_epochs))
		# Train
		model.train()
		thresh=[]
		batch = 0
		running_loss = 0
		last_loss = 0
		running_vloss =0
		last_vloss =0
		for  x, y in train_loader:
			optimizer.zero_grad()
			inputs= augment_data(x.numpy())
			inputs = torch.Tensor(inputs)
			outputs = model(inputs.float().cuda(), edge_index.cuda())
			loss = loss_criterion(outputs, y.float().cuda())
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()

			running_loss += loss.item()

			outputs = outputs.cpu().data.numpy()
			labels = y.cpu().data.numpy()

			thresholds = np.linspace(0.01, 0.99, 99)
			f1_scores = [f1_score(labels, np.where(outputs >= t, 1, 0), average='weighted', zero_division=0) for t in thresholds]
			auc_scores = [average_precision_score(labels, np.where(outputs >= t, 1, 0), average='weighted') for t in thresholds]
			scores = f1_scores + auc_scores
			best_index = np.argmax(scores)
			best_t = thresholds[best_index]
			thresh.append(best_t)

			tf1_score = f1_scores[best_index]
			tauc_score = auc_scores[best_index]

			batch +=1

			if batch % log_interval == log_interval-1:
				last_loss = running_loss / log_interval
				print('  batch {}  training loss: {} and training F1_score: {}'.format(batch + 1, last_loss, tf1_score))
				tb_x = batch * len(train_loader) + batch + 1
				print('Loss/train', last_loss, tb_x)
				running_loss = 0

		train_losses.append(last_loss)
		train_f1_scores.append(tf1_score)
		train_auc_scores.append(tauc_score)

		scheduler.step()

		avg_t = sum(thresh) / len(thresh)

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

				voutputs = np.where(voutputs >= avg_t, 1, 0)

				vauc_score = average_precision_score(vlabels, voutputs, average='weighted')

				vf1_score = f1_score(vlabels, voutputs, average='weighted', zero_division=0)

				batch += 1

				if batch % log_interval == log_interval-1:
					last_vloss = running_vloss / log_interval # loss per batch
					print('  batch {}  test loss: {} and test F1_score: {}'.format(batch + 1, last_vloss, vf1_score))
					vb_x = batch * len(train_loader) + batch + 1
					print('Loss/test', last_vloss, vb_x)
					running_loss = 0

			test_losses.append(last_vloss)
			test_f1_scores.append(vf1_score)
			test_auc_scores.append(vauc_score)


		os.makedirs(os.path.join(model_dir, "classifier", "model"), exist_ok=True)

		improvement = False
		if last_vloss < best_vloss:
			best_vloss = last_vloss
			chosen_thresh = avg_t
			improvement = True
		if vf1_score > best_f1_score:
			best_f1_score = vf1_score
			chosen_thresh = avg_t
			improvement = True
		if vauc_score > best_auc_score:
			best_auc_score = vauc_score
			chosen_thresh = avg_t
			improvement = True
		if improvement:
			no_improvement_counter = 0
			path = os.path.join(model_dir, "classifier", "model/model{}_{}".format(timestamp, (epoch_number + 1)))
			torch.save(model.state_dict(), path)
		else:
			no_improvement_counter +=1

		if no_improvement_counter >= patience:
			print(f'Early stopping triggered after {epoch_number + 1} epochs')
			break

		epoch_number += 1
		
		print('')

	final_path = os.path.join(model_dir, "classifier", "model.pt")
	model.load_state_dict(torch.load(path))
	plot_path = os.path.join(model_dir, "classifier", "graphs")
	os.makedirs(plot_path, exist_ok=True)
	
	model.eval()
	with torch.no_grad():

		cmoutputs = model(test_x.float().cuda(), edge_index.cuda())

		cmoutputs = cmoutputs.cpu().data.numpy()
		cmoutputs = np.where(cmoutputs>=chosen_thresh, 1, 0)

		cm = multilabel_confusion_matrix(test_y.data.numpy(), cmoutputs)
		f, axes = plt.subplots(1, 2)
		f.tight_layout(pad=5.0)
		axes = axes.ravel()
		for i in range(2):
			disp = ConfusionMatrixDisplay(confusion_matrix=cm[i,:])
			disp.plot(ax=axes[i], values_format='.4g')
				
	disp.figure_.savefig(os.path.join(plot_path, "conf.tif"),dpi=600)

	f.clear()
	plt.close(f)

	plt.figure(figsize=(10, 5))
	plt.plot(range(epoch_number), train_losses, label='Train')
	plt.plot(range(epoch_number), test_losses, label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('Loss')
	plt.savefig(os.path.join(plot_path, "loss.tif"),dpi=600)

	plt.figure(figsize=(10, 5))
	plt.plot(range(epoch_number), train_f1_scores, label='Train')
	plt.plot(range(epoch_number), test_f1_scores, label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('F1 Score')
	plt.legend()
	plt.title('F1 Score')
	plt.savefig(os.path.join(plot_path, "f1.tif"),dpi=600)

	plt.figure(figsize=(10, 5))
	plt.plot(range(epoch_number), train_auc_scores, label='Train')
	plt.plot(range(epoch_number), test_auc_scores, label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('AUC Score')
	plt.legend()
	plt.title('AUC Score')
	plt.savefig(os.path.join(plot_path, "auc.tif"),dpi=600)


	plt.clf()
	
	plt.figure(figsize=(10, 5))
	plt.plot(range(epoch_number), savgol_filter(train_losses, window_length=6, polyorder=3), label='Train')
	plt.plot(range(epoch_number), savgol_filter(test_losses, window_length=6, polyorder=3), label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('Loss')
	plt.savefig(os.path.join(plot_path, "loss-smooth.tif"),dpi=600)

	plt.figure(figsize=(10, 5))
	plt.plot(range(epoch_number), savgol_filter(train_f1_scores, window_length=6, polyorder=3), label='Train')
	plt.plot(range(epoch_number), savgol_filter(test_f1_scores, window_length=6, polyorder=3), label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('F1 Score')
	plt.legend()
	plt.title('F1 Score')
	plt.savefig(os.path.join(plot_path, "f1-smooth.tif"),dpi=600)

	plt.figure(figsize=(10, 5))
	plt.plot(range(epoch_number), savgol_filter(train_auc_scores, window_length=6, polyorder=3), label='Train')
	plt.plot(range(epoch_number), savgol_filter(test_auc_scores, window_length=6, polyorder=3), label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('AUC Score')
	plt.legend()
	plt.title('AUC Score')
	plt.savefig(os.path.join(plot_path, "auc-smooth.tif"),dpi=600)


	plt.clf()

	torch.save(model.state_dict(), final_path)

	threshold_file = os.path.join(model_dir, "classifier", "model.thresh")
	with open(threshold_file, 'w') as file:
		file.write(f"{chosen_thresh}")
	
	print('Training complete...')
	print('')

