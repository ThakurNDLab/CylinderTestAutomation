import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GCNAttn(nn.Module):
	def __init__(self, hidden_channels, out_channels):
		super(GCNAttn, self).__init__()
		self.conv1 = GATv2Conv(-1, hidden_channels, heads=2, negative_slope=0.1)
		self.conv2 = GATv2Conv(-1, out_channels, heads=2, negative_slope=0.1)
		self.out_channels = out_channels

	def forward(self, x, edge_index):
		x = x.permute(0,2,1)
		batch_size, num_nodes, num_features = x.shape
		h = torch.empty(batch_size, num_nodes, 2 * (self.out_channels), device=x.device)
		for i in range(x.shape[0]):
			graph_features = x[i,:,:]
			hi = self.conv1(graph_features, edge_index)
			hi = self.conv2(hi, edge_index)
			h[i] = hi
		x = h.permute(0,2,1)
		return x

class LSTMAttn(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(LSTMAttn, self).__init__()
		self.hidden_size = hidden_size
		self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
		self.attention = nn.Linear(2 * hidden_size, 1)
		self.leaky = nn.LeakyReLU(0.1)
		self.hidden_state = None
	def reset_hidden_state(self):
		self.hidden_state = None
	def forward(self, x):
		output, (ht, ct) = self.lstm(x)
		weights = torch.softmax(self.attention(output), dim=1)
		lstm_attn_out = torch.sum(weights * output, dim=1)
		lstm_attn_out = self.leaky(lstm_attn_out)
		return lstm_attn_out

class BehaviorModel(nn.Module):
	def __init__(self, lstm_dim, hidden_dim, num_labels):
		super(BehaviorModel, self).__init__()
		self.gcn1 = GCNAttn(hidden_dim, hidden_dim)
		self.gcn2 = GCNAttn(hidden_dim, hidden_dim)
		self.lstm = LSTMAttn(lstm_dim, hidden_dim)
		self.dropout = nn.Dropout(0.2)
		self.classifier = nn.Sequential(
			nn.LazyLinear(512),
			nn.LeakyReLU(0.1),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.1),
			nn.Linear(256, 256),
			nn.LeakyReLU(0.1),
			nn.Dropout(0.2),
			nn.Linear(256, 256),
			nn.LeakyReLU(0.1),
			nn.Dropout(0.2),
			nn.Linear(256, 64),
			nn.LeakyReLU(0.1),
			nn.Linear(64, 32),
			nn.LeakyReLU(0.1),
			nn.Linear(32, num_labels)
		)
		
		self.sigmoid = nn.Sigmoid()

	def reset_hidden_state(self):
		self.lstm.reset_hidden_state()

	def forward(self, x, edge_index):
		xg = x[:,-1,:,:]
		xl = x.view(x.shape[0], x.shape[1], (x.shape[2])*(x.shape[3]))
		xg = self.gcn1(xg, edge_index)
		xg = self.gcn2(xg, edge_index)
		xg = xg.reshape(xg.shape[0], -1)
		xl = self.lstm(xl)
		x = torch.cat((xg, xl), dim=1)
		x = x.view(x.shape[0],-1)
		x = self.classifier(x)
		x = self.sigmoid(x)
		
		return x

