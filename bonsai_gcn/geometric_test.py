from trainer import GNNTrainer

encoding = {'layertype1':['gatconv', 'gcnconv'], 'layertype2':'gatconv', 'act1':'sigmoid', 'width1':256, 'skips': ['no','yes'], 'merge':'add'}
hyperparams = {'lr':0.01, 'wd':0.00001, 'opt':'adam', 'dropout':0.5}

t = GNNTrainer(encoding, hyperparams)
results = t.train()
print(results)
# from torch_geometric.datasets import Planetoid

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, GraphConv, GATConv, SAGEConv

# dataset = Planetoid(root='/tmp/Cora', name='CiteSeer')

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()

#         self.combine_method = 'cat'

#         # Layer block one
#         self.conv1 = SAGEConv(dataset.num_node_features, 32)
#         self.conv1_2 = GATConv(dataset.num_node_features, 32, heads=8, concat=False)

#         # Layer block two
#         self.conv2 = GCNConv(32, dataset.num_classes)

#         # Weighted combine layer
#         # The input size here needs to be output features * num layers
#         self.combine = nn.Linear(64, 32)

#     def forward(self, data):
#         data_x, edge_index = data.x, data.edge_index

#         x_1 = self.conv1(data_x, edge_index)
#         x_2 = self.conv1_2(data_x, edge_index)
#         if self.combine_method == 'add':
#             x = x_1 + x_2
#             x = F.relu(x)

#         elif self.combine_method == 'cat':
#             x = torch.cat([F.relu(x_1), F.tanh(x_2)], dim=1)
#             # x = F.relu(x)
#             x = self.combine(x)
#             x = F.relu(x)

#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)

#         return F.log_softmax(x, dim=1)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
# data = dataset[0].to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# print(f'Model has {sum([x.numel() for x in model.parameters()])} parameters')
# model.train()

# for epoch in range(500):

#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()

#     print(loss.item())

# model.eval()
# _, pred = model(data).max(dim=1)
# correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
# acc = correct / data.test_mask.sum().item()
# print('Accuracy: {:.4f}'.format(acc))