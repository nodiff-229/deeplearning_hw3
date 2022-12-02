import os.path as osp
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


device = torch.device('cpu')
model = GAT(dataset.num_features, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


epoch_train_accuracy = []
epoch_valid_accuracy = []
epoch_test_accuracy = []
num_epochs_array = [i + 1 for i in range(1,201)]
for epoch in range(1, 201):
    train(data)
    train_acc, val_acc, test_acc = test(data)
    epoch_train_accuracy.append(train_acc)
    epoch_valid_accuracy.append(val_acc)
    epoch_test_accuracy.append(test_acc)

    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')


# 绘图
# 绘制训练曲线图
plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(111)
plt.xlabel('epochs')  # x轴标签
plt.ylabel('accuracy')  # y轴标签
# 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
plt.plot(num_epochs_array, epoch_train_accuracy, linewidth=1, linestyle="solid", label="train accuracy")
plt.plot(num_epochs_array, epoch_valid_accuracy, linewidth=1, linestyle="solid", label="valid accuracy", color='black')
plt.plot(num_epochs_array, epoch_test_accuracy, linewidth=1, linestyle="solid", label="test accuracy", color='green')
plt.legend()
plt.title('GAT accuracy curve')

plt.savefig("../result/GAT_result.png")