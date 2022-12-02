import os.path as osp
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F

from torch.nn import Linear, LayerNorm, ReLU
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import DeepGCNLayer,GCNConv

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class DeepGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super(DeepGCN, self).__init__()

        self.node_encoder = Linear(data.x.size(-1), hidden_channels)


        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):  # 层数
            conv = GCNConv(hidden_channels, hidden_channels, cached=True, normalize=True)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)


        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)






device = torch.device('cpu')
model = DeepGCN(hidden_channels=128, num_layers=3).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index, data.edge_attr), []
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
plt.title('DeepGCN accuracy curve')

plt.savefig("../result/DeepGCN_result.png")