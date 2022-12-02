import os.path as osp

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = Planetoid(path, dataset)
data = dataset[0]

device = torch.device('cpu')
model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                 context_size=10, walks_per_node=10,
                 num_negative_samples=1, p=1, q=1, sparse=True).to(device)

loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask],
                     max_iter=150)
    return acc


epoch_train_loss = []
epoch_test_accuracy = []
num_epochs_array = [i + 1 for i in range(1,101)]

for epoch in range(1, 101):
    loss = train()
    acc = test()
    epoch_train_loss.append(loss)
    epoch_test_accuracy.append(acc)

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')


@torch.no_grad()
def visualize(color_list):
    model.eval()
    z = model(torch.arange(data.num_nodes, device=device))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=color_list[i])
    plt.axis('off')
    plt.savefig('visualization.png')


color_list = [
    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
    '#ffd700'
]
visualize(color_list)
# 绘图
# 绘制训练曲线图
plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(111)
plt.xlabel('epochs')  # x轴标签
plt.ylabel('loss')  # y轴标签
# 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
plt.plot(num_epochs_array, epoch_train_loss, linewidth=1, linestyle="solid", label="train loss")
plt.plot(num_epochs_array, epoch_test_accuracy, linewidth=1, linestyle="solid", label="test accuracy",color='red')


plt.legend()
plt.title('node2vec curve')


plt.savefig("../result/node2vec_result.png")