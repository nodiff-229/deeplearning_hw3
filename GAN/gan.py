# Setup cell.
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from gan_pytorch import preprocess_img, deprocess_img, rel_error, count_params, ChunkSampler
from gan_pytorch import Flatten, Unflatten, initialize_weights
from gan_pytorch import discriminator
from gan_pytorch import generator
from gan_pytorch import bce_loss, discriminator_loss, generator_loss
from gan_pytorch import get_optimizer, run_a_gan
from gan_pytorch import sample_noise
from gan_pytorch import build_dc_classifier,build_dc_generator

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # Set default size of plots.
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# device = 'mps'
device = 'cuda'

def show_images(images):

    images = np.reshape(images, [images.shape[0], -1])  # Images reshape to (batch_size, D).
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return


# answers = dict(np.load('gan-checks.npz'))
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

NUM_TRAIN = 50000
NUM_VAL = 5000

NOISE_DIM = 96
batch_size = 128

mnist_train = dset.MNIST(
    './datasets/MNIST_data',
    train=True,
    download=True,
    transform=T.ToTensor()
)
loader_train = DataLoader(
    mnist_train,
    batch_size=batch_size,
    sampler=ChunkSampler(NUM_TRAIN, 0)
)

mnist_val = dset.MNIST(
    './datasets/MNIST_data',
    train=True,
    download=True,
    transform=T.ToTensor()
)
loader_val = DataLoader(
    mnist_val,
    batch_size=batch_size,
    sampler=ChunkSampler(NUM_VAL, NUM_TRAIN)
)

imgs = loader_train.__iter__().next()[0].view(batch_size, 784).numpy().squeeze()


# def test_sample_noise():
#     batch_size = 3
#     dim = 4
#     torch.manual_seed(231)
#     z = sample_noise(batch_size, dim)
#     np_z = z.cpu().numpy()
#     assert np_z.shape == (batch_size, dim)
#     assert torch.is_tensor(z)
#     assert np.all(np_z >= -1.0) and np.all(np_z <= 1.0)
#     assert np.any(np_z < 0.0) and np.any(np_z > 0.0)
#     print('All tests passed!')
#
#
# def test_discriminator_loss(logits_real, logits_fake, d_loss_true):
#     d_loss = discriminator_loss(torch.Tensor(logits_real).type(dtype),
#                                 torch.Tensor(logits_fake).type(dtype)).cpu().numpy()
#     print("Maximum error in d_loss: %g" % rel_error(d_loss_true, d_loss))
#
#
# test_discriminator_loss(
#     answers['logits_real'],
#     answers['logits_fake'],
#     answers['d_loss_true']
# )
#
#
# def test_generator_loss(logits_fake, g_loss_true):
#     g_loss = generator_loss(torch.Tensor(logits_fake).type(dtype)).cpu().numpy()
#     print("Maximum error in g_loss: %g" % rel_error(g_loss_true, g_loss))
#
#
# test_generator_loss(
#     answers['logits_fake'],
#     answers['g_loss_true']
# )

# Make the discriminator
# D = discriminator().type(dtype)
# D = discriminator().to(device)
D = build_dc_classifier().to(device)


# Make the generator
# G = generator().type(dtype)
# G = generator().to(device)
G = build_dc_generator().to(device)
# Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
D_solver = get_optimizer(D)
G_solver = get_optimizer(G)

# Run it!
images = run_a_gan(
    D,
    G,
    D_solver,
    G_solver,
    discriminator_loss,
    generator_loss,
    loader_train,
    num_epochs=30
)

show_images(images[-1])
# plt.savefig("../result/GAN.png")
plt.savefig("./checkpoints/result.png")

