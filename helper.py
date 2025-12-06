import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import json


def test_network(net, trainloader):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Create Variables for the inputs and targets
    inputs = Variable(images)
    targets = Variable(images)

    # Clear the gradients from all Variables
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = net.forward(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    return True


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def view_recon(img, recon):
    ''' Function for displaying an image (as a PyTorch Tensor) and its
        reconstruction also a PyTorch Tensor
    '''

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')

with open("dataset/labels.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

def view_classify(img, ps, top_k=10):
    '''Visualiza una imagen y muestra las top_k predicciones del modelo junto con su correspondencia.'''
    ps = ps.detach().cpu().numpy().squeeze()

    # Ordenar probabilidades de mayor a menor
    topk_idx = np.argsort(ps)[-top_k:][::-1]
    topk_ps = ps[topk_idx]

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.numpy().squeeze(), cmap='gray')
    ax1.axis('off')

    # 🔹 Ajustar las etiquetas al rango 1–101
    class_numbers = topk_idx + 1

    ax2.barh(np.arange(top_k), topk_ps[::-1])
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(top_k))
    ax2.set_yticklabels(class_numbers[::-1])  # ahora muestra 1–101
    ax2.set_title('Top Predicted Classes')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

    # 🔹 Texto de correspondencias correctas
    correspondencias = "\n".join([
        f"{num} = {label_map.get(str(num), '?')}" for num in class_numbers
    ])

    plt.figtext(0.5, 0.02, correspondencias, wrap=True,
                horizontalalignment='center', fontsize=12)

    plt.show()