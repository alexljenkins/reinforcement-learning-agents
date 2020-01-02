from torchvision.utils import save_image
import torch
import torchvision

rand_tensor= torch.rand(64, 3,28,28)

img1 = rand_tensor[0]
# img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
save_image(img1, 'img1.png')

rand_tensor.shape
img1.shape

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion

img = np.array(img1)

m = img - binary_erosion(img)

plt.matshow(m, cmap=plt.cm.gray)

img
