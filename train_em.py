import numpy as np
import torch
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

train_data = torch.load("./data/train/train_data.pt").cpu()
gm = GaussianMixture(n_components=36, random_state=0).fit(train_data.cpu().numpy())

x, y = gm.sample(15000)

plt.scatter(x[:, 0], x[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of samples')
plt.savefig(f"./figs/train_em.png")
plt.close()
