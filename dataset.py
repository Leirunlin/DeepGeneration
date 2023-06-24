import torch
import torch.distributions as D
from torch.distributions import MixtureSameFamily, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import math

# Define the means and covariance matrix
num_points = 36
radius = 75.0
center_x = 0
center_y = 0
theta = torch.linspace(0, 2 * math.pi, num_points)
x = center_x + radius * torch.cos(theta)
y = center_y + radius * torch.sin(theta)
mean = torch.stack([x, y], dim=1)
covariance = torch.ones(36, 2)* 1  # Identity matrix
mix = D.Categorical(torch.ones(36,))
comp = D.Independent(D.Normal(
       mean, covariance), 1)
gmm = MixtureSameFamily(mix, comp)

# Generate samples from the mixture distribution
train_samples = gmm.sample(sample_shape=(15000,))
torch.save(train_samples, "./data/train/train_data.pt")

val_samples = gmm.sample(sample_shape=(15000,))
torch.save(val_samples, "./data/val/val_data.pt")

test_samples = gmm.sample(sample_shape=(15000,))
torch.save(test_samples, "./data/test/test_data.pt")

plt.scatter(train_samples[:, 0], train_samples[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of samples')
plt.savefig("./figs/train.png")

"""
# Create the component distributions
dist1 = MultivariateNormal(mean1, covariance)
dist2 = MultivariateNormal(mean2, covariance)

# Create the mixture distribution
mixture = MixtureSameFamily(Categorical(torch.tensor([0.5, 0.5])), dist1, dist2)

# Generate samples from the mixture distribution
samples = mixture.sample(sample_shape=(100,))

print(samples)
"""