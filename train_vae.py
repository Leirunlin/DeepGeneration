import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical
import matplotlib.pyplot as plt

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, device):
        super(VAE, self).__init__()

        self.device=device
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.scale = None
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
        )
        
        # Latent layers
        self.mean = nn.Linear(64, latent_dim)
        self.log_var = nn.Linear(64, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        mean = self.mean(h)
        log_var = self.log_var(h)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        if self.scale is None:
            self.scale = torch.max(x) 
        x = x / self.scale
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decode(z) * self.scale
        return x_hat, mean, log_var

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        samples = self.decode(z) * self.scale
        return samples


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare the data
train_data = torch.load("./data/train/train_data.pt").to(device)
#scale = torch.max(train_data)
train_data = train_data
plt.scatter(train_data[:, 0].cpu().detach().numpy(), train_data[:, 1].cpu().detach().numpy())
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of samples')
plt.savefig(f"./figs/train.png")
plt.close()

# Set hyperparameters
input_dim = 2
latent_dim = 2
epochs = 10000

# Initialize the VAE model
vae = VAE(input_dim, latent_dim, device).to(device)

# Define the loss function
reconstruction_loss = nn.()

# Define the optimizer
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    vae.train()
    optimizer.zero_grad()
    inputs = train_data
    x_hat, mean, log_var = vae(inputs)
    if epoch % 1000 == 0:
        plt.scatter(x_hat[:, 0].cpu().detach().numpy(), x_hat[:, 1].cpu().detach().numpy())
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Scatter plot of samples')
        plt.savefig(f"./figs/train_vae_{epoch}.png")
        plt.close()
    recon_loss = reconstruction_loss(x_hat, inputs)
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    loss = recon_loss + 0.01*kl_loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item()}")


# Generate samples
vae.eval()
generated_samples = vae.sample(15000).to('cpu').detach().numpy()
plt.scatter(generated_samples[:, 0], generated_samples[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of samples')
plt.savefig("./figs/vae.png")

# Evaluate on test set
test_data = torch.load("./data/test/test_data.pt").to(device)
#test_loader = DataLoader(test_data, batch_size=256)

total_loss = 0
with torch.no_grad():
    inpus = test_data
    x_hat, mean, log_var = vae(inputs)
    recon_loss = reconstruction_loss(x_hat, inputs)
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    loss = recon_loss + kl_loss 
    total_loss += loss.item() * inputs.size(0)
        
    average_loss = total_loss / len(test_data)
    print(f"Test Loss: {average_loss}")
