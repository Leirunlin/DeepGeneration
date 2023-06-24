import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Generator,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self,z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        output = self.net(x)
        return output.view(-1, 1).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = torch.load("./data/train/train_data.pt").to(device)

data_dim = 2
netG = Generator(input_dim=data_dim, hidden_dim=64).to(device)
netD = Discriminator(input_dim=data_dim, hidden_dim=64).to(device)
optimizerG = optim.Adam(netG.parameters(), lr=0.0002)
optimizerD = optim.Adam(netD.parameters(), lr=0.0002)
criterion = nn.BCELoss()
batch_size = 15000
fixed_noise = torch.randn(batch_size, 2, device=device)
real_label = 1
fake_label = 0

for epoch in range(100000):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real
    netD.zero_grad()
    real_cpu = train_data.to(device)
    label = torch.full((batch_size,), real_label,
                        dtype=real_cpu.dtype, device=device)

    output = netD(real_cpu)
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()

    # train with fake
    noise = torch.randn(batch_size, data_dim, device=device)
    fake = netG(noise)
    label.fill_(fake_label)
    output = netD(fake.detach())
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    output = netD(fake)
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    optimizerG.step()

    print(errD.item())

    if epoch % 5000 == 0:
        x = netG(fixed_noise).cpu().detach().numpy()
        plt.scatter(x[:, 0], x[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Scatter plot of samples')
        plt.savefig(f"./figs/train_gan_{epoch}.png")
        plt.close()

