import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import imageio

# Parameters
IMAGE_SIZE = 64
BATCH_SIZE = 128
LATENT_DIM = 100
CHANNELS_IMG = 3
FEATURES_GEN = 64
FEATURES_DISC = 64
EPOCHS = 20   # Increase for better results
lr = 0.0002
beta1, beta2 = 0.5, 0.999
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directories
os.makedirs("outputs/samples", exist_ok=True)

# Data Preparation
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DATA_DIR = "C:\\Users\\akhil\\Downloads\\Anime_Face_Generation_using_GANs\\animefacedataset"
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Models
import torch.nn as nn

# Latent vector size (input noise)
LATENT_DIM = 100
CHANNELS_IMG = 3         # RGB
FEATURES_GEN = 64        # generator feature maps
FEATURES_DISC = 64       # discriminator feature maps

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g, channels_img, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# Initialize Models
gen = Generator(LATENT_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(DEVICE)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

gen.apply(init_weights)
disc.apply(init_weights)

criterion = nn.BCELoss()
optimizer_gen = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_disc = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, beta2))

# Training Loop
if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs("outputs/samples", exist_ok=True)

    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)

    G_losses, D_losses = [], []
    print("Starting Training Loop...")

    for epoch in range(EPOCHS):
        for i, (real, _) in enumerate(dataloader):
            real = real.to(DEVICE)
            b_size = real.size(0)

            # --- Train Discriminator ---
            noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake = gen(noise)

            label_real = torch.full((b_size,), 0.9, device=DEVICE)  # smoothed labels
            label_fake = torch.zeros(b_size, device=DEVICE)

            # Ensure outputs are (batch,)
            output_real = disc(real).mean([1, 2, 3])
            lossD_real = criterion(output_real, label_real)

            output_fake = disc(fake.detach()).mean([1, 2, 3])
            lossD_fake = criterion(output_fake, label_fake)

            lossD = (lossD_real + lossD_fake) / 2
            optimizer_disc.zero_grad()
            lossD.backward()
            optimizer_disc.step()

            # --- Train Generator ---
            label_gen = torch.ones(b_size, device=DEVICE)
            output = disc(fake).mean([1, 2, 3])  # flatten
            lossG = criterion(output, label_gen)

            optimizer_gen.zero_grad()
            lossG.backward()
            optimizer_gen.step()

        # Track losses
        G_losses.append(lossG.item())
        D_losses.append(lossD.item())

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss D: {lossD:.4f}, Loss G: {lossG:.4f}")

        # Save generated samples
        with torch.no_grad():
            fake = gen(fixed_noise).detach().cpu()
        vutils.save_image(fake, f"outputs/samples/epoch_{epoch+1}.png", normalize=True, nrow=8)

    print("Training complete.")


    # --- Save Loss Curves ---
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("outputs/loss_curves.png")
    plt.close()


# Create GIF from epoch samples
images = []
for epoch in range(1, EPOCHS+1):
    img_path = f"outputs/samples/epoch_{epoch}.png"
    if os.path.exists(img_path):
        images.append(imageio.imread(img_path))
if images:
    imageio.mimsave("outputs/training_progress.gif", images, fps=2)
    print("GIF saved at outputs/training_progress.gif")

# Evaluation (IS & FID)
print("Evaluating model...")
gen.eval()
inception = InceptionScore(normalize=True).to(DEVICE)
fid = FrechetInceptionDistance(feature=64).to(DEVICE)

# Fake images
noise = torch.randn(500, LATENT_DIM, 1, 1, device=DEVICE)
fake_images = gen(noise).detach()
fake_images_uint8 = ((fake_images * 0.5 + 0.5) * 255).clamp(0,255).to(torch.uint8)

# Inception Score
inception.update(fake_images_uint8)
is_mean, is_std = inception.compute()

# Real images for FID
for real, _ in dataloader:
    real_uint8 = ((real * 0.5 + 0.5) * 255).clamp(0,255).to(torch.uint8).to(DEVICE)
    fid.update(real_uint8, real=True)
    break  # subset for speed

# Fake images for FID
fid.update(fake_images_uint8.to(DEVICE), real=False)
fid_score = fid.compute()

print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
print(f"FID Score: {fid_score:.2f}")
