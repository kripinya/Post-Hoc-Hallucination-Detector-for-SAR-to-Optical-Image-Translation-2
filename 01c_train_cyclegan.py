"""
CycleGAN Training Script for SAR-to-Optical Image Translation
Architecture: Zhu et al., 2017 - "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
Dataset: SEN12MS (same geographic ROI splits as Pix2Pix for fair comparison)

Key Difference from Pix2Pix:
- CycleGAN does NOT use paired supervision. It treats SAR and Optical as two independent domains.
- Instead of pixel-level L1 loss, it enforces Cycle Consistency: SAR -> Optical -> SAR should reconstruct the original.
- This tests whether unsupervised learning produces more or fewer hallucinations than supervised Pix2Pix.
"""

import os
import json
import torch
import rasterio
import itertools
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import logging
from pathlib import Path

# --- CONFIGURATION ---
TRAIN_SPLIT_JSON = "./splits/train_files.json"
VAL_SPLIT_JSON = "./splits/val_files.json"
CHECKPOINT_FILE = "cyclegan_checkpoint.pth"
FINAL_MODEL = "cyclegan_gen_global.pth"
BEST_MODEL = "cyclegan_gen_best.pth"
OUTPUT_IMAGES = Path("./training_progress_images_cyclegan")

EPOCHS = 15
BATCH_SIZE = 4  # CycleGAN uses 4 generators/discriminators, needs more VRAM per batch
LR = 0.0002
B1 = 0.5
B2 = 0.999
LAMBDA_CYCLE = 10.0   # Cycle consistency weight
LAMBDA_IDENTITY = 5.0 # Identity loss weight (helps preserve color)

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("cyclegan_training.log"),
        logging.StreamHandler()
    ]
)

# ============================
# MODEL ARCHITECTURES
# ============================

class ResidualBlock(nn.Module):
    """Residual block with two conv layers and a skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    """
    ResNet-based Generator (9 residual blocks for 256x256 images).
    Maps from domain A (SAR, 2ch) to domain B (Optical, 3ch) or vice-versa.
    """
    def __init__(self, in_channels, out_channels, n_residual_blocks=9):
        super().__init__()

        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Sigmoid(),  # Output [0, 1] to match our optical normalization
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class CycleGANDiscriminator(nn.Module):
    """PatchGAN Discriminator (70x70 receptive field)."""
    def __init__(self, in_channels):
        super().__init__()

        def block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.model(x)


# ============================
# REPLAY BUFFER (Stabilizes GAN training)
# ============================

class ReplayBuffer:
    """
    Stores previously generated images and randomly replays them to the discriminator.
    This prevents the discriminator from overfitting to the generator's most recent output.
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        result = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if np.random.uniform() > 0.5:
                    i = np.random.randint(0, self.max_size)
                    result.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    result.append(element)
        return torch.cat(result)


# ============================
# DATASET (Identical to Pix2Pix for fair comparison)
# ============================

class SAR2OptDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.sar_files = json.load(f)

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        s1_path = self.sar_files[idx]
        s2_path = s1_path.replace('_s1_', '_s2_').replace('/s1_', '/s2_').replace('\\s1_', '\\s2_')

        with rasterio.open(s1_path) as src:
            sar = src.read().astype(np.float32)

        with rasterio.open(s2_path) as src:
            opt = src.read()
            if opt.shape[0] >= 3:
                opt = opt[[3, 2, 1], :, :] if opt.shape[0] == 13 else opt[:3, :, :]
            opt = opt.astype(np.float32) / 10000.0

        return torch.from_numpy(sar), torch.from_numpy(opt)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting CycleGAN Training on {device}")

    train_dataset = SAR2OptDataset(TRAIN_SPLIT_JSON)
    val_dataset = SAR2OptDataset(VAL_SPLIT_JSON)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # CycleGAN has TWO generators and TWO discriminators
    # G_AB: SAR (2ch) -> Optical (3ch)   [This is the one we care about for hallucination detection]
    # G_BA: Optical (3ch) -> SAR (2ch)   [Only needed for cycle consistency]
    G_AB = CycleGANGenerator(in_channels=2, out_channels=3).to(device)
    G_BA = CycleGANGenerator(in_channels=3, out_channels=2).to(device)
    D_A = CycleGANDiscriminator(in_channels=2).to(device)   # Judges SAR images
    D_B = CycleGANDiscriminator(in_channels=3).to(device)   # Judges Optical images

    # Losses
    criterion_GAN = nn.MSELoss()   # LSGAN loss (more stable than BCE)
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Optimizers (joint optimization of both generators)
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=LR, betas=(B1, B2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=LR, betas=(B1, B2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=LR, betas=(B1, B2))

    # Mixed precision scalers
    scaler_G = torch.amp.GradScaler('cuda')
    scaler_D_A = torch.amp.GradScaler('cuda')
    scaler_D_B = torch.amp.GradScaler('cuda')

    # Replay buffers
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    start_epoch = 0

    # SAFE RESUME LOGIC
    if os.path.exists(CHECKPOINT_FILE):
        logging.info(f"Found checkpoint {CHECKPOINT_FILE}. Restoring ALL states to safely resume...")
        ckpt = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
        G_AB.load_state_dict(ckpt['G_AB'])
        G_BA.load_state_dict(ckpt['G_BA'])
        D_A.load_state_dict(ckpt['D_A'])
        D_B.load_state_dict(ckpt['D_B'])
        optimizer_G.load_state_dict(ckpt['opt_G'])
        optimizer_D_A.load_state_dict(ckpt['opt_D_A'])
        optimizer_D_B.load_state_dict(ckpt['opt_D_B'])
        scaler_G.load_state_dict(ckpt['scaler_G'])
        scaler_D_A.load_state_dict(ckpt['scaler_D_A'])
        scaler_D_B.load_state_dict(ckpt['scaler_D_B'])
        start_epoch = ckpt['epoch'] + 1
        logging.info(f"Resuming from Epoch {start_epoch + 1}")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, EPOCHS):
        G_AB.train(); G_BA.train()
        D_A.train(); D_B.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for real_A, real_B in pbar:
            real_A = real_A.to(device, non_blocking=True)  # SAR
            real_B = real_B.to(device, non_blocking=True)  # Optical

            valid = torch.ones((real_A.size(0), 1, 16, 16), device=device, requires_grad=False)
            fake_label = torch.zeros((real_A.size(0), 1, 16, 16), device=device, requires_grad=False)

            # =====================
            # Train Generators
            # =====================
            optimizer_G.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                # --- BUG FIX: Commented out original identity loss calculation ---
                # Identity loss (helps preserve color when input is already in target domain)
                # id_B = G_AB(real_A)  # G_AB should generate optical-like from SAR
                # id_A = G_BA(real_B)  # G_BA should generate SAR-like from optical
                # loss_identity = (criterion_identity(id_A, real_B) * LAMBDA_IDENTITY * 0.5 +
                #                  criterion_identity(id_B, real_B) * LAMBDA_IDENTITY * 0.5)
                # -----------------------------------------------------------------

                # Note: identity loss on G_AB uses real_B as target only when channels match.
                # Since SAR(2ch) != Optical(3ch), we skip the strict identity and only use
                # the cycle consistency to enforce structure preservation.
                loss_identity = torch.tensor(0.0, device=device)  # Disable identity (channel mismatch)

                # GAN loss
                fake_B = G_AB(real_A)       # SAR -> Fake Optical
                pred_fake_B = D_B(fake_B)
                loss_GAN_AB = criterion_GAN(pred_fake_B, valid)

                fake_A = G_BA(real_B)       # Optical -> Fake SAR
                pred_fake_A = D_A(fake_A)
                loss_GAN_BA = criterion_GAN(pred_fake_A, valid)

                # Cycle consistency loss
                recovered_A = G_BA(fake_B)  # SAR -> Optical -> SAR (should match original SAR)
                loss_cycle_A = criterion_cycle(recovered_A, real_A) * LAMBDA_CYCLE

                recovered_B = G_AB(fake_A)  # Optical -> SAR -> Optical (should match original Optical)
                loss_cycle_B = criterion_cycle(recovered_B, real_B) * LAMBDA_CYCLE

                # Total generator loss
                loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_identity

            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            # =====================
            # Train Discriminator A (judges SAR images)
            # =====================
            optimizer_D_A.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                loss_real_A = criterion_GAN(D_A(real_A), valid)
                fake_A_replay = fake_A_buffer.push_and_pop(fake_A.detach())
                loss_fake_A = criterion_GAN(D_A(fake_A_replay), fake_label)
                loss_D_A = 0.5 * (loss_real_A + loss_fake_A)

            scaler_D_A.scale(loss_D_A).backward()
            scaler_D_A.step(optimizer_D_A)
            scaler_D_A.update()

            # =====================
            # Train Discriminator B (judges Optical images)
            # =====================
            optimizer_D_B.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                loss_real_B = criterion_GAN(D_B(real_B), valid)
                fake_B_replay = fake_B_buffer.push_and_pop(fake_B.detach())
                loss_fake_B = criterion_GAN(D_B(fake_B_replay), fake_label)
                loss_D_B = 0.5 * (loss_real_B + loss_fake_B)

            scaler_D_B.scale(loss_D_B).backward()
            scaler_D_B.step(optimizer_D_B)
            scaler_D_B.update()

            pbar.set_postfix(
                G=loss_G.item(),
                D_A=loss_D_A.item(),
                D_B=loss_D_B.item(),
                Cyc=loss_cycle_A.item() + loss_cycle_B.item()
            )

        # --- Validation & Visual Progress ---
        G_AB.eval()
        val_l1_loss = 0.0
        with torch.no_grad():
            for i, (sar, real_opt) in enumerate(val_loader):
                sar, real_opt = sar.to(device), real_opt.to(device)
                with torch.amp.autocast('cuda'):
                    fake_opt = G_AB(sar)
                    val_l1_loss += nn.L1Loss()(fake_opt, real_opt).item()

                if i == 0:
                    img_sample = torch.cat((real_opt.data[:4], fake_opt.data[:4]), -2)
                    save_image(img_sample, OUTPUT_IMAGES / f"epoch_{epoch + 1}.png", nrow=4, normalize=False)

        val_l1_loss /= len(val_loader)
        logging.info(f"Epoch {epoch + 1} Complete. Validation L1 Pixel Loss: {val_l1_loss:.4f}")

        # --- SAVE CHECKPOINT ---
        ckpt = {
            'epoch': epoch,
            'G_AB': G_AB.state_dict(),
            'G_BA': G_BA.state_dict(),
            'D_A': D_A.state_dict(),
            'D_B': D_B.state_dict(),
            'opt_G': optimizer_G.state_dict(),
            'opt_D_A': optimizer_D_A.state_dict(),
            'opt_D_B': optimizer_D_B.state_dict(),
            'scaler_G': scaler_G.state_dict(),
            'scaler_D_A': scaler_D_A.state_dict(),
            'scaler_D_B': scaler_D_B.state_dict(),
        }
        torch.save(ckpt, CHECKPOINT_FILE)

        # --- SAVE BEST MODEL (lowest val L1 loss) ---
        if val_l1_loss < best_val_loss:
            best_val_loss = val_l1_loss
            torch.save(G_AB.state_dict(), BEST_MODEL)
            logging.info(f"New best model saved at Epoch {epoch + 1}! Val L1: {best_val_loss:.4f} -> {BEST_MODEL}")
        else:
            logging.info("Checkpoint saved safely (All CycleGAN components).")

    # Export the final epoch SAR->Optical generator (for comparison against best)
    torch.save(G_AB.state_dict(), FINAL_MODEL)
    logging.info(f"Training finished! Final model: {FINAL_MODEL} | Best model: {BEST_MODEL} (lowest val L1: {best_val_loss:.4f})")


if __name__ == "__main__":
    main()
