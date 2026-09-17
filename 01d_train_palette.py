"""
Palette (Conditional DDPM) Training Script for SAR-to-Optical Image Translation
Architecture: Saharia et al., 2022 - "Palette: Image-to-Image Diffusion Models" (Google Research)
Dataset: SEN12MS (same geographic ROI splits as Pix2Pix and CycleGAN for fair comparison)

Key Difference from GANs:
- Instead of adversarial training, Palette learns to iteratively denoise a noisy image.
- The SAR image is concatenated with the noisy optical image as conditioning input.
- During inference, the model starts from pure Gaussian noise and iteratively removes it
  over T timesteps to produce a clean optical image, guided by the SAR conditioning.
- This produces highly photorealistic outputs but is much slower at inference time.
"""

import os
import json
import math
import copy
import torch
import rasterio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import logging
from pathlib import Path

# --- CONFIGURATION ---
TRAIN_SPLIT_JSON = "./splits/train_files.json"
VAL_SPLIT_JSON = "./splits/val_files.json"
CHECKPOINT_FILE = "palette_checkpoint.pth"
FINAL_MODEL = "palette_gen_global.pth"
BEST_MODEL = "palette_gen_best.pth"
OUTPUT_IMAGES = Path("./training_progress_images_palette")

EPOCHS = 1        # Changed to 1 for initial test run
BATCH_SIZE = 16   # Increased from 8 — if OOM, drop back to 8
LR = 1e-4         # Lower LR than GANs (standard for diffusion)
T = 1000          # Number of diffusion timesteps (standard)
BETA_START = 1e-4
BETA_END = 0.02
INFERENCE_STEPS = 25  # Guided DDIM preview steps
EMA_DECAY = 0.9999    # Exponential Moving Average smoothing factor
GRAD_CLIP_NORM = 1.0  # Max gradient norm to prevent NaN spikes

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("palette_training.log"),
        logging.StreamHandler()
    ]
)


# ============================
# DIFFUSION SCHEDULE
# ============================

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from Ho et al., 2020."""
    return torch.linspace(beta_start, beta_end, timesteps)


def get_diffusion_params(betas):
    """Pre-compute all diffusion parameters from the beta schedule."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'posterior_variance': posterior_variance,
    }


# ============================
# UNET MODEL (Palette-style conditional denoiser)
# ============================

class SinusoidalPositionEmbeddings(nn.Module):
    """Encodes the diffusion timestep as a sinusoidal embedding vector."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ConvBlock(nn.Module):
    """Double convolution block with GroupNorm."""
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        else:
            self.time_mlp = None

    def forward(self, x, t_emb=None):
        h = self.act(self.norm1(self.conv1(x)))
        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h


class AttentionBlock(nn.Module):
    """Self-attention block for the UNet bottleneck."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Force attention computation in FP32 to prevent half-precision dot product overflow
        with torch.amp.autocast('cuda', enabled=False):
            q_f = q.float()
            k_f = k.float()
            v_f = v.float()
            attn = torch.einsum('bci,bcj->bij', q_f, k_f) * self.scale
            attn = attn.softmax(dim=-1)
            out = torch.einsum('bij,bcj->bci', attn, v_f).to(x.dtype).reshape(B, C, H, W)

        return x + self.proj(out)


class PaletteUNet(nn.Module):
    """
    Conditional UNet for Palette diffusion.
    Input: concatenation of [noisy_optical (3ch), sar_condition (2ch)] = 5 channels
    Output: predicted noise (3 channels, same size as optical)
    """
    def __init__(self, in_channels=5, out_channels=3, time_emb_dim=128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64, time_emb_dim)
        self.enc2 = ConvBlock(64, 128, time_emb_dim)
        self.enc3 = ConvBlock(128, 256, time_emb_dim)
        self.enc4 = ConvBlock(256, 512, time_emb_dim)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot = ConvBlock(512, 1024, time_emb_dim)
        self.attn = AttentionBlock(1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512, time_emb_dim)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64, time_emb_dim)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        e4 = self.enc4(self.pool(e3), t_emb)

        # Bottleneck
        b = self.bot(self.pool(e4), t_emb)
        b = self.attn(b)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1), t_emb)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)

        return self.final(d1)


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

        # 1. Normalize Sentinel-1 SAR dB to [-1.0, 1.0]
        with rasterio.open(s1_path) as src:
            sar = src.read().astype(np.float32)
            vv = np.clip(sar[0:1], -25.0, 0.0)
            vh = np.clip(sar[1:2], -32.5, 0.0)
            vv_norm = 2.0 * (vv - (-25.0)) / 25.0 - 1.0
            vh_norm = 2.0 * (vh - (-32.5)) / 32.5 - 1.0
            sar_norm = np.concatenate([vv_norm, vh_norm], axis=0).astype(np.float32)

        # 2. Normalize Sentinel-2 Surface Reflectance [0, 10000] to [-1.0, 1.0]
        with rasterio.open(s2_path) as src:
            opt = src.read()
            if opt.shape[0] >= 3:
                opt = opt[[3, 2, 1], :, :] if opt.shape[0] == 13 else opt[:3, :, :]
            opt_scaled = (opt.astype(np.float32) / 10000.0) * 2.0 - 1.0
            opt_norm = np.clip(opt_scaled, -1.0, 1.0).astype(np.float32)

        return torch.from_numpy(sar_norm), torch.from_numpy(opt_norm)


# ============================
# FORWARD DIFFUSION (Add noise)
# ============================

def q_sample(x_start, t, diffusion_params, noise=None):
    """Add noise to x_start at timestep t (forward diffusion process)."""
    if noise is None:
        noise = torch.randn_like(x_start)

    t_cpu = t.cpu()  # diffusion schedule tensors are on CPU; index with CPU t
    sqrt_alphas_cumprod_t = diffusion_params['sqrt_alphas_cumprod'][t_cpu][:, None, None, None].to(x_start.device)
    sqrt_one_minus_t = diffusion_params['sqrt_one_minus_alphas_cumprod'][t_cpu][:, None, None, None].to(x_start.device)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_t * noise


# ============================
# REVERSE DIFFUSION (DDPM Sampling)
# ============================

@torch.no_grad()
def p_sample_loop(model, sar_condition, diffusion_params, device, num_steps=25, guidance_scale=2.0):
    """
    DDIM deterministic sampling with Classifier-Free Guidance (CFG).
    Operates stably in [-1.0, 1.0] space.
    """
    b = sar_condition.shape[0]
    img = torch.randn(b, 3, 256, 256, device=device)
    T = len(diffusion_params['alphas_cumprod'])

    step_size = max(1, T // num_steps)
    timesteps = list(range(T - 1, -1, -step_size))
    if timesteps[-1] != 0:
        timesteps.append(0)

    alphas_cumprod = diffusion_params['alphas_cumprod'].to(device)
    null_sar = torch.zeros_like(sar_condition)

    for idx, t_val in enumerate(timesteps):
        t = torch.full((b,), t_val, device=device, dtype=torch.long)

        # CFG: predict conditional and unconditional noise
        if guidance_scale > 1.0:
            model_input = torch.cat([
                torch.cat([img, sar_condition], dim=1),
                torch.cat([img, null_sar], dim=1)
            ], dim=0)
            t_both = torch.cat([t, t], dim=0)
            pred_both = model(model_input, t_both)
            pred_cond, pred_uncond = pred_both.chunk(2, dim=0)
            predicted_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            predicted_noise = model(torch.cat([img, sar_condition], dim=1), t)

        alpha_t = alphas_cumprod[t_val]

        # Predicted clean image x_0 in [-1.0, 1.0]
        pred_x0 = (img - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        if t_val == 0:
            img = pred_x0
        else:
            t_prev_val = timesteps[idx + 1]
            alpha_t_prev = alphas_cumprod[t_prev_val]
            img = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1.0 - alpha_t_prev) * predicted_noise

    return img.clamp(-1.0, 1.0)


# ============================
# MAIN TRAINING LOOP
# ============================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Palette with Proper Normalization & CFG")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Total epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting Palette (Conditional DDPM) Training on {device} with [-1, 1] Normalization, EMA & CFG")

    train_dataset = SAR2OptDataset(TRAIN_SPLIT_JSON)
    val_dataset = SAR2OptDataset(VAL_SPLIT_JSON)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Initialize model: input = [noisy_optical(3) + sar(2)] = 5 channels, output = predicted noise (3 channels)
    model = PaletteUNet(in_channels=5, out_channels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    # EMA Model: maintained as a smoothed copy for evaluation and final deployment
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    # Pre-compute diffusion schedule
    betas = linear_beta_schedule(T, BETA_START, BETA_END)
    diffusion_params = get_diffusion_params(betas)

    start_epoch = 0
    best_val_loss = float('inf')

    # SAFE RESUME LOGIC (Restores all optimizer, scaler, EMA, and best_val_loss states)
    if args.resume and os.path.exists(CHECKPOINT_FILE):
        logging.info(f"Found checkpoint {CHECKPOINT_FILE}. Restoring ALL states to safely resume...")
        ckpt = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])

        if 'ema_model' in ckpt:
            ema_model.load_state_dict(ckpt['ema_model'])
            logging.info("EMA weights restored from checkpoint.")
        else:
            ema_model.load_state_dict(model.state_dict())
            logging.info("No EMA in checkpoint. Initialized EMA from model weights.")

        if 'best_val_loss' in ckpt:
            best_val_loss = ckpt['best_val_loss']
            logging.info(f"Restored best validation loss: {best_val_loss:.6f}")

        start_epoch = ckpt['epoch'] + 1
        logging.info(f"Resuming from Epoch {start_epoch + 1}")
    elif os.path.exists(CHECKPOINT_FILE) and not args.resume:
        legacy_backup = "palette_checkpoint_legacy.pth"
        logging.info(f"Starting FRESH run. Backing up legacy checkpoint {CHECKPOINT_FILE} -> {legacy_backup}")
        if os.path.exists(legacy_backup):
            os.remove(legacy_backup)
        os.rename(CHECKPOINT_FILE, legacy_backup)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for sar, real_opt in pbar:
            sar = sar.to(device, non_blocking=True)
            real_opt = real_opt.to(device, non_blocking=True)

            # Sample random timesteps
            t = torch.randint(0, T, (sar.size(0),), device=device).long()

            # Add noise (forward diffusion in [-1, 1] space)
            noise = torch.randn_like(real_opt)
            noisy_opt = q_sample(real_opt, t, diffusion_params, noise)

            # CFG: Per-sample condition dropout (10% independent per patch)
            mask = (torch.rand(sar.size(0), 1, 1, 1, device=device) >= 0.10).float()
            sar_input = sar * mask

            model_input = torch.cat([noisy_opt, sar_input], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                predicted_noise = model(model_input, t)
                loss = F.mse_loss(predicted_noise, noise)

            scaler.scale(loss).backward()

            # Gradient clipping with scaler unscaling to prevent NaN explosions
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()

            # Update EMA weights
            with torch.no_grad():
                for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(EMA_DECAY).add_(model_p.data, alpha=1.0 - EMA_DECAY)

            total_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Real Validation Evaluation on val_loader (50 batches = 800 patches) ---
        ema_model.eval()
        total_val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for sar_val, real_opt_val in val_loader:
                sar_val = sar_val.to(device, non_blocking=True)
                real_opt_val = real_opt_val.to(device, non_blocking=True)

                t_val = torch.randint(0, T, (sar_val.size(0),), device=device).long()
                noise_val = torch.randn_like(real_opt_val)
                noisy_opt_val = q_sample(real_opt_val, t_val, diffusion_params, noise_val)

                model_input_val = torch.cat([noisy_opt_val, sar_val], dim=1)
                with torch.amp.autocast('cuda'):
                    pred_val = ema_model(model_input_val, t_val)
                    val_batch_loss = F.mse_loss(pred_val, noise_val)

                total_val_loss += val_batch_loss.item()
                val_batches += 1
                if val_batches >= 50:
                    break

        avg_val_loss = total_val_loss / max(1, val_batches)
        logging.info(f"Epoch {epoch + 1} Complete. Train MSE: {avg_train_loss:.6f} | Val MSE (EMA): {avg_val_loss:.6f}")

        # --- Generate Validation Visual Progress with Guided DDIM ---
        with torch.no_grad():
            for i, (sar, real_opt) in enumerate(val_loader):
                if i > 0:
                    break
                sar = sar.to(device)
                real_opt = real_opt.to(device)

                fake_opt = p_sample_loop(ema_model, sar[:4], diffusion_params, device,
                                         num_steps=INFERENCE_STEPS, guidance_scale=2.0)

                # Rescale from [-1.0, 1.0] to [0.0, 1.0] for saving
                real_vis = ((real_opt[:4] + 1.0) / 2.0).clamp(0.0, 1.0)
                fake_vis = ((fake_opt + 1.0) / 2.0).clamp(0.0, 1.0)

                img_sample = torch.cat((real_vis.data, fake_vis.data), -2)
                save_image(img_sample, OUTPUT_IMAGES / f"epoch_{epoch + 1}.png", nrow=4, normalize=False)

        # --- SAVE CHECKPOINT (includes EMA weights and best_val_loss) ---
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'ema_model': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(ckpt, CHECKPOINT_FILE)

        # --- SAVE BEST MODEL (lowest validation loss tracked using EMA weights) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ema_model.state_dict(), BEST_MODEL)
            logging.info(f"New best EMA model saved at Epoch {epoch + 1}! Val Loss: {best_val_loss:.6f} -> {BEST_MODEL}")
        else:
            logging.info("Checkpoint saved safely.")

    # Export final epoch weights
    torch.save(ema_model.state_dict(), FINAL_MODEL)
    logging.info(f"Training finished! Final EMA model: {FINAL_MODEL} | Best EMA model: {BEST_MODEL} (lowest val loss: {best_val_loss:.6f})")


if __name__ == "__main__":
    main()
