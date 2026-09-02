"""
Multi-Step Recurrent Gaze Policy & Dynamic Scanning Network (REINFORCE + LSTM)
Extends single-glance active perception to sequential multi-step (2-3 glance) scanning
when ground-truth bounding boxes are absent.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------------------------------------
# 1. Dynamic Crop Extractor (Spatial Transformer Network Glimpse)
# ---------------------------------------------------------
def extract_glimpse(high_res: torch.Tensor, center: torch.Tensor, crop_size: int = 224, scale: float = 0.3):
    """
    Extracts affine high-resolution crop centered at relative normalized coordinates center (cx, cy) in [-1, 1].
    """
    B = high_res.shape[0]
    device = high_res.device

    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = center[:, 0]
    theta[:, 1, 2] = center[:, 1]

    grid = F.affine_grid(theta, torch.Size([B, 3, crop_size, crop_size]), align_corners=False)
    crop = F.grid_sample(high_res, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return crop


# ---------------------------------------------------------
# 2. Recurrent Gaze / Attention Network (Glance & Scan Policy)
# ---------------------------------------------------------
class RecurrentGazePolicyNetwork(nn.Module):
    """
    Multi-Step Sequential Active Scanner.
    - Uses a low-res glance thumbnail (128x128) to produce initial recurrent state (h_0, c_0).
    - At each timestep t in [1..T]:
        1. Predicts Gaussian policy N(mu_t, std_t) over normalized coordinates [-1, 1]^2.
        2. Samples action location a_t.
        3. Extracts high-res visual crop at a_t.
        4. Fuses visual feature + location embedding into glimpse vector g_t.
        5. Updates recurrent memory state (h_t, c_t).
    - After T steps, predicts final object classification logits.
    """
    def __init__(self, num_classes: int = 10, hidden_dim: int = 128, crop_size: int = 224):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.crop_size = crop_size

        # 1. Initial Glance Downsampled Thumbnail Feature Extractor
        self.glance_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 64 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.init_h = nn.Linear(32, hidden_dim)
        self.init_c = nn.Linear(32, hidden_dim)

        # 2. Policy Location Head (Predicts center mu for next glance)
        self.policy_mu = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh()  # Bounds location to [-1, 1]
        )
        self.log_std = nn.Parameter(torch.zeros(2))  # Exploration standard deviation

        # 3. High-Resolution Focus Crop Encoder
        self.focus_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 4. Location Coordinate Embedder
        self.location_embed = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32)
        )

        # 5. Glimpse Fusion & Recurrent State Core (LSTM Cell)
        self.glimpse_fc = nn.Sequential(
            nn.Linear(64 + 32, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim)

        # 6. Task Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def init_recurrent_state(self, low_res_img: torch.Tensor):
        """Encodes low-resolution image to initialize LSTM (h, c)."""
        glance_feat = self.glance_conv(low_res_img)
        h_0 = torch.tanh(self.init_h(glance_feat))
        c_0 = torch.tanh(self.init_c(glance_feat))
        return h_0, c_0

    def predict_action(self, h_state: torch.Tensor):
        """Outputs action location distribution N(mu, std) given current hidden state."""
        mu = self.policy_mu(h_state)
        std = torch.exp(self.log_std).expand_as(mu)
        dist = Normal(mu, std)
        
        # Sample action location
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        
        # Log probability across 2D dimensions
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, dist

    def encode_glimpse(self, high_res_img: torch.Tensor, action: torch.Tensor):
        """Extracts high-res crop at action center and computes fused glimpse embedding."""
        crops = extract_glimpse(high_res_img, action.detach(), crop_size=self.crop_size, scale=0.3)
        crop_feat = self.focus_conv(crops)
        loc_feat = self.location_embed(action.detach())
        
        glimpse_in = torch.cat([crop_feat, loc_feat], dim=-1)
        glimpse_emb = self.glimpse_fc(glimpse_in)
        return glimpse_emb, crops

    def forward_sequence(self, high_res_img: torch.Tensor, num_glances: int = 3):
        """
        Executes a multi-step sequence of num_glances.
        Returns:
            final_logits: (B, num_classes)
            actions_history: List of (B, 2) tensors
            log_probs_history: List of (B,) tensors
            crops_history: List of (B, 3, H, W) crop tensors
        """
        B = high_res_img.shape[0]
        device = high_res_img.device
        
        # Prepare low-resolution peripheral thumbnail (128x128)
        low_res_img = F.interpolate(high_res_img, size=(128, 128), mode='bilinear', align_corners=False)

        # Initialize recurrent state from global peripheral glance
        h_t, c_t = self.init_recurrent_state(low_res_img)

        actions_history = []
        log_probs_history = []
        crops_history = []

        for step in range(num_glances):
            # 1. Policy head predicts location center
            action, log_prob, _ = self.predict_action(h_t)
            
            # 2. Extract high-res glimpse crop & fuse features
            glimpse_emb, crop = self.encode_glimpse(high_res_img, action)
            
            # 3. Update LSTM state
            h_t, c_t = self.lstm_cell(glimpse_emb, (h_t, c_t))

            actions_history.append(action)
            log_probs_history.append(log_prob)
            crops_history.append(crop)

        # Final classification logits after num_glances
        final_logits = self.classifier(h_t)
        return final_logits, actions_history, log_probs_history, crops_history


# ---------------------------------------------------------
# 3. Multi-Step REINFORCE Joint Optimization Step
# ---------------------------------------------------------
def train_step_multistep(recurrent_model, optimizer, high_res_batch, labels, 
                         baseline_value=None, num_glances=3, momentum=0.9, gamma=0.95):
    """
    Single optimization step using Multi-Step REINFORCE with Exponential Moving Average Baseline.
    """
    recurrent_model.train()

    # 1. Run multi-step recurrent scanning loop
    final_logits, actions_hist, log_probs_hist, crops_hist = recurrent_model.forward_sequence(
        high_res_batch, num_glances=num_glances
    )

    # 2. Supervised Task Cross-Entropy Loss
    ce_loss = F.cross_entropy(final_logits, labels)

    # 3. Predict final classes & compute Reward
    pred_classes = torch.argmax(final_logits, dim=-1)
    rewards = (pred_classes == labels).float() # (Batch,)

    # 4. Update Exponential Moving Average Baseline
    batch_mean_reward = rewards.mean().item()
    if baseline_value is None:
        baseline_value = batch_mean_reward
    else:
        baseline_value = momentum * baseline_value + (1 - momentum) * batch_mean_reward

    # Advantage: (R - b)
    advantages = rewards - baseline_value

    # 5. Multi-Step REINFORCE Policy Loss:
    # J(theta) = sum_{t=1}^T log pi(a_t | s_t) * Advantage
    seq_log_probs = torch.stack(log_probs_hist, dim=0).sum(dim=0) # (Batch,)
    policy_loss = -(seq_log_probs * advantages.detach()).mean()

    # Total Joint Objective
    total_loss = ce_loss + policy_loss

    # 6. Backward Pass & Step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    accuracy = (rewards.sum() / len(rewards)).item()
    return ce_loss.item(), policy_loss.item(), accuracy, baseline_value


# ---------------------------------------------------------
# 4. Standalone Execution & Verification Demo
# ---------------------------------------------------------
def run_recurrent_gaze_demo(num_glances: int = 3, num_steps: int = 5):
    print("=" * 75)
    print(f"  [Python RL Policy] Multi-Step Recurrent Scanning ({num_glances} Glances)")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Compute Device   : {device}")
    print(f"  - Sequence Length  : T = {num_glances} recursive gaze glances")
    print(f"  - Policy Network   : Recurrent Gaze LSTM + Spatial Transformer Network")

    model = RecurrentGazePolicyNetwork(num_classes=10, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic High-Res Camera Batch: 8 images (1024x1024)
    dummy_images = torch.randn(8, 3, 1024, 1024, device=device)
    dummy_labels = torch.randint(0, 10, (8,), device=device)

    baseline = None
    for step in range(num_steps):
        ce_l, pol_l, acc, baseline = train_step_multistep(
            model, optimizer, dummy_images, dummy_labels, 
            baseline_value=baseline, num_glances=num_glances
        )
        print(f"  Step {step+1:02d}/{num_steps:02d} | CE Loss: {ce_l:.4f} | Policy Loss: {pol_l:+.4f} | Batch Acc: {acc*100:.1f}% | Baseline: {baseline:.3f}")

    print(f"[SUCCESS] Recurrent Active Scanner Multi-Step Training Operational.")
    return {
        "num_glances": num_glances,
        "final_accuracy": acc,
        "final_baseline": baseline
    }


if __name__ == "__main__":
    run_recurrent_gaze_demo(num_glances=3, num_steps=5)
