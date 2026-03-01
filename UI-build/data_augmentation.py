import numpy as np
import torch
import random

# mislabelling
def add_label_noise(dataset, noise_ratio=0.45, num_classes=10, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(dataset.targets)
    n_noisy = int(noise_ratio * n)

    noisy_indices = torch.randperm(n)[:n_noisy]

    original = dataset.targets[noisy_indices]
    noise = torch.randint(0, num_classes - 1, size=(n_noisy,))
    noise = (noise + (noise >= original)).long()

    dataset.targets[noisy_indices] = noise

    return dataset

# mixup augmentation
def mixup_data(x, y, alpha):
    lamb = np.random.beta(alpha, alpha)

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lamb * x + (1 - lamb) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lamb

def mixup_criterion(criterion, preds, y_a, y_b, lamb):
    return lamb * criterion(preds, y_a) + (1 - lamb) * criterion(preds, y_b)

# cutout augmentation
class CutoutAugmentation():
    def __init__(self, K=16):
        self.K = K

    def __call__(self, img):
        # tensor shape: c, h, w
        if random.random() < 0.5:
            return img

        C, H, W = img.shape
        half = self.K // 2

        # random center
        cx = random.randint(0, W - 1)
        cy = random.randint(0, H - 1)

        # bounding box
        x1 = max(cx - half, 0)
        x2 = min(cx + half, W)
        y1 = max(cy - half, 0)
        y2 = min(cy + half, H)

        img[:, y1:y2, x1:x2] = 0.0
        return img
    
# standard augmentation
class RandomShift():
    def __init__(self, K):
        self.K = K

    def __call__(self, img):
        k1 = random.randint(-self.K, self.K)
        k2 = random.randint(-self.K, self.K)

        C, H, W = img.shape
        shifted = torch.zeros_like(img)

        h_start = max(0, k1)
        h_end = min(H, H + k1)
        w_start = max(0, k2)
        w_end = min(W, W + k2)

        orig_h_start = max(0, -k1)
        orig_h_end = orig_h_start + (h_end - h_start)
        orig_w_start = max(0, -k2)
        orig_w_end = orig_w_start + (w_end - w_start)

        shifted[:, h_start:h_end, w_start:w_end] = img[:, orig_h_start:orig_h_end, orig_w_start:orig_w_end]

        img = shifted

        # horizontal flip
        # if random.random() < 0.5:
        #     img = torch.flip(img, dims=[2])

        return img
