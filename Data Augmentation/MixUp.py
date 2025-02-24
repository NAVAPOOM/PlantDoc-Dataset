import torch

class MixUp(torch.nn.Module):
    def __init__(self, alpha):
        self.alpha = alpha
        self.beta_dist = torch.distributions.beta.Beta(alpha, alpha)

    def __call__(self, image, label):
        lam = self.beta_dist.sample().item()
        image = image.clone()
        batch_size = image.size(0)
    
        # Ensure label is a tensor of shape (batch_size,)
        label = label.view(-1)

        index = torch.randperm(batch_size)

        mixed_image = lam * image + (1 - lam) * image[index, :]
        label_a, label_b = label, label[index]
        mixed_label = lam * label_a + (1 - lam) * label_b

        return mixed_image, mixed_label

    def mixup_criterion(criterion, pred, label_a, label_b, lam):
        return lam * criterion(pred, label_a) + (1 - lam) * criterion(pred, label_b)
    
import torch
import numpy as np

class CutMix(torch.nn.Module):
    def __init__(self, alpha):
        self.alpha = alpha
        self.beta_dist = torch.distributions.beta.Beta(alpha, alpha)

    def __call__(self, image, label):
        lam = self.beta_dist.sample().item()
        image = image.clone()
        batch_size = image.size(0)
    
        # Ensure label is a tensor of shape (batch_size,)
        label = label.view(-1)

        index = torch.randperm(batch_size)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.size(), lam)
        image[:, :, bbx1:bbx2, bby1:bby2] = image[index, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
        label_a, label_b = label, label[index]
        mixed_label = lam * label_a + (1 - lam) * label_b

        return image, mixed_label


    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[1]
        cut_rat = torch.sqrt(1. - torch.tensor(lam))
        cut_w = int(W * cut_rat.item())
        cut_h = int(H * cut_rat.item())
        # uniform
        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()
        bbx1 = torch.clamp(torch.tensor(cx) - cut_w // 2, 0, W)
        bby1 = torch.clamp(torch.tensor(cy) - cut_h // 2, 0, H)
        bbx2 = torch.clamp(torch.tensor(cx) + cut_w // 2, 0, W)
        bby2 = torch.clamp(torch.tensor(cy) + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
    
    def cutmix_criterion(criterion, pred, label_a, label_b, lam):
        return lam * criterion(pred, label_a) + (1 - lam) * criterion(pred, label_b)