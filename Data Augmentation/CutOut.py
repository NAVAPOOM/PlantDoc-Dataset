import torch
import numpy as np

class CutOut(torch.nn.Module):
    def __init__(self, length=50):
        super(CutOut, self).__init__()
        self.length = length

    def forward(self, img):
        img = img.clone()
        if len(img.shape) == 4:
            img = img.squeeze(0)
        c, h, w = img.size()
        mask = torch.ones((h, w), dtype=torch.float32)
        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()

        y1 = max(y - self.length // 2, 0)
        y2 = min(y + self.length // 2, h)
        x1 = max(x - self.length // 2, 0)
        x2 = min(x + self.length // 2, w)

        mask[y1: y2, x1: x2] = 0.
        mask = mask.expand_as(img)
        img *= mask
        return img