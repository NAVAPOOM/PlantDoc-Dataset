import torch
import torch.nn as nn
import random

class HideAndSeek(torch.nn.Module):
    def __init__(self):
        super(HideAndSeek, self).__init__()
        # possible grid size, 0 means no hiding
        self.grid_sizes = [0, 16, 32, 44, 56]
        # hiding probability
        self.hide_prob = 0.5

    def forward(self, img):
        img = img.clone()
        # randomly choose one grid size
        grid_size = random.choice(self.grid_sizes)
        if len(img.shape) == 4:
            img = img.squeeze(0)

        c, h, w = img.size() 

        # hide the patches
        if grid_size == 0:
            return img
        for x in range(0, w, grid_size):
            for y in range(0, h, grid_size):
                x_end = min(w, x + grid_size)
                y_end = min(h, y + grid_size)
                if (random.random() <= self.hide_prob):
                    img[:, x:x_end, y:y_end] = 0

        return img