import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class GridMask(torch.nn.Module):
    def __init__(self, d_min=64, d_max=128, rotate=1, ratio=0.5, mode=0, prob=1):
        super(GridMask, self).__init__()
        self.d_min = d_min
        self.d_max = d_max
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob

    def forward(self, img):
        img = img.clone()
        if np.random.rand() > self.prob:
            return img

        h, w = img.size()[-2:]
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d_min, self.d_max)
        self.l = int(d * self.ratio + 0.5)

        mask = np.zeros((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] = 1

        for i in range(-1, ww // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, ww), 0)
            t = max(min(t, ww), 0)
            mask[:, s:t] = 1

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).float()
        mask = mask.expand_as(img)
        img *= mask

        return img