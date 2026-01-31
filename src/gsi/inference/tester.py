from pathlib import Path

from tqdm import tqdm

import torch
from torch import nn

class Tester:
    def __init__(self, 
                 model: nn.Module,
                 test_dataloader: torch.utils.data.DataLoader,
                 acc_fn):
        self.model = model
        self.test_dataloader = test_dataloader
        self.acc_fn = acc_fn

    def eval(self):
        pred_list = []
        gt_list = []

        self.model.eval()
        with torch.inference_mode():
            for images, labels in tqdm(self.test_dataloader, desc=f'[{self.__class__.__name__.upper()}] Processing test items', unit='img'):
                logits = self.model(images)
                pred_list.extend(logits.softmax(dim=1).argmax(dim=1).tolist())
                gt_list.extend(labels.tolist())

        return self.acc_fn(torch.Tensor(pred_list), torch.Tensor(gt_list))
    