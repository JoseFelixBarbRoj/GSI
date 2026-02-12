from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt

from gsi.inference.metrics import acc_fn

class Tester:
    def __init__(self, 
                 model: nn.Module,
                 test_dataloader: torch.utils.data.DataLoader,
                 indices: list[int],
                 output_path: Path | str,
                 device):
        self.model = model.to(device)
        self.test_dataloader = test_dataloader
        self.indices = indices
        self.output_path = output_path
        self.device = device
        plt.style.use('ggplot')
    def eval(self):
        pred_list = []
        gt_list = []

        self.model.eval()
        with torch.inference_mode():
            for images, labels in tqdm(self.test_dataloader, desc=f'[{self.__class__.__name__.upper()}] Processing test items', unit='img'):
                logits = self.model(images.to(self.device))
                pred_list.extend(logits.softmax(dim=1).cpu())
                gt_list.extend(labels.tolist())
            results = [acc_fn(np.array(pred_list), np.array(gt_list), i) for i in self.indices]
            
            plt.figure(figsize=(10,7))
            plt.title(self.model.__class__.__name__)
            plt.bar(x=[f'top-{i} accuracy' for i in self.indices],
                    height=results)
            plt.savefig(self.output_path / 'testing.png')
            plt.close()
