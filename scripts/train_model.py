import os
from pathlib import Path
from sys import argv

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.transforms import v2
from torch import nn
from torch.utils.data import DataLoader

from gsi.dataset.butterfly_dataset import ButterFlyDataset
from gsi.train.trainer import Trainer
from gsi.models.baseline_model import BaselineModel
from gsi.models.extended_baseline_model import ExtendedBaselineModel
from gsi.inference.metrics import acc_fn


if __name__ == '__main__':
    if len(argv) != 3:
        print('[SYSTEM] Usage python3 train_model.py model_class num_epochs')
        exit(1)
        
    BATCH_SIZE = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('[SYSTEM] Device being used: ', device)
    data_path = Path('data')
    csv_path = data_path / 'data.csv'
    df = pd.read_csv(csv_path)

    transforms = v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05,
                hue=0.02
            )
        ]
    )
        
    train_data = ButterFlyDataset(df, data_path, 'train', transform=transforms)
    val_data = ButterFlyDataset(df, data_path, 'val')

    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
    val_dataloader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())

    match(argv[1]):
        case 'BaselineModel':
            print('[SYSTEM] Starting training with BaselineModel architecture...')
            model = BaselineModel(in_channels=3, num_classes=len(train_data.class_name_to_idx))
        case 'ExtendedBaselineModel':
            print('[SYSTEM] Starting training with ExtendedBaselineModel architecture...')
            model = ExtendedBaselineModel(in_channels=3, num_classes=len(train_data.class_name_to_idx))
        case _:
            print('[SYSTEM] Error. Class should be: BaselineModel, ExtendedBaselineModel') # Completar
            exit(1)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    model_path = Path('models')
    trainer = Trainer(epochs=int(argv[2]),
                      model=model,
                      model_name=argv[1],
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      optimizer=optim,
                      output_path='models',
                      loss_fn=loss_fn,
                      acc_fn=acc_fn)
    trainer.train()
    
