import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.custom_dataset import CustomDataset
from src.trainer import Trainer
from src.baseline_model import BaselineModel

def acc_fn(y_preds, y_true):
    return len(y_preds[y_preds == y_true]) / len(y_preds)

if __name__ == '__main__':
    BATCH_SIZE = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('[SYSTEM] Device being used: ', device)
    data_path = Path('data')
    csv_path = data_path / 'data.csv'
    df = pd.read_csv(csv_path)

        
    train_data = CustomDataset(df, data_path, 'train')
    sample = train_data[3]

    print('[SYSTEM] Showing an image from training daset (Close it to continue)')
    plt.imshow(sample[0].numpy().transpose(1, 2, 0))
    plt.axis(False)
    plt.title(f'Class {sample[1]} ({train_data.class_idx_to_name[sample[1]]})')
    plt.show()
    plt.close()

    test_data = CustomDataset(df, data_path, 'test')
    val_data = CustomDataset(df, data_path, 'val')

    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
    val_dataloader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())

    print('[SYSTEM] Showing an image from training daseloader (Close it to continue)')
    dataloader_imgs, dataloader_label = next(iter(train_dataloader))
    plt.imshow(dataloader_imgs[0].cpu().numpy().transpose(1, 2, 0))
    plt.axis(False)
    plt.title(f'Class {dataloader_label[0]} ({train_data.class_idx_to_name[dataloader_label.cpu().numpy()[0]]})')
    plt.show()
    plt.close()

    print('[SYSTEM] Beggining training for baseline model')
    baseline = BaselineModel(in_channels=3, num_classes=len(train_data.class_name_to_idx))
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(baseline.parameters(), lr=0.01)
    model_path = Path('models')
    trainer = Trainer(epochs=100,
                      model=baseline,
                      model_name='baseline',
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      optimizer=optim,
                      output_path='models',
                      loss_fn=loss_fn,
                      acc_fn=acc_fn)
    trainer.train()
    
