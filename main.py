from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.custom_dataset import CustomDataset

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('[SYSTEM] Device being used: ', device)
    data_path = Path('data')
    csv_path = data_path / 'data.csv'
    df = pd.read_csv(csv_path)

        
    train_data = CustomDataset(df, data_path, 'val')
    sample = train_data[3]

    plt.imshow(sample[0].numpy().transpose(1, 2, 0))
    plt.axis(False)
    plt.title(f'Class {sample[1]} ({train_data.class_idx_to_name[sample[1]]})')
    plt.show()
    plt.close()
