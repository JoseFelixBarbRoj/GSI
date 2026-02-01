import os
from pathlib import Path
from sys import argv

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from gsi.dataset.butterfly_dataset import ButterFlyDataset
from gsi.inference.tester import Tester
from gsi.models.baseline_model import BaselineModel
from gsi.models.extended_baseline_model import ExtendedBaselineModel
from gsi.inference.metrics import acc_fn

if __name__ == '__main__':
    if len(argv) != 3:
        print('[SYSTEM] Usage python3 test_model.py model_class models_folder')
        exit(1)

    BATCH_SIZE = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('[SYSTEM] Device being used: ', device)
    data_path = Path('data')
    csv_path = data_path / 'data.csv'
    df = pd.read_csv(csv_path)

    test_data = ButterFlyDataset(df, data_path, 'test')
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())

    print('[SYSTEM] Beggining testing for baseline model')
    match(argv[1]):
        case 'BaselineModel':
            model = BaselineModel(in_channels=3, num_classes=len(test_data.class_name_to_idx))
        case 'ExtendedBaselineModel':
            model = ExtendedBaselineModel(in_channels=3, num_classes=len(test_data.class_name_to_idx))
        case _:
            print('[SYSTEM] Error. Class should be: BaselineModel, ExtendedBaselineModel') # Completar
            exit(1)
    loss_fn = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(Path(argv[2]) / argv[1] / 'best.pth', weights_only=True))
    tester = Tester(model=model,
                    test_dataloader=test_dataloader,
                    acc_fn=acc_fn,
                    device=device)
    acc = tester.eval()

    print(f'[SYSTEM] Accuracy: {acc:.4f}')
    