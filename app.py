import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import pandas as pd
from gsi.models.baseline_model import BaselineModel
from gsi.models.extended_baseline_model import ExtendedBaselineModel
from gsi.models.efficientnet_v2_s import EfficientNetV2
from gsi.dataset.butterfly_dataset import ButterFlyDataset

def main():
    if len(sys.argv) != 3:
        print('[USAGE] python3 app.py <model_class> <image_or_dir_path>')
        return

    model_name = sys.argv[1]
    input_path = Path(sys.argv[2])
    data_path = Path('data')
    csv_path = data_path / 'data.csv'
    weights_path = Path('models') / model_name / 'best.pth'

    if not weights_path.exists():
        print(f'[ERROR] Weights file not found: {weights_path}')
        sys.exit(1)
        
    if not input_path.exists():
        print(f'[ERROR] Path not found: {input_path}')
        sys.exit(1)

    if not csv_path.exists():
        print(f'[ERROR] data.csv not found at {csv_path}')
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df_base = pd.read_csv(csv_path)
    aux_dataset = ButterFlyDataset(df_base, data_path, 'train')
    idx_to_class = aux_dataset.class_idx_to_name
    num_classes = len(idx_to_class)

    match model_name:
        case 'BaselineModel':
            model = BaselineModel(in_channels=3, num_classes=num_classes)
        case 'ExtendedBaselineModel':
            model = ExtendedBaselineModel(in_channels=3, num_classes=num_classes)
        case 'EfficientNetV2':
            model = EfficientNetV2(num_classes=num_classes)
        case _:
            print(f'[ERROR] Unknown model class: {model_name}')
            sys.exit(1)

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    def predict_image(img_p):
        image = cv2.imread(str(img_p), cv2.IMREAD_COLOR_RGB)
        if image is None:
            return None, None
        image = image.transpose(2, 0, 1).astype(np.float32)
        image /= 255.0
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        
        with torch.inference_mode():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction_idx = torch.max(probabilities, dim=1)
            
        return idx_to_class[prediction_idx.item()], confidence.item()

    if input_path.is_file():
        class_name, conf = predict_image(input_path)
        if class_name:
            print(f'Prediction: {class_name}')
            print(f'Confidence: {conf:.4f}')
        else:
            print(f'[ERROR] Could not process image: {input_path}')

    elif input_path.is_dir():
        results = []
        extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in extensions:
            image_files.extend(list(input_path.rglob(ext)))

        if not image_files:
            print(f'[WARNING] No images found in {input_path}')
            return

        for img_f in image_files:
            class_name, conf = predict_image(img_f)
            if class_name:
                results.append({
                    'filename': img_f.name,
                    'prediction': class_name,
                    'confidence': round(conf, 4)
                })

        output_df = pd.DataFrame(results)
        output_file = Path('output') / f'predictions_{model_name}.csv'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_file, index=False)
        print(f'[SYSTEM] Batch processing complete. Results saved in {output_file}')

if __name__ == '__main__':
    main()