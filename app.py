import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from gsi.models.baseline_model import BaselineModel
from gsi.models.extended_baseline_model import ExtendedBaselineModel
from gsi.models.efficientnet_v2_s import EfficientNetV2

def get_num_classes_from_checkpoint(checkpoint_path, model_name):

    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    key_to_check = None
    match model_name:
        case 'BaselineModel':
            key_to_check = 'fc1.weight'
        case 'ExtendedBaselineModel':
            key_to_check = 'fc2.weight'
        case 'EfficientNetV2':
            key_to_check = 'head.2.weight'
            
    if key_to_check and key_to_check in state_dict:
        return state_dict[key_to_check].shape[0]
    
    return 10

def main():
    if len(sys.argv) != 3:
        print('[USAGE] python3 app.py <model_class> <image_path>')
        return

    model_name = sys.argv[1]
    img_path = Path(sys.argv[2])
    
    weights_path = Path('models') / model_name / 'best.pth'

    if not weights_path.exists():
        print(f'[ERROR] Weights file not found: {weights_path}')
        sys.exit(1)
        
    if not img_path.exists():
        print(f'[ERROR] Image not found: {img_path}')
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_classes = get_num_classes_from_checkpoint(weights_path, model_name)

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

    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)
    if image is None:
        print(f'[ERROR] Failed to read image: {img_path}')
        sys.exit(1)

    image = image.transpose(2, 0, 1).astype(np.float32)
    
    image /= 255.0
    
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        
        probabilities = torch.softmax(logits, dim=1)
        
        confidence, prediction_idx = torch.max(probabilities, dim=1)
        
        prediction_idx = prediction_idx.item()
        confidence = confidence.item()

    print(f'Prediction: {prediction_idx}')
    print(f'Confidence: {confidence:.4f}')

if __name__ == '__main__':
    main()