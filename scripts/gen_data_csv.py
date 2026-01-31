from pathlib import Path
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    DATASET_PATH = Path('data')
    labelled_imgs_path = DATASET_PATH /  'train'

    image_data = []
    train_csv = pd.read_csv(DATASET_PATH / 'Training_set.csv')

    for file in labelled_imgs_path.rglob('*.jpg'):
        image = cv2.imread(file)
        image_class = str(train_csv[train_csv['filename'] == file.name]['label'].values[0])
        image_data.append((file.stem, file.relative_to(DATASET_PATH), 'train', image.shape[0], image.shape[1], image_class))

    initial_df = pd.DataFrame(image_data, columns=['id', 'path', 'partition', 'height', 'width', 'class'])

    X = initial_df[initial_df['partition'] == 'train']
    train, val = train_test_split(X, test_size=0.1)
    train = pd.DataFrame(train)
    val = pd.DataFrame(val).sort_index()

    initial_df.loc[val.index, 'partition'] = 'val'

    X = initial_df[initial_df['partition'] == 'train']
    train, test = train_test_split(X, test_size=0.1/0.9) # Since now train is 90% of original size
    train = pd.DataFrame(train)
    test = pd.DataFrame(test).sort_index()

    initial_df.loc[test.index, 'partition'] = 'test'
    print(initial_df['partition'].value_counts())
    initial_df.to_csv(DATASET_PATH / 'data.csv', index=False)

