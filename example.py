import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader

class ImageNetDataset(Dataset):
    def __init__(self, data_path, labels_path=None):
        self.data = np.memmap(data_path, dtype='uint8', mode='r', shape=(1_281_152, 4097))
        if labels_path is not None:
            with open(labels_path, 'r') as f:
                self.labels_txt = [(line.strip()) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image, label = image[:-1], image[-1]
        image = image.astype(np.float32).reshape(4, 32, 32)
        image = (image / 255.0 - 0.5) * 24.0
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        if hasattr(self, 'labels_txt'):
            return image, label, self.labels_txt[idx]
        return image, label

data_path = 'inet.npy'
labels_path = None#'inet.txt'
dataset = ImageNetDataset(data_path, labels_path)
dataloader = DataLoader(dataset, batch_size=128)

for images, labels in tqdm.tqdm(dataloader):
    # print(images.shape, labels.shape)
    pass