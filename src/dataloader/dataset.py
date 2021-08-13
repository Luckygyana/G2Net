from torch.utils.data import Dataset
import torch
import numpy as np
from nnAudio.Spectrogram import CQT1992v2


class G2Dataset(Dataset):
    def __init__(self, images_filepaths, targets, transform=None):
        self.images_filepaths = images_filepaths
        self.targets = targets
        self.transform = transform
        self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=64, verbose=True)

    def __len__(self):
        return len(self.images_filepaths)

    def apply_cqt(self, image, transform):
        image = np.hstack(image)
        image = image / np.max(image)
        image = torch.from_numpy(image).float()
        image = transform(image)
        return image

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = np.load(image_filepath)
        image = self.apply_cqt(image, self.wave_transform)

        if self.transform is not None:
            image = image.squeeze().numpy()
            image = self.transform(image=image)["image"]
        else:
            image = image[np.newaxis, :, :]
            image = torch.from_numpy(image).float()

        label = torch.tensor(self.targets[idx]).float()
        return image, label
