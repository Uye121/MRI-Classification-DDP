import torch
from torch.utils.data import Dataset
from PIL import Image

class BrainMRIDataset(Dataset):

    def __init__(self, df, base_path='.', transforms=None):
        self.transforms = transforms
        self.base_path = base_path
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        mask_path = self.df.iloc[idx]['mask_path']
        label = self.df.iloc[idx]['mask']

        # Load image and mask
        img = Image.open(self.base_path + img_path).convert('RGB')
        # Single channel/grayscale
        mask = Image.open(self.base_path + mask_path).convert('L')

        if self.transforms:
            img = self.transforms[0](img)
            mask = self.transforms[1](mask)

        # Binarize mask
        mask = (torch.max(mask) > 0).long()

        return img, mask