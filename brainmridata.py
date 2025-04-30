import torch
from typing import Any, Tuple
from torch.utils.data import Dataset
from PIL import Image

class BrainMRIDataset(Dataset):

    def __init__(self, df, base_path='.', transforms=None) -> None:
        """
        Constructor for loading brain MRI images and their corresponding masks.

        Args:
            df (Dataframe): the dataframe to process and use in deep leanring model training
            base_path (str): the base directory path to the brain MRI images.
            transforms (Tuple(Compose, Compose)): the tuple containing the functions to transform
            MRI image and mask image.
        """
        self.transforms = transforms
        self.base_path = base_path
        self.df = df

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            - int: the number of samples in the dataset
        """
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[Any, torch.Tensor]:
        """
        Load and return an image and its corresponding mask at a specific index

        Args:
            idx (int): index of the data to fetch

        Returns:
            - Tuple(Any, torch.Tensor): 
        """
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