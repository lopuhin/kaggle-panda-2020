from pathlib import Path

import jpeg4py
import pandas as pd
from torch.utils.data import Dataset


class PandaDataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame, transform):
        self.df = df
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image = jpeg4py.JPEG(self.root / f'{item.image_id}.jpeg').decode()
        x = self.transform(image)
        x = x[:256, :256]  # TODO
        y = item.isup_grade
        return x, y
