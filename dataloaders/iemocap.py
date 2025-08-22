import logging
import numpy as np
import pandas as pd
import torch
from typing import List
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class IemocapDataset(Dataset):
    """
    PyTorch Dataset wrapper for processed IEMOCAP speech audio and transcription data.
    """
    def __init__(
        self,
        dataset_path: str,
        dataframe: pd.DataFrame,
        emotions: List[str],
        split: str,
    ):
        self._dataset_path = dataset_path
        self._emotions = np.array(emotions)
        self._dataframe = dataframe.copy()

        self._dataframe = self._dataframe.loc[self._dataframe["emotion"].isin(emotions)]

        rows_80_percent = int(0.8 * len(self._dataframe))
        if split == "train":
            self._dataframe = self._dataframe.iloc[:rows_80_percent, :]
        elif split == "test":
            self._dataframe = self._dataframe.iloc[rows_80_percent:, :]
        else:
            raise ValueError(f"Invalid dataset split: {split}. Choose 'train' or 'test'.")

        logger.info(f"Loaded {split} dataset split. Size: {len(self)} samples.")

    def __getitem__(self, index: int):
        audio, text, emotion = self._dataframe.iloc[index]
        emotion_matches = np.where(self._emotions == emotion)[0]
        emotion_index = torch.tensor(emotion_matches[0]) if len(emotion_matches) > 0 else torch.tensor(0)
        return torch.tensor(audio, dtype=torch.float32), torch.tensor(text, dtype=torch.long), emotion_index

    def __len__(self):
        return len(self._dataframe)


def IemocapDataLoader(
    dataset_path: str,
    dataframe: pd.DataFrame,
    emotions: List[str],
    split: str,
    **kwargs,
):
    dataset = IemocapDataset(dataset_path, dataframe, emotions, split)
    return DataLoader(dataset, **kwargs)
