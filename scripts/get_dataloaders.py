import os
import pandas as pd
from typing import List, Union

from core.config import CONFIG
from dataloaders.iemocap import IemocapDataLoader


def get_dataloader(
    split: Union[str, List[str]] = None,
    shuffle: Union[bool, List[bool]] = None,
):
    if split is None:
        split = ["train", "test"]
    if shuffle is None:
        shuffle = [True, False]

    prep_file = os.path.join(CONFIG.dataset_preprocessed_dir_path(), "w2v2_and_tokens.pkl")
    if not os.path.exists(prep_file):
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {prep_file}. Please run data preprocessing first."
        )

    dataframe = pd.read_pickle(prep_file)

    if isinstance(split, str):
        return IemocapDataLoader(
            CONFIG.dataset_path(),
            dataframe,
            CONFIG.dataset_emotions(),
            split,
            **CONFIG.dataloader_dict(),
            shuffle=shuffle if isinstance(shuffle, bool) else shuffle[0],
        )
    else:
        return [
            IemocapDataLoader(
                CONFIG.dataset_path(),
                dataframe,
                CONFIG.dataset_emotions(),
                split_item,
                **CONFIG.dataloader_dict(),
                shuffle=shuffle_item,
            )
            for split_item, shuffle_item in zip(split, shuffle)
        ]
