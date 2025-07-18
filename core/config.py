import os
import torch
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"


class CONFIG:
    _dict: dict = {}

    @classmethod
    def load_config(cls, filename: str = "config.yaml"):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Configuration file not found: {filename}")
        with open(filename, "r", encoding="utf-8") as file:
            cls._dict = yaml.safe_load(file)

    @classmethod
    def _path_from_data(cls, path: str):
        return os.path.join(cls._dict.get("data_source", {}).get("path", "data"), path)

    @classmethod
    def dataset_path(cls):
        dataset_name = cls._dict.get("dataset", {}).get("name", "IEMOCAP")
        return cls._path_from_data(dataset_name)

    @classmethod
    def dataloader_dict(cls):
        return cls._dict.get("dataloader", {"batch_size": 32, "num_workers": 4})

    @classmethod
    def dataset_preprocessed_dir_path(cls):
        dataset_name = cls._dict.get("dataset", {}).get("name", "IEMOCAP")
        prep_dir = cls._dict.get("dataset_specific", {}).get(dataset_name, {}).get("preprocessed_dir", "Preprocessed")
        return os.path.join(cls.dataset_path(), prep_dir)

    @classmethod
    def dataset_emotions(cls):
        dataset_name = cls._dict.get("dataset", {}).get("name", "IEMOCAP")
        return cls._dict.get("dataset_specific", {}).get(dataset_name, {}).get("emotions", ["ang", "fru", "neu", "sad", "exc", "hap"])

    @classmethod
    def saved_models_location(cls):
        save_path = cls._dict.get("models", {}).get("save_location", "saved_models")
        os.makedirs(save_path, exist_ok=True)
        return save_path

    @classmethod
    def pretrained_alexnet_url(cls):
        return cls._dict.get("models", {}).get("pretrained_alexnet", "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth")

    @classmethod
    def save_plots_location(cls):
        save_path = cls._dict.get("plots", {}).get("save_location", "pictures")
        os.makedirs(save_path, exist_ok=True)
        return save_path
