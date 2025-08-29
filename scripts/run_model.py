import logging
import os
import torch

from audio.trainer import AudioTrainer
from audio.wav2vec2 import Wav2Vec2
from core.config import CONFIG, device
from core.trainer import AbstractTrainer
from fusion.model import FusionModel
from fusion.trainer import FusionTrainer
from scripts.get_dataloaders import get_dataloader
from text.deberta import DebertaV3
from text.trainer import TextTrainer

logger = logging.getLogger(__name__)


class TrainerOps:
    @classmethod
    def create_or_load_text_trainer(cls, load_path=None):
        num_classes = len(CONFIG.dataset_emotions())
        model = DebertaV3(num_classes).to(device)
        return TextTrainer(model)

    @classmethod
    def create_or_load_audio_trainer(cls, load_path=None):
        num_classes = len(CONFIG.dataset_emotions())
        model = Wav2Vec2(num_classes).to(device)
        return AudioTrainer(model)

    @classmethod
    def create_or_load_fusion_trainer(
        cls,
        load_path: str = None,
        audio_model: Wav2Vec2 = None,
        text_model: DebertaV3 = None,
    ):
        num_classes = len(CONFIG.dataset_emotions())
        if audio_model is None:
            audio_model = Wav2Vec2(num_classes).to(device)
        if text_model is None:
            text_model = DebertaV3(num_classes).to(device)
        model = FusionModel(num_classes, text_model, audio_model).to(device)
        return FusionTrainer(model)

    @classmethod
    def train(cls, trainer: AbstractTrainer):
        train_dataloader = get_dataloader("train", shuffle=True)
        trainer.train(train_dataloader)

    @classmethod
    def save(cls, trainer: AbstractTrainer, save_path: str):
        full_path = os.path.join(CONFIG.saved_models_location(), save_path)
        torch.save(trainer.model.state_dict(), full_path)

    @classmethod
    def evaluate(cls, trainer: AbstractTrainer):
        emotions = CONFIG.dataset_emotions()
        test_dataloader = get_dataloader("test", False)
        return trainer.eval(test_dataloader, emotions)
