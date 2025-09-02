import argparse
import logging
import random
import numpy as np
import torch

from core.config import CONFIG
from scripts.run_model import TrainerOps
from text.deberta import DebertaV3Tokenizer
from audio.extractor import Wav2Vec2Extractor
from scripts.preprocess_data import (
    process_raw_data_to_pickle,
    process_audio_data_to_pickle,
    process_text_data_to_pickle,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline(preprocess: bool = False, model_type: str = "fusion"):
    set_seed(42)
    CONFIG.load_config("config.yaml")

    if preprocess:
        logger.info("Step 1: Extracting raw audio and text from IEMOCAP...")
        process_raw_data_to_pickle("audio_and_text.pkl")

        logger.info("Step 2: Processing acoustic features with Wav2Vec2...")
        process_audio_data_to_pickle(
            "audio_and_text.pkl", "w2v2_and_text.pkl", Wav2Vec2Extractor()
        )

        logger.info("Step 3: Tokenizing text transcripts with DeBERTa-v3...")
        process_text_data_to_pickle(
            "w2v2_and_text.pkl", "w2v2_and_tokens.pkl", DebertaV3Tokenizer()
        )

    if model_type in ["text", "all"]:
        logger.info("=== Training DeBERTa-V3 Text Model ===")
        text_trainer = TrainerOps.create_or_load_text_trainer()
        TrainerOps.train(text_trainer)
        TrainerOps.save(text_trainer, "deberta_model.pt", save_state_dict=True)
        TrainerOps.evaluate(text_trainer)

    if model_type in ["audio", "all"]:
        logger.info("=== Training Wav2Vec2 Audio Model ===")
        audio_trainer = TrainerOps.create_or_load_audio_trainer()
        TrainerOps.train(audio_trainer)
        TrainerOps.save(audio_trainer, "wav2vec2_model.pt", save_state_dict=True)
        TrainerOps.evaluate(audio_trainer)

    if model_type in ["fusion", "all"]:
        logger.info("=== Training Multimodal Fusion Model ===")
        fusion_trainer = TrainerOps.create_or_load_fusion_trainer()
        TrainerOps.train(fusion_trainer)
        TrainerOps.save(fusion_trainer, "fusion_model.pt", save_state_dict=True)
        TrainerOps.evaluate(fusion_trainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Speech Emotion Recognition")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run full data preprocessing pipeline before model execution",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fusion",
        choices=["text", "audio", "fusion", "all"],
        help="Which model pipeline to run (default: fusion)",
    )
    args = parser.parse_args()
    run_pipeline(preprocess=args.preprocess, model_type=args.model)
