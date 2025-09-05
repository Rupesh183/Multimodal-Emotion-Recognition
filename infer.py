import argparse
import torch
import numpy as np
from core.config import CONFIG, device
from text.deberta import DebertaV3Tokenizer, DebertaV3
from audio.wav2vec2 import Wav2Vec2
from audio.extractor import Wav2Vec2Extractor
from fusion.model import FusionModel


def predict_emotion(text_input: str = None, audio_path: str = None):
    CONFIG.load_config("config.yaml")
    emotions = CONFIG.dataset_emotions()
    num_classes = len(emotions)

    print(f"Loaded {num_classes} emotion categories: {', '.join(emotions)}")
    print(f"Target Device: {device}")

    text_model = DebertaV3(num_classes).to(device)
    audio_model = Wav2Vec2(num_classes).to(device)
    fusion_model = FusionModel(num_classes, text_model, audio_model, freeze_weights=False).to(device)
    fusion_model.eval()

    tokenizer = DebertaV3Tokenizer()
    if text_input is not None:
        tokens = tokenizer.encode(
            text_input,
            add_special_tokens=True,
            truncation=True,
            max_length=64,
            padding="max_length",
            return_tensors="pt"
        ).to(device)
    else:
        tokens = torch.zeros((1, 64), dtype=torch.long).to(device)

    if audio_path is not None:
        raw_feat = Wav2Vec2Extractor.extract(audio_path).unsqueeze(0).to(device)
    else:
        raw_feat = torch.randn(1, 16000).to(device)

    with torch.no_grad():
        output = fusion_model(tokens, raw_feat)
        probs = output.squeeze().cpu().numpy()

    print("\n" + "=" * 45)
    print("      PREDICTED EMOTION PROBABILITIES       ")
    print("=" * 45)
    for emo, prob in zip(emotions, probs):
        bar = "█" * int(prob * 30)
        print(f"  {emo:<5} : {prob * 100:>6.2f}%  | {bar}")
    print("=" * 45)
    pred_idx = int(np.argmax(probs))
    print(f"  Top Prediction: {emotions[pred_idx].upper()} ({probs[pred_idx] * 100:.2f}%)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Emotion Recognition Inference")
    parser.add_argument("--text", type=str, default=None, help="Input utterance transcription")
    parser.add_argument("--audio", type=str, default=None, help="Path to input audio file (.wav)")
    args = parser.parse_args()

    default_text = args.text if args.text else "I am absolutely thrilled and amazed by this achievement!"
    predict_emotion(text_input=default_text, audio_path=args.audio)
