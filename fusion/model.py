import torch
import torch.nn as nn
from typing import List

from audio.wav2vec2 import Wav2Vec2
from text.deberta import DebertaV3


class FusionModel(nn.Module):
    """
    Multimodal Speech Emotion Recognition Fusion Model.
    Fuses dense acoustic features (Wav2Vec2) and linguistic features (DeBERTa-V3)
    along with unimodal predictions through a deep Multi-Layer Perceptron.
    """
    def __init__(
        self,
        num_classes: int,
        deberta_model: DebertaV3,
        wav2vec2_model: Wav2Vec2,
        hidden_layers: List[int] = None,
        freeze_weights: bool = True,
    ):
        super(FusionModel, self).__init__()
        if freeze_weights:
            for param in deberta_model.parameters():
                param.requires_grad = False
            for param in wav2vec2_model.parameters():
                param.requires_grad = False

        if hidden_layers is None:
            hidden_layers = [512, 128, 32]

        self.num_classes = num_classes
        self.text_model = deberta_model
        self.audio_model = wav2vec2_model

        input_dim = 170 * 768 + 768 + num_classes * 2

        layers = [nn.Linear(input_dim, hidden_layers[0]), nn.ReLU()]
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())
        self.mlp_head = nn.Sequential(*layers)

        self.cls_head = nn.Linear(hidden_layers[-1], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text: torch.Tensor, audio: torch.Tensor):
        audio_features = self.audio_model.flatten(
            self.audio_model.wav2vec2(audio).last_hidden_state
        )
        audio_classification = self.audio_model.softmax(
            self.audio_model.cls_head(self.audio_model.lm_head(audio_features))
        )

        deberta_out = self.text_model.deberta(text).last_hidden_state
        text_features = self.text_model.pooler(deberta_out)
        text_classification = self.text_model.dropout(
            self.text_model.classifier(text_features)
        )

        fused = torch.cat(
            [audio_features, audio_classification, text_features, text_classification],
            dim=1,
        )

        x = self.mlp_head(fused)
        x = self.cls_head(x)
        x = self.softmax(x)
        return x
