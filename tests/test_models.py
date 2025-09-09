import unittest
import torch
from core.config import CONFIG
from audio.timnet import TimNetClassifier
from audio.alexnet import ModifiedAlexNet
from audio.wav2vec2 import Wav2Vec2
from text.deberta import DebertaV3
from fusion.model import FusionModel


class TestMultimodalModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CONFIG.load_config("config.yaml")
        cls.emotions = CONFIG.dataset_emotions()
        cls.num_classes = len(cls.emotions)

    def test_timnet_classifier_forward(self):
        model = TimNetClassifier(in_filters=39, num_classes=self.num_classes)
        dummy_input = torch.randn(2, 39, 100)
        out = model(dummy_input)
        self.assertEqual(out.shape, (2, self.num_classes))
        self.assertTrue(torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-4))

    def test_alexnet_forward(self):
        model = ModifiedAlexNet(num_classes=self.num_classes)
        dummy_spec = torch.randn(2, 3, 128, 128)
        out = model(dummy_spec)
        self.assertEqual(out.shape, (2, self.num_classes))

    def test_fusion_model_architecture(self):
        text_model = DebertaV3(self.num_classes)
        audio_model = Wav2Vec2(self.num_classes)
        fusion = FusionModel(self.num_classes, text_model, audio_model, freeze_weights=True)

        dummy_text = torch.randint(0, 1000, (2, 32))
        dummy_audio = torch.randn(2, 16000)
        out = fusion(dummy_text, dummy_audio)
        self.assertEqual(out.shape, (2, self.num_classes))
        self.assertTrue(torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
