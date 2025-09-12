# Multimodal Speech Emotion Recognition

An end-to-end deep learning framework for **Multimodal Speech Emotion Recognition (SER)** combining acoustic speech representations (**Wav2Vec 2.0**) and contextual linguistic representations (**DeBERTa-V3**). 

The system leverages intermediate feature fusion through a hierarchical Multi-Layer Perceptron (MLP) to classify speech into distinct emotional states on the **IEMOCAP** benchmark dataset.

---

## Architecture Overview

The framework processes spoken audio and conversational transcriptions in parallel:
1. **Acoustic Branch:** Extracts dense representations from raw speech audio using `facebook/wav2vec2-base-960h`.
2. **Linguistic Branch:** Extracts contextual embeddings from text transcripts using `microsoft/deberta-v3-small`.
3. **Multimodal Fusion:** Concatenates intermediate latent features and unimodal class probabilities, routing them through an MLP classifier to produce the final emotion prediction.

<p align="center">
  <img src="architecture.png" alt="Model Architecture" width="90%">
</p>

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/Rupesh183/Multimodal-Emotion-Recognition.git
cd Multimodal-Emotion-Recognition

# Install dependencies
pip install -r requirements.txt
```

## License
Licensed under the [MIT License](LICENSE).
