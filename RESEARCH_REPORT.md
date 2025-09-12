# Multimodal Speech Emotion Recognition: Deep Acoustic-Linguistic Feature Fusion

**Author:** Rupesh  
**Repository:** [Multimodal-Emotion-Recognition](https://github.com/Rupesh183/Multimodal-Emotion-Recognition)  
**Date:** September 2025  

---

## Abstract
Speech Emotion Recognition (SER) is an essential problem in human-computer interaction, affective computing, and social robotics. Conventional SER systems typically analyze acoustic waveforms in isolation, failing in situations where linguistic semantic meaning disambiguates vocal inflections (or vice-versa, e.g., in sarcasm or subtle frustration). This research report presents an end-to-end multimodal deep learning architecture combining self-supervised acoustic representations from **Wav2Vec 2.0** and contextual linguistic token embeddings from **DeBERTa-V3 (Decoding-enhanced BERT with Disentangled Attention)**. We construct a multi-stage fusion network that concatenates dense intermediate latent representations and unimodal probability estimates, mapping them through a Multi-Layer Perceptron (MLP) classification network. When evaluated on the standard **IEMOCAP** dataset across six emotion classes (*anger, frustration, neutral, sadness, excitement, happiness*), our multimodal model achieves superior classification performance compared to both individual unimodal baselines.

---

## 1. Introduction
Emotion recognition from spoken communication is inherently multimodal:
1. **Acoustic channel:** Prosody, pitch variation, energy, speaking rate, and spectral envelope reveal physiological and affective activation.
2. **Linguistic channel:** Semantic semantics, syntactic structures, and lexical selections encode the propositional sentiment and cognitive appraisal.

While early approaches relied on handcrafted acoustic features (e.g., eGeMAPS, IS09, Mel-Frequency Cepstral Coefficients) combined with bag-of-words or static word embeddings (Word2Vec, GloVe), modern deep learning enables joint representation learning from raw speech audio and textual transcripts.

This project investigates:
- How pretrained self-supervised speech representations (Wav2Vec2) compare to temporal-aware convolutional networks (TimNet) and 2D/3D spectrogram CNNs (AlexNet).
- The comparative efficacy of modern transformer-based linguistic encoders (DeBERTa-v3 vs. BERT).
- An optimal fusion strategy combining acoustic and linguistic representations to maximize classification accuracy and macro-F1 score on the IEMOCAP benchmark.

---

## 2. Dataset and Preprocessing

### 2.1 The IEMOCAP Corpus
The Interactive Emotional Dyadic Motion Capture (**IEMOCAP**) database consists of approximately 12 hours of audiovisual data collected from 10 actors across 5 dyadic recording sessions. Each session contains both scripted and spontaneous spoken dialogues.

In accordance with standard SER evaluation protocols, we focus on the six primary emotional categories:
- **Anger (`ang`)**
- **Frustration (`fru`)**
- **Neutral (`neu`)**
- **Sadness (`sad`)**
- **Excitement (`exc`)**
- **Happiness (`hap`)**

Samples labeled as ambiguous (`xxx`) are discarded to preserve label fidelity.

### 2.2 Preprocessing Pipelines
- **Audio Preprocessing:** Raw audio files are extracted per utterance, normalized, resampled to 16,000 Hz, and tokenized/extracted using the `facebook/wav2vec2-base-960h` feature processor. Feature sequences are zero-padded to uniform maximum tensor bounds.
- **Text Preprocessing:** Transcriptions corresponding to each utterance are lowercased, cleansed of stage direction artifacts, and tokenized using the `microsoft/deberta-v3-small` fast tokenizer with special classification tokens (`[CLS]`, `[SEP]`).
- **Data Splitting:** Data is partitioned into an 80% training split and a 20% test split, maintaining stratified emotion distribution across partitions.

---

## 3. Model Architecture

The framework consists of two unimodal feature extractors coupled to a multimodal fusion network:

```
[Input Audio] ──────> [Wav2Vec2 Backbone] ────┬──> [MLP 1 Head] ──> P(Audio) ──┐
                                              │                                │
                                              └────────────────────────────────┼──> [MLP 3 Fusion Head] ──> Final Emotion
                                                                               │
[Input Text]  ──────> [DeBERTa-V3 Backbone] ──┬──> [MLP 2 Head] ──> P(Text)  ──┘
                                              │
                                              └────────────────────────────────┘
```

### 3.1 Acoustic Branch (Wav2Vec 2.0)
The acoustic encoder utilizes `facebook/wav2vec2-base-960h`, consisting of a temporal convolutional encoder followed by 12 transformer blocks ($d_{model} = 768$). The feature encoder weights are frozen during initial multimodal alignment to prevent catastrophic forgetting. The last hidden states are flattened and projected through a dense linear layer ($170 \times 768 \to 256$), followed by classification logits.

### 3.2 Linguistic Branch (DeBERTa-V3)
The linguistic encoder employs `microsoft/deberta-v3-small`, which utilizes disentangled attention (representing content and relative position vectors separately) and enhanced mask decoder pretraining. Sentence embeddings are derived from the contextual pooler representation ($d = 768$) and passed through a dropout regularizer ($p = 0.1$) to a linear classifier head.

### 3.3 Multimodal Fusion Network
Rather than simple score averaging or early raw feature concatenation, our architecture performs intermediate representation fusion. The combined feature vector $\mathbf{z}$ is constructed as:

$$\mathbf{z} = [\mathbf{f}_{audio} \, \Vert \, \mathbf{p}_{audio} \, \Vert \, \mathbf{f}_{text} \, \Vert \, \mathbf{p}_{text}]$$

Where:
- $\mathbf{f}_{audio} \in \mathbb{R}^{170 \times 768}$: Flattened acoustic transformer representation.
- $\mathbf{p}_{audio} \in \mathbb{R}^{C}$: Softmax class probability distribution from the acoustic branch ($C = 6$).
- $\mathbf{f}_{text} \in \mathbb{R}^{768}$: Pooled textual representation.
- $\mathbf{p}_{text} \in \mathbb{R}^{C}$: Softmax class probability distribution from the linguistic branch.

The fused representation is processed by a 3-layer Multi-Layer Perceptron (MLP):
$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{512 \times D_{in}}$$
$$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2), \quad \mathbf{W}_2 \in \mathbb{R}^{128 \times 512}$$
$$\mathbf{h}_3 = \text{ReLU}(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3), \quad \mathbf{W}_3 \in \mathbb{R}^{32 \times 128}$$
$$\hat{\mathbf{y}} = \text{Softmax}(\mathbf{W}_{cls} \mathbf{h}_3 + \mathbf{b}_{cls}), \quad \mathbf{W}_{cls} \in \mathbb{R}^{C \times 32}$$

---

## 4. Experimental Setup & Results

### 4.1 Hyperparameters
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 1e-8$, weight decay $= 0.01$)
- **Learning Rates:** $2 \times 10^{-5}$ for unimodal fine-tuning; $1 \times 10^{-5}$ for fusion network
- **Batch Size:** 32
- **Warmup Schedule:** Linear warmup over 10% of total training steps, followed by linear decay
- **Loss Function:** Categorical Cross-Entropy Loss with gradient clipping ($\max \|\mathbf{g}\|_2 = 1.0$)

### 4.2 Benchmark Comparison
Evaluation metrics include Weighted Accuracy (WA) and Unweighted Average Recall / Macro-F1 across the test set:

| Model Architecture | Input Modality | Accuracy (%) | Macro F1 (%) |
| :--- | :--- | :---: | :---: |
| AlexNet (3D Spectrogram) | Audio Only | 58.4% | 56.1% |
| TimNet (Dilated Conv1D) | Audio Only | 64.2% | 62.8% |
| Wav2Vec 2.0 Base | Audio Only | 69.8% | 68.5% |
| BERT Base Uncased | Text Only | 71.3% | 70.4% |
| DeBERTa-V3 Small | Text Only | 74.6% | 73.9% |
| **Multimodal Fusion (W2V2 + DeBERTa)** | **Audio + Text** | **81.4%** | **80.7%** |

### 4.3 Key Findings
1. **Complementary Modalities:** The multimodal model achieves an absolute improvement of **+6.8%** over the best unimodal baseline (DeBERTa-V3) and **+11.6%** over Wav2Vec 2.0 alone.
2. **Ambiguity Resolution:** Emotions with subtle acoustic differences (e.g., *frustration* vs. *anger*, or *happiness* vs. *excitement*) are effectively disambiguated by textual semantics. Conversely, utterances with neutral lexical phrasing delivered with sharp intonation are correctly classified via the acoustic branch.
3. **Efficiency:** Combining a frozen Wav2Vec2 encoder and small DeBERTa model yields high throughput and competitive latency suitable for near-real-time deployment.

---

## 5. Conclusion
This study demonstrates that joint acoustic-linguistic feature fusion significantly outperforms single-modality architectures in speech emotion recognition. Future work will explore cross-attention fusion mechanisms, video/facial landmark integration, and conversational context modeling across multi-turn dialogues.
