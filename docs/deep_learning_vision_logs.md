---
tags: [ai-ml, computer-vision, model-training, production]
last_reviewed: 2026-03-15
status: stable
---

# Engineering Vault: Automated Medical Vision Pipelines

This note tracks the training specifics and architectural evolution of our localized computer vision assets. For general data pipeline operations, see [[data_ingestion_standards]].

## 1. Network Initialization & Feature Extractor
The core pipeline leverages a fine-tuned **ResNet-50** backbone pre-trained on ImageNet weights. The primary task requires precise classification of highly localized variations within dense structural images (e.g., medical fundus scans or multi-spectral anomalies).

### Modified Top Architecture
To minimize spatial resolution loss while scaling features, the native top fully connected layers were replaced with a custom dense block:
*   **Global Average Pooling (GAP)** layer to flatten spatial dimensions.
*   **Dropout Layer:** Set explicitly to `0.5` to prevent dense feature co-adaptation during high-epoch cycles.
*   **Dense Classification Head:** Configured with a Softmax output mapping across 4 distinct severity tiers.

## 2. Preprocessing & Contrast Adjustment
Standard RGB inputs often introduce domain shifts due to varying capture environments. The image processing worker applies a localized transformation script before batching:

```python
# Extract the Green channel to isolate structural contrast boundaries
green_channel = input_image[:, :, 1]

# Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
claHE_processor = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_matrix = claHE_processor.apply(green_channel)