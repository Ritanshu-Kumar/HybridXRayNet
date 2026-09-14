# HybridXRayNet

### Hybrid CNN–Transformer Network for Pneumonia Detection from Chest X-rays

HybridXRayNet is a deep learning research project for binary chest X-ray classification: **Normal** vs **Pneumonia**.

The model combines **multi-scale convolutional feature extraction**, **residual channel and spatial attention**, a lightweight **Transformer encoder for global context**, and a learned **adaptive fusion gate** that combines local CNN features with global Transformer representations. The project also includes **Grad-CAM explainability** and a lightweight **Flask inference application** for interactive predictions.

> **Research / educational project only.** This system is not clinically validated and must not be used as a medical diagnostic tool.

---

## Why a Hybrid CNN–Transformer?

Chest X-rays contain both fine-grained local patterns and broader spatial relationships. HybridXRayNet is designed around that distinction:

* **CNN branch:** captures local and multi-scale visual features.
* **Attention block:** emphasizes informative channels and spatial regions.
* **Transformer branch:** models long-range relationships across the feature map.
* **Adaptive fusion:** learns how much to rely on the CNN and Transformer representations for each input.
* **Grad-CAM:** provides a visual indication of image regions that influenced the prediction.

---

## Architecture

![HybridXRayNet Architecture](assets/hybridxraynet_architecture1.png)

### Model Components

**1. Multi-Scale CNN**

Three parallel convolution branches with **3×3, 5×5, and 7×7 kernels** capture patterns at different receptive-field sizes. Their outputs are concatenated and refined into a 128-channel feature map.

**2. Residual Channel + Spatial Attention**

The attention module first learns which feature channels are important using channel attention, followed by spatial attention that emphasizes informative regions. A residual connection preserves the original feature representation.

**3. Transformer Context Module**

The attended feature map is reduced to **14×14**, producing **196 tokens** before entering a 2-layer Transformer encoder with **8 attention heads**.

This reduces the computational cost compared with applying self-attention directly to the full CNN feature map.

**4. Adaptive Fusion**

A learned sigmoid gate dynamically determines the contribution of the CNN and Transformer representations:

```text
fused = gate × CNN + (1 − gate) × Transformer
```

The resulting 128-dimensional representation is passed through an MLP classifier to produce the final prediction.

---

## Dataset

The training pipeline expects the dataset to follow this structure:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The dataset is **not included** in this repository.

If a validation directory is not available, the training pipeline can automatically create a **10% validation split** from the training data.

The test set remains separate from training and validation.

### Preprocessing

All X-ray images are:

* Converted to grayscale
* Resized to **224×224**
* Converted to tensors
* Normalized using mean `0.5` and standard deviation `0.5`

### Training Augmentation

The training pipeline applies lightweight augmentation:

* Random horizontal flip
* Random rotation up to 7°
* Small translations
* Small scale jitter (`0.95–1.05`)

---

## Results

Performance on the held-out test set:

| Metric        |     Result |
| ------------- | ---------: |
| **Accuracy**  | **83.17%** |
| **Precision** | **79.26%** |
| **Recall**    | **98.97%** |
| **F1 Score**  | **88.03%** |
| **ROC-AUC**   | **96.67%** |

### Confusion Matrix

```text
                 Predicted
              NORMAL  PNEUMONIA
NORMAL          133      101
PNEUMONIA        4       386
```

The model correctly identifies **386 of 390 pneumonia cases**, resulting in very high pneumonia recall.

The primary trade-off is the relatively high number of false positives for the Normal class.

> Results may vary depending on random seed, hardware, preprocessing, and dataset version.

---

## Explainability with Grad-CAM

HybridXRayNet includes **Grad-CAM** to provide a visual explanation of model predictions.

The application generates a heatmap from the model's attention feature representation and overlays it on the input X-ray.

This provides an indication of which regions contributed most strongly to the predicted class.

> Grad-CAM is provided for interpretability and research purposes. It does not establish clinical validity or provide medical evidence.

---

## Training Configuration

The configuration used for the reported results:

| Setting        | Value         |
| -------------- | ------------- |
| Optimizer      | AdamW         |
| Learning Rate  | `3e-4`        |
| Weight Decay   | `1e-4`        |
| Batch Size     | `4`           |
| Early Stopping | Validation F1 |
| Image Size     | `224×224`     |
| Random Seed    | `42`          |
| Epochs         | Configurable  |

The batch size was selected with a **4 GB RTX 3050** in mind, but can be adjusted depending on available hardware.

---

## Project Structure

```text
HybridXRayNet/
├── train_model.py              # Model architecture, training and evaluation
├── app.py                      # Flask inference application + Grad-CAM
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html              # Web interface
├── assets/
│   └── hybridxraynet_architecture.png
├── outputs/                    # Model weights and evaluation artifacts
└── README.md
```

### Generated Outputs

Training produces artifacts such as:

```text
outputs/
├── best_model.pth
├── training_history.json
├── model_config.json
├── test_metrics.json
├── confusion_matrix.png
└── roc_curve.png
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ritanshu-Kumar/HybridXRayNet.git
cd HybridXRayNet
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

For the CUDA 12.1 PyTorch build used during development:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The project uses:

* PyTorch 2.3.1
* torchvision 0.18.1
* NumPy 1.26.4
* Pillow 10.4.0
* scikit-learn 1.5.1
* Matplotlib 3.9.2
* Flask 3.0.3

> For CPU-only environments or a different CUDA version, install the appropriate PyTorch and torchvision builds first, then install the remaining dependencies from `requirements.txt`.

---

## Train the Model

Place the dataset inside the project directory as:

```text
chest_xray/
```

Then run:

```bash
python train_model.py --data-dir chest_xray --epochs 15 --batch-size 4
```

The best model checkpoint is saved to:

```text
outputs/best_model.pth
```

Training also generates evaluation metrics, plots, configuration files, and training history inside `outputs/`.

---

## Run the Web Application

After training successfully generates:

```text
outputs/best_model.pth
```

start the Flask application:

```bash
python app.py
```

The application will be available at:

```text
http://127.0.0.1:5001
```

Upload a chest X-ray through the web interface to receive:

* **Predicted class:** NORMAL or PNEUMONIA
* **Confidence score**
* **Inference device:** CUDA or CPU
* **Grad-CAM visualization**

The application automatically uses CUDA when available and falls back to CPU otherwise.

---

## Reproducibility

The training pipeline initializes random seeds for:

* Python
* NumPy
* PyTorch
* CUDA

The default seed is:

```text
42
```

This improves experiment reproducibility, although exact results may still vary across hardware, CUDA/cuDNN versions, and software environments.

---

## Limitations

This project has several important limitations:

* It is a **research and educational prototype**, not a clinically validated diagnostic system.
* Model performance depends heavily on the training dataset.
* The model may not generalize to X-rays from different hospitals, scanners, populations, or acquisition protocols.
* The current results show a relatively high number of **false positives for the Normal class**.
* The model has not been externally validated on an independent dataset.
* Grad-CAM visualizations should be treated as model explanations rather than medical evidence.
* Real-world clinical deployment would require extensive external validation, calibration, clinical evaluation, robust dataset design, and appropriate regulatory review.

---

## Tech Stack

* **Python**
* **PyTorch**
* **torchvision**
* **scikit-learn**
* **NumPy**
* **Pillow**
* **Matplotlib**
* **Flask**

---

## License

For educational and research use.
