# Skin Lesion Image Generation and Evaluation - DFTA

This repository presents **DFTA (Dual Flow Transformation Architecture)**, a novel skin lesion image generation method. It provides a comprehensive framework for skin lesion image generation, evaluation, and downstream task validation using the HAM10000 dataset. The framework includes a complete pipeline for training, generation, quality assessment, as well as downstream classification and segmentation tasks. Additionally, it implements end-to-end comparison with multiple state-of-the-art generation methods including ArSDM, ControlNet, T2I-Adapter, DreamBooth, DFMGAN, LesionGen, Derm-T2IM, LF-VAR, Siamese Diffusion, and Skin-Disease-Diffusion.

---

## Directory Conventions (Default Paths)

| Purpose | Path |
|---------|------|
| Generated Images | `/root/autodl-tmp/output/generate/<MODEL>/<CLASS>/images` |
| Evaluation Output | `/root/autodl-tmp/output/metric` |
| Model Cache | `/root/autodl-tmp/model` |
| Real Data | `/root/autodl-tmp/data/HAM10000/input` |

---

## Core Academic and Visual Content










---
## Quick Start

```bash
# Environment
conda env create -f main/environment.yaml
conda activate flow

# 1. Download data
bash scripts/dataset_downloader.sh

# 2. Train Flow model
bash scripts/compare_Main.sh

# 3. Run metrics
bash scripts/run_metric.sh

# 4. Train downstream classifier
bash scripts/run_classifier.sh

# 5. Train downstream segmentation
bash scripts/run_segmentation.sh
```

## 1. Environment Setup

### 1.1 Create Conda Environment

Use the provided environment file at `main/environment.yaml`:

```bash
# Install dependencies from environment file
conda env create -f main/environment.yaml

# Activate the environment (named "xcontrol" in yaml)
conda activate flow
```

### 1.2 Directory Structure

```
/root/autodl-tmp/
├── main/                          # Flow-matching code and configuration
│   ├── train.py                   # Training entry (PyTorch Lightning)
│   ├── inference.py               # Inference entry
│   ├── generate.py                # Batch generation for HAM10000
│   ├── dataset.py                 # Data loading logic
│   ├── config/flow_matching.yaml  # Model configuration
│   ├── modules/                   # Model modules (cldm/, ldm/)
│   ├── share.py                   # Common settings
│   └── data/prompt.json           # Training/prompt list
├── scripts/                        # Shell scripts for all tasks
├── checkpoint/flow/                # Training checkpoints (Lightning)
├── output/generate2/flow/         # Generated images (by class)
├── compare_main/                  # Comparison models
└── metric/                        # Evaluation metrics
```

### 1.3 Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/compare_Main.sh` | One-click Flow training + generation |
| `scripts/dataset_downloader.sh` | Download HAM10000 dataset |
| `scripts/run_metric.sh` | Run generation quality metrics |
| `scripts/run_classifier.sh` | Train downstream classifier |
| `scripts/run_segmentation.sh` | Train downstream segmentation |

---

## 2. Data Download

Download and prepare the HAM10000 dataset using the provided script.

### Script
```bash
bash scripts/dataset_downloader.sh
```

This script will:
- Download the original HAM10000 data (with mirror support)
- Split into train/val sets
- Resize images to 512×512
- Organize by class into `HAM10000_img_class` / `HAM10000_seg_class`

### Expected Output
After running, the following directories should exist:
- `data/HAM10000/input/train/HAM10000_img_class`
- `data/HAM10000/input/val/HAM10000_img_class`

### HAM10000 Classes
The dataset contains 7 skin lesion classes:
| Code | Full Name |
|------|-----------|
| akiec | Actinic Keratoses |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevi |
| vasc | Vascular Lesions |

---

## 3. Data Preprocessing

### 3.1 Hair Removal (DullRazor)

Optional preprocessing to remove hair artifacts from dermoscopy images.

```bash
bash scripts/run_dullrazor_ham10000.sh
```

This processes images in `data/HAM10000/input` for cleaner generation inputs.

---

## 4. Flow Model Training

Flow-Matching (ControlLDM) training and generation pipeline using `scripts/compare_Main.sh`.

### 4.1 One-Click Training + Generation

Run the complete pipeline with one command:

```bash
bash scripts/compare_Main.sh
```

This script handles:
- Training the Flow-Matching model
- Auto-finding the latest checkpoint
- Generating images for all 7 HAM10000 classes

### 4.2 Control Generation Only

```bash
# Skip training, only run generation
TRAIN_ENABLED=false bash scripts/compare_Main.sh
```

### 4.3 Quick Test Mode

```bash
# Test mode: 5 training steps, 1 image per class
TEST_MODE=1 bash scripts/compare_Main.sh
```

### 4.4 Using Pre-trained Weights

```bash
# Continue training with pre-trained weights
USE_PRETRAINED=1 bash scripts/compare_Main.sh

# Use pre-trained for generation only (skip training)
TRAIN_ENABLED=false USE_PRETRAINED=1 bash scripts/compare_Main.sh
```

### 4.5 Key Parameters (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIN_ENABLED` | Enable/disable training stage | `true` |
| `RUN_ENABLED` | Enable/disable generation stage | `true` |
| `TEST_MODE` | Quick test mode (5 steps, 1 img/class) | `false` |
| `USE_PRETRAINED` | Load pre-trained weights for initialization | `false` |
| `NUM_IMAGES_PER_CLASS` | Number of images to generate per class | 1500 |
| `MAX_TRAIN_STEPS` | Maximum training steps | 5000 |
| `NUM_WORKERS` | Data loading workers | 4 |
| `RESUME_CKPT` | Specify checkpoint path to resume from | - |

### 4.6 Input Data Format

`main/data/prompt.json` - One JSON per line:
```json
{"source": "/path/to/mask.png", "target": "/path/to/image.jpg", "prompt": "A photo of skin lesion"}
```

### 4.7 Output Locations

| Output | Path |
|--------|------|
| Training Checkpoint | `checkpoint/flow/lightning_logs/...` |
| Merged Model | `checkpoint/flow/PRETRAINED/merged_pytorch_model.pth` |
| Generated Images | `output/generate2/flow/<class>/images/` |

---

## 5. Comparison Model Training

Ten comparison models are available for benchmarking. Each has a one-click training + generation script.

### 5.1 Model Comparison Table

| # | Model | Script | Description |
|---|-------|--------|-------------|
| 1 | **DreamBooth** | `scripts/compare_DreamBooth.sh` | Subject-driven generation using mixed training across all classes |
| 2 | **ControlNet Depth** | `scripts/compare_ControlNet.sh` | Depth-map conditioned generation with LoRA fine-tuning |
| 3 | **T2I-Adapter** | `scripts/compare_T2i_adapter.sh` | Lightweight adapter using depth maps as conditioning |
| 4 | **ArSDM** | `scripts/compare_ArSDM.sh` | Two-stage training: base model + class-specific LoRA |
| 5 | **Siamese-Diffusion** | `scripts/compare_Siamese.sh` | Siamese architecture for paired image generation |
| 6 | **LF-VAR** | `scripts/compare_LF-VAR.sh` | Latent Variable AutoRegressive model for synthesis |
| 7 | **Skin-Disease-Diffusion** | `scripts/compare_skin-disease-diffusion.sh` | Specialized diffusion for skin disease images |
| 8 | **LesionGen** | `scripts/compare_LesionGen.sh` | Lesion-specific generation framework |
| 9 | **DFMGAN** | `scripts/compare_DFMGAN.sh` | Deep Feature Matching GAN approach |
| 10 | **Derm-T2IM** | `scripts/compare_Derm_T2IM.sh` | Text-to-image model for dermoscopy |

### 5.2 Common Usage Pattern

```bash
# Full training + generation (15000 steps, 1500 images per class)
bash scripts/compare_<Model>.sh

# Test mode (quick validation)
TEST_MODE=1 bash scripts/compare_<Model>.sh

# Training only (skip generation)
RUN_ENABLED=false bash scripts/compare_<Model>.sh

# Generation only (requires existing model)
TRAIN_ENABLED=false bash scripts/compare_<Model>.sh

# Custom parameters
MAX_TRAIN_STEPS=10000 NUM_IMAGES_PER_CLASS=500 bash scripts/compare_<Model>.sh
```

### 5.3 Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIN_ENABLED` | Enable training stage | `true` |
| `RUN_ENABLED` | Enable generation stage | `true` |
| `TEST_MODE` | Quick test mode | `false` |
| `MAX_TRAIN_STEPS` | Training steps | 15000 |
| `NUM_IMAGES_PER_CLASS` | Images per class | 1500 |

---

## 6. Medical Text Prompts

Some comparison models (e.g., ControlNet, T2I-Adapter) require text prompts for generation. This project provides pre-generated prompts for all HAM10000 classes.

### Pre-provided Prompts

The project includes ready-to-use prompts in CSV format:
- `data/metadata_train_llava.csv`
- `data/metadata_val_llava.csv`
- `data/metadata_test_llava.csv`

These contain image paths, segmentation masks, and disease descriptions for each class.

### Custom Prompt Generation (Optional)

If you want to generate custom medical prompts using LLaVA, run:

```bash
bash scripts/run_generate_llava_med.sh
```

This will:
1. Clone LLaVA repository if not present
2. Generate medical prompts from HAM10000 metadata
3. Save to `/root/autodl-tmp/model` and `/root/autodl-tmp/data`

#### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_ID` | LLaVA model identifier | - |
| `MODEL_DIR` | Local model directory | - |
| `LIMIT` | Maximum prompts to generate | - |

---

## 7. Basic Metrics Evaluation

Evaluate generation quality using multiple metrics.

### Metrics Included

| Metric | Description |
|--------|-------------|
| KID | Kernel Inception Distance |
| Density & Coverage | Distribution coverage metrics |
| MS-SSIM | Multi-Scale Structural Similarity |
| LPIPS | Learned Perceptual Image Patch Similarity |
| Inception Score | Generation quality score |
| Vendi | Diversity score |

### 7.1 Run Metrics

```bash
bash scripts/run_metric.sh
```

### 7.2 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEN_ROOT` | Generated images root | `/root/autodl-tmp/output/generate` |
| `REAL_ROOT` | Real images root | `/root/autodl-tmp/data/HAM10000/input` |
| `REAL_SPLIT` | Data split (train/val) | `val` |
| `OUT_DIR` | Output directory | `/root/autodl-tmp/output/metric` |
| `BATCH_SIZE` | Batch size | 16 |
| `MODEL_LIST` | Specific models to evaluate | All |

### 7.3 Output

- JSON results: `output/metric/metrics_*.json`
- Summary CSV: `output/metric/summary_metrics.csv`

### 7.4 Pre-downloaded Weights

To avoid repeated downloads, ensure Inception weights exist at:
```
/root/autodl-tmp/model/inception_v3_google-0cc3c7bd.pth
```

Set `TORCH_HOME=/root/autodl-tmp/model` to cache models.

---

## 8. Downstream Classification

Train and evaluate classification models on generated images.

### 8.1 Run Classifier

```bash
bash scripts/run_classifier.sh
```

### 8.2 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REAL_TRAIN` | Real training data | `data/HAM10000/input/train/HAM10000_img_class` |
| `REAL_VAL` | Real validation data | `data/HAM10000/input/val/HAM10000_img_class` |
| `SYN_ROOT` | Generated images root | `output/generate` |
| `MIX_ROOT` | Mixed dataset root | `output/mixed_datasets/classifier` |
| `OUT_ROOT` | Output root | `output/classifier` |
| `EPOCHS` | Training epochs | 30 |
| `TRAIN_BATCH` | Training batch size | 64 |
| `SAMPLE` | Sample mode | 0 |

### 8.3 Supported Models

- ResNet18
- ResNet50
- EfficientNet-B0

### 8.4 Output

- Checkpoints: `output/classifier/checkpoints/`
- Logs: `output/classifier/logs/`
- Metrics: `output/classifier/metrics/`

---

## 9. Downstream Segmentation

Train and evaluate segmentation models on generated images.

### 9.1 Run Segmentation

```bash
bash scripts/run_segmentation.sh
```

### 9.2 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REAL_ROOT` | Real data root | `data/HAM10000/input` |
| `SYN_ROOT` | Generated images root | `output/generate` |
| `MIX_ROOT` | Mixed dataset root | `output/mixed_datasets/segmentation` |
| `OUT_ROOT` | Output root | `output/segmentation` |
| `MAX_STEPS` | Training max steps | 15000 |
| `VAL_INTERVAL` | Validation interval | 3000 |
| `BATCH_SIZE` | Batch size | 4 |
| `IMAGE_SIZE` | Image resolution | 512 |

### 9.3 Output

- Summary JSON: `output/segmentation/summary/metrics_summary.json`
- Summary CSV: `output/segmentation/summary/metrics_summary.csv`
- Summary Excel: `output/segmentation/summary/metrics_summary.xlsx`


---

## 10. Open Source and Academic Standards












---
