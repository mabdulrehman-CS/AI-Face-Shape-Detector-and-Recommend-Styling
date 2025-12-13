# Facial Anthropometry and Grooming Recommendation System

## Overview
This system uses computer vision and deep learning to analyze face shapes and provide personalized grooming recommendations (hairstyles, beards, glasses).
It combines **MediaPipe Face Mesh** (geometric analysis) with **EfficientNetV2** (deep learning classification) for robust inference.

## Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Pipeline Stages
### Phase 1: Data Infrastructure
- **Download & Augment**: Downloads Niten Lama (5k) and CelebA (2k male) datasets.
- **Process**: Extracts landmarks, aligns eyes, crops with padding.
- **Run**:
  ```bash
  run_phase1.bat
  ```

### Phase 2: Model Training
- **Train**: Trains EfficientNetV2-Small with Focal Loss.
  - Stage 1: Frozen Backbone (10 epochs)
  - Stage 2: Fine-Tuning (50 epochs, unfreeze 30%)
- **Run**:
  ```bash
  run_training.bat
  ```
- **Output**: `models/final_model.keras`

### Phase 3: Inference
- **Engine**: `src/recommendation/engine.py`
- Uses a hybrid voting mechanism (CNN + Geometry).
- Rules defined in `src/recommendation/rules.json`.

## Usage
(Coming Soon: API/UI)
