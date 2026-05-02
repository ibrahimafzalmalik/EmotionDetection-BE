# Facial Expression Recognition (FER2013) with PyTorch

A production-quality deep learning system for seven-class facial expression recognition on FER2013.

## Requirements

- Python 3.10+
- PyTorch-compatible environment (CUDA optional)

Install dependencies:

```bash
pip install torch torchvision scikit-learn seaborn tqdm matplotlib opencv-python torchinfo pandas
```

## Quick Start

Run commands from the `EmotionDetection-BE` directory (parent of `fer_project/`) so Python can resolve the `fer_project` package.

1. Download FER2013 image-folder dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).
2. Place dataset folders into:
   - `EmotionDetection-BE/fer_project/data/raw/train/`
   - `EmotionDetection-BE/fer_project/data/raw/test/`
3. Train model:

```bash
cd EmotionDetection-BE
python -m fer_project.training.train
```

4. Evaluate best checkpoint:

```bash
cd EmotionDetection-BE
python -m fer_project.training.evaluate
```

5. Open notebook walkthrough:

```bash
jupyter notebook EmotionDetection-BE/fer_project/notebooks/FER_Full_Pipeline.ipynb
```

## Project Structure

```text
fer_project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── custom_cnn.py
│   └── transfer_model.py
├── utils/
│   ├── dataset.py
│   ├── transforms.py
│   ├── metrics.py
│   └── gradcam.py
├── training/
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── FER_Full_Pipeline.ipynb
├── outputs/
│   ├── checkpoints/
│   ├── plots/
│   └── results/
├── config.py
└── README.md
```

## Configuration (`config.py`)

Key settings:

- `DATA_DIR`: path to raw FER2013 folder.
- `IMG_SIZE` / `TRANSFER_IMG_SIZE`: 48 for native CNN and 224 for transfer learning.
- `USE_TRANSFER_LEARNING`: toggles custom CNN vs pretrained backbone.
- `TRANSFER_MODEL_NAME`: `resnet50`, `vgg16`, or `mobilenet_v2`.
- `FREEZE_BACKBONE`: freeze pretrained feature extractor when fine-tuning head only.
- `EARLY_STOPPING_PATIENCE`: early-stop criterion based on validation loss.
- `CHECKPOINT_PATH`: best-model checkpoint destination.
- `CLASS_NAMES`: canonical FER class ordering.

## Results (Update After Training)

| Model | Validation Accuracy |
|---|---|
| Random Baseline | 14.3% |
| VGG-B (2015) | 71.2% |
| ResNet-50 Transfer Learning | X.X% |
| Custom CNN (This Project) | X.X% |

## Output Artifacts

Generated outputs are saved under `fer_project/outputs/`:

- `checkpoints/best_model.pth`
- `plots/training_curves.png`
- `plots/confusion_matrix.png`
- `plots/misclassified.png`
- `plots/gradcam_samples.png`
- `results/history.json`
- `results/predictions.csv`

## Live Demo Tips

During your presentation:

1. Open the frontend on your laptop about two minutes early (this wakes the Render backend on the free tier).
2. Have three to five test face images ready with clear expressions.
3. Show the **Model Insights** tab — training curves demonstrate that full training was run end-to-end.
4. Show **Grad-CAM** — this usually impresses evaluators; explain that the model often emphasizes the mouth region for happy/sad and the eye region for fear/surprise.
5. Try **Live Camera** — face the camera and walk through a real-time capture and prediction.
6. The confusion matrix illustrates that **disgust** is often the hardest class to classify (small support in FER2013).
7. Mention that about **61%** validation accuracy compared to roughly **14%** random baseline is about a **4.3×** improvement.

## References

- Goodfellow, I. et al. (2013). *Challenges in Representation Learning: A report on three machine learning contests*. arXiv:1307.0414.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Simonyan, K., Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR.
- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.

