
# Malaria Cell Classification: Supervised vs. Unsupervised vs. Semi-Supervised Learning

A comparative machine learning project that classifies malaria-infected vs. uninfected blood cell images using three different learning paradigms — **supervised**, **unsupervised**, and **semi-supervised** — built on deep-learning feature extraction.

## 🔍 Overview

This project explores how much labeled data actually matters for medical image classification. Instead of just training one model, it trains three, under three very different assumptions about label availability, and compares them head-to-head on the same feature space.

| Approach | Label Usage | Core Idea |
|---|---|---|
| **Supervised** | 100% labels | Random Forest trained on fully labeled data |
| **Unsupervised** | 0% labels | KMeans clustering, labels only used post-hoc for evaluation |
| **Semi-Supervised** | 15% labels | Self-Training Classifier that iteratively labels the rest |

## 📊 Dataset

- **Source:** [Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) (Kaggle)
- **Classes:** `Parasitized`, `Uninfected`
- **Total images:** 27,558
- **Split:** 80% train (22,046) / 20% test (5,512), stratified

## 🧠 Approach

1. **Feature Extraction:** Each image is resized to 96×96 and passed through a pretrained **MobileNetV2** (ImageNet weights, no top layer, global average pooling) to produce a 1,280-dimensional feature vector. This turns raw images into a compact, meaningful representation before any classical ML model sees them.
2. **Supervised Learning:** A `RandomForestClassifier` is trained directly on the extracted features using full labels.
3. **Unsupervised Learning:** `KMeans` (k=2) clusters the same features with no label information at all. Cluster-to-class mapping is done post-hoc (majority vote) purely for evaluation purposes — the model itself never sees a label during training.
4. **Semi-Supervised Learning:** Only 15% of training labels are kept; the rest are masked as unlabeled (`-1`). A `SelfTrainingClassifier` wrapped around a Random Forest iteratively labels high-confidence unlabeled samples and retrains.

## 📈 Results

| Metric | Supervised (Random Forest) | Unsupervised (KMeans) | Semi-Supervised (Self-Training) |
|---|---|---|---|
| Accuracy | **0.9162** | 0.6508 | 0.8784 |
| MCC | **0.8324** | 0.3016 | 0.7571 |
| AUROC | **0.9721** | 0.6508 | 0.9543 |

**Unsupervised-only metrics:** Silhouette Score: 0.1100 · ARI: 0.0908 · NMI: 0.0666

**Key takeaway:** Full supervision performs best, as expected — but semi-supervised learning gets remarkably close using only 15% of the labels, showing strong potential for scenarios where labeled medical data is scarce or expensive to obtain. Pure unsupervised clustering, while informative, is not reliable enough on its own for a diagnostic task like this.

## 🛠️ Tech Stack

- Python, NumPy, OpenCV
- TensorFlow / Keras (MobileNetV2 for feature extraction)
- scikit-learn (RandomForestClassifier, KMeans, SelfTrainingClassifier)
- pandas (results comparison)

## 📓 Notebook

This project was originally developed and run on Kaggle (with GPU acceleration):
🔗 **[View the notebook on Kaggle](https://www.kaggle.com/code/alinazubair353-ai/supervised-learning-project-on-kaggle)**

## 🚀 Running Locally

```bash
pip install numpy opencv-python scikit-learn tensorflow pandas scipy
```

Update `DATASET_PATH` to point to your local copy of the dataset, then run the notebook cells in order.

## 👤 Author

**Alina Zubair**
- GitHub: [@alinazubair353-AI](https://github.com/alinazubair353-AI)
- LinkedIn: [Alina Zubair](https://linkedin.com/in/alina-zubair-6b957938a)
- Kaggle: [alinazubair353-ai](https://www.kaggle.com/alinazubair353-ai)
