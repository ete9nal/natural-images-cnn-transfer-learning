# 🌿 Natural Images CNN & Transfer Learning

An image classification project that compares a custom CNN against VGG16 and MobileNetV2 transfer learning models on 8 natural image categories.

---

## 📌 Problem

Given a photo of a natural object, can a model correctly identify whether it's an airplane, car, cat, dog, flower, fruit, motorbike, or person? This project compares building a CNN from scratch vs reusing pretrained ImageNet weights.

---

## 📁 Project Structure

```
natural-images-cnn-transfer-learning/
│
├── app/
│   └── streamlit_app.py         # Streamlit demo app
│
├── history/
│   ├── history_cnn.pkl
│   ├── history_vgg.pkl
│   ├── history_vgg_tuned.pkl
│   ├── history_mob.pkl
│   └── history_mob_tuned.pkl
│
├── models/
│   └── .gitkeep                 # Models not included — see Models section
│
├── notebooks/
│   └── natural_images.ipynb     # Full training pipeline
│
├── .gitignore
└── README.md
```

---

## 📊 Dataset

> **Source:** [prasunroy/natural-images](https://www.kaggle.com/datasets/prasunroy/natural-images) via Kaggle

| Property         | Value                  |
|------------------|------------------------|
| Total images     | 6,899                  |
| Train / Val      | 5,520 / 1,379 (80/20)  |
| Image size       | 160×160                |
| Number of classes| 8                      |
| Classes          | airplane, car, cat, dog, flower, fruit, motorbike, person |

---

## 🤖 Models & Results

| Model        | Val Accuracy | Fine-tuning         |
|--------------|-------------|---------------------|
| Custom CNN   | ~92%        | —                   |
| VGG16        | ~99%        | block5 unfrozen     |
| MobileNetV2  | ~95%+       | block_16 + Conv_1 unfrozen |

> **Note:** Model `.keras` files are not included due to size (50–151 MB).
> To reproduce: run `notebooks/natural_images.ipynb` in Google Colab with Kaggle API token.

---

## 💡 Key Insights

**VGG16 outperforms** the custom CNN significantly — from ~92% to ~99% val accuracy with just 10 epochs of feature extraction and 15 epochs of fine-tuning.

**Transfer learning works even on small datasets** — only 5,520 training images, yet pretrained ImageNet features generalize well to natural image categories.

**MobileNetV2 is a strong alternative** — nearly as accurate as VGG16 but ~25% smaller in file size, making it more practical for deployment.

**Custom CNN still performs well** — ~92% val accuracy from scratch, but requires more epochs and careful tuning to compete with pretrained models.

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/ete9nal/natural-images-cnn-transfer-learning.git
cd natural-images-cnn-transfer-learning

# Install dependencies
pip install tensorflow streamlit pillow matplotlib

# Train models (Google Colab recommended)
# Open notebooks/natural_images.ipynb and run all cells
# Models will be saved to models/ directory

# Run Streamlit app (after training)
cd app
streamlit run streamlit_app.py
```

---

## 🛠️ Tech Stack

- **Python 3.11**
- **TensorFlow / Keras** — model building and training
- **NumPy / Pillow** — image processing
- **Matplotlib** — training history visualization
- **Streamlit** — demo web app
- **Google Colab** — training environment
- **Kaggle** — dataset source
