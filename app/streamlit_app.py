import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image

# Назви класів
CLASS_NAMES = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

st.title("Natural Images Classifier")
st.write("Класифікація зображень за допомогою CNN, VGG16 та MobileNetV2")

# ── Завантаження моделей ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    import tensorflow as tf
    models = {}
    models['CNN'] = tf.keras.models.load_model('../models/model_cnn.keras')
    models['VGG16'] = tf.keras.models.load_model('../models/model_vgg.keras')
    models['MobileNetV2'] = tf.keras.models.load_model('../models/model_mob.keras')
    return models

models = load_models()

# ── Завантаження історій ──────────────────────────────────────────────────────
def load_history(filename):
    with open(f'../history/{filename}', 'rb') as f:
        return pickle.load(f)

# ── Таби ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Класифікація", "Графіки навчання"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Класифікація
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    uploaded_file = st.file_uploader("Завантаж зображення", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Завантажене зображення", width=300)

        img_array = np.array(img.resize((160, 160)), dtype=np.float32)

        from keras.applications.vgg16 import preprocess_input as vgg_preprocess
        from keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess

        img_cnn = np.expand_dims(img_array / 255.0, axis=0)
        img_vgg = np.expand_dims(vgg_preprocess(img_array.copy()), axis=0)
        img_mob = np.expand_dims(mob_preprocess(img_array.copy()), axis=0)

        st.subheader("Результати:")

        pred_cnn = models['CNN'].predict(img_cnn, verbose=0)[0]
        st.write(f"**CNN:** {CLASS_NAMES[np.argmax(pred_cnn)]} ({np.max(pred_cnn)*100:.1f}%)")

        pred_vgg = models['VGG16'].predict(img_vgg, verbose=0)[0]
        st.write(f"**VGG16:** {CLASS_NAMES[np.argmax(pred_vgg)]} ({np.max(pred_vgg)*100:.1f}%)")

        pred_mob = models['MobileNetV2'].predict(img_mob, verbose=0)[0]
        st.write(f"**MobileNetV2:** {CLASS_NAMES[np.argmax(pred_mob)]} ({np.max(pred_mob)*100:.1f}%)")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Графіки навчання
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Графіки навчання")

    model_choice = st.selectbox("Обери модель", ['CNN', 'VGG16', 'MobileNetV2'])

    if model_choice == 'CNN':
        h = load_history('history_cnn.pkl')
        acc, val_acc = h['accuracy'], h['val_accuracy']
        loss, val_loss = h['loss'], h['val_loss']

    elif model_choice == 'VGG16':
        h1 = load_history('history_vgg.pkl')
        h2 = load_history('history_vgg_tuned.pkl')
        acc = h1['accuracy'] + h2['accuracy']
        val_acc = h1['val_accuracy'] + h2['val_accuracy']
        loss = h1['loss'] + h2['loss']
        val_loss = h1['val_loss'] + h2['val_loss']

    elif model_choice == 'MobileNetV2':
        h1 = load_history('history_mob.pkl')
        h2 = load_history('history_mob_tuned.pkl')
        acc = h1['accuracy'] + h2['accuracy']
        val_acc = h1['val_accuracy'] + h2['val_accuracy']
        loss = h1['loss'] + h2['loss']
        val_loss = h1['val_loss'] + h2['val_loss']

    st.write("**Accuracy**")
    st.line_chart(pd.DataFrame({'Train accuracy': acc, 'Val accuracy': val_acc}))

    st.write("**Loss**")
    st.line_chart(pd.DataFrame({'Train loss': loss, 'Val loss': val_loss}))

    st.subheader("Фінальні метрики")
    col1, col2 = st.columns(2)
    col1.metric("Val Accuracy", f"{max(val_acc)*100:.2f}%")
    col2.metric("Val Loss", f"{min(val_loss):.4f}")