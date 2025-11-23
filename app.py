import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd

# ======================================
# Configuración de la página
# ======================================
st.set_page_config(
    page_title="Detección de Fibrosis Hepática",
    page_icon="🩺",
    layout="centered"
)

# ======================================
# Carga del modelo (cacheado)
# ======================================
@st.cache_resource
def load_model():
    """
    Descarga (si hace falta) y carga el modelo entrenado desde Hugging Face.
    """
    import requests  # lo importo aquí para no usarlo si no se llama a la función

    model_url = "https://huggingface.co/Pinzon98/proyectoDL/resolve/main/best_model.keras"
    model_path = "best_model.keras"

    if not os.path.exists(model_path):
        with st.spinner("Descargando el modelo... esto puede tomar un momento."):
            response = requests.get(model_url)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(response.content)

    model = tf.keras.models.load_model(model_path)
    return model


model = load_model()

# ======================================
# Utilidades de preprocesado
# ======================================
def get_target_size_from_model(model):
    """
    Obtiene el tamaño de entrada esperado por el modelo (alto, ancho).
    Ejemplo típico: (225, 225)
    """
    input_shape = model.input_shape  # (None, H, W, C)
    if len(input_shape) == 4:
        return (input_shape[1], input_shape[2])
    # Valor por defecto por si acaso
    return (128, 128)


TARGET_SIZE = get_target_size_from_model(model)


def preprocess_image(_image, target_size=TARGET_SIZE):
    """Preprocesar imagen para el modelo de fibrosis hepática."""
    # Convertir a RGB si es necesario
    if _image.mode != "RGB":
        image = _image.convert("RGB")
    else:
        image = _image

    # Redimensionar
    image = image.resize(target_size)

    # Convertir a array numpy
    img_array = np.array(image)          # valores 0–255

    # Mismo rango que en entrenamiento → NO normalizar
    img_array = img_array.astype(np.float32)

    # Agregar dimensión batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array



# ======================================
# Predicción
# ======================================
CLASS_NAMES = [
    "Fase 0 (Sin fibrosis)",
    "Fase 1 (Fibrosis portal leve)",
    "Fase 2 (Fibrosis significativa)",
    "Fase 3 (Fibrosis avanzada)",
    "Fase 4 (Cirrosis)",
]


def predict_image(model, image):
    """Hacer predicción con el modelo de fibrosis hepática."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)

    predicted_class_idx = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][predicted_class_idx])

    return CLASS_NAMES[predicted_class_idx], confidence, prediction[0]


# ======================================
# Imágenes de ejemplo (Kaggle y Santa Fe)
# ======================================
# Suponemos que estos archivos están en la misma carpeta que app.py.
# Usa la versión "_1" de cada fase.
KAGGLE_IMAGES = {
    "F0": "K_F0_1.jpg",
    "F1": "K_F1_1.jpg",
    "F2": "K_F2_1.png",
    "F3": "K_F3_1.png",
    "F4": "K_F4_1.jpg",
}

SANTAFE_IMAGES = {
    "F0": "SF_F0_1.jpg",
    "F1": "SF_F1_1.jpg",
    "F2": "SF_F2_1.jpg",
    "F3": "SF_F3_1.png",
    "F4": "SF_F4_1.jpg",
}


@st.cache_data
def load_sample_image_local(path: str):
    """Carga y cachea una imagen de ejemplo desde el disco."""
    try:
        return Image.open(path)
    except Exception as e:
        st.error(f"Error cargando imagen de ejemplo '{path}': {e}")
        return None


def set_sample_image_from_path(path: str):
    image = load_sample_image_local(path)
    if image:
        st.session_state.image_to_predict = image


def render_example_row(title: str, images_dict: dict, button_prefix: str):
    """
    Dibuja una fila de imágenes de ejemplo (F0..F4) con sus botones.
    title: texto que se muestra encima (Kaggle / Santa Fe)
    images_dict: diccionario fase -> ruta
    button_prefix: prefijo para los keys de los botones (para que no choquen)
    """
    st.markdown(f"**{title}**")
    fases_ordenadas = ["F0", "F1", "F2", "F3", "F4"]
    cols = st.columns(len(fases_ordenadas))

    for i, fase in enumerate(fases_ordenadas):
        with cols[i]:
            path = images_dict.get(fase)
            if path is None:
                continue
            img = load_sample_image_local(path)
            if img:
                st.image(
                    img,
                    caption=f"{fase}",
                    use_container_width=True
                )
                if st.button(
                    f"Usar {fase}",
                    key=f"{button_prefix}_{fase}"
                ):
                    set_sample_image_from_path(path)


# ======================================
# Interfaz principal
# ======================================
def main():
    st.title("🩺 Clasificador de Fibrosis Hepática (F0–F4)")
    st.markdown(
        """
Este demo utiliza una red neuronal convolucional (CNN) para **clasificar fibrosis hepática en 5 fases (0–4)** a partir de imágenes histológicas.
"""
    )
    st.markdown("---")

    if model is None:
        st.error("❌ No se pudo cargar el modelo.")
        return

    st.success("✅ Modelo cargado correctamente")

    # --------------------------
    # Sidebar
    # --------------------------
    with st.sidebar:
        st.header("📊 Información del Modelo")
        st.info(
            f"""
**Tipo de modelo**: CNN para clasificación de fibrosis hepática

**Entrada**: Imagen {TARGET_SIZE[0]}x{TARGET_SIZE[1]} píxeles

**Número de clases**: 5 fases (F0–F4)

**Salida (softmax)**: Probabilidad para cada fase.
"""
        )

        if st.checkbox("Ver arquitectura del modelo"):
            st.text("Capas del modelo:")
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text("\n".join(model_summary))

    # --------------------------
    # Estado de sesión
    # --------------------------
    if "image_to_predict" not in st.session_state:
        st.session_state.image_to_predict = None

    # --------------------------
    # Subir imagen
    # --------------------------
    st.header("📤 Subir imagen de biopsia / estudio")

    uploaded_file = st.file_uploader(
        "Selecciona una imagen (por ejemplo, una lámina de biopsia hepática)",
        type=["jpg", "jpeg", "png"],
        help="Formatos soportados: JPG, JPEG, PNG",
    )

    if uploaded_file is not None:
        st.session_state.image_to_predict = Image.open(uploaded_file)

    # --------------------------
    # Imágenes de ejemplo
    # --------------------------
    st.markdown("### Imágenes de ejemplo")

    # Fila 1: Kaggle
    render_example_row("Fila 1: Imágenes Kaggle (K_)", KAGGLE_IMAGES, "kaggle")

    # Fila 2: Santa Fe
    render_example_row("Fila 2: Imágenes Santa Fe (SF_)", SANTAFE_IMAGES, "santafe")

    image_to_predict = st.session_state.image_to_predict

    # --------------------------
    # Mostrar imagen original y procesada
    # --------------------------
    if image_to_predict:
        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📷 Imagen a Analizar")
            st.image(image_to_predict, use_container_width=True)

        with col2:
            st.subheader("🔍 Imagen Procesada")
            processed_display = preprocess_image(image_to_predict)
            st.image(
                processed_display[0],
                caption=f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]} normalizada",
                use_container_width=True,
            )

    # --------------------------
    # Botón de predicción
    # --------------------------
    if image_to_predict is not None:
        if st.button("🚀 Clasificar Fibrosis", type="primary"):
            with st.spinner("🧠 Analizando imagen..."):
                predicted_class, confidence, all_predictions = predict_image(
                    model, image_to_predict
                )

                st.markdown("---")
                st.subheader("📋 Resultados")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fase Predicha", predicted_class)
                with col2:
                    st.metric("Confianza del modelo", f"{confidence:.2%}")

                st.progress(confidence)

                # Gráfico de probabilidades
                st.subheader("📊 Distribución de Probabilidades (Fases 0–4)")

                prob_data = {
                    "Fase": [f"F{i}" for i in range(len(all_predictions))],
                    "Probabilidad": [float(p) for p in all_predictions],
                }

                df = pd.DataFrame(prob_data)
                st.bar_chart(data=df.set_index("Fase")["Probabilidad"])

                # Interpretación NO clínica
                st.subheader("🧠 Interpretación (NO clínica)")

                fase_idx = CLASS_NAMES.index(predicted_class)

                if fase_idx == 0:
                    st.success(
                        "🔵 El modelo sugiere **ausencia de fibrosis significativa (F0)**."
                    )
                elif fase_idx == 1:
                    st.info("🟢 El modelo sugiere **fibrosis leve (F1)**.")
                elif fase_idx == 2:
                    st.warning("🟡 El modelo sugiere **fibrosis significativa (F2)**.")
                elif fase_idx == 3:
                    st.error("🟠 El modelo sugiere **fibrosis avanzada (F3)**.")
                else:
                    st.error("🔴 El modelo sugiere **fibrosis severa / cirrosis (F4)**.")


# Ejecutar app
if __name__ == "__main__":
    main()
