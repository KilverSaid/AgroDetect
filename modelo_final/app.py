import json
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# 1. Configuración de página (OBLIGATORIAMENTO LA PRIMERA LÍNEA DE STREAMLIT)
st.set_page_config(
    page_title="AgroDetect — Detección en Café",
    page_icon="🌿",
    layout="centered"
)

# Intentar importar e inicializar Google Generative AI
try:
    import google.generativeai as genai
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini_disponible = True
    else:
        gemini_disponible = False
except Exception:
    gemini_disponible = False

# Constantes predeterminadas
DEFAULT_CLASSES = ["sana", "roya", "cercospora", "phoma", "arana_roja", "minador"]
DEFAULT_IMG_SIZE = (224, 224)
DEFAULT_UMBRAL = 0.60


def generar_recomendacion_ia(enfermedad, probabilidad):
    """Genera recomendaciones agronómicas probando modelos de Gemini con manejo de errores."""
    if not gemini_disponible:
        return "⚠️ La integración con Gemini no está disponible. Verifica tu API Key en los Secrets de Streamlit."

    prompt = f"""
    Eres un agrónomo experto en el cultivo de café en Honduras.
    Un modelo de visión por computadora identificó la siguiente afección en una hoja de café:
    - Diagnóstico: {enfermedad}
    - Certidumbre del modelo: {probabilidad*100:.1f}%

    Proporciona un plan de tratamiento claro y estructurado para un productor agrícola local:
    1. Descripción breve y gravedad de la afección.
    2. Medidas de control cultural / preventivo.
    3. Manejo o control biológico / químico recomendado en la región.
    4. Advertencias o precauciones inmediatas.

    Mantén un tono profesional, accesible y práctico.
    """

    # Modelos estándares para probar en orden
    modelos_a_probar = ["gemini-1.5-flash", "gemini-1.5-pro"]

    # Intentar cargar modelos dinámicos si están disponibles
    try:
        modelos_remotos = [
            m.name.replace("models/", "") 
            for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        # Filtrar modelos no deseados o de prueba
        modelos_filtrados = [
            m for m in modelos_remotos 
            if not any(bad in m for bad in ["2.5", "0.0.1"])
        ]
        if modelos_filtrados:
            modelos_a_probar = modelos_filtrados + modelos_a_probar
    except Exception:
        pass

    ultimo_error = None
    for nombre_modelo in modelos_a_probar:
        try:
            model = genai.GenerativeModel(nombre_modelo)
            response = model.generate_content(prompt)
            return response.text
        except Exception as err:
            ultimo_error = err
            continue

    return f"No se pudo consultar la API de Gemini. Detalle: {ultimo_error}"

@st.cache_resource
def cargar_modelo_y_config():
    """Carga el modelo TensorFlow (.keras o .h5) y el archivo de configuración JSON."""
    config = {
        "clases": DEFAULT_CLASSES,
        "img_size": DEFAULT_IMG_SIZE,
        "umbral_decision": DEFAULT_UMBRAL,
        "arquitectura_seleccionada": "MobileNetV2"
    }

    if os.path.exists("config_app.json"):
        try:
            with open("config_app.json", "r", encoding="utf-8") as f:
                config.update(json.load(f))
        except Exception as e:
            st.warning(f"No se pudo cargar config_app.json: {e}. Usando valores por defecto.")

    model_paths = [
        f"agrodetect_{config['arquitectura_seleccionada'].lower()}.keras",
        f"agrodetect_{config['arquitectura_seleccionada'].lower()}.h5",
        "agrodetect_mobilenetv2.keras",
        "agrodetect_mobilenetv2.h5",
        "modelo_final/agrodetect_mobilenetv2.keras",
        "modelo_final/agrodetect_mobilenetv2.h5"
    ]

    model = None
    path_usado = None

    for path in model_paths:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path, compile=False)
                path_usado = path
                break
            except Exception:
                continue

    return model, config, path_usado


def preprocesar_y_predecir(image, model, config):
    """Preprocesa la imagen de acuerdo a MobileNetV2 y realiza la predicción."""
    img_size = tuple(config["img_size"])
    image = image.convert("RGB").resize(img_size)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    probs = model.predict(img_preprocessed, verbose=0)[0]
    idx_pred = int(np.argmax(probs))
    confianza = float(probs[idx_pred])

    return idx_pred, confianza, probs


# --- INTERFAZ GRÁFICA (STREAMLIT) ---

st.title("🌿 AgroDetect")
st.subheader("Sistema Inteligente para la Detección de Enfermedades y Plagas Foliares en Café")
st.markdown(
    "**UNAH Campus Comayagua (UNAH-CURC)**  \n"
    "*Proyecto de Inteligencia Artificial*"
)
st.divider()

model, config, path_usado = cargar_modelo_y_config()

if model is None:
    st.error("No se encontró ningún archivo de modelo guardado (.keras o .h5).")
    st.info("Asegúrate de colocar tu modelo entrenado en la ruta adecuada dentro del repositorio.")
    st.stop()

# Menú lateral
with st.sidebar:
    st.header("⚙️ Configuración")
    st.write(f"**Arquitectura:** {config.get('arquitectura_seleccionada', 'MobileNetV2')}")
    st.write(f"**Modelo cargado:** `{path_usado}`")
    
    umbral = st.slider(
        "Umbral de confianza mínima",
        min_value=0.30,
        max_value=0.95,
        value=float(config["umbral_decision"]),
        step=0.05,
        help="Si la probabilidad principal es menor a este valor, se marcará el resultado como incierto."
    )
    
    st.divider()
    st.markdown("### Clases detectables:")
    for c in config["clases"]:
        st.markdown(f"- `{c}`")

# Carga de imagen
uploaded_file = st.file_uploader(
    "Selecciona o toma una fotografía de la hoja de café:",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Imagen ingresada", use_column_width=True)

    with st.spinner("Analizando hoja de café..."):
        idx_pred, confianza_max, probs = preprocesar_y_predecir(image, model, config)
        clases = config["clases"]
        clase_principal = clases[idx_pred]

    with col2:
        st.markdown("### Diagnóstico y Porcentajes")

        # Verificar umbral
        if confianza_max < umbral:
            st.warning("⚠️ **Diagnóstico Incierto**")
            st.write(
                f"El nivel de confianza del modelo ({confianza_max:.1%}) está por debajo del umbral mínimo configurado ({umbral:.1%})."
            )
            st.info("💡 **Recomendación:** Consultar directamente con un técnico del **IHCAFE**.")
        else:
            if clase_principal.lower() == "sana":
                st.success(f"🌱 **Hoja Sana** (Certeza: {confianza_max:.1%})")
                st.write("La planta no muestra patrones significativos de plagas o enfermedades.")
            else:
                st.error(f"⚠️ **Detección Principal: {clase_principal.upper()}** ({confianza_max:.1%})")

        st.divider()

        # Mostrar desglose de barras
        st.markdown("#### Porcentaje de presencia detectada:")
        resultados_ordenados = sorted(zip(clases, probs), key=lambda x: x[1], reverse=True)

        for nombre_clase, porcentaje in resultados_ordenados:
            pct_val = float(porcentaje)
            etiqueta = nombre_clase.replace("_", " ").title()
            st.write(f"• **{etiqueta}:** `{pct_val:.2%}`")
            st.progress(pct_val)

    # --- SECCIÓN DE RECOMENDACIÓN TÉCNICA VÍA GEMINI ---
    if confianza_max >= umbral and clase_principal.lower() != "sana":
        st.divider()
        st.markdown("### 📋 Plan Recomendado de Tratamiento (Asistente Agrónomo IA - Gemini)")
        
        with st.spinner("Consultando recomendaciones agronómicas especializadas..."):
            try:
                recomendacion = generar_recomendacion_ia(clase_principal, confianza_max)
                st.info(recomendacion)
            except Exception as e:
                st.error(f"No se pudo generar la recomendación automatizada: {e}")
