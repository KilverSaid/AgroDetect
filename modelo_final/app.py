import json
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import google.generativeai as genai

# Configuración inicial
st.set_page_config(
    page_title="AgroDetect — Detección en Café",
    page_icon="🌿",
    layout="centered"
)

# Configurar API Key global de Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def generar_recomendacion_ia(enfermedad, probabilidad):
    """Genera recomendaciones agronómicas mediante la API estable de Gemini."""
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

    # Inicializar el modelo con la librería estándar
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text
