import streamlit as st

def mostrar_pagina_principal():
    # Configuración del tema verde oscuro
    st.markdown("""
        <style>
        .main {
            background-color: #1e3d2f;
            color: white;
        }
        .stButton>button {
            background-color: #2c5a44;
            color: white;
        }
        .stExpander {
            background-color: #2c5a44;
        }
        .css-1d391kg {
            background-color: #1e3d2f;
        }
        footer {
            visibility: hidden;
        }
        .logo-text {
            position: fixed;
            right: 20px;
            bottom: 20px;
            color: #4CAF50;
            font-size: 18px;
            font-weight: bold;
            font-style: italic;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        </style>
        """, unsafe_allow_html=True)

    # Título principal con emoji
    st.write("# 📊 Bienvenido al Analizador de Datos")
    
    # Descripción principal
    st.markdown("""
    ### ¡Bienvenido/a! Esta herramienta está diseñada para transformar tus datos en insights accionables. Aquí encontrarás:
    
    #### 📈 Análisis y Entrenamiento
    - Carga de archivos CSV y Excel
    - Selección inteligente de variables
    - Generación de modelos predictivos
    - Análisis estadístico completo
    
    #### 📊 Graficos
    - Gráficos de barras interactivos
    - Gráficos de radar para comparaciones
    - Diagramas de dispersión
    
    #### 🔍 Carga de Cvs
    - Manipulación visual mediante previsualización interactiva
    - Personalización profunda 
    - Descargas ágiles en múltiples formatos
    - Memoria de configuraciones para mantener tus preferencias
    """)
    
 
    
    # Logo Juan Calvo
    st.markdown("""
        <div class='logo-text'>
            Juan Calvo
        </div>
        """, unsafe_allow_html=True)