import streamlit as st

def mostrar_pagina_principal():
    # Solo mantiene el logo y oculta el footer
    st.markdown("""
        <style>
        footer {
            visibility: hidden;
        }
        .logo-text {
            position: fixed;
            right: 20px;
            bottom: 20px;
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