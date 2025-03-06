import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modulos.principal import mostrar_pagina_principal
from modulos.Reporte import preparar_datos, generar_reporte_completo
from modulos.graficos import generar_grafico_barras, generar_radar_chart, generar_grafico_dispersion
from modulos.cargararchivo import cargar_csv, filtrar_datos


# Configuración de la página
st.set_page_config(
    page_title="Visualizador de Excel",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("Visualizador de archivo Excel")

# Barra lateral
st.sidebar.header("Panel de Control")
opcion = st.sidebar.selectbox(
    "Selecciona una opción",
    ["Inicio", "Análisis y Entrenamiento", "Gráficos", "Cargar CSV"]
)

# Contenido principal
if opcion == "Inicio":
    mostrar_pagina_principal()

elif opcion == "Cargar CSV":
    df = cargar_csv()
    if df is not None:
        df_filtrado = filtrar_datos(df)

elif opcion == "Gráficos":
    archivo = st.file_uploader("Selecciona un archivo (CSV o Excel)", type=['csv', 'xlsx'])
    if archivo is not None:
        try:
            # Cargar el archivo
            if archivo.name.endswith('.csv'):
                df = pd.read_csv(archivo)
            else:
                df = pd.read_excel(archivo)
            
            # Generar los tres tipos de gráficos
            generar_grafico_barras(df)
            generar_radar_chart(df)
            generar_grafico_dispersion(df)
                
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")
            
elif opcion == "Análisis y Entrenamiento":
    st.write("## Análisis y Entrenamiento del Modelo")
    archivo = st.file_uploader("Selecciona un archivo (CSV o Excel)", type=['csv', 'xlsx'])
    
    if archivo is not None:
        try:
            # Cargar el archivo
            if archivo.name.endswith('.csv'):
                df = pd.read_csv(archivo)
            else:
                df = pd.read_excel(archivo)
                
            # Guardar en session_state
            st.session_state['datos'] = df
            
            # Mostrar vista previa
            st.write("### Vista previa de los datos")
            st.dataframe(df.head())
            
            # Preparar datos para el análisis
            X, y, vars_pred, var_obj = preparar_datos(df)
            
            if X is not None:
                if st.button("Generar Reporte Completo"):
                    with st.spinner("Generando análisis completo..."):
                        exito = generar_reporte_completo(X, y, vars_pred, var_obj)
                        if exito:
                            st.success("¡Reporte generado exitosamente!")
                        else:
                            st.error("Error al generar el reporte")
                            
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")