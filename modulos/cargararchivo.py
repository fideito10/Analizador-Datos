import streamlit as st
import pandas as pd

def cargar_csv():
    st.write("## Carga y visualización de archivos CSV")
    
    archivo_csv = st.file_uploader("Selecciona un archivo CSV", type=['csv'])
    
    if archivo_csv is not None:
        try:
            df = pd.read_csv(archivo_csv)
            return df
                
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")
            return None

def filtrar_datos(df):
    if df is not None:
        st.write("## Filtrado de datos")
        
        columnas = df.columns.tolist()
        columnas_seleccionadas = st.multiselect(
            "Selecciona las columnas para filtrar:",
            options=columnas
        )
        
        df_filtrado = df.copy()
        
        for columna in columnas_seleccionadas:
            st.write(f"### Filtrar valores para: {columna}")
            valores_unicos = sorted(df[columna].unique())
            valores_seleccionados = st.multiselect(
                f"Selecciona los valores para {columna}:",
                options=valores_unicos,
                default=None
            )
            
            if valores_seleccionados:
                df_filtrado = df_filtrado[df_filtrado[columna].isin(valores_seleccionados)]
        
        if len(columnas_seleccionadas) > 0:
            st.write("### Resultados del filtrado:")
            st.write(f"Filas después del filtrado: {len(df_filtrado)}")
            st.dataframe(df_filtrado)
            
            if st.button("Descargar datos filtrados"):
                csv = df_filtrado.to_csv(index=False)
                st.download_button(
                    label="Descargar como CSV",
                    data=csv,
                    file_name="datos_filtrados.csv",
                    mime="text/csv"
                )
        
        return df_filtrado