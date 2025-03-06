import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def generar_grafico_barras(df):
    st.write("## Gráfico de Barras")
    
    # Selección de columnas para el gráfico de barras
    columnas_seleccionadas = st.multiselect(
        "Selecciona las columnas para el gráfico de barras:",
        options=df.columns.tolist(),
        key='barras_cols'
    )
    
    if columnas_seleccionadas:
        col_x = st.selectbox("Selecciona columna para eje X:", columnas_seleccionadas, key='bar_x')
        col_y = st.selectbox("Selecciona columna para eje Y:", columnas_seleccionadas, key='bar_y')
        
        # Slider para número de registros
        num_registros = st.slider(
            "Cantidad de registros a mostrar",
            min_value=2,
            max_value=len(df),
            value=min(10, len(df)),
            key='bar_slider'
        )
        
        try:
            df_filtrado = df[columnas_seleccionadas].head(num_registros)
            fig = px.bar(df_filtrado, x=col_x, y=col_y,
                        title=f'Gráfico de Barras: {col_x} vs {col_y}')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error en gráfico de barras: {str(e)}")

def generar_radar_chart(df):
    st.write("## Radar Chart")
    
    # Selección de columnas para el radar chart
    columnas_seleccionadas = st.multiselect(
        "Selecciona las columnas para el radar chart:",
        options=df.columns.tolist(),
        key='radar_cols'
    )
    
    if columnas_seleccionadas:
        col_theta = st.selectbox("Selecciona columna para theta:", columnas_seleccionadas, key='radar_theta')
        col_r = st.selectbox("Selecciona columna para radio:", columnas_seleccionadas, key='radar_r')
        
        num_registros = st.slider(
            "Cantidad de registros a mostrar",
            min_value=2,
            max_value=len(df),
            value=min(10, len(df)),
            key='radar_slider'
        )
        
        try:
            df_filtrado = df[columnas_seleccionadas].head(num_registros)
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=df_filtrado[col_r],
                theta=df_filtrado[col_theta],
                fill='toself'
            ))
            fig.update_layout(title=f'Radar Chart: {col_theta} vs {col_r}')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error en radar chart: {str(e)}")

def generar_grafico_dispersion(df):
    st.write("## Gráfico de Dispersión")
    
    # Selección de columnas para el gráfico de dispersión
    columnas_seleccionadas = st.multiselect(
        "Selecciona las columnas para el gráfico de dispersión:",
        options=df.columns.tolist(),
        key='scatter_cols'
    )
    
    if columnas_seleccionadas:
        col_x = st.selectbox("Selecciona columna para eje X:", columnas_seleccionadas, key='scatter_x')
        col_y = st.selectbox("Selecciona columna para eje Y:", columnas_seleccionadas, key='scatter_y')
        
        num_registros = st.slider(
            "Cantidad de registros a mostrar",
            min_value=2,
            max_value=len(df),
            value=min(10, len(df)),
            key='scatter_slider'
        )
        
        try:
            df_filtrado = df[columnas_seleccionadas].head(num_registros)
            fig = px.scatter(df_filtrado, x=col_x, y=col_y,
                           title=f'Gráfico de Dispersión: {col_x} vs {col_y}')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error en gráfico de dispersión: {str(e)}")

def generar_graficos(df):
    with st.expander("Gráfico de Barras", expanded=True):
        generar_grafico_barras(df)
        
    with st.expander("Radar Chart", expanded=True):
        generar_radar_chart(df)
        
    with st.expander("Gráfico de Dispersión", expanded=True):
        generar_grafico_dispersion(df)