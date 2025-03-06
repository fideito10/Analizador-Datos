import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

def preparar_datos(df):
    """
    Prepara los datos para el análisis, separando variables numéricas y no numéricas.
    """
    if df is not None:
        st.write("### Selección de variables")
        
        # Separar columnas numéricas y no numéricas
        columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()
        columnas_no_numericas = df.select_dtypes(exclude=np.number).columns.tolist()
        
        # Verificar si hay columnas numéricas
        if not columnas_numericas:
            st.error("No hay columnas numéricas en el dataset")
            return None, None, None, None
            
        # Mostrar información sobre las columnas
        st.info(f"Columnas numéricas disponibles: {', '.join(columnas_numericas)}")
        if columnas_no_numericas:
            st.warning(f"Columnas no numéricas (no se usarán para el modelo): {', '.join(columnas_no_numericas)}")
        
        # Seleccionar variable objetivo (solo numérica)
        variable_objetivo = st.selectbox(
            "Selecciona la variable NUMÉRICA objetivo (y):",
            options=columnas_numericas
        )
        
        # Seleccionar variables predictoras (solo numéricas)
        variables_predictoras = st.multiselect(
            "Selecciona variables predictoras NUMÉRICAS (X):",
            options=[col for col in columnas_numericas if col != variable_objetivo]
        )
        
        if variables_predictoras:
            # Verificar el número de registros disponibles
            total_registros = len(df)
            
            if total_registros < 3:
                st.error("Se necesitan al menos 3 registros para entrenar el modelo.")
                return None, None, None, None
            
            # Ajustar valores del slider según cantidad de datos
            min_registros = min(3, total_registros)
            valor_default = min(100, total_registros)
            
            # Número de registros a usar
            num_registros = st.slider(
                "Selecciona la cantidad de registros a utilizar",
                min_value=min_registros,
                max_value=total_registros,
                value=valor_default
            )
            
            # Preparar los datos (solo columnas numéricas)
            X = df[variables_predictoras][:num_registros]
            y = df[variable_objetivo][:num_registros]
            
            # Verificar que no haya valores nulos
            nulos_X = X.isnull().sum()
            nulos_y = y.isnull().sum()
            
            if nulos_X.any() or nulos_y > 0:
                st.warning("⚠️ Se detectaron valores nulos:")
                if nulos_X.any():
                    st.write("Variables predictoras con nulos:")
                    for col, nulos in nulos_X[nulos_X > 0].items():
                        st.write(f"- {col}: {nulos} valores nulos")
                if nulos_y > 0:
                    st.write(f"Variable objetivo '{variable_objetivo}': {nulos_y} valores nulos")
                
                # Opción para manejar nulos
                manejo_nulos = st.radio(
                    "¿Cómo deseas manejar los valores nulos?",
                    ["Eliminar filas con nulos", "Rellenar con la media", "Mantener los nulos"]
                )
                
                if manejo_nulos == "Eliminar filas con nulos":
                    mask = X.notnull().all(axis=1) & y.notnull()
                    X = X[mask]
                    y = y[mask]
                    st.info(f"Se eliminaron {(~mask).sum()} filas con valores nulos")
                elif manejo_nulos == "Rellenar con la media":
                    X = X.fillna(X.mean())
                    y = y.fillna(y.mean())
                    st.info("Se rellenaron los valores nulos con la media")
            
            # Verificar datos finales
            if len(X) < 3:
                st.error("No hay suficientes datos válidos después de procesar los nulos")
                return None, None, None, None
                
            # Mostrar resumen final
            st.success(f"""
            ✅ Datos preparados exitosamente:
            - Registros seleccionados: {len(X)}
            - Variables predictoras: {len(variables_predictoras)}
            - Variable objetivo: {variable_objetivo}
            """)
            
            return X, y, variables_predictoras, variable_objetivo
    
    return None, None, None, None




def generar_reporte_completo(X, y, variables_predictoras, variable_objetivo):
    """
    Versión simplificada del reporte con análisis esencial y visualizaciones claras
    """
    try:
        st.header("📊 Reporte Analítico Rápido")
        
        # 1. Análisis estadístico básico
        st.subheader("1. Estadísticas Clave")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Muestra Total", len(X))
            st.write(f"Variables Predictoras: {len(variables_predictoras)}")
            
        with col2:
            st.metric(f"Media de {variable_objetivo}", f"{y.mean():.2f}")
            st.metric(f"Rango de {variable_objetivo}", f"{y.min():.2f} - {y.max():.2f}")
        
        # 2. Correlaciones
        st.subheader("2. Correlaciones Principales")
        numeric_df = X.select_dtypes(include=np.number).join(y)
        corr_matrix = numeric_df.corr().round(2)
        fig = px.imshow(corr_matrix, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. Modelo rápido
        st.subheader("3. Modelo Predictivo")
        modelo = entrenar_modelo_simplificado(X, y)
        
        if modelo:
            # 4. Importancia de variables
            st.subheader("4. Variables más Importantes")
            importancia = obtener_importancia_variables(modelo, X)
            st.bar_chart(importancia.set_index('Variable'))
            
            # 5. Conclusiones clave
            st.subheader("5. Conclusiones")
            r2 = modelo.r2 if hasattr(modelo, 'r2') else 0
            conclusiones = [
                f"🧠 El modelo explica el {r2*100:.1f}% de la variabilidad ({variable_objetivo})",
                f"📌 Variable más importante: {importancia.iloc[0]['Variable']}",
                "💡 Recomendación: " + ("Mejorar con más datos" if r2 < 0.7 else "Modelo confiable")
            ]
            st.success("\n\n".join(conclusiones))
            
        return True
        
    except Exception as e:
        st.error(f"Error generando reporte: {str(e)}")
        return False

# Funciones auxiliares simplificadas
def entrenar_modelo_simplificado(X, y):
    try:
        # Identificar columnas numéricas
        columnas_numericas = X.select_dtypes(include=['int64', 'float64']).columns
        
        if len(columnas_numericas) == 0:
            st.warning("No hay variables numéricas para entrenar el modelo")
            return None
            
        # Usar solo las columnas numéricas
        X_num = X[columnas_numericas]
        
        # Verificar que la variable objetivo sea numérica
        if not pd.to_numeric(y, errors='coerce').notnull().all():
            st.warning("La variable objetivo debe ser numérica")
            return None
            
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2)
        
        # Entrenar modelo
        modelo = RandomForestRegressor().fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        # Calcular métricas
        modelo.mse = mean_squared_error(y_test, y_pred)
        modelo.r2 = r2_score(y_test, y_pred)
        
        # Guardar las columnas usadas
        modelo.columnas_usadas = columnas_numericas
        
        # Mostrar información
        st.info(f"Variables numéricas utilizadas: {', '.join(columnas_numericas)}")
        
        return modelo
        
    except Exception as e:
        st.warning(f"No se pudo entrenar el modelo: {str(e)}")
        return None
    
def obtener_importancia_variables(modelo, X):
    # Usar solo las columnas que se usaron en el entrenamiento
    return pd.DataFrame({
        'Variable': modelo.columnas_usadas,
        'Importancia': modelo.feature_importances_
    }).sort_values('Importancia', ascending=False).head(5)