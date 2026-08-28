import time
import pandas as pd

from sklearn.metrics import mean_squared_error
#=================================================================
def mostrar_nan(df, columna=None, mostrar=False, num_filas=10):
    """
    Imprime la cantidad de valores Nan por columna del DataFrame
    Muestra las filas completas que contienen valores NaN
    
    Parámetros:
    df: DataFrame a analizar
    columna: nombre de columna específica (opcional)
    num_filas: cuántas filas mostrar (por defecto 10)
    """
    name = str(df)
    lista= df.columns

    #Para una sola columna:
    if columna:  
        filas_nan = df[df[columna].isna()] #crea DF de puros valores NaN
        if mostrar:
            print(f"\nFilas con NaN en '{columna}': {len(filas_nan)} encontradas") # imprime nombre de la columna y cantidad de NaN
        if not filas_nan.empty: # Si el DF filas_nan tiene algo adentro
            if mostrar:
                print("\nPrimeras", min(num_filas, len(filas_nan)), "filas completas conteniendo NaN para: ", "'",columna,"'")
                print(filas_nan.head(num_filas))
                print('-'*50) #separador visual
    else:
        filas_nan =df[df.isna().any(axis=1)]
        if mostrar:
            # 1. Mostrar total de filas con NaN
            print(f"\nTotal de filas con NaN: {len(filas_nan)}")
    
            # 2. Mostrar NaN por cada columna del DataFrame ORIGINAL
            print(f"\nCantidad de NaN por columna:")
            for column in df.columns:  # df.columns, no filas_nan.columns
                nan_count = df[column].isna().sum()  # Contar NaN en cada columna
                print(f"  {column}: {nan_count}")
    
            # 3. Mostrar las primeras filas con NaN
            if not filas_nan.empty:
                print(f"\nPrimeras {min(num_filas, len(filas_nan))} filas con NaN:")
                print(filas_nan.head(num_filas))
                print('-'*50)
    return filas_nan

#========================================
def evaluar_modelo(
    modelo,
    X_train,
    y_train,
    X_valid,
    y_valid,
    nombre_modelo,
    parametros=None,
    cat_features=None
):
    """
    Entrena un modelo y devuelve sus métricas de desempeño.

    Parámetros
    ----------
    modelo : estimador de sklearn o compatible
    X_train, y_train : datos de entrenamiento
    X_valid, y_valid : datos de validación
    nombre_modelo : str
    parametros : dict, opcional
        Diccionario con los hiperparámetros utilizados.

    Retorna
    -------
    DataFrame con una fila de resultados.
    """

    # ==========================
    # Entrenamiento
    # ==========================
    inicio_train = time.perf_counter()

    if cat_features is not None:
        modelo.fit(
            X_train,
            y_train,
            cat_features=cat_features
        )
    else:
        modelo.fit(
            X_train,
            y_train
        )

    tiempo_train = time.perf_counter() - inicio_train


    # ==========================
    # Predicción
    # ==========================
    inicio_pred = time.perf_counter()

    predicciones = modelo.predict(X_valid)

    tiempo_pred = time.perf_counter() - inicio_pred


    # ==========================
    # RMSE
    # ==========================
    rmse = mean_squared_error(y_valid, predicciones, squared=False)
    
    
    # ==========================
    # Formato de hiperparámetros
    # ==========================
    
    if parametros is None:
        parametros_txt = "Default"
    else:
        parametros_txt = ", ".join(
            f"{k}={v}" for k, v in parametros.items()
        )

    
    # ==========================
    # Resultados
    # ==========================
    resultados = pd.DataFrame({
        'Modelo': [nombre_modelo],
        'Parámetros': [parametros_txt],
        'RMSE': [rmse],
        'Tiempo entrenamiento (s)': [tiempo_train],
        'Tiempo predicción (s)': [tiempo_pred]
    })

    return resultados, modelo
#====================================================
def probar_hiperparametros(
    modelo_base,
    lista_parametros,
    X_train,
    y_train,
    X_valid,
    y_valid,
    nombre_modelo
):

    resultados = []

    mejor_modelo = None
    mejor_rmse = float("inf")


    for parametros in lista_parametros:

        modelo = modelo_base(**parametros)

        resultado, modelo_entrenado = evaluar_modelo(
            modelo=modelo,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            nombre_modelo=nombre_modelo,
            parametros=parametros
        )

        resultados.append(resultado)

        if resultado.loc[0, 'RMSE'] < mejor_rmse:

            mejor_rmse = resultado.loc[0, 'RMSE']
            mejor_modelo = modelo_entrenado


    resultados = (
        pd.concat(resultados, ignore_index=True)
          .sort_values("RMSE")
          .reset_index(drop=True)
    )

    return resultados, mejor_modelo

