# Predicción del Precio de Vehículos Usados

> English version: [README.md](README.md)

Proyecto de regresión que predice **el valor de mercado de autos usados** a partir de sus especificaciones técnicas, para la función de valuación de una app de autos usados.

El proyecto limpia 354,369 anuncios de autos, crea una variable de antigüedad del vehículo y compara **Regresión Lineal, Random Forest, XGBoost y LightGBM** con tres criterios: calidad de la predicción (RMSE), tiempo de entrenamiento y tiempo de predicción. El modelo elegido, **LightGBM (400 árboles, tasa de aprendizaje 0.05)**, obtuvo un **RMSE de 1,642.76 en validación** y de **1,649.63 en el conjunto de prueba independiente**: el menor error de todos los modelos, entrenando en segundos en lugar de minutos.

---

# Contexto de Negocio

Rusty Bargain, un mercado de autos usados, está creando una app en la que los usuarios pueden conocer rápidamente el valor de mercado de su auto. A la empresa le importan tres cosas:

| Criterio | Por qué importa |
| --- | --- |
| **Calidad de la predicción** | Valuaciones justas y realistas generan confianza |
| **Velocidad de predicción** | La app debe responder de inmediato |
| **Tiempo de entrenamiento** | El modelo debe poder reentrenarse fácilmente con anuncios nuevos |

El modelo debe predecir una variable objetivo:

| Objetivo | Descripción |
| --- | --- |
| `Price` | Precio del anuncio en euros |

---

# Conceptos Clave

| Término | Significado |
| --- | --- |
| Antigüedad del vehículo | `2019 − RegistrationYear`; representa la depreciación de forma más directa que el año |
| One-Hot Encoding | Convierte cada categoría en una columna de 0/1; lo necesitan Random Forest, XGBoost y la Regresión Lineal |
| Manejo nativo de categorías | LightGBM puede dividir directamente columnas de tipo `category`, sin One-Hot Encoding |
| Gradient boosting | Construye árboles uno tras otro, cada uno corrigiendo los errores de los anteriores (LightGBM, XGBoost) |
| Línea base | Un modelo sencillo (aquí, la Regresión Lineal) para comprobar que los modelos complejos aportan valor |

---

# Métrica de Evaluación

La métrica principal es la **raíz del error cuadrático medio (RMSE)**:

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

El RMSE está en la misma unidad que el precio (euros) y castiga más los errores grandes que los pequeños, lo cual importa porque una valuación muy equivocada es lo que más afecta la confianza del usuario. El **tiempo de entrenamiento** y el **tiempo de predicción** se midieron con `time.perf_counter()` para cada modelo.

---

# Dataset

| Archivo | Filas | Columnas | Descripción |
| --- | ---: | ---: | --- |
| `data/car_data.csv` | 354,369 | 16 | Anuncios de autos usados recopilados en marzo–abril de 2016 |

| Columna | Descripción |
| --- | --- |
| `Price` | Precio en euros (objetivo) |
| `VehicleType` | Tipo de carrocería |
| `RegistrationYear`, `RegistrationMonth` | Fecha de matriculación |
| `Gearbox` | Manual o automática |
| `Power` | Potencia (CV) |
| `Model`, `Brand` | Modelo y marca |
| `Mileage` | Kilometraje |
| `FuelType` | Tipo de combustible |
| `NotRepaired` | Si el auto tiene daños sin reparar |
| `DateCrawled`, `DateCreated`, `LastSeen` | Fechas relacionadas con el anuncio |
| `NumberOfPictures`, `PostalCode` | Número de fotos y código postal del dueño |

**Fuente de los datos.** El dataset fue proporcionado por **TripleTen** como parte de su Bootcamp de Data Science y se incluye en este repositorio con fines educativos y de reproducibilidad.

---

# Flujo del Proyecto

```text
Datos originales (354,369 anuncios)
            │
            ▼
1. Preparación: valores faltantes, valores fuera de rango, VehicleAge
            │
            ▼
2. División: 70% train / 15% validación / 15% test
            │
            ▼
3. Preprocesamiento por modelo (imputación ajustada en train)
   ├── LightGBM: columnas categóricas nativas
   ├── Random Forest y XGBoost: One-Hot Encoding
   └── Regresión Lineal: One-Hot Encoding (drop_first) + StandardScaler
            │
            ▼
4. Entrenar y cronometrar cada modelo con una pequeña rejilla de hiperparámetros
            │
            ▼
5. Evaluar una sola vez el mejor modelo en el conjunto de prueba
```

---

# 1. Preparación de los Datos

## Valores Faltantes

| Columna | Faltantes | Tratamiento |
| --- | ---: | --- |
| `NotRepaired` | 71,154 | `"unknown"` |
| `VehicleType` | 37,490 | `"unknown"` |
| `FuelType` | 32,895 | `"unknown"` |
| `Gearbox` | 19,833 | `"unknown"` |
| `Model` | 19,705 | `"unknown"` |

108,555 filas (30.6%) tenían al menos un valor faltante, así que rellenarlas con `"unknown"` permitió conservar los datos en lugar de eliminar casi un tercio.

## Valores Fuera de Rango

| Columna | Problema | Tratamiento |
| --- | --- | --- |
| `Price` | 36,054 anuncios (10.2%) por debajo de €500, incluidos 10,772 con precio 0 | Filas eliminadas, para que el modelo no aprenda de precios irreales |
| `RegistrationYear` | Valores de 1000 a 9999 | Los años fuera de 1910–2019 se marcaron como faltantes y después se imputaron con la mediana de entrenamiento |
| `Power` | 40,225 anuncios con 0 CV | Se reemplazaron por la mediana de la potencia |
| `RegistrationMonth` | Mes 0 | Se trató como "mes desconocido" (0) |

## Ingeniería y Selección de Variables

* **`VehicleAge` = 2019 − `RegistrationYear`**, y se eliminó `RegistrationYear`.
* Columnas eliminadas: `DateCrawled`, `DateCreated`, `LastSeen` (fechas del anuncio), `NumberOfPictures` (siempre 0) y `PostalCode`.
* Datos finales: **318,315 filas** y **10 variables** (6 categóricas y 4 numéricas).

---

# 2. División y Preprocesamiento

```text
318,315 anuncios
│
├── 70% TRAIN ─────────→ 222,947 → entrenamiento
├── 15% VALIDACIÓN ────→  47,748 → selección de modelo e hiperparámetros
└── 15% TEST ──────────→  47,620 → evaluación final (se usa una sola vez)
```

Los imputadores, el One-Hot Encoder y el escalador se **ajustaron solo con el conjunto de entrenamiento** y después se aplicaron a validación y test.

| Familia de modelos | Entrada |
| --- | --- |
| LightGBM | Numéricas imputadas + categóricas como `category` (10 variables) |
| Random Forest, XGBoost | Numéricas imputadas + One-Hot Encoding (317 variables) |
| Regresión Lineal | One-Hot Encoding con `drop_first` + `StandardScaler` (311 variables) |

---

# 3. Modelado

| Modelo | Configuración | RMSE validación | Tiempo de entrenamiento | Tiempo de predicción |
| --- | --- | ---: | ---: | ---: |
| **LightGBM** | `n_estimators=400, learning_rate=0.05` | **1,642.76** | 7 s | 0.72 s |
| LightGBM | `n_estimators=200, learning_rate=0.1` | 1,647.50 | 3 s | 0.31 s |
| XGBoost | `n_estimators=400, learning_rate=0.05, max_depth=8` | 1,655.91 | 911 s | 0.57 s |
| XGBoost | `n_estimators=200, learning_rate=0.1, max_depth=6` | 1,715.73 | 346 s | 0.34 s |
| Random Forest | `n_estimators=100, max_depth=15` | 1,746.21 | 123 s | 0.34 s |
| Random Forest | `n_estimators=100, max_depth=10` | 1,969.91 | 101 s | 0.18 s |
| Random Forest | `n_estimators=50, max_depth=10` | 1,972.17 | 53 s | 0.17 s |
| Regresión Lineal (línea base) | Por defecto | 2,852.67 | 8 s | 0.01 s |

*Tiempos de la ejecución del notebook; cambian un poco entre ejecuciones y equipos. Los tiempos de predicción son para todo el conjunto de validación (47,748 autos).*

* **Todos los modelos basados en árboles superaron claramente a la línea base** (RMSE de 2,853), así que las relaciones no lineales entre las variables y el precio importan.
* **LightGBM tuvo el menor error y además estuvo entre los más rápidos de entrenar**: unos 7 segundos, mientras que XGBoost necesitó unos 15 minutos para llegar a un error un poco mayor.
* Random Forest mejoró con más profundidad, pero quedó detrás de los modelos de boosting.
* **CatBoost** está en el notebook pero comentado, porque requería más capacidad de cómputo de la disponible. En una ejecución aparte obtuvo un RMSE de cerca de 1,700, que no quedó guardado en el notebook.

## Resultado Final

El modelo elegido se evaluó **una sola vez** en el conjunto de prueba:

```text
Modelo:              LightGBM (n_estimators=400, learning_rate=0.05)
RMSE validación:     1,642.76
RMSE test:           1,649.63
Tiempo de predicción: 0.68 s para 47,620 autos
```

El error en test es solo 0.4% mayor que en validación, así que el modelo generaliza bien a anuncios nuevos.

---

# Conclusiones

1. **La calidad de los datos requirió varias decisiones**: casi un tercio de las filas tenía valores faltantes, el 10% tenía precios irreales y había años de matriculación y potencias imposibles.

2. **El gradient boosting dio los mejores resultados.** LightGBM y XGBoost tuvieron los errores más bajos, y la Regresión Lineal quedó muy atrás.

3. **LightGBM ofreció el mejor equilibrio entre los tres criterios de negocio**: el menor RMSE, entrenamiento en segundos y predicciones rápidas.

4. **El resultado es estable**: el RMSE en test (1,649.63) es casi igual al de validación (1,642.76).

## Interpretación de Negocio

LightGBM puede valuar un auto casi al instante y reentrenarse rápido conforme llegan anuncios nuevos. Un error promedio de cerca de €1,650 debe presentarse a los usuarios como un valor aproximado, no como un precio exacto.

## Limitaciones y Siguientes Pasos

* **Años de matriculación posteriores a 2016:** los anuncios se recopilaron en 2016, pero 14,425 autos (4%) tienen año de matriculación entre 2017 y 2019, lo cual no es posible. Limitar el rango a 2016 y calcular la antigüedad respecto a la fecha del anuncio haría más precisa `VehicleAge`.
* **Potencia:** los valores de 0 CV se reemplazaron con la mediana de todo el dataset antes de dividir los datos (una pequeña fuga de información), y no se trataron 302 valores mayores a 1,000 CV (hasta 20,000). La mediana debería calcularse solo con el conjunto de entrenamiento y los valores extremos deberían acotarse.
* Ajustar más LightGBM (número de hojas, mínimo de muestras por hoja) con validación cruzada.
* Ejecutar CatBoost con suficiente capacidad de cómputo y guardar sus resultados.
* Analizar la importancia de las variables para explicar qué características determinan el precio.

---

# Estructura del Proyecto

```text
used-vehicles-price-prediction/
│
├── data/
│   └── car_data.csv
│
├── notebooks/
│   └── used_vehicle_price_prediction.ipynb
│
├── src/
│   └── funciones_personales.py
│
├── .gitignore
├── environment.yml
├── requirements.txt
├── LEEME.md
└── README.md
```

`src/funciones_personales.py` contiene funciones auxiliares para revisar valores faltantes (`mostrar_nan`), entrenar y cronometrar un modelo (`evaluar_modelo`) y probar una lista de combinaciones de hiperparámetros (`probar_hiperparametros`).

---

# Instalación

INSTALLATION_PLACEHOLDER

---

# Tecnologías

* Python
* Pandas
* NumPy
* scikit-learn (Regresión Lineal, Random Forest, One-Hot Encoding, StandardScaler, SimpleImputer)
* LightGBM
* XGBoost
* CatBoost
* Jupyter Notebook

---

# Habilidades Demostradas

* Regresión con datos tabulares
* Limpieza de datos: valores faltantes, valores fuera de rango y objetivos irreales
* Ingeniería de variables (antigüedad del vehículo)
* Preprocesamiento específico por modelo (codificadores e imputadores ajustados con el conjunto de entrenamiento)
* Comparación de Regresión Lineal, Random Forest, XGBoost y LightGBM
* Ajuste de hiperparámetros
* Medición de tiempos de entrenamiento y predicción
* Elección de un modelo equilibrando precisión y velocidad

---

## Autor

**Pablo Andrés Vázquez Calva**

Data Scientist

GitHub: https://github.com/pablo-vc87

LinkedIn: https://www.linkedin.com/in/pablo-avc/
