# Plataforma Inteligente de Análisis Económico Territorial

Aplicación interactiva desarrollada con **Streamlit** para explorar y comparar indicadores económicos y sociales de los departamentos de Colombia. Incluye visualizaciones estadísticas y herramientas de minería de datos para analizar patrones territoriales.

> **Aviso sobre los datos:** gran parte de los indicadores se generan con **datos simulados** para fines demostrativos. El código también intenta obtener información real del PIB departamental desde datos.gov.co (DANE). Los resultados simulados no deben utilizarse como estadísticas oficiales ni como evidencia para decisiones de política pública.

## Funcionalidades implementadas

### 1. Análisis exploratorio de datos (EDA)

- Selección de uno o varios departamentos, indicador y rango de años.
- Estadísticas descriptivas: media, mediana, mínimo, máximo y cantidad de observaciones.
- Gráfico de tendencias anuales por departamento.
- Histograma de distribución y diagrama de caja para visualizar valores atípicos.
- Glosario sencillo para algunos indicadores.

### 2. Minería de datos (IA)

- Selección de variables y año de análisis.
- Matriz de correlación entre indicadores seleccionados.
- Estandarización de variables con `StandardScaler`.
- Agrupamiento territorial mediante **K-Means**.
- Reducción de dimensionalidad mediante **análisis de componentes principales (PCA)**.
- Visualización de agrupamientos e información sobre la varianza explicada por los dos primeros componentes.

## Indicadores disponibles

El proyecto genera datos simulados para los siguientes indicadores:

1. Tasa de desempleo (%).
2. Tasa de ocupación (%).
3. Informalidad laboral (%).
4. Índice Departamental de Competitividad (0–10).
5. Índice Departamental de Innovación (0–10).
6. Pobreza monetaria (%).
7. Índice de Gini (desigualdad).
8. Creación de empresas (tasa).
9. Inflación anual (%).

Además, intenta cargar **PIB departamental a precios constantes de 2015** desde una API de datos.gov.co. Su disponibilidad depende del acceso a internet, la respuesta del servicio y la estructura de los datos recibidos.

## Tecnologías

- **Streamlit:** interfaz web interactiva.
- **Pandas y NumPy:** procesamiento, simulación y análisis de datos.
- **Plotly:** gráficos interactivos.
- **Requests:** consulta de datos de fuentes externas.
- **Scikit-learn:** K-Means, PCA y estandarización.

## Instalación y ejecución

Se recomienda usar Python 3.10 o posterior.

1. Coloca `observatorio.py`, `requirements.txt` y `README.md` en la misma carpeta.
2. **Antes de ejecutar la aplicación, elimina la línea `!pip install -q streamlit` de `observatorio.py`.** Esa instrucción pertenece a Google Colab/Jupyter y provoca un error de sintaxis cuando se ejecuta como un archivo `.py` normal.
3. Desde una terminal en la carpeta del proyecto, instala las dependencias:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Inicia la aplicación:

   ```bash
   streamlit run observatorio.py
   ```

5. Abre la dirección local que muestra Streamlit en la terminal (habitualmente `http://localhost:8501`).

## Publicación en Streamlit Community Cloud

1. Sube los archivos `observatorio.py`, `requirements.txt` y `README.md` a un repositorio de GitHub.
2. Comprueba que hayas retirado la instrucción `!pip install -q streamlit` del script.
3. En Streamlit Community Cloud, selecciona el repositorio y define `observatorio.py` como archivo principal (*main file path*).
4. Despliega la aplicación. El servicio instalará las dependencias indicadas en `requirements.txt`.

## Consideraciones técnicas

- Los datos simulados abarcan **2010–2023** y se generan con una semilla aleatoria fija para facilitar la reproducibilidad.
- La API del PIB se consulta al cargar los datos; si la solicitud falla, la aplicación continúa solo con datos simulados.
- El filtro de años de la interfaz está definido para 2010–2023, incluso si la fuente real tiene otros años.
- Los métodos K-Means y PCA identifican patrones estadísticos; **no permiten establecer causalidad**.
- El código actual no incorpora todavía modelos VAR, VECM, econometría espacial, predicciones ni simulaciones de políticas públicas.

## Archivo principal

`observatorio.py`: contiene la carga y generación de datos, las herramientas de análisis y las dos pestañas de la aplicación Streamlit.
