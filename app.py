# -*- coding: utf-8 -*-
"""OBSERVATORIO V2"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import unicodedata
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURACIÓN DE PÁGINA Y NORMALIZACIÓN
# ============================================================
st.set_page_config(
    page_title="Plataforma de Análisis Económico", 
    page_icon="🇨🇴", 
    layout="wide",
    initial_sidebar_state="expanded"
)

DEPARTAMENTOS_CANONICOS = {
    "amazonas": "Amazonas", "antioquia": "Antioquia", "arauca": "Arauca",
    "atlantico": "Atlántico", "bogota": "Bogotá D.C.", "bogota d c": "Bogotá D.C.",
    "bolivar": "Bolívar", "boyaca": "Boyacá", "caldas": "Caldas", "caqueta": "Caquetá",
    "casanare": "Casanare", "cauca": "Cauca", "cesar": "Cesar", "choco": "Chocó",
    "cordoba": "Córdoba", "cundinamarca": "Cundinamarca", "guainia": "Guainía",
    "guaviare": "Guaviare", "huila": "Huila", "la guajira": "La Guajira",
    "magdalena": "Magdalena", "meta": "Meta", "narino": "Nariño",
    "norte de santander": "Norte de Santander", "putumayo": "Putumayo",
    "quindio": "Quindío", "risaralda": "Risaralda", "san andres": "San Andrés y Providencia",
    "santander": "Santander", "sucre": "Sucre", "tolima": "Tolima",
    "valle del cauca": "Valle del Cauca", "vaupes": "Vaupés", "vichada": "Vichada",
}

def normalizar_departamento(nombre):
    if pd.isna(nombre) or nombre is None: return ""
    t = str(nombre).strip().lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return DEPARTAMENTOS_CANONICOS.get(t.replace(",", "").replace(".", " ").strip(), str(nombre).strip().title())

# ============================================================
# 2. GENERACIÓN Y DESCARGA DE DATOS
# ============================================================
@st.cache_data
def generar_datos_simulados():
    np.random.seed(42)
    deptos = list(set(DEPARTAMENTOS_CANONICOS.values()))
    anios = list(range(2010, 2024))

    indicadores_sim = {
        "Tasa de Desempleo (%)": (7.0, 25.0),
        "Tasa de Ocupación (%)": (45.0, 65.0),
        "Informalidad Laboral (%)": (35.0, 85.0),
        "Índice Departamental de Competitividad (0-10)": (2.5, 8.5),
        "Índice Departamental de Innovación (0-10)": (1.0, 7.5),
        "Pobreza Monetaria (%)": (15.0, 65.0),
        "Índice de Gini (Desigualdad)": (0.40, 0.65),
        "Creación de Empresas (Tasa)": (5.0, 15.0),
        "Inflación Anual (%)": (2.0, 13.0)
    }

    datos = []
    for d in deptos:
        factor_depto = np.random.uniform(0.8, 1.2)
        for a in anios:
            tendencia = (a - 2010) * 0.05
            for ind, (vmin, vmax) in indicadores_sim.items():
                ruido = np.random.normal(0, (vmax-vmin)*0.05)
                val_base = np.random.uniform(vmin, vmax)
                val_final = np.clip(val_base * factor_depto + tendencia + ruido, vmin, vmax)
                datos.append([d, a, val_final, ind, "Simulación Estructural"])

    return pd.DataFrame(datos, columns=["departamento", "anio", "valor", "indicador", "fuente"])

@st.cache_data
def cargar_apis_y_simulaciones():
    partes = [generar_datos_simulados()]
    try:
        url = "https://www.datos.gov.co/resource/kgyi-qc7j.json?$limit=50000"
        df = pd.DataFrame(requests.get(url, timeout=10).json())
        df["anio"] = pd.to_numeric(df["a_o"], errors="coerce")
        df["valor"] = pd.to_numeric(df["valor_miles_de_millones_de"], errors="coerce")
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        df = df[df["tipo_de_precios"].str.contains("constantes", case=False, na=False)]
        agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
        agg["indicador"], agg["fuente"] = "PIB (miles de millones, const 2015) [REAL]", "DANE"
        partes.append(agg)
    except: pass

    res = pd.concat(partes, ignore_index=True)
    res["anio"] = res["anio"].astype(int)
    return res

DB_MAESTRA = cargar_apis_y_simulaciones()
lista_deptos = sorted(DB_MAESTRA["departamento"].dropna().unique())
lista_inds = sorted(DB_MAESTRA["indicador"].unique())

# ============================================================
# 3. DICCIONARIO ECONÓMICO
# ============================================================
def consultar_glosario(indicador):
    diccionario = {
        "Tasa de Desempleo (%)": "Porcentaje de la fuerza de trabajo que no tiene empleo y busca activamente.",
        "Tasa de Ocupación (%)": "Relación porcentual entre la población ocupada y el número de personas que integran la población en edad de trabajar.",
        "Informalidad Laboral (%)": "Proporción de trabajadores que no cuentan con seguridad social (salud y pensión) derivada de su vínculo laboral.",
        "Índice Departamental de Competitividad (0-10)": "Evalúa condiciones transversales como infraestructura, educación, salud, instituciones e innovación empresarial.",
        "Índice Departamental de Innovación (0-10)": "Mide las capacidades y resultados de innovación tecnológica y empresarial en las regiones.",
        "Pobreza Monetaria (%)": "Porcentaje de la población con ingresos insuficientes para adquirir una canasta básica de bienes y servicios.",
        "Índice de Gini (Desigualdad)": "Medida de la desigualdad en la distribución del ingreso (0 = igualdad perfecta, 1 = desigualdad máxima).",
        "Creación de Empresas (Tasa)": "Dinámica de registro de nuevas unidades productivas formales en la región.",
        "Inflación Anual (%)": "Variación porcentual anual del Índice de Precios al Consumidor (IPC).",
        "PIB (miles de millones, const 2015) [REAL]": "Valor monetario de todos los bienes y servicios finales producidos en el departamento, descontando la inflación."
    }
    return diccionario.get(indicador, "No hay descripción disponible para este indicador.")

# ============================================================
# 4. INTERFAZ DE USUARIO (STREAMLIT)
# ============================================================

# --- MENSAJE DE BIENVENIDA INMERSIVO ---
st.markdown("""
<div style='background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%); padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
    <h1 style='color: white; margin:0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>👋 ¡Bienvenido al Observatorio Económico Territorial!</h1>
    <p style='color: #f1f1f1; font-size: 1.2em; margin-top: 10px;'>Explora, analiza y comprende las dinámicas socioeconómicas de los departamentos de Colombia mediante datos estructurados y modelos de Inteligencia Artificial.</p>
</div>
""", unsafe_allow_html=True)

# --- BARRA LATERAL (CONTROLES) ---
with st.sidebar:
    st.header("⚙️ Panel de Control")
    st.markdown("Ajusta los parámetros para explorar los datos.")
    
    ind_sel = st.selectbox("📊 Selecciona un Indicador", lista_inds)
    deptos_sel = st.multiselect("📍 Departamentos", ["Todos"] + lista_deptos, default=["Todos"])
    anio_min, anio_max = st.slider("📅 Rango de años", 2010, 2023, (2015, 2023))
    
    st.markdown("---")
    st.markdown("💡 *Los datos se actualizan automáticamente al cambiar los filtros.*")
    st.markdown("---")
    st.markdown("👨‍💻 **Desarrollado por:** EDGAR FERNANDO DOMINGUEZ TORRES")
    st.markdown("🎓 *Programa de Economía - Universidad de Nariño*")

# --- ÁREA PRINCIPAL ---
tab1, tab2 = st.tabs(["📈 1. Análisis Exploratorio (EDA)", "🤖 2. Minería de Datos (IA)"])

# PESTAÑA 1: EDA
with tab1:
    # Tarjeta de significado del indicador
    st.info(f"**📖 ¿Qué es este indicador?**\n\n**{ind_sel}:** {consultar_glosario(ind_sel)}")

    # Filtrado de datos
    d = DB_MAESTRA.copy()
    if "Todos" not in deptos_sel and deptos_sel:
        d = d[d["departamento"].isin(deptos_sel)]
    d = d[(d["indicador"] == ind_sel) & (d["anio"] >= anio_min) & (d["anio"] <= anio_max)]

    if d.empty:
        st.warning("⚠️ No hay datos para los filtros seleccionados.")
    else:
        st.markdown("### 📊 Evolución Histórica")
        fig_tendencia = px.line(d, x="anio", y="valor", color="departamento", markers=True, template="plotly_white")
        fig_tendencia.update_layout(xaxis_title="Año", yaxis_title="Valor", legend_title="Departamento")
        st.plotly_chart(fig_tendencia, use_container_width=True)

        ultimo_anio = d["anio"].max()
        d_ultimo = d[d["anio"] == ultimo_anio]

        st.markdown(f"### 🔍 Radiografía del año {ultimo_anio}")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_hist = px.histogram(d_ultimo, x="valor", nbins=15, title="Distribución de frecuencias", template="plotly_white", color_discrete_sequence=['#4CAF50'])
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_g2:
            fig_box = px.box(d_ultimo, x="valor", hover_data=["departamento"], title="Dispersión y Valores Atípicos", template="plotly_white", points="all", color_discrete_sequence=['#FF9800'])
            st.plotly_chart(fig_box, use_container_width=True)

# PESTAÑA 2: MACHINE LEARNING
with tab2:
    st.markdown("### 🧠 Segmentación de Territorios (Clustering)")
    st.write("**¿Qué hace esta sección?** Utilizamos minería de datos (algoritmo **K-Means**) para encontrar patrones ocultos y agrupar automáticamente los departamentos que tienen condiciones económicas y sociales similares. Además, usamos **PCA** (Análisis de Componentes Principales) para resumir todas las variables cruzadas en un gráfico 2D fácil de interpretar.")

    col_ml1, col_ml2 = st.columns([1, 2.5])

    with col_ml1:
        st.markdown("#### Configuración del Modelo")
        inds_ml = st.multiselect("Variables a cruzar", lista_inds, default=lista_inds[:3], key="ml_inds")
        
        # Selector de departamentos exclusivo para Minería de Datos
        deptos_ml = st.multiselect("Departamentos a analizar", ["Todos"] + lista_deptos, default=["Todos"], key="ml_deptos")
        
        anio_ml = st.slider("Año de corte transversal", 2010, 2023, 2022, key="ml_anio")
        n_clusters = st.slider("Número de Clústeres (Grupos)", 2, 6, 3)
        ejecutar_ml = st.button("🚀 Ejecutar Modelo", use_container_width=True)

    with col_ml2:
        if ejecutar_ml:
            if len(inds_ml) < 2:
                st.error("⚠️ Debes seleccionar al menos 2 indicadores para ejecutar el modelo de Machine Learning.")
            else:
                # Filtrar la base de datos según los departamentos seleccionados
                df_ml = DB_MAESTRA.copy()
                if "Todos" not in deptos_ml and deptos_ml:
                    df_ml = df_ml[df_ml["departamento"].isin(deptos_ml)]
                    
                pivot = df_ml[(df_ml["anio"] == anio_ml) & (df_ml["indicador"].isin(inds_ml))].pivot(index="departamento", columns="indicador", values="valor").dropna()
                
                if len(pivot) < n_clusters or len(pivot) < 3:
                    st.warning(f"⚠️ No hay suficientes departamentos con datos para crear {n_clusters} grupos. Selecciona más departamentos en el filtro (al menos {max(n_clusters, 3)}) o cambia el año.")
                else:
                    tab_clust, tab_corr = st.tabs(["🎯 Segmentación K-Means", "🔗 Mapa de Correlaciones"])

                    with tab_clust:
                        X_scaled = StandardScaler().fit_transform(pivot)
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        pivot["Cluster"] = kmeans.fit_predict(X_scaled).astype(str)

                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        pivot["PC1"], pivot["PC2"] = X_pca[:, 0], X_pca[:, 1]

                        fig_cluster = px.scatter(pivot.reset_index(), x="PC1", y="PC2", color="Cluster", text="departamento", title="Mapa de Similitud Departamental", template="plotly_white")
                        fig_cluster.update_traces(textposition='top center', marker=dict(size=12))
                        fig_cluster.update_layout(xaxis_title="Componente Principal 1", yaxis_title="Componente Principal 2")
                        st.plotly_chart(fig_cluster, use_container_width=True)

                        var_exp = pca.explained_variance_ratio_ * 100
                        st.success(f"**Interpretación del Modelo:** Los departamentos agrupados bajo el mismo color comparten características socioeconómicas similares. El algoritmo K-Means identificó {n_clusters} perfiles distintos, y el gráfico resume el **{var_exp[0]+var_exp[1]:.1f}%** de toda la información cruzada.")

                    with tab_corr:
                        st.info("**¿Qué significa esto?** El mapa de calor muestra cómo se relacionan los indicadores entre sí. Los tonos rojos indican una relación directa (si uno sube, el otro también). Los tonos azules indican una relación inversa (si uno sube, el otro baja).")
                        fig_corr = px.imshow(pivot.corr().round(2), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", origin="lower")
                        st.plotly_chart(fig_corr, use_container_width=True)
