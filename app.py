import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from funciones_recommendation_system import get_datasets
from funciones_plot_validation import obtain_data_agg_methods, get_plot_comparison_agg_methods, get_plot_comparison_methods

# Función para cargar datos una vez
@st.cache_data(persist=True)
def load_data():
    path = '/export/data_ml4ds/AI4U/Datasets/'
    version_wp = '20240510'
    version_rp = '20240321'
    
    df_publications, df_projects, df_publications_researchers, df_projects_researchers, df_researchers, df_calls = get_datasets(path, version_wp, version_rp)
    
    return df_publications, df_projects, df_publications_researchers, df_projects_researchers, df_researchers, df_calls

# Cargar datos una vez
df_publications, df_projects, df_publications_researchers, df_projects_researchers, df_researchers, df_calls = load_data()

# Cargar el dataset de validación
df_val = pd.read_excel('/export/usuarios_ml4ds/mafuello/Github/recommendation_system_validation/validation_set.xlsx')
df_val['id_researcher'] = df_val['id_researcher'].astype(str)

# Listas con los métodos de agregación y métodos de recomendación
agg_methods = ['sum', 'mean', 'mean_imp']
methods = ['BERT', 'bhattacharyya', 'separated', 'semiseparated']

# Función para obtener datos agregados (cacheada)
@st.cache_data(persist=True)
def obtain_data_cached(df_val, agg_methods, method, df_researchers, df_calls):
    return obtain_data_agg_methods(df_val, agg_methods, method, df_researchers, df_calls)

# Obtener datos agregados una vez
researchers_sum_bert, researchers_mean_bert, researchers_mean_imp_bert, calls_sum_bert, calls_mean_bert, calls_mean_imp_bert = obtain_data_cached(df_val, agg_methods, 'BERT', df_researchers, df_calls)
researchers_sum_bhattacharyya, researchers_mean_bhattacharyya, researchers_mean_imp_bhattacharyya, calls_sum_bhattacharyya, calls_mean_bhattacharyya, calls_mean_imp_bhattacharyya = obtain_data_cached(df_val, agg_methods, 'bhattacharyya', df_researchers, df_calls)
researchers_sum_separated, researchers_mean_separated, researchers_mean_imp_separated, calls_sum_separated, calls_mean_separated, calls_mean_imp_separated = obtain_data_cached(df_val, agg_methods, 'separated', df_researchers, df_calls)
researchers_sum_semiseparated, researchers_mean_semiseparated, researchers_mean_imp_semiseparated, calls_sum_semiseparated, calls_mean_semiseparated, calls_mean_imp_semiseparated = obtain_data_cached(df_val, agg_methods, 'semiseparated', df_researchers, df_calls)

# Supongamos que scores_researchers y scores_calls están definidos en alguna parte
# Aquí se simulan los datos para el ejemplo
scores_researchers = np.random.rand(3, 3)  # Reemplaza con los datos reales
scores_calls = np.random.rand(3, 3)  # Reemplaza con los datos reales

# Crear los DataFrames de resultados
results_researchers = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers)
results_calls = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls)

# Tabs para comparar métodos de agregación y métodos de recomendación
tab1, tab2 = st.tabs(["Compare aggregation methods", "Compare recommendations methods"])
with tab1:
    # Selección el método a comparar
    selectbox_key1 = st.empty()  # Genera un contenedor vacío para poder actualizar el selectbox más tarde
    with selectbox_key1:
        recommendations = st.selectbox('Select which recommendations to obtain:', ['Researchers', 'Calls'], key='selectbox_tab1')  # Añade una key única

    # Título de la aplicación
    st.title('Comparison of different aggregation methods when recommending {}'.format(recommendations))
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(15, 10))  
        method = methods[0]
        
        if recommendations == 'Researchers':
            get_plot_comparison_agg_methods(ax, method, researchers_sum_bert, researchers_mean_bert, researchers_mean_imp_bert)
            st.table(results_researchers)
        elif recommendations == 'Calls':
            get_plot_comparison_agg_methods(ax, method, calls_sum_bert, calls_mean_bert, calls_mean_imp_bert)
            st.table(results_calls)

        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(15, 10))  
        method = methods[1]
        
        if recommendations == 'Researchers':
            get_plot_comparison_agg_methods(ax, method, researchers_sum_bhattacharyya, researchers_mean_bhattacharyya, researchers_mean_imp_bhattacharyya)
            st.table(results_researchers)
        elif recommendations == 'Calls':
            get_plot_comparison_agg_methods(ax, method, calls_sum_bhattacharyya, calls_mean_bhattacharyya, calls_mean_imp_bhattacharyya)
            st.table(results_calls)

        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots(figsize=(15, 10))  
        method = methods[2]
        
        if recommendations == 'Researchers':
            get_plot_comparison_agg_methods(ax, method, researchers_sum_separated, researchers_mean_separated, researchers_mean_imp_separated)
            st.table(results_researchers)
        elif recommendations == 'Calls':
            get_plot_comparison_agg_methods(ax, method, calls_sum_separated, calls_mean_separated, calls_mean_imp_separated)
            st.table(results_calls)

        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(15, 10))  
        method = methods[3]
        
        if recommendations == 'Researchers':
            get_plot_comparison_agg_methods(ax, method, researchers_sum_semiseparated, researchers_mean_semiseparated, researchers_mean_imp_semiseparated)
            st.table(results_researchers)
        elif recommendations == 'Calls':
            get_plot_comparison_agg_methods(ax, method, calls_sum_semiseparated, calls_mean_semiseparated, calls_mean_imp_semiseparated)
            st.table(results_calls)

        st.pyplot(fig)

with tab2:
    # Selección el método a comparar
    selectbox_key2 = st.empty()  # Genera un contenedor vacío para poder actualizar el selectbox más tarde
    with selectbox_key2:
        recommendations = st.selectbox('Select which recommendations to obtain:', ['Researchers', 'Calls'], key='selectbox_tab2')  # Añade una key única
    
    # Título de la aplicación
    st.title('Comparison of different methods when recommending {}'.format(recommendations))

    for agg_method in agg_methods:
        with st.columns(1)[0]:
            fig, ax = plt.subplots(figsize=(15, 10))  
            
            if recommendations == 'Researchers':
                if agg_method == 'sum':
                    get_plot_comparison_methods(ax, agg_method, researchers_sum_bert, researchers_sum_bhattacharyya, researchers_sum_separated, researchers_sum_semiseparated)
                    st.table(results_researchers)
                elif agg_method == 'mean':
                    get_plot_comparison_methods(ax, agg_method, researchers_mean_bert, researchers_mean_bhattacharyya, researchers_mean_separated, researchers_mean_semiseparated)
                    st.table(results_researchers)
                elif agg_method == 'mean_imp':
                    get_plot_comparison_methods(ax, agg_method, researchers_mean_imp_bert, researchers_mean_imp_bhattacharyya, researchers_mean_imp_separated, researchers_mean_imp_semiseparated)
                    st.table(results_researchers)
            elif recommendations == 'Calls':
                if agg_method == 'sum':
                    get_plot_comparison_methods(ax, agg_method, calls_sum_bert, calls_sum_bhattacharyya, calls_sum_separated, calls_sum_semiseparated)
                    st.table(results_calls)
                elif agg_method == 'mean':
                    get_plot_comparison_methods(ax, agg_method, calls_mean_bert, calls_mean_bhattacharyya, calls_mean_separated, calls_mean_semiseparated)
                    st.table(results_calls)
                elif agg_method == 'mean_imp':
                    get_plot_comparison_methods(ax, agg_method, calls_mean_imp_bert, calls_mean_imp_bhattacharyya, calls_mean_imp_separated, calls_mean_imp_semiseparated)
                    st.table(results_calls)

            st.pyplot(fig)
