import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from funciones_recommendation_system import get_datasets
from fuciones_plot_validation import obtain_data_agg_methods, get_plot_comparison_agg_methods, get_plot_comparison_methods
# load databases
path = '/export/data_ml4ds/AI4U/Datasets/'
version_wp = '20240510'
version_rp = '20240321'

df_publications, df_projects, df_publications_researchers,df_projects_researchers, df_researchers, df_calls = get_datasets(path, version_wp, version_rp)

# load the validation set
df_val = pd.read_excel('/export/usuarios_ml4ds/mafuello/Github/recommendation_system_validation/validation_set.xlsx')
df_val['id_researcher'] = df_val['id_researcher'].astype(str)

# load the lists with the data to plot
agg_methods = ['sum', 'mean', 'mean_imp'] 
methods = ['BERT', 'bhattacharyya', 'separated', 'semiseparated']

researchers_sum_bert, researchers_mean_bert, researchers_mean_imp_bert, calls_sum_bert, calls_mean_bert, calls_mean_imp_bert = obtain_data_agg_methods(df_val, agg_methods, 'BERT', df_researchers, df_calls)
researchers_sum_bhattacharyya, researchers_mean_bhattacharyya, researchers_mean_imp_bhattacharyya, calls_sum_bhattacharyya, calls_mean_bhattacharyya, calls_mean_imp_bhattacharyya = obtain_data_agg_methods(df_val, agg_methods, 'bhattacharyya', df_researchers, df_calls)
researchers_sum_separated, researchers_mean_separated, researchers_mean_imp_separated, calls_sum_separated, calls_mean_separated, calls_mean_imp_separated = obtain_data_agg_methods(df_val, agg_methods, 'separated', df_researchers, df_calls)
researchers_sum_semiseparated, researchers_mean_semiseparated, researchers_mean_imp_semiseparated, calls_sum_semiseparated, calls_mean_semiseparated, calls_mean_imp_semiseparated = obtain_data_agg_methods(df_val, agg_methods, 'semiseparated', df_researchers, df_calls)

tab1, tab2 = st.tabs(["Comparar métodos de agregación", "Comparar métodos de recomendación"])

with tab1:
    # Selección el método a comparar
    selectbox_key1 = st.empty()  # Genera un contenedor vacío para poder actualizar el selectbox más tarde
    with selectbox_key1:
        recommendations = st.selectbox('Select which recommendations to obtain:', ['Researchers', 'Calls'], key='selectbox_tab1')  # Añade una key única
        

    # Título de la aplicación
    st.title('Comparison of different aggregation methods  when recommending {}'.format(recommendations))
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(15, 10))  
        method = methods[0]
        
        if recommendations == 'Researchers':
            get_plot_comparison_agg_methods(ax, method, researchers_sum_bert, researchers_mean_bert, researchers_mean_imp_bert)
        elif recommendations == 'Calls':
            get_plot_comparison_agg_methods(ax, method, calls_sum_bert, calls_mean_bert, calls_mean_imp_bert)

        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(15, 10))  
        method = methods[1]
        
        if recommendations == 'Researchers':
            get_plot_comparison_agg_methods(ax, method, researchers_sum_bhattacharyya, researchers_mean_bhattacharyya, researchers_mean_imp_bhattacharyya)
        elif recommendations == 'Calls':
            get_plot_comparison_agg_methods(ax, method, calls_sum_bhattacharyya, calls_mean_bhattacharyya, calls_mean_imp_bhattacharyya)    
        
        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots(figsize=(15, 10))  
        method = methods[2]
        
        if recommendations == 'Researchers':
            get_plot_comparison_agg_methods(ax, method, researchers_sum_separated, researchers_mean_separated, researchers_mean_imp_separated)
        elif recommendations == 'Calls':
            get_plot_comparison_agg_methods(ax, method, calls_sum_separated, calls_mean_separated, calls_mean_imp_separated)    
        
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(15, 10))  
        method = methods[3]
        
        if recommendations == 'Researchers':
            get_plot_comparison_agg_methods(ax, method, researchers_sum_semiseparated, researchers_mean_semiseparated, researchers_mean_imp_semiseparated)
        elif recommendations == 'Calls':
            get_plot_comparison_agg_methods(ax, method, calls_sum_semiseparated, calls_mean_semiseparated, calls_mean_imp_semiseparated)
        
        st.pyplot(fig)

with tab2:
    # Selección el método a comparar
    selectbox_key2 = st.empty()  # Genera un contenedor vacío para poder actualizar el selectbox más tarde
    with selectbox_key2:
        recommendations = st.selectbox('Select which recommendations to obtain:', ['Researchers', 'Calls'], key='selectbox_tab2')  # Añade una key única
    
    # Título de la aplicación
    st.title('Comparison of different methods  when recommending {}'.format(recommendations))

    with st.columns(1)[0]:
        fig, ax = plt.subplots(figsize=(15, 10))  
        agg_method = agg_methods[0]
        
        if recommendations == 'Researchers':
            get_plot_comparison_methods(ax, agg_method, researchers_sum_bert, researchers_sum_bhattacharyya, researchers_sum_separated, researchers_sum_semiseparated)
        elif recommendations == 'Calls':
            get_plot_comparison_methods(ax, agg_method, calls_sum_bert, calls_sum_bhattacharyya, calls_sum_separated, calls_sum_semiseparated)

        st.pyplot(fig)

    with st.columns(1)[0]:
        fig, ax = plt.subplots(figsize=(15, 10))  
        agg_method = agg_methods[1]
        
        if recommendations == 'Researchers':
            get_plot_comparison_methods(ax, agg_method, researchers_mean_bert, researchers_mean_bhattacharyya, researchers_mean_separated, researchers_mean_semiseparated)
        elif recommendations == 'Calls':
            get_plot_comparison_methods(ax, agg_method, calls_mean_bert, calls_mean_bhattacharyya, calls_mean_separated, calls_mean_semiseparated)

        st.pyplot(fig)

    with st.columns(1)[0]:
        fig, ax = plt.subplots(figsize=(15, 10))  
        agg_method = agg_methods[2]
        
        if recommendations == 'Researchers':
            get_plot_comparison_methods(ax, agg_method, researchers_mean_imp_bert, researchers_mean_imp_bhattacharyya, researchers_mean_imp_separated, researchers_mean_imp_semiseparated)
        elif recommendations == 'Calls':
            get_plot_comparison_methods(ax, agg_method, calls_mean_imp_bert, calls_mean_imp_bhattacharyya, calls_mean_imp_separated, calls_mean_imp_semiseparated)

        st.pyplot(fig)