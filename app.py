import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from funciones_recommendation_system import get_datasets
from funciones_plot_validation import obtain_data_agg_methods, get_plot_comparison_agg_methods, get_plot_comparison_methods

def convert_to_str(val):
    if isinstance(val, float):
        return str(int(val))  # Elimina el .0 convirtiendo a int primero
    return str(val)  # Deja los strings como están

# Función para cargar datos una vez
@st.cache_data(persist=True)
def load_data():
    path = '/export/data_ml4ds/AI4U/Datasets/'
    version_wp = '20240510'
    version_rp = '20240321'
    
    df_publications, df_projects, df_publications_researchers, df_projects_researchers, df_researchers, df_calls = get_datasets(path, version_wp, version_rp)
    df_project_publication_researcher = pd.concat([df_publications_researchers[['id_paper', 'id_researcher']], df_projects_researchers[['actID', 'id_researcher']].rename(columns={'actID':'id_paper'})], ignore_index=True)
    df_project_publication_researcher['id_paper'] = df_project_publication_researcher['id_paper'].apply(convert_to_str)
    df_project_publication_researcher['id_researcher'] = df_project_publication_researcher['id_researcher'].astype(str)
    
    return df_publications, df_projects, df_publications_researchers, df_projects_researchers, df_researchers, df_calls, df_project_publication_researcher

# Cargar datos una vez
df_publications, df_projects, df_publications_researchers, df_projects_researchers, df_researchers, df_calls, df_project_publication_researcher = load_data()

# Cargar el dataset de validación
df_val = pd.read_excel('/export/usuarios_ml4ds/mafuello/Github/recommendation_system_validation/validation_set.xlsx')
df_val['id_researcher'] = df_val['id_researcher'].astype(str)

# Listas con los métodos de agregación y métodos de recomendación
agg_methods = ['sum', 'mean', 'mean_imp']
methods = ['BERT', 'bhattacharyya', 'separated', 'semiseparated']

# Función para obtener datos agregados (cacheada)
@st.cache_data(persist=False)
def obtain_data_cached(df_val, agg_methods, method, df_researchers, df_calls, df_project_publication_researcher, num_publis, num_IP, combined):
    return obtain_data_agg_methods(df_val, agg_methods, method, df_researchers, df_calls, df_project_publication_researcher, num_publis, num_IP, combined)

# Obtener datos agregados una vez
num_publis = None
num_IP = None
combined = None

# data without filters
researchers_sum_bert, researchers_mean_bert, researchers_mean_imp_bert, calls_sum_bert, calls_mean_bert, calls_mean_imp_bert, scores_researchers_bert, scores_calls_bert = obtain_data_cached(df_val, agg_methods, 'BERT', df_researchers, df_calls,  df_project_publication_researcher, num_publis, num_IP, combined)
researchers_sum_bhattacharyya, researchers_mean_bhattacharyya, researchers_mean_imp_bhattacharyya, calls_sum_bhattacharyya, calls_mean_bhattacharyya, calls_mean_imp_bhattacharyya, scores_researchers_bhattacharyya, scores_calls_bhattacharyya = obtain_data_cached(df_val, agg_methods, 'bhattacharyya', df_researchers, df_calls, df_project_publication_researcher, num_publis, num_IP, combined)
researchers_sum_separated, researchers_mean_separated, researchers_mean_imp_separated, calls_sum_separated, calls_mean_separated, calls_mean_imp_separated, scores_researchers_separated, scores_calls_separated = obtain_data_cached(df_val, agg_methods, 'separated', df_researchers, df_calls, df_project_publication_researcher, num_publis, num_IP, combined)
researchers_sum_semiseparated, researchers_mean_semiseparated, researchers_mean_imp_semiseparated, calls_sum_semiseparated, calls_mean_semiseparated, calls_mean_imp_semiseparated, scores_researchers_semiseparated, scores_calls_semiseparated = obtain_data_cached(df_val, agg_methods, 'semiseparated', df_researchers, df_calls, df_project_publication_researcher, num_publis, num_IP, combined)

# data with filters
researchers_sum_bert_num_publis, researchers_mean_bert_num_publis, researchers_mean_imp_bert_num_publis, calls_sum_bert_num_publis, calls_mean_bert_num_publis, calls_mean_imp_bert_num_publis, scores_researchers_bert_num_publis, scores_calls_bert_num_publis = obtain_data_cached(df_val, agg_methods, 'BERT', df_researchers, df_calls,  df_project_publication_researcher, num_publis=True, num_IP=False, combined=False)
researchers_sum_bhattacharyya_num_publis, researchers_mean_bhattacharyya_num_publis, researchers_mean_imp_bhattacharyya_num_publis, calls_sum_bhattacharyya_num_publis, calls_mean_bhattacharyya_num_publis, calls_mean_imp_bhattacharyya_num_publis, scores_researchers_bhattacharyya_num_publis, scores_calls_bhattacharyya_num_publis = obtain_data_cached(df_val, agg_methods, 'bhattacharyya', df_researchers, df_calls, df_project_publication_researcher, num_publis=True, num_IP=False, combined=False)
researchers_sum_separated_num_publis, researchers_mean_separated_num_publis, researchers_mean_imp_separated_num_publis, calls_sum_separated_num_publis, calls_mean_separated_num_publis, calls_mean_imp_separated_num_publis, scores_researchers_separated_num_publis, scores_calls_separated_num_publis = obtain_data_cached(df_val, agg_methods, 'separated', df_researchers, df_calls, df_project_publication_researcher, num_publis=True, num_IP=False, combined=False)
researchers_sum_semiseparated_num_publis, researchers_mean_semiseparated_num_publis, researchers_mean_imp_semiseparated_num_publis, calls_sum_semiseparated_num_publis, calls_mean_semiseparated_num_publis, calls_mean_imp_semiseparated_num_publis, scores_researchers_semiseparated_num_publis, scores_calls_semiseparated_num_publis = obtain_data_cached(df_val, agg_methods, 'semiseparated', df_researchers, df_calls, df_project_publication_researcher, num_publis=True, num_IP=False, combined=False)

researchers_sum_bert_num_IP, researchers_mean_bert_num_IP, researchers_mean_imp_bert_num_IP, calls_sum_bert_num_IP, calls_mean_bert_num_IP, calls_mean_imp_bert_num_IP, scores_researchers_bert_num_IP, scores_calls_bert_num_IP = obtain_data_cached(df_val, agg_methods, 'BERT', df_researchers, df_calls,  df_project_publication_researcher, num_publis=False, num_IP=True, combined=False)
researchers_sum_bhattacharyya_num_IP, researchers_mean_bhattacharyya_num_IP, researchers_mean_imp_bhattacharyya_num_IP, calls_sum_bhattacharyya_num_IP, calls_mean_bhattacharyya_num_IP, calls_mean_imp_bhattacharyya_num_IP, scores_researchers_bhattacharyya_num_IP, scores_calls_bhattacharyya_num_IP = obtain_data_cached(df_val, agg_methods, 'bhattacharyya', df_researchers, df_calls, df_project_publication_researcher, num_publis=False, num_IP=True, combined=False)
researchers_sum_separated_num_IP, researchers_mean_separated_num_IP, researchers_mean_imp_separated_num_IP, calls_sum_separated_num_IP, calls_mean_separated_num_IP, calls_mean_imp_separated_num_IP, scores_researchers_separated_num_IP, scores_calls_separated_num_IP = obtain_data_cached(df_val, agg_methods, 'separated', df_researchers, df_calls, df_project_publication_researcher, num_publis=False, num_IP=True, combined=False)
researchers_sum_semiseparated_num_IP, researchers_mean_semiseparated_num_IP, researchers_mean_imp_semiseparated_num_IP, calls_sum_semiseparated_num_IP, calls_mean_semiseparated_num_IP, calls_mean_imp_semiseparated_num_IP, scores_researchers_semiseparated_num_IP, scores_calls_semiseparated_num_IP = obtain_data_cached(df_val, agg_methods, 'semiseparated', df_researchers, df_calls, df_project_publication_researcher, num_publis=False, num_IP=True, combined=False)

researchers_sum_bert_combined, researchers_mean_bert_combined, researchers_mean_imp_bert_combined, calls_sum_bert_combined, calls_mean_bert_combined, calls_mean_imp_bert_combined, scores_researchers_bert_combined, scores_calls_bert_combined = obtain_data_cached(df_val, agg_methods, 'BERT', df_researchers, df_calls,  df_project_publication_researcher, num_publis=False, num_IP=False, combined=True)
researchers_sum_bhattacharyya_combined, researchers_mean_bhattacharyya_combined, researchers_mean_imp_bhattacharyya_combined, calls_sum_bhattacharyya_combined, calls_mean_bhattacharyya_combined, calls_mean_imp_bhattacharyya_combined, scores_researchers_bhattacharyya_combined, scores_calls_bhattacharyya_combined = obtain_data_cached(df_val, agg_methods, 'bhattacharyya', df_researchers, df_calls, df_project_publication_researcher, num_publis=False, num_IP=False, combined=True)
researchers_sum_separated_combined, researchers_mean_separated_combined, researchers_mean_imp_separated_combined, calls_sum_separated_combined, calls_mean_separated_combined, calls_mean_imp_separated_combined, scores_researchers_separated_combined, scores_calls_separated_combined = obtain_data_cached(df_val, agg_methods, 'separated', df_researchers, df_calls, df_project_publication_researcher, num_publis=False, num_IP=False, combined=True)
researchers_sum_semiseparated_combined, researchers_mean_semiseparated_combined, researchers_mean_imp_semiseparated_combined, calls_sum_semiseparated_combined, calls_mean_semiseparated_combined, calls_mean_imp_semiseparated_combined, scores_researchers_semiseparated_combined, scores_calls_semiseparated_combined = obtain_data_cached(df_val, agg_methods, 'semiseparated', df_researchers, df_calls, df_project_publication_researcher, num_publis=False, num_IP=False, combined=True)


# Crear los DataFrames de resultados
results_researchers_bert = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_bert)
results_calls_bert = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_bert)

results_researchers_bert_num_publis = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_bert_num_publis)
results_calls_bert_num_publis = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_bert_num_publis)

results_researchers_bert_num_IP = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_bert_num_IP)
results_calls_bert_num_IP = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_bert_num_IP)

results_researchers_bert_combined = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_bert_combined)
results_calls_bert_combined = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_bert_combined)


results_researchers_bhattacharyya = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_bhattacharyya)
results_calls_bhattacharyya = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_bhattacharyya)

results_researchers_bhattacharyya_num_publis = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_bhattacharyya_num_publis)
results_calls_bhattacharyya_num_publis = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_bhattacharyya_num_publis)

results_researchers_bhattacharyya_num_IP = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_bhattacharyya_num_IP)
results_calls_bhattacharyya_num_IP = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_bhattacharyya_num_IP)

results_researchers_bhattacharyya_combined = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_bhattacharyya_combined)
results_calls_bhattacharyya_combined = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_bhattacharyya_combined)


results_researchers_separated = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_separated)
results_calls_separated = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_separated)

results_researchers_separated_num_publis = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_separated_num_publis)
results_calls_separated_num_publis = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_separated_num_publis)

results_researchers_separated_num_IP = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_separated_num_IP)
results_calls_separated_num_IP = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_separated_num_IP)

results_researchers_separated_combined = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_separated_combined)
results_calls_separated_combined = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_separated_combined)


results_researchers_semiseparated = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_semiseparated)
results_calls_semiseparated = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_semiseparated)

results_researchers_semiseparated_num_publis = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_semiseparated_num_publis)
results_calls_semiseparated_num_publis = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_semiseparated_num_publis)

results_researchers_semiseparated = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_semiseparated)
results_calls_semiseparated = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_semiseparated)

results_researchers_semiseparated_num_IP = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_semiseparated_num_IP)
results_calls_semiseparated_num_IP = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_semiseparated_num_IP)

results_researchers_semiseparated_combined = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_department'], data=scores_researchers_semiseparated_combined)
results_calls_semiseparated_combined = pd.DataFrame(index=agg_methods, columns=['score_similarity', 'score_position', 'score_cluster'], data=scores_calls_semiseparated_combined)


# Tabs para comparar métodos de agregación y métodos de recomendación
tab1, tab2, tab3, tab4 = st.tabs(["Validation researchers analysis", "Compare aggregation methods", "Compare recommendations methods", "Compare Filters - Aggregation Methods"])

with tab1:
    df_plot = df_val.merge(df_researchers, on='id_researcher')[['id_researcher', 'no_publis', 'Projects_IP', 'Projects_no_IP']]

    st.title('Analysis of number of publications and projects')

    col1, col2 = st.columns([3, 1])
    with col1:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 30))

        # Histograma num publis
        ax1.hist(df_plot['no_publis'].dropna(), bins=60, color='skyblue', edgecolor='black')
        ax1.set_title('Number of Publications distribution', fontsize=30)
        ax1.set_xlabel('Number of publications', fontsize=26)
        ax1.set_ylabel('Frequency', fontsize=26)
        ax1.tick_params(axis='both', which='major', labelsize=20)

        # Histograma num projects IP
        ax2.hist(df_plot['Projects_IP'].dropna(), bins=60, color='skyblue', edgecolor='black')
        ax2.set_title('Number of Projects as Principal Investigator distribution', fontsize=30)
        ax2.set_xlabel('Number of Projects IP', fontsize=26)
        ax2.set_ylabel('Frequency', fontsize=26)
        ax2.tick_params(axis='both', which='major', labelsize=20)

        # Histograma num projects no IP
        ax3.hist(df_plot['Projects_no_IP'].dropna(), bins=60, color='skyblue', edgecolor='black')
        ax3.set_title('Number of Projects not as Principal Investigator distribution', fontsize=30)
        ax3.set_xlabel('Number of Projects no IP', fontsize=26)
        ax3.set_ylabel('Frequency', fontsize=26)
        ax3.tick_params(axis='both', which='major', labelsize=20)

        st.pyplot(fig)

    with col2:
        stats_no_publis = df_plot['no_publis'].describe().drop(['min', 'max', 'count']).to_frame().rename(columns={'no_publis': 'Number of Publications'})
        stats_projects_ip = df_plot['Projects_IP'].describe().drop(['min', 'max', 'count']).to_frame().rename(columns={'Projects_IP': 'Number of Projects IP'})
        stats_projects_no_ip = df_plot['Projects_no_IP'].describe().drop(['min', 'max', 'count']).to_frame().rename(columns={'Projects_no_IP': 'Number of Projects no IP'})
    
        st.table(stats_no_publis)
        st.table(stats_projects_ip)
        st.table(stats_projects_no_ip)
with tab2:
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
            st.table(results_researchers_bert)
        elif recommendations == 'Calls':
            get_plot_comparison_agg_methods(ax, method, calls_sum_bert, calls_mean_bert, calls_mean_imp_bert)
            st.table(results_calls_bert)

        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(15, 10))  
        method = methods[1]
        
        if recommendations == 'Researchers':
            get_plot_comparison_agg_methods(ax, method, researchers_sum_bhattacharyya, researchers_mean_bhattacharyya, researchers_mean_imp_bhattacharyya)
            st.table(results_researchers_bhattacharyya)
        elif recommendations == 'Calls':
            get_plot_comparison_agg_methods(ax, method, calls_sum_bhattacharyya, calls_mean_bhattacharyya, calls_mean_imp_bhattacharyya)
            st.table(results_calls_bhattacharyya)

        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots(figsize=(15, 10))  
        method = methods[2]
        
        if recommendations == 'Researchers':
            get_plot_comparison_agg_methods(ax, method, researchers_sum_separated, researchers_mean_separated, researchers_mean_imp_separated)
            st.table(results_researchers_separated)
        elif recommendations == 'Calls':
            get_plot_comparison_agg_methods(ax, method, calls_sum_separated, calls_mean_separated, calls_mean_imp_separated)
            st.table(results_calls_separated)

        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(15, 10))  
        method = methods[3]
        
        if recommendations == 'Researchers':
            get_plot_comparison_agg_methods(ax, method, researchers_sum_semiseparated, researchers_mean_semiseparated, researchers_mean_imp_semiseparated)
            st.table(results_researchers_semiseparated)
        elif recommendations == 'Calls':
            get_plot_comparison_agg_methods(ax, method, calls_sum_semiseparated, calls_mean_semiseparated, calls_mean_imp_semiseparated)
            st.table(results_calls_semiseparated)

        st.pyplot(fig)
with tab3:
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
                elif agg_method == 'mean':
                    get_plot_comparison_methods(ax, agg_method, researchers_mean_bert, researchers_mean_bhattacharyya, researchers_mean_separated, researchers_mean_semiseparated)
                elif agg_method == 'mean_imp':
                    get_plot_comparison_methods(ax, agg_method, researchers_mean_imp_bert, researchers_mean_imp_bhattacharyya, researchers_mean_imp_separated, researchers_mean_imp_semiseparated)
            
            elif recommendations == 'Calls':
                if agg_method == 'sum':
                    get_plot_comparison_methods(ax, agg_method, calls_sum_bert, calls_sum_bhattacharyya, calls_sum_separated, calls_sum_semiseparated)
                elif agg_method == 'mean':
                    get_plot_comparison_methods(ax, agg_method, calls_mean_bert, calls_mean_bhattacharyya, calls_mean_separated, calls_mean_semiseparated)
                elif agg_method == 'mean_imp':
                    get_plot_comparison_methods(ax, agg_method, calls_mean_imp_bert, calls_mean_imp_bhattacharyya, calls_mean_imp_separated, calls_mean_imp_semiseparated)

            st.pyplot(fig)
with tab4:
    print('tab 4')
    # Selección el método a comparar
    selectbox_key1 = st.empty()  # Genera un contenedor vacío para poder actualizar el selectbox más tarde
    with selectbox_key1:
        recommendations = st.selectbox('Select which recommendations to obtain:', ['Researchers', 'Calls'], key='selectbox_tab4')  # Añade una key única

    selectbox_key2 = st.empty()  # Genera un contenedor vacío para poder actualizar el selectbox más tarde
    with selectbox_key2:
        method = st.selectbox('Select a method to obtain recommendations:', methods, key='selectbox_methods')  # Añade una key única

    # Título de la aplicación
    st.title('Comparison of the effects of different filters on the aggegation methods when recommending {} with {}'.format(recommendations, method))
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(15, 10))  
        
        if recommendations == 'Researchers':
            if method == 'BERT':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_bert, researchers_mean_bert, researchers_mean_imp_bert)
                st.table(results_researchers_bert)
            
            if method == 'bhattacharyya':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_bhattacharyya, researchers_mean_bhattacharyya, researchers_mean_imp_bhattacharyya)
                st.table(results_researchers_bhattacharyya)
            
            if method == 'separated':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_separated, researchers_mean_separated, researchers_mean_imp_separated)
                st.table(results_researchers_separated)
            
            if method =='semiseparated':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_semiseparated, researchers_mean_semiseparated, researchers_mean_imp_semiseparated)
                st.table(results_researchers_semiseparated)

        elif recommendations == 'Calls':
            if method == 'BERT':
                get_plot_comparison_agg_methods(ax, method, calls_sum_bert, calls_mean_bert, calls_mean_imp_bert)
                st.table(results_calls_bert)
            
            if method == 'bhattacharyya':
                get_plot_comparison_agg_methods(ax, method, calls_sum_bhattacharyya, calls_mean_bhattacharyya, calls_mean_imp_bhattacharyya)
                st.table(results_calls_bhattacharyya)
            
            if method == 'separated':
                get_plot_comparison_agg_methods(ax, method, calls_sum_separated, calls_mean_separated, calls_mean_imp_separated)
                st.table(results_calls_separated)
            
            if method =='semiseparated':
                get_plot_comparison_agg_methods(ax, method, calls_sum_semiseparated, calls_mean_semiseparated, calls_mean_imp_semiseparated)
                st.table(results_calls_semiseparated)

        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(15, 10))          
        if recommendations == 'Researchers':
            if method == 'BERT':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_bert_num_publis, researchers_mean_bert_num_publis, researchers_mean_imp_bert_num_publis)
                st.table(results_researchers_bert_num_publis)
            
            if method == 'bhattacharyya':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_bhattacharyya_num_publis, researchers_mean_bhattacharyya_num_publis, researchers_mean_imp_bhattacharyya_num_publis)
                st.table(results_researchers_bhattacharyya_num_publis)
            
            if method == 'separated':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_separated_num_publis, researchers_mean_separated_num_publis, researchers_mean_imp_separated_num_publis)
                st.table(results_researchers_separated_num_publis)
            
            if method =='semiseparated':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_semiseparated_num_publis, researchers_mean_semiseparated_num_publis, researchers_mean_imp_semiseparated_num_publis)
                st.table(results_researchers_semiseparated_num_publis)

        elif recommendations == 'Calls':
            if method == 'BERT':
                get_plot_comparison_agg_methods(ax, method, calls_sum_bert_num_publis, calls_mean_bert_num_publis, calls_mean_imp_bert_num_publis)
                st.table(results_calls_bert_num_publis)
            
            if method == 'bhattacharyya':
                get_plot_comparison_agg_methods(ax, method, calls_sum_bhattacharyya_num_publis, calls_mean_bhattacharyya_num_publis, calls_mean_imp_bhattacharyya_num_publis)
                st.table(results_calls_bhattacharyya_num_publis)
            
            if method == 'separated':
                get_plot_comparison_agg_methods(ax, method, calls_sum_separated_num_publis, calls_mean_separated_num_publis, calls_mean_imp_separated_num_publis)
                st.table(results_calls_separated_num_publis)
            
            if method =='semiseparated':
                get_plot_comparison_agg_methods(ax, method, calls_sum_semiseparated_num_publis, calls_mean_semiseparated_num_publis, calls_mean_imp_semiseparated_num_publis)
                st.table(results_calls_semiseparated_num_publis)

        st.pyplot(fig)
    
    with col3:
        fig, ax = plt.subplots(figsize=(15, 10))  
        if recommendations == 'Researchers':
            if method == 'BERT':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_bert_num_IP, researchers_mean_bert_num_IP, researchers_mean_imp_bert_num_IP)
                st.table(results_researchers_bert_num_IP)
            
            if method == 'bhattacharyya':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_bhattacharyya_num_IP, researchers_mean_bhattacharyya_num_IP, researchers_mean_imp_bhattacharyya_num_IP)
                st.table(results_researchers_bhattacharyya_num_IP)
            
            if method == 'separated':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_separated_num_IP, researchers_mean_separated_num_IP, researchers_mean_imp_separated_num_IP)
                st.table(results_researchers_separated_num_IP)
            
            if method =='semiseparated':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_semiseparated_num_IP, researchers_mean_semiseparated_num_IP, researchers_mean_imp_semiseparated_num_IP)
                st.table(results_researchers_semiseparated_num_IP)

        elif recommendations == 'Calls':
            if method == 'BERT':
                get_plot_comparison_agg_methods(ax, method, calls_sum_bert_num_IP, calls_mean_bert_num_IP, calls_mean_imp_bert_num_IP)
                st.table(results_calls_bert_num_IP)
            
            if method == 'bhattacharyya':
                get_plot_comparison_agg_methods(ax, method, calls_sum_bhattacharyya_num_IP, calls_mean_bhattacharyya_num_IP, calls_mean_imp_bhattacharyya_num_IP)
                st.table(results_calls_bhattacharyya_num_IP)
            
            if method == 'separated':
                get_plot_comparison_agg_methods(ax, method, calls_sum_separated_num_IP, calls_mean_separated_num_IP, calls_mean_imp_separated_num_IP)
                st.table(results_calls_separated_num_IP)
            
            if method =='semiseparated':
                get_plot_comparison_agg_methods(ax, method, calls_sum_semiseparated_num_IP, calls_mean_semiseparated_num_IP, calls_mean_imp_semiseparated_num_IP)
                st.table(results_calls_semiseparated_num_IP)


            st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(15, 10))  
        
        if recommendations == 'Researchers':
            if method == 'BERT':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_bert_combined, researchers_mean_bert_combined, researchers_mean_imp_bert_combined)
                st.table(results_researchers_bert_combined)
            
            if method == 'bhattacharyya':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_bhattacharyya_combined, researchers_mean_bhattacharyya_combined, researchers_mean_imp_bhattacharyya_combined)
                st.table(results_researchers_bhattacharyya_combined)
            
            if method == 'separated':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_separated_combined, researchers_mean_separated_combined, researchers_mean_imp_separated_combined)
                st.table(results_researchers_separated_combined)
            
            if method =='semiseparated':
                get_plot_comparison_agg_methods(ax, method, researchers_sum_semiseparated_combined, researchers_mean_semiseparated_combined, researchers_mean_imp_semiseparated_combined)
                st.table(results_researchers_semiseparated_combined)

        elif recommendations == 'Calls':
            if method == 'BERT':
                get_plot_comparison_agg_methods(ax, method, calls_sum_bert_combined, calls_mean_bert_combined, calls_mean_imp_bert_combined)
                st.table(results_calls_bert_combined)
            
            if method == 'bhattacharyya':
                get_plot_comparison_agg_methods(ax, method, calls_sum_bhattacharyya_combined, calls_mean_bhattacharyya_combined, calls_mean_imp_bhattacharyya_combined)
                st.table(results_calls_bhattacharyya_combined)
            
            if method == 'separated':
                get_plot_comparison_agg_methods(ax, method, calls_sum_separated_combined, calls_mean_separated_combined, calls_mean_imp_separated_combined)
                st.table(results_calls_separated_combined)
            
            if method =='semiseparated':
                get_plot_comparison_agg_methods(ax, method, calls_sum_semiseparated_combined, calls_mean_semiseparated_combined, calls_mean_imp_semiseparated_combined)
                st.table(results_calls_semiseparated_combined)

        st.pyplot(fig)




