from funciones_score_computation import get_score_similarity, get_score_position, get_score_cluster
from funciones_recommendation_system import get_datasets, match_researcher_call, recommendation_system_researcher_call, match_call_researcher, recommendation_system_call_researcher
import numpy as np
from matplotlib import pyplot as plt 

def get_dictionaries_compare_agg_methods(df, agg_methods, method, df_researchers, df_calls):
    '''
    Function for obtaining the dictionary of the scores and the postions results for recommendations of calls or researchers

    df -> Dataset containing the validation calls and researchers
    agg_methods -> List containing the different aggregation methods to compare
    method -> Method to obtain the recommendations
    df_researchers, df_calls -> Datasets containing the information regarding researchers and calls
    '''
    dict_results_researchers = {}
    dict_results_calls = {}
    scores_researchers = []
    scores_calls = []
    
    errores = []
    for agg_method in agg_methods:
        positions_researchers, positions_calls, scores_similarity_researchers, scores_similarity_calls, scores_position_researchers, scores_position_calls, scores_department_researchers,  scores_cluster_calls = [], [], [], [], [], [], [], []
        for i in df.index:
            try:
                invID = df['id_researcher'][i]
                call = df['Línea prioritia/panel/topic'][i]   

                ranking_researchers = recommendation_system_call_researcher(method=method, agg_method=agg_method, call=call, researchers=df_researchers)
                ranking_calls = recommendation_system_researcher_call(method=method, agg_method=agg_method, researcher=invID, calls=df_calls)

                # score similarity
                scores_similarity_researchers.append(get_score_similarity(ranking_researchers, invID, call))
                scores_similarity_calls.append(get_score_similarity(ranking_calls, invID, call))

                # score position
                scores_position_researchers.append(get_score_position(ranking_researchers, invID, call))
                scores_position_calls.append(get_score_position(ranking_calls, invID, call))

                # score department/cluster
                scores_department_researchers.append(get_score_cluster(ranking_researchers, invID, call))
                scores_cluster_calls.append(get_score_cluster(ranking_calls, invID, call))

                # get position
                indice_valor_exacto = ranking_researchers.loc[ranking_researchers['id_researcher'] == invID].index[0]
                ranking_aux = ranking_researchers.iloc[:indice_valor_exacto + 1]
                positions_researchers.append(ranking_aux.shape[0])

                indice_valor_exacto = ranking_calls.loc[ranking_calls['Call'] == call].index[0]
                ranking_aux = ranking_calls.iloc[:indice_valor_exacto + 1]
                positions_calls.append(ranking_aux.shape[0])

            except Exception as e:
                print(f'Error: {e}')
                #errores.append(i)

            
        scores_researchers.append([np.mean(scores_similarity_researchers), np.mean(scores_position_researchers), np.mean(scores_department_researchers)])
        scores_calls.append([np.mean(scores_similarity_calls), np.mean(scores_position_calls), np.mean(scores_cluster_calls)])

        dict_results_researchers['{}_{}'.format(method, agg_method)] = contar_repeticiones(positions_researchers)
        dict_results_calls['{}_{}'.format(method, agg_method)] = contar_repeticiones(positions_calls)

    return scores_researchers, scores_calls, dict_results_researchers, dict_results_calls

def obtain_data_agg_methods(df, agg_methods, method, df_researchers, df_calls):
    '''    
    Function for obtaining the lists with the data to plot for comparing the different aggregation methods

    df -> Dataset containing the validation calls and researchers
    agg_methods -> List containing the different aggregation methods to compare
    method -> Method to obtain the recommendations

    '''
    # obtain the dictionaries 
    scores_researchers, scores_calls, dict_results_researchers, dict_results_calls = get_dictionaries_compare_agg_methods(df, agg_methods, method, df_researchers, df_calls)

    researchers_sum = dict_results_researchers['{}_sum'.format(method)]
    researchers_mean = dict_results_researchers['{}_mean'.format(method)]
    researchers_mean_imp = dict_results_researchers['{}_mean_imp'.format(method)]

    calls_sum = dict_results_calls['{}_sum'.format(method)]
    calls_mean = dict_results_calls['{}_mean'.format(method)]
    calls_mean_imp = dict_results_calls['{}_mean_imp'.format(method)]
    
    
    #return researchers_sum, calls_sum
    return researchers_sum, researchers_mean, researchers_mean_imp, calls_sum, calls_mean, calls_mean_imp

def get_plot_comparison_agg_methods(ax, method, sum, mean, mean_imp):
    '''
    Function for plotting the comparison graph of the different aggregation methods for a given method of recommendation

    ax -> plot 
    method -> desired method to plot
    sum, mean, mean_imp -> dictionaries containg the frequency of the ranking aparitions for each of the agg methods
    '''
    keys_sum, values_sum = list(sum.keys()), list(sum.values())
    keys_mean, values_mean = list(mean.keys()), list(mean.values())
    keys_mean_imp, values_mean_imp = list(mean_imp.keys()), list(mean_imp.values())

    cumulative_sum = np.cumsum(values_sum)
    cumulative_mean = np.cumsum(values_mean)
    cumulative_mean_imp = np.cumsum(values_mean_imp)

    ax.plot(keys_sum, cumulative_sum, marker='o', color='blue', label='{}_sum'.format(method))
    ax.plot(keys_mean, cumulative_mean, marker='s', color='red', label='{}_mean'.format(method))
    ax.plot(keys_mean_imp, cumulative_mean_imp, marker='^', color='green', label='{}_mean_imp'.format(method))

    ax.set_xlabel('Depth', fontsize=26)
    ax.set_ylabel('Cumulative Frequency', fontsize=26)
    ax.set_title('Cumulative Accuracy with {} method'.format(method), fontsize=30)
    ax.legend(fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)

def get_plot_comparison_methods(ax, agg_method, bert, bhattacharyya, separated, semiseparated):
    '''
    Function for plotting the comparison graph of the different methods of recommendation for a given aggregation method

    ax -> plot 
    agg_method -> desired aggregation method to plot
    bert, bhattacharyya, separated, semiseparated -> dictionaries containg the frequency of the ranking aparitions for each of the methods
    '''
    keys_BERT, values_BERT = list(bert.keys()), list(bert.values())
    keys_bhattacharyya, values_bhattacharyya = list(bhattacharyya.keys()), list(bhattacharyya.values())
    keys_separated, values_separated = list(separated.keys()), list(separated.values())
    keys_semiseparated, values_semiseparated = list(semiseparated.keys()), list(semiseparated.values())


    cumulative_BERT = np.cumsum(values_BERT)
    cumulative_bhattacharyya = np.cumsum(values_bhattacharyya)
    cumulative_separated = np.cumsum(values_separated)
    cumulative_semiseparated = np.cumsum(values_semiseparated)

    ax.plot(keys_BERT, cumulative_BERT, marker='o', color='blue', label='BERT_{}'.format(agg_method))
    ax.plot(keys_bhattacharyya, cumulative_bhattacharyya, marker='s', color='red', label='bhattacharyya_{}'.format(agg_method))
    ax.plot(keys_separated, cumulative_separated, marker='^', color='green', label='separated_{}'.format(agg_method))
    ax.plot(keys_semiseparated, cumulative_semiseparated, marker='+', color='orange', label='semiseparated_{}'.format(agg_method))

    ax.set_xlabel('Depth', fontsize=26)
    ax.set_ylabel('Cumulative Frequency', fontsize=26)
    ax.set_title('Cumulative Accuracy with {} aggregation method'.format(agg_method), fontsize=30)
    ax.legend(fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)


def contar_repeticiones(lista):
    '''
    Function for counting the repetitions of a number in a list

    lista -> lista containing the numbers
    '''
    diccionario = {}
    for numero in lista:
        if numero in diccionario:
            diccionario[numero] += 1
        else:
            diccionario[numero] = 1
    
    return {k: v for k, v in sorted(diccionario.items(), key=lambda item: item[0])}